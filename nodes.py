"""BSAI-ComfyUI-FastH3 — 快速生成 H3 视频（4 步 DMD2 蒸馏 FastH3）节点套件。

围绕 FastVideo FastH3 4-step 预览模型（33B 双模态 MiniMax-H3 的无数据 DMD2
蒸馏 + 90% VSA 稀疏注意力）封装的一站式 ComfyUI 节点：

  * BSAIFastH3Loader          FastH3 专用模型加载器（Dense 兼容 + 严格校验）
  * BSAIFastH3NativeVSA       FastH3 原生 VSA 稀疏注意力补丁
  * BSAIFastH3Timesteps       FastH3 精确时间步（显式训练阶梯, 默认 [999,749,500,250]）
  * BSAIFastH3EulerSampler    FastH3 Euler 4 步采样器（音视频双调度自适应）
  * BSAIFastH3VSAStats        VSA 命中统计（只读诊断）

全部节点仅通过 ModelPatcher.clone() / model_options 注入，不修改 ComfyUI
内部源码，升级无碍。
"""

import logging
import math
import os

import torch
import torch.nn.functional as F
from tqdm.auto import trange

import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths

from . import fast_h3_vsa as _vsa

# 兼容两种 Python 包导入形态（直接 import vs 以 custom_nodes 包导入）
try:
    from .fast_h3_vsa import vsa_stats, reset_vsa_stats
except ImportError:                                            # pragma: no cover
    from fast_h3_vsa import vsa_stats, reset_vsa_stats

SHIFT_V, SHIFT_A = 12.0, 3.0                                   # FastH3 视频/音频 flow shift

# ---------------------------------------------------------------------------
# 1) FastH3 专用模型加载器
# ---------------------------------------------------------------------------

_FAST_H3_MARKERS = ("fastvideo", "fasth3", "fast_h3", "4step")


def _is_fast_h3_filename(name):
    base = os.path.splitext(os.path.basename(name).lower())[0]
    return any(k in base for k in _FAST_H3_MARKERS)


class BSAIFastH3Loader:
    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("diffusion_models")
        return {"required": {
            "model": (sorted(models),),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                             {"advanced": True}),
            "strict_fast_h3_check": ("BOOLEAN", {
                "default": True,
                "label_on": "校验 FastH3 文件名",
                "label_off": "允许任意 H3 权重",
                "tooltip": "开启时仅接受文件名含 fastvideo/fasth3/_4step 的 4 步蒸馏权重，"
                           "避免误加载 50 步基座模型。"}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "BSAI/FastH3"
    DESCRIPTION = ("加载 FastVideo FastH3 4 步蒸馏权重（如 "
                   "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors）。"
                   "文件放入 ComfyUI/models/diffusion_models。")

    def load_model(self, model, weight_dtype="default", strict_fast_h3_check=True):
        if strict_fast_h3_check and not _is_fast_h3_filename(model):
            raise ValueError(
                f"[BSAI FastH3] {model} 文件名不含 FastH3/4 步标记（fastvideo/fasth3/_4step）。"
                "FastH3 使用 4 步蒸馏权重；如需强行加载请关闭 strict_fast_h3_check。")
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        if weight_dtype != "default":
            logging.warning("[BSAI FastH3] FastH3 权重通常为 int8_convrot 量化；"
                            "weight_dtype 选 default 最稳，fp8 仅在确需时使用。")
        path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        m = comfy.sd.load_diffusion_model(path, model_options=model_options)
        print(f"[BSAI FastH3 Loader] {model} | weight_dtype={weight_dtype} "
              f"| strict_fast_h3_check={strict_fast_h3_check}", flush=True)
        return (m,)


# ---------------------------------------------------------------------------
# 2) FastH3 原生 VSA 稀疏注意力
# ---------------------------------------------------------------------------

class BSAIFastH3NativeVSA:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {
                "default": True,
                "forceInput": True,
                "tooltip": "关闭时透传模型，不安装 VSA 补丁。可接 BSAI H3 MotionFix.vsa_enabled 自动驱动。"}),
            "video_keep_percent": ("FLOAT", {
                "default": 10.0, "min": 0.5, "max": 100.0, "step": 0.5,
                "forceInput": True,
                "tooltip": "视频 token 中保留精确注意力的 tile 百分比。FastVideo 官方约 10%。"
                           "越小越省显存/越快，越低细节损失越大。可接 BSAI H3 MotionFix.video_keep_percent 自动驱动。"}),
            "start_percent": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "采样进度在此之前的步运行 Dense（高温预热）。"}),
            "end_percent": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "采样进度在此之后运行 Dense。"}),
            "min_tokens": ("INT", {
                "default": 8192, "min": 0, "max": 1048576, "step": 256,
                "tooltip": "序列长度小于此值的注意力调用保持 Dense（短序列稀疏无收益）。"}),
            "sink_conditioning": (["exact_kv", "exact_kv_and_rows", "off"], {
                "default": "exact_kv",
                "tooltip": "exact_kv: 所有视频 query 都精确看到 text/audio/ref 条件行（~3% 开销）。"
                           "exact_kv_and_rows: 额外让条件 query 行也跑 Dense（音频更稳）。"
                           "off: 只保留视频块 top-k，内存最省但条件对齐变弱。"}),
            "backend": (["auto", "native", "torch"], {
                "default": "auto",
                "tooltip": "auto: 有 comfy_kitchen.sol_attn 用 native（Blackwell 内核），否则 torch。"
                           "native: 强制 comfy_kitchen 内核。torch: 纯 PyTorch 块稀疏（40 系可用）。"}),
            "strict_native_backend": ("BOOLEAN", {
                "default": True,
                "tooltip": "True: native 不可用时直接报错。False: 自动降级到 torch 稀疏。"}),
            "verbose": ("BOOLEAN", {"default": True,
                                    "tooltip": "打印每次 VSA 命中的形状与后端信息。"}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "apply_vsa"
    CATEGORY = "BSAI/FastH3"
    DESCRIPTION = ("FastH3 原生 VSA（Video Sparse Attention）：按 64-token tile 打分，"
                   "仅保留 top-k 视频块精确注意力，条件行保持精确。约省 90% 视频自注意力成本。")

    def apply_vsa(self, model, enabled, video_keep_percent, start_percent, end_percent,
                  min_tokens, sink_conditioning, backend, strict_native_backend,
                  verbose):
        if backend == "auto":
            backend = "native" if _vsa._ck is not None and hasattr(_vsa._ck, "sol_attn") else "torch"
        m = _vsa.apply_vsa(
            model, enabled=enabled, keep_percent=video_keep_percent, tau=None,
            start_percent=start_percent, end_percent=end_percent,
            min_tokens=min_tokens, sink_conditioning=sink_conditioning,
            backend=backend, strict_native_backend=strict_native_backend,
            verbose=verbose)
        return (m,)


# ---------------------------------------------------------------------------
# 3) FastH3 精确时间步（显式训练阶梯）
# ---------------------------------------------------------------------------

class BSAIFastH3Timesteps:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "ladder": ("STRING", {
                "default": "999,749,500,250", "multiline": False,
                "forceInput": True,
                "tooltip": "显式训练的 4 步阶梯（v0.2 卡片要求用训练跳点采样，勿用均匀网格）。"
                           "逗号分隔的 timestep（0-1000）。可接 BSAI H3 MotionFix.ladder 自动驱动。"}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "BSAI/FastH3"
    DESCRIPTION = ("把 FastH3 显式训练阶梯 timestep 换算成 SamplerCustomAdvanced 的 SIGMAS。"
                   "默认 [999,749,500,250]，末尾自动补 0（最终去噪）。")

    def get_sigmas(self, model, ladder="999,749,500,250"):
        values = []
        for part in str(ladder).replace("，", ",").replace("[", "").replace("]", "").split(","):
            part = part.strip()
            if part:
                values.append(float(part))
        if len(values) < 2:
            raise ValueError(f"[BSAI FastH3] 阶梯至少需要 2 个 timestep: {ladder!r}")
        ms = model.get_model_object("model_sampling")
        sig = []
        for t in values:
            s = float(ms.sigma(torch.tensor(t, dtype=torch.float32)))
            sig.append(s)
        sig.append(0.0)                                        # 最终一步 -> 干净 latent
        sigmas = torch.tensor(sig, dtype=torch.float32, device="cpu")
        print(f"[BSAI FastH3 Timesteps] ladder={values} -> sigmas="
              f"{[round(float(s), 4) for s in sigmas]}", flush=True)
        return (sigmas,)


# ---------------------------------------------------------------------------
# 4) FastH3 Euler 4 步采样器
# ---------------------------------------------------------------------------

def _time_shift_sigma(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return to * base / (1.0 + (to - 1.0) * base)


def _time_shift_slope(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return (to * (1.0 + (fr - 1.0) * base) ** 2) / (fr * (1.0 + (to - 1.0) * base) ** 2)


def _audio_sigma(sv, shift_v, shift_a):
    return _time_shift_sigma(sv, shift_v, shift_a)


def _audio_slope(sv, shift_v, shift_a):
    return _time_shift_slope(sv, shift_v, shift_a)


def _latent_shapes(model):
    """[video_shape, audio_shape] the sampler is packing over."""
    guider = getattr(model, "inner_model", model)
    conds = getattr(guider, "conds", None)
    if conds:
        for cond_list in conds.values():
            for c in (cond_list or []):
                mc = c.get("model_conds", {}) if isinstance(c, dict) else {}
                if "latent_shapes" in mc:
                    return mc["latent_shapes"].cond
    return None


def _model_sampling(model):
    for chain in (("inner_model", "inner_model", "model_sampling"),
                  ("inner_model", "model_sampling"),
                  ("model_sampling",)):
        o = model
        try:
            for a in chain:
                o = getattr(o, a)
        except AttributeError:
            continue
        if o is not None:
            return o
    return None


def _native_av_schedule(model):
    """True 时该 ComfyUI 由 ModelSamplingAV 原生处理 H3 音视频双调度，单调度 Euler 即可。"""
    ms = _model_sampling(model)
    if ms is None:
        return False
    if getattr(ms, "audio_shift", None) is not None:
        return True
    av = getattr(comfy.model_sampling, "ModelSamplingAV", None)
    return av is not None and isinstance(ms, av)


@torch.no_grad()
def _fast_h3_euler(model, x, sigmas, extra_args=None, callback=None, disable=None,
                   shift_video=SHIFT_V, shift_audio=SHIFT_A, schedule_mode="auto", **kwargs):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    _rms = lambda t: float(t.float().pow(2).mean().sqrt())

    if schedule_mode == "native" or (schedule_mode == "auto" and _native_av_schedule(model)):
        print(f"[BSAI FastH3 Euler] native ModelSamplingAV -> 单调度 Euler  "
              f"sigmas={[round(float(s), 4) for s in sigmas]}  x={tuple(x.shape)}", flush=True)
        for i in trange(len(sigmas) - 1, disable=disable):
            sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            d = (x - denoised) / sigmas[i]
            x = x + (sv_n - sv) * d
            print(f"[BSAI FastH3 step {i}] {sv:.4f}->{sv_n:.4f}  "
                  f"denoised_rms={_rms(denoised):.4f} x_rms={_rms(x):.4f}", flush=True)
            if callback is not None:
                callback({"i": i, "denoised": denoised, "x": x,
                          "sigma": sigmas[i], "sigma_hat": sigmas[i]})
        return x

    # 旧版 ComfyUI：视频/音频各自 flow 调度（video shift 12 / audio shift 3），分别推进
    shapes = _latent_shapes(model)
    if not shapes or len(shapes) < 2:
        raise RuntimeError(
            "BSAI FastH3 Euler 需要 MiniMax-H3 视频+音频 latent "
            "(EmptyMiniMaxH3LatentAV / MiniMaxH3ImageToVideo 输出)。")
    v_numel = math.prod(shapes[0][1:])
    a_numel = x.shape[-1] - v_numel
    print(f"[BSAI FastH3 Euler] legacy 双调度 (无 ModelSamplingAV)  "
          f"v_numel={v_numel} a_numel={a_numel} shapes={shapes}", flush=True)
    for i in trange(len(sigmas) - 1, disable=disable):
        sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        out = (x - denoised) / sigmas[i]
        xv, ov = x[..., :v_numel], out[..., :v_numel]
        xa, oa = x[..., v_numel:], out[..., v_numel:]
        xv = xv + (sv_n - sv) * ov
        sl = _audio_slope(max(sv, 1e-6), shift_video, shift_audio)
        xa = xa + (_audio_sigma(sv_n, shift_video, shift_audio)
                   - _audio_sigma(sv, shift_video, shift_audio)) * (oa / sl)
        x = torch.cat([xv, xa], dim=-1)
        print(f"[BSAI FastH3 step {i}] {sv:.4f}->{sv_n:.4f}  video_rms={_rms(xv):.4f} "
              f"audio_rms={_rms(xa):.4f} slope={sl:.4f}", flush=True)
        if callback is not None:
            callback({"i": i, "denoised": denoised, "x": x,
                      "sigma": sigmas[i], "sigma_hat": sigmas[i]})
    return x


class BSAIFastH3EulerSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shift_video": ("FLOAT", {
                "default": SHIFT_V, "min": 0.01, "max": 100.0, "step": 0.01,
                "tooltip": "视频流 flow shift（FastH3 官方 shift-12 整流调度）。"}),
            "shift_audio": ("FLOAT", {
                "default": SHIFT_A, "min": 0.01, "max": 100.0, "step": 0.01,
                "tooltip": "音频流 flow shift（FastH3 官方 shift-3）。"}),
            "schedule_mode": (["auto", "native", "legacy_dual"], {
                "default": "auto",
                "tooltip": "auto: 检测 ModelSamplingAV，有则单调度 Euler，否则音视频双调度。"
                           "native: 强制单调度。legacy_dual: 强制双调度。"}),
        }}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "BSAI/FastH3"
    DESCRIPTION = ("FastH3 4 步 Euler 采样器。ComfyUI 0.31+ 原生 ModelSamplingAV 下按"
                   "单调度推进；旧版自动按视频 shift 12 / 音频 shift 3 双调度推进。"
                   "接入 SamplerCustomAdvanced.sampler，与 FastH3 精确时间步搭配使用。")

    def get_sampler(self, shift_video=SHIFT_V, shift_audio=SHIFT_A, schedule_mode="auto"):
        sampler = comfy.samplers.KSAMPLER(
            lambda model, x, sigmas, extra_args=None, callback=None, disable=None, **kw:
            _fast_h3_euler(model, x, sigmas, extra_args=extra_args, callback=callback,
                           disable=disable, shift_video=shift_video,
                           shift_audio=shift_audio, schedule_mode=schedule_mode, **kw))
        return (sampler,)


# ---------------------------------------------------------------------------
# 5) VSA 命中统计（只读诊断）
# ---------------------------------------------------------------------------

class BSAIFastH3VSAStats:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("stats",)
    FUNCTION = "get_stats"
    CATEGORY = "BSAI/FastH3"
    OUTPUT_NODE = True
    DESCRIPTION = ("只读查看 FastH3 VSA 的命中统计（sparse/dense/native/torch/errors）。"
                   "sparse 计数 > 0 说明 VSA 真实参与了采样。")

    def get_stats(self, model):
        native_ok = _vsa._ck is not None and hasattr(_vsa._ck, "sol_attn")
        st = vsa_stats()
        cfg = model.model_options.get("transformer_options", {}).get("bsai_fasth3_vsa")
        lines = [
            "[BSAI FastH3 VSA Stats]",
            f"  native backend available : {native_ok}",
            f"  configured               : {cfg}",
            f"  sparse calls             : {st['sparse']}",
            f"  dense  calls             : {st['dense']}",
            f"  native / torch           : {st['native']} / {st['torch']}",
            f"  errors (dense fallback)  : {st['errors']}",
        ]
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "BSAIFastH3Loader": BSAIFastH3Loader,
    "BSAIFastH3NativeVSA": BSAIFastH3NativeVSA,
    "BSAIFastH3Timesteps": BSAIFastH3Timesteps,
    "BSAIFastH3EulerSampler": BSAIFastH3EulerSampler,
    "BSAIFastH3VSAStats": BSAIFastH3VSAStats,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAIFastH3Loader": "BSAI FastH3 Loader · 专用加载器 (Dense 兼容)",
    "BSAIFastH3NativeVSA": "BSAI FastH3 Native VSA · 原生稀疏注意力",
    "BSAIFastH3Timesteps": "BSAI FastH3 Timesteps · 精确时间步 [999,749,500,250]",
    "BSAIFastH3EulerSampler": "BSAI FastH3 Euler · 4步采样器",
    "BSAIFastH3VSAStats": "BSAI FastH3 VSA Stats · 命中统计",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
