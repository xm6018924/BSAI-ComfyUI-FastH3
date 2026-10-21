"""BSAI-ComfyUI-FastH3 — 快速生成 H3 视频（FastVideo FastH3 蒸馏 + TaoMate 3步 + OmniRef 8步）节点套件。

围绕 FastVideo FastH3 系列蒸馏模型 + VSA 视频稀疏注意力封装的一站式
ComfyUI 节点，内置四套配方：

  * 4 步 Preview v1 / v0.2 —— 阶梯 [999,749,500,250]，视频/音频 shift 12/3，
    VSA 90% 稀疏（keep 10%），官方 12.5× 前向压缩。
  * 8 步 V2（2026-09-15 发布）—— 阶梯 [999,874,749,624,500,375,250,125]，
    视频/音频 shift 10/3（视频 shift 与 v1 不同！），VSA 80% 稀疏（keep 20%），
    更稳的 8 前向高质量路线。
  * 8 步 OmniRef（2026-10-20 发布）—— 阶梯 [999,874,749,624,500,375,250,125]，
    视频/音频 shift 12/3（回到 v1 值），VSA 90% 稀疏（keep 10%），reference-conditioned
    蒸馏，支持最多 9 图 + 3 视频 + 3 音频的多参考条件生成。
  * 3 步 TaoMate（阿里淘天 TaoLive AIGC）—— 阶梯 [999,750,500]，
    视频/音频 shift 12/3，keep 100%（Dense；TaoMate 是原版 MiniMax-H3 的
    3 步蒸馏 LoRA，配合 TaoMate-H3-step3000-ComfyUI-*.safetensors 使用，
    底座模型用 minimax_h3_fl2va_* 原版权重，社区实测 Euler + CFG 1.2~1.5）。

节点清单：

  * BSAIFastH3Loader          FastH3 专用模型加载器（Dense 兼容 + 严格校验，支持原版 H3 底座）
  * BSAIFastH3NativeVSA       FastH3 原生 VSA 稀疏注意力补丁（recipe 内建 shift/keep）
  * BSAIFastH3Timesteps       FastH3 精确时间步（recipe 内建 3 步 / 4 步 / 8 步阶梯）
  * BSAIFastH3EulerSampler    FastH3 Euler 采样器（recipe 内建 shift，音视频双调度自适应）
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

# ==== BSAI 插件协同 SDK：加载即自动注册（失败不拖垮插件） ====
try:
    import sys as _bsai_sys, os as _bsai_os
    _BSAI_ORCH_DIR = _bsai_os.path.join(
        _bsai_os.path.dirname(_bsai_os.path.abspath(__file__)),
        "..", "BSAI-ComfyUI-Orchestrator")
    if _bsai_os.path.isdir(_BSAI_ORCH_DIR) and _BSAI_ORCH_DIR not in _bsai_sys.path:
        _bsai_sys.path.insert(0, _BSAI_ORCH_DIR)
    from bsai_orch_client import BSAIOrch  # noqa: E402
    BSAIOrch.register(
        name="BSAI-FastH3",
        kind="sampling",
        hardware=["cuda"],
    )
except Exception as _bsai_e:  # 注册失败不得拖垮插件
    print(f"[BSAI SDK] BSAI-FastH3 注册失败(忽略): {_bsai_e}")
# ==== BSAI SDK 块结束 ====

from . import fast_h3_vsa as _vsa

# 兼容两种 Python 包导入形态（直接 import vs 以 custom_nodes 包导入）
try:
    from .fast_h3_vsa import vsa_stats, reset_vsa_stats
except ImportError:                                            # pragma: no cover
    from fast_h3_vsa import vsa_stats, reset_vsa_stats

SHIFT_V, SHIFT_A = 12.0, 3.0                                   # FastH3 v1/v0.2 视频/音频 flow shift

# ---------------------------------------------------------------------------
# FastH3 官方配方 / official recipes（2026-10-21 跟踪，新增 OmniRef）
#   * 4-step Preview v1/v0.2：ladder [999,749,500,250]，shift 12/3，VSA keep 10%
#   * 8-step V2：ladder [999,874,749,624,500,375,250,125]，shift 10/3，VSA keep 20%
#     （V2 视频 shift 为 10，与 v1 的 12 不同；80% 稀疏 = keep 20%）
#   * 8-step OmniRef（2026-10-20）：ladder 同 V2，shift 12/3，VSA keep 10%
#     reference-conditioned 蒸馏；官方模板 shift 12/3 + VSA keep 10% + 8 step；
#     支持最多 9 images + 3 videos + 3 audios 多参考条件
#   * 3-step TaoMate：ladder [999,750,500]，shift 12/3，keep 100%（Dense）
#     阿里淘天 TaoLiveAIGC 发布的 MiniMax-H3 3 步蒸馏 LoRA（rank128/alpha128，
#     step3000 EMA）；底座为原版 minimax_h3_fl2va_*（TaoMate 官方目标模型），
#     ComfyUI 实测：Euler 采样器 + CFG 1.2~1.5 效果最佳，勿叠加旧缓存插件。
# ---------------------------------------------------------------------------
RECIPE_4STEP = "4-step Preview (999,749,500,250 · shift 12/3 · keep 10%)"
RECIPE_8STEP_V2 = "8-step V2 (999,874,749,624,500,375,250,125 · shift 10/3 · keep 20%)"
RECIPE_8STEP_OMNI = "8-step OmniRef (999,874,749,624,500,375,250,125 · shift 12/3 · keep 10%)"
RECIPE_STEP3_TAOMATE = "3-step TaoMate (999,750,500 · shift 12/3 · keep 100% Dense)"
RECIPE_CUSTOM = "custom"
RECIPES = (RECIPE_4STEP, RECIPE_8STEP_V2, RECIPE_8STEP_OMNI, RECIPE_STEP3_TAOMATE, RECIPE_CUSTOM)

_RECIPE_LADDERS = {
    RECIPE_4STEP: "999,749,500,250",
    RECIPE_8STEP_V2: "999,874,749,624,500,375,250,125",
    RECIPE_8STEP_OMNI: "999,874,749,624,500,375,250,125",
    RECIPE_STEP3_TAOMATE: "999,750,500",
}
_RECIPE_SHIFTS = {
    RECIPE_4STEP: (12.0, 3.0),
    RECIPE_8STEP_V2: (10.0, 3.0),
    RECIPE_8STEP_OMNI: (12.0, 3.0),
    RECIPE_STEP3_TAOMATE: (12.0, 3.0),
}
_RECIPE_KEEP = {
    RECIPE_4STEP: 10.0,
    RECIPE_8STEP_V2: 20.0,
    RECIPE_8STEP_OMNI: 10.0,
    RECIPE_STEP3_TAOMATE: 100.0,
}


def _recipe_ladder(recipe, ladder):
    """recipe 非 custom 时返回内建官方阶梯，否则返回用户 ladder 字符串。"""
    if recipe != RECIPE_CUSTOM and recipe in _RECIPE_LADDERS:
        return _RECIPE_LADDERS[recipe]
    return ladder


def _patch_av_shift(model, shift_video, shift_audio):
    """把模型的 ModelSamplingAV 视频/音频 shift 覆盖为配方值（与 ComfyUI 原生
    MiniMaxH3SigmaShift 节点同构）。返回 clone 后的模型；model_options 深拷贝，
    后续 clone 不受影响。"""
    m = model.clone()

    class _ModelSamplingAV(comfy.model_sampling.ModelSamplingAV,
                           comfy.model_sampling.CONST):
        pass

    original = m.get_model_object("model_sampling")
    ms = _ModelSamplingAV(m.model.model_config)
    ms.set_parameters(shift=shift_video, audio_shift=shift_audio)
    if hasattr(original, "noise_scale"):
        ms.set_noise_scale(original.noise_scale)
    m.add_object_patch("model_sampling", ms)
    to = m.model_options.setdefault("transformer_options", {})
    to["minimax_h3_sigma_shift_video"] = shift_video
    to["minimax_h3_sigma_shift_audio"] = shift_audio
    print(f"[BSAI FastH3] recipe shift -> video {shift_video} / audio {shift_audio}", flush=True)
    return m

# ---------------------------------------------------------------------------
# 1) FastH3 专用模型加载器
# ---------------------------------------------------------------------------

_FAST_H3_MARKERS = ("fastvideo", "fasth3", "fast_h3", "omniref", "omni_ref", "omni",
                     "4step", "8step", "v2")
# OmniRef 专用标记（8 步 reference-conditioned 蒸馏，2026-10-20 发布）
_OMNI_REF_MARKERS = ("omniref", "omni_ref")
# 原版 MiniMax-H3 基座（TaoMate-H3 3 步 LoRA 的官方目标模型）文件名标记
_BASE_H3_MARKERS = ("minimax_h3_fl2va", "minimax_h3_ref2va", "minimax_h3_hybrid")


def _is_fast_h3_filename(name):
    base = os.path.splitext(os.path.basename(name).lower())[0]
    return any(k in base for k in _FAST_H3_MARKERS)


def _is_omni_ref_filename(name):
    base = os.path.splitext(os.path.basename(name).lower())[0]
    return any(k in base for k in _OMNI_REF_MARKERS)


def _is_base_h3_filename(name):
    base = os.path.splitext(os.path.basename(name).lower())[0]
    return any(k in base for k in _BASE_H3_MARKERS)


class BSAIFastH3Loader:
    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("diffusion_models")
        return {"required": {
            "model": (sorted(models),),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                             {"advanced": True}),
            "model_type": (["auto", "fast_h3_distill", "omni_ref", "base_h3_fl2va"], {
                "default": "auto",
                "tooltip": "auto: auto-detect from the filename. fast_h3_distill: only accept FastH3 distilled weights "
                           "(fastvideo/fasth3/omniref/omni/4step/8step/v2). omni_ref: accept only FastH3 OmniRef "
                           "(fasth3_omniref_*). base_h3_fl2va: accept the original "
                           "MiniMax-H3 base (minimax_h3_fl2va/ref2va/hybrid, the official base for the "
                           "TaoMate-H3 3-step LoRA)."}),
            "strict_fast_h3_check": ("BOOLEAN", {
                "default": True,
                "label_on": "Validate model filename",
                "label_off": "Allow any H3 weights",
                "tooltip": "When on, validates against model_type: fast_h3_distill only accepts distilled weights with "
                           "fastvideo/fasth3/omniref/omni/_4step/_8step/v2 in the filename (avoids loading the 50-step "
                           "base model); omni_ref only accepts fasth3_omniref_* (the 8-step reference-conditioned "
                           "checkpoint); base_h3_fl2va accepts minimax_h3_fl2va/ref2va/hybrid base "
                           "(for the TaoMate-H3 3-step LoRA)."}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "BSAI/FastH3"
    DESCRIPTION = ("Load FastVideo FastH3 distilled weights (4-step v1, 8-step V2, "
                   "8-step OmniRef) or the original MiniMax-H3 base. "
                   "OmniRef (2026-10-20) = fasth3_omniref_8step_* with shift 12/3 + VSA keep 10%; "
                   "V2 = minimax_h3_fastvideo_8step_v2_* with shift 10/3 + keep 20%; "
                   "4-step v1/v0.2 = minimax_h3_fastvideo_vsa_*; "
                   "with model_type=base_h3_fl2va it loads minimax_h3_fl2va_* (pair with the "
                   "TaoMate-H3 3-step LoRA, recipe = 3-step TaoMate). Put files in "
                   "ComfyUI/models/diffusion_models.")

    def load_model(self, model, weight_dtype="default", model_type="auto",
                   strict_fast_h3_check=True):
        if strict_fast_h3_check:
            if model_type == "omni_ref":
                if not _is_omni_ref_filename(model):
                    raise ValueError(
                        f"[BSAI FastH3] {model} filename lacks the OmniRef marker "
                        "(omniref / omni_ref). OmniRef recipe pairs with "
                        "fasth3_omniref_8step_* checkpoints; to force-load anyway, "
                        "turn off strict_fast_h3_check or set model_type=auto.")
            elif model_type == "base_h3_fl2va":
                if not _is_base_h3_filename(model):
                    raise ValueError(
                        f"[BSAI FastH3] {model} filename lacks the original H3 base marker "
                        "(minimax_h3_fl2va/ref2va/hybrid). The TaoMate recipe needs the original base; "
                        "to force-load anyway, turn off strict_fast_h3_check or set model_type=auto.")
            elif model_type == "fast_h3_distill":
                if not _is_fast_h3_filename(model):
                    raise ValueError(
                        f"[BSAI FastH3] {model} filename lacks the FastH3/distilled marker "
                        "(fastvideo/fasth3/omniref/omni/_4step/_8step/v2). FastH3 uses distilled weights; "
                        "to force-load anyway, turn off strict_fast_h3_check.")
            elif not (_is_fast_h3_filename(model) or _is_base_h3_filename(model)):
                raise ValueError(
                    f"[BSAI FastH3] {model} filename has neither the FastH3/distilled marker "
                    "(fastvideo/fasth3/omniref/omni/_4step/_8step/v2) nor the original H3 base marker "
                    "(minimax_h3_fl2va/ref2va/hybrid). To force-load anyway, turn off "
                    "strict_fast_h3_check.")
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
              f"| model_type={model_type} | strict_fast_h3_check={strict_fast_h3_check}", flush=True)
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
                "tooltip": "When off, passes the model through without installing the VSA patch. Can be driven by BSAI H3 MotionFix.vsa_enabled."}),
            "video_keep_percent": ("FLOAT", {
                "default": 10.0, "min": 0.5, "max": 100.0, "step": 0.5,
                "tooltip": "Percentage of video tokens that keep exact attention. Official: ~10% for 4-step v1/v0.2 "
                           "(90% sparse); ~20% for 8-step V2 (80% sparse). Lower = less VRAM / faster, "
                           "but more detail loss. Can be driven by BSAI H3 MotionFix.video_keep_percent."}),
            "start_percent": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Steps before this sampling progress run Dense (high-temperature warmup)."}),
            "end_percent": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Steps after this sampling progress run Dense."}),
            "min_tokens": ("INT", {
                "default": 8192, "min": 0, "max": 1048576, "step": 256,
                "tooltip": "Attention calls with sequence length below this stay Dense (sparsity has no benefit on short sequences)."}),
            "sink_conditioning": (["exact_kv", "exact_kv_and_rows", "off"], {
                "default": "exact_kv",
                "tooltip": "exact_kv: all video queries see text/audio/ref conditioning rows exactly (~3% overhead). "
                           "exact_kv_and_rows: additionally runs Dense on conditioning query rows (more stable audio). "
                           "off: only video-block top-k is kept, lowest memory but weaker conditioning alignment."}),
            "backend": (["auto", "native", "torch"], {
                "default": "auto",
                "tooltip": "auto: uses native (Blackwell kernel) when comfy_kitchen.sol_attn is available, else torch. "
                           "native: force the comfy_kitchen kernel. torch: pure PyTorch block sparsity (works on 40-series)."}),
            "strict_native_backend": ("BOOLEAN", {
                "default": True,
                "tooltip": "True: error out when native is unavailable. False: automatically fall back to torch sparsity."}),
            "verbose": ("BOOLEAN", {"default": True,
                                    "tooltip": "Print the shape and backend info of every VSA hit."}),
            "recipe": (list(RECIPES), {
                "default": RECIPE_8STEP_OMNI,
                "tooltip": "Official one-click recipe presets. 4-step: shift 12/3 + keep 10%. "
                           "8-step V2: shift 10/3 + keep 20% (auto-patches the model ModelSamplingAV "
                           "video shift to 10). 8-step OmniRef (2026-10-20): shift 12/3 + keep 10% "
                           "(reference-to-video checkpoint, 9 images + 3 videos + 3 audios). "
                           "3-step TaoMate: shift 12/3 + keep 100% "
                           "(Dense; pairs with the original minimax_h3_fl2va base + TaoMate LoRA, not sparse). "
                           "custom: fully manual (keep and shift managed by you)."}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "apply_vsa"
    CATEGORY = "BSAI/FastH3"
    DESCRIPTION = ("FastH3 native VSA (Video Sparse Attention): scores 64-token tiles, "
                   "keeps only top-k video blocks exact and conditioning rows exact. "
                   "4-step recipe saves ~90% video self-attention; 8-step V2 saves ~80% "
                   "and auto-fixes shift 10/3; 8-step OmniRef saves ~90% (keep 10%, shift 12/3, "
                   "for reference-to-video); 3-step TaoMate keeps 100% (Dense, not sparse).")

    def apply_vsa(self, model, enabled, video_keep_percent, start_percent, end_percent,
                  min_tokens, sink_conditioning, backend, strict_native_backend,
                  verbose, recipe=RECIPE_4STEP):
        if recipe != RECIPE_CUSTOM:
            video_keep_percent = _RECIPE_KEEP.get(recipe, video_keep_percent)
            shift_video, shift_audio = _RECIPE_SHIFTS.get(
                recipe, (SHIFT_V, SHIFT_A))
            # 8 步 V2 的视频 shift 为 10（非 v1 的 12）：先修模型调度，Timesteps/
            # 采样器共用该模型即全链路一致；VSA 关闭（enabled=False）时也生效。
            if (shift_video, shift_audio) != (SHIFT_V, SHIFT_A):
                model = _patch_av_shift(model, shift_video, shift_audio)
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
                "tooltip": "Explicitly trained ladder (the official requirement is to sample at training jump points, not a uniform grid). "
                           "4-step v1/v0.2: 999,749,500,250; 8-step V2: "
                           "999,874,749,624,500,375,250,125; 3-step TaoMate: 999,750,500. "
                           "Effective when recipe=custom. "
                           "Can be driven by BSAI H3 MotionFix.ladder."}),
            "recipe": (list(RECIPES), {
                "default": RECIPE_8STEP_OMNI,
                "tooltip": "Official one-click recipe ladders. 8-step V2 uses the training ladder "
                           "999,874,749,624,500,375,250,125 (9 sigma points = 8 forwards); "
                           "8-step OmniRef uses the same 8-step ladder; "
                           "4-step Preview: 999,749,500,250; "
                           "3-step TaoMate uses 999,750,500 (4 sigma points = 3 forwards). "
                           "custom: use the ladder string above."}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "BSAI/FastH3"
    DESCRIPTION = ("Convert FastH3 explicitly-trained ladder timesteps into SIGMAS for SamplerCustomAdvanced. "
                   "4-step [999,749,500,250]; 8-step V2 [999,874,749,624,500,375,250,125]; "
                   "3-step TaoMate [999,750,500]; 0 is appended at the end (final denoise). "
                   "8-step V2 needs model shift 10/3 (auto-patched by the VSA node recipe).")

    def get_sigmas(self, model, ladder="999,749,500,250", recipe=RECIPE_4STEP):
        ladder = _recipe_ladder(recipe, ladder)
        values = []
        for part in str(ladder).replace("，", ",").replace("[", "").replace("]", "").split(","):
            part = part.strip()
            if part:
                values.append(float(part))
        if len(values) < 2:
            raise ValueError(f"[BSAI FastH3] ladder needs at least 2 timesteps: {ladder!r}")
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

    # ---- BSAI 协同：GPU 主采样租约（短任务 watchdog=False；失败不阻断采样） ----
    _bsai_alloc = None
    try:
        _bsai_alloc = BSAIOrch.allocate("sampling", requester="8191", watchdog=False)
        if not _bsai_alloc.ok:
            print(f"[BSAI FastH3] SDK 未取得 gpu1_sampling 租约({_bsai_alloc.reason})，仍按原逻辑采样", flush=True)
    except Exception as _bsai_ae:
        print(f"[BSAI FastH3] SDK allocate 异常(忽略): {_bsai_ae}", flush=True)
        _bsai_alloc = None

    if schedule_mode == "native" or (schedule_mode == "auto" and _native_av_schedule(model)):
        print(f"[BSAI FastH3 Euler] native ModelSamplingAV -> 单调度 Euler  "
              f"sigmas={[round(float(s), 4) for s in sigmas]}  x={tuple(x.shape)}", flush=True)
        _shapes = _latent_shapes(model)
        _vnumel = math.prod(_shapes[0][1:]) if (_shapes and len(_shapes) >= 1) else None
        for i in trange(len(sigmas) - 1, disable=disable):
            sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            d = (x - denoised) / sigmas[i]
            x = x + (sv_n - sv) * d
            print(f"[BSAI FastH3 step {i}] {sv:.4f}->{sv_n:.4f}", flush=True)
            if callback is not None:
                callback({"i": i, "denoised": denoised, "x": x,
                          "sigma": sigmas[i], "sigma_hat": sigmas[i]})
        if _bsai_alloc is not None:
            try:
                _bsai_alloc.release()
            except Exception:
                pass
        return x

    # 旧版 ComfyUI：视频/音频各自 flow 调度（video shift 12 / audio shift 3），分别推进
    shapes = _latent_shapes(model)
    if not shapes or len(shapes) < 2:
        raise RuntimeError(
            "BSAI FastH3 Euler requires a MiniMax-H3 video+audio latent "
            "(EmptyMiniMaxH3LatentAV / MiniMaxH3ImageToVideo output).")
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
    if _bsai_alloc is not None:
        try:
            _bsai_alloc.release()
        except Exception:
            pass
    return x


class BSAIFastH3EulerSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shift_video": ("FLOAT", {
                "default": SHIFT_V, "min": 0.01, "max": 100.0, "step": 0.01,
                "tooltip": "Video-stream flow shift (FastH3 official shift-12 rectified schedule)."}),
            "shift_audio": ("FLOAT", {
                "default": SHIFT_A, "min": 0.01, "max": 100.0, "step": 0.01,
                "tooltip": "Audio-stream flow shift (FastH3 official shift-3)."}),
            "schedule_mode": (["auto", "native", "legacy_dual"], {
                "default": "auto",
                "tooltip": "auto: detect ModelSamplingAV; single-schedule Euler if present, otherwise dual audio/video schedule. "
                           "native: force single schedule. legacy_dual: force dual schedule."}),
            "recipe": (list(RECIPES), {
                "default": RECIPE_8STEP_OMNI,
                "tooltip": "Official one-click recipe shift presets. 4-step / OmniRef: video 12 / audio 3; "
                           "8-step V2: video 10 / audio 3 (note V2 video shift is NOT 12!); "
                           "3-step TaoMate: video 12 / audio 3. "
                           "custom: use the shift parameters above."}),
        }}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "BSAI/FastH3"
    DESCRIPTION = ("FastH3 Euler sampler (any step count, advances by SIGMAS length). "
                   "4-step recipe shift 12/3; 8-step V2 shift 10/3; 8-step OmniRef shift 12/3; "
                   "3-step TaoMate shift 12/3. Under native ModelSamplingAV (ComfyUI 0.31+) "
                   "it advances with a single schedule; on older builds it auto-advances with "
                   "dual video/audio schedules. Plug into SamplerCustomAdvanced.sampler "
                   "together with FastH3 Timesteps.")

    def get_sampler(self, shift_video=SHIFT_V, shift_audio=SHIFT_A, schedule_mode="auto",
                    recipe=RECIPE_4STEP):
        if recipe != RECIPE_CUSTOM:
            shift_video, shift_audio = _RECIPE_SHIFTS.get(
                recipe, (shift_video, shift_audio))
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
    DESCRIPTION = ("Read-only view of FastH3 VSA hit statistics (sparse/dense/native/torch/errors). "
                   "A sparse count > 0 means VSA actually participated in sampling.")

    def get_stats(self, model):
        native_ok = _vsa._ck is not None and hasattr(_vsa._ck, "sol_attn")
        st = vsa_stats()
        cfg = model.model_options.get("transformer_options", {}).get("bsai_fasth3_vsa")
        ms = _model_sampling(model)
        shift_v = getattr(ms, "shift", None)
        shift_a = getattr(ms, "audio_shift", None)
        lines = [
            "[BSAI FastH3 VSA Stats]",
            f"  native backend available : {native_ok}",
            f"  configured               : {cfg}",
            f"  model shift (v/a)        : {shift_v} / {shift_a}",
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
    "BSAIFastH3Loader": "BSAI FastH3 Loader (4-step / 8-step V2 / OmniRef)",
    "BSAIFastH3NativeVSA": "BSAI FastH3 Native VSA (recipe)",
    "BSAIFastH3Timesteps": "BSAI FastH3 Timesteps (4-step / 8-step / OmniRef)",
    "BSAIFastH3EulerSampler": "BSAI FastH3 Euler Sampler (4-step / 8-step / OmniRef)",
    "BSAIFastH3VSAStats": "BSAI FastH3 VSA Stats",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
