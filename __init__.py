"""BSAI-ComfyUI-FastH3 — 快速生成 H3 视频（FastVideo FastH3 蒸馏 + TaoMate 3步 + OmniRef 8步）节点套件。

围绕 FastVideo FastH3 4-step / 8-step V2 / 8-step OmniRef 蒸馏模型 + TaoMate-H3 3 步 LoRA
封装的一站式节点（recipe 一键切换）：
  * BSAIFastH3Loader        FastH3 专用模型加载器（支持 OmniRef / 原版 H3 基座）
  * BSAIFastH3NativeVSA     FastH3 原生 VSA 稀疏注意力（OmniRef keep 10%, shift 12/3）
  * BSAIFastH3Timesteps     FastH3 精确时间步 [4/8/OmniRef 阶梯 / TaoMate 3步]
  * BSAIFastH3EulerSampler  FastH3 Euler 采样器（4/8/OmniRef/TaoMate 配方）
  * BSAIFastH3VSAStats      VSA 命中统计

OmniRef（2026-10-20 发布）支持最多 9 images + 3 videos + 3 audios 的多参考条件，
使用 ComfyUI 原生 MiniMaxH3ReferenceToVideo conditioning 节点。

安装：把本目录放到 ComfyUI/custom_nodes/BSAI-ComfyUI-FastH3，重启 ComfyUI。
可选依赖：comfy_kitchen（含 sol_attn 内核时启用 native VSA，否则自动用 torch 稀疏）。
"""

# === BSAI H3 VAE 全量加载统一补丁（宿主 BSAI-ComfyUI-Sol-H3）===
# 修复 H3 AudioVAE/VideoVAE 走 DynamicVRAM 权重量流导致的解码极慢。
# try/except 容错，宿主缺失则 no-op。
try:
    import sys as _sys, os as _os
    _SOL = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "..", "BSAI-ComfyUI-Sol-H3")
    if _SOL not in _sys.path:
        _sys.path.insert(0, _SOL)
    import bsai_h3_vae_full as _h3v
    _h3v.apply_global_patch("BSAI-ComfyUI-FastH3")
except Exception:
    pass

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
