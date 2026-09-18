"""BSAI-ComfyUI-FastH3 — 快速生成 H3 视频（FastVideo FastH3 蒸馏 + TaoMate 3步）节点套件。

围绕 FastVideo FastH3 4-step / 8-step V2 蒸馏模型 + TaoMate-H3 3 步 LoRA
封装的一站式节点（recipe 一键切换）：
  * BSAIFastH3Loader        FastH3 专用模型加载器（支持原版 H3 基座）
  * BSAIFastH3NativeVSA     FastH3 原生 VSA 稀疏注意力
  * BSAIFastH3Timesteps     FastH3 精确时间步 [999,749,500,250] / V2 / TaoMate 3步
  * BSAIFastH3EulerSampler  FastH3 Euler 采样器（3/4/8 步配方）
  * BSAIFastH3VSAStats      VSA 命中统计

安装：把本目录放到 ComfyUI/custom_nodes/BSAI-ComfyUI-FastH3，重启 ComfyUI。
可选依赖：comfy_kitchen（含 sol_attn 内核时启用 native VSA，否则自动用 torch 稀疏）。
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
