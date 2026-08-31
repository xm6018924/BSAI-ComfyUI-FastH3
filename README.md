# BSAI-ComfyUI-FastH3

> **BSAI · ComfyUI 快速生成 H3 视频插件套件 / Fast H3 Video Generation Nodes for ComfyUI**
> 4 步蒸馏 FastH3 + Native VSA 视频稀疏注意力，文生 / 图生 / 多参考（多参）/ 4K 超分 全链路一键出片。
> One-stop nodes for **fast H3 video (with synced audio)** in ComfyUI — 4-step distilled **FastH3** +
> **VSA (Video Sparse Attention)**, covering Text-to-Video / Image-to-Video / Multi-reference / 4K Upscale.

基于（Based on）：
- FastVideo `FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`（MiniMax-H3 33B 双模态扩散 Transformer 的无数据 DMD2 蒸馏 / data-free DMD2 distillation）
- Kijai `MiniMax-H3-experimental`（权重 / weights）
- closerAI minimaxH3 Helper（加载器 / VSA / 时间步 / Euler 命名启发，名称启发 inspiration）
- ComfyUI 原生 MiniMax-H3 节点（0.31+）

---

## 目录 / Table of Contents

1. [插件简介 / Introduction](#1-插件简介--introduction)
2. [节点清单 / Node Reference](#2-节点清单--node-reference)
3. [安装 / Installation](#3-安装--installation)
4. [模型下载 / Model Weights](#4-模型下载--model-weights)
5. [示例工作流 1：文生视频+音频（T2VA）/ Workflow 1: Text-to-Video+Audio](#5-示例工作流-1文生视频音频t2va--workflow-1-text-to-videoaudio)
6. [示例工作流 2：文生+图生+多参+4K 超分 / Workflow 2: T2V+I2V+Multi-ref+4K](#6-示例工作流-2文生图生多参4k-超分--workflow-2-t2vi2vmulti-ref4k)
7. [与官方/其它方案对比 / Comparison](#7-与官方其它方案对比--comparison)
8. [原理速览 / How It Works](#8-原理速览--how-it-works)
9. [FAQ / 排障 / FAQ & Troubleshooting](#9-faq--排障--faq--troubleshooting)
10. [测试记录 / Test Records](#10-测试记录--test-records)
11. [许可证 / License](#11-许可证--license)

---

## 1. 插件简介 / Introduction

**中文**：在 ComfyUI 中**快速生成 H3 视频（含同步音轨）**。FastH3 把 50 步 H3 基座蒸馏成 **4 步**（官方约省 **12.5×** 模型前向次数），再叠加 **VSA 视频稀疏注意力**（约 **90%** 稀疏）进一步压显存与耗时。一次 pipeline 同时输出**视频 + 音频**。

**English**: Generate **H3 videos with a synced audio track** fast inside ComfyUI. FastH3 distills the 50-step H3 base model down to **4 steps** (≈ **12.5×** fewer model forwards per the official model card), and **VSA (Video Sparse Attention)** cuts memory and latency by keeping only ~**90%**-sparse exact attention. One pipeline produces **video + audio** at once.

| 痛点 Problem | 解法 Solution |
|---|---|
| H3 基座要 50 去噪步，很慢 / 50-step base is slow | FastH3 蒸馏为 **4 步**（shift-12 整流调度）/ distilled to **4 steps** |
| 长视频注意力 O(N²) 显存爆炸 / O(N²) attention blows VRAM | **VSA 稀疏注意力**：64-token 分块，top-k 视频块精确注意力 / 64-token tiled top-k |
| 4 步必须走训练阶梯 / 4-step needs the trained ladder | `BSAIFastH3Timesteps` 输出 `[999,749,500,250]` 对应 SIGMAS |
| 音视频双调度易错 / dual schedule error-prone | `BSAIFastH3EulerSampler` 自动检测原生 `ModelSamplingAV`，单/双调度自动切换 |

---

## 2. 节点清单 / Node Reference

| 节点 Node | 作用 Purpose | 关键参数（默认）Key params (default) |
|---|---|---|
| `BSAIFastH3Loader` | FastH3 专用加载器（Dense 兼容 + 文件名严格校验）/ FastH3 loader (Dense-compatible + strict filename check) | `weight_dtype=default`，`strict_fast_h3_check=True` |
| `BSAIFastH3NativeVSA` | VSA 稀疏注意力补丁（native / torch 双后端自动切换）/ VSA sparse-attention patch (native/torch auto) | `video_keep_percent=10.0`，`sink_conditioning=exact_kv`，`backend=auto`，`min_tokens=8192` |
| `BSAIFastH3Timesteps` | 精确时间步：显式训练阶梯 → SIGMAS / exact ladder → SIGMAS | `ladder="999,749,500,250"` |
| `BSAIFastH3EulerSampler` | FastH3 4 步 Euler 采样器（音视频双调度自适应）/ 4-step Euler sampler (video/audio schedule adaptive) | `shift_video=12.0`，`shift_audio=3.0`，`schedule_mode=auto` |
| `BSAIFastH3VSAStats` | 只读诊断：sparse/dense/native/torch/errors 命中统计 / read-only VSA hit stats | — |

### BSAIFastH3NativeVSA 参数详解 / Parameter details

- `video_keep_percent`（视频保留百分比 / video tokens kept %）：视频 token 中保留精确注意力的 tile 百分比，官方约 **10%**。越小越快/越省显存，细节损失越大。Lower = faster/lighter, more detail loss.
- `sink_conditioning`（条件行处理 / conditioning rows）：
  - `exact_kv`（默认 / default）：所有视频 query 都精确看到文本/音频/参考条件行（约 3% 开销）→ VSA 官方推荐。All video queries see exact conditioning KV.
  - `exact_kv_and_rows`：额外让条件 query 行也跑 Dense（音频更稳）。Conditioning query rows also dense (more stable audio).
  - `off`：只保留视频块 top-k，最省内存但条件对齐弱。Cheapest, weaker conditioning alignment.
- `backend`：`auto`（有 `comfy_kitchen.sol_attn` 用 native 内核，否则 torch）/ native / torch。
- `strict_native_backend=True`：native 不可用直接报错；`False` 自动降级 torch 稀疏。Fail hard if native missing / auto-fallback to torch.
- `min_tokens`：序列长度小于该值直接 Dense（短序列稀疏无收益，默认 8192）。Short sequences go dense.
- `start_percent` / `end_percent`：采样进度窗口，窗口外的步跑 Dense（高温预热）。Steps outside window run dense (hot-start).

---

## 3. 安装 / Installation

**方法 A：ComfyUI-Manager（推荐 / recommended）**
> 在 Manager → “Custom Nodes Manager” 中搜索 `BSAI-ComfyUI-FastH3` 安装，或在
> “Install Custom Nodes” → “Git URL” 粘贴仓库地址后安装。安装后重启 ComfyUI。
> Search `BSAI-ComfyUI-FastH3` in ComfyUI-Manager, or paste the Git URL in "Install Custom Nodes". Restart ComfyUI after install.

**方法 B：手动 git clone / Manual clone**
```bat
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/BSAI-ComfyUI-FastH3.git
```
重启 ComfyUI 即可。无强制第三方依赖。/ Restart ComfyUI. No mandatory third-party deps.

**可选依赖（强烈建议，启用 native VSA 内核，速度最快）/ Optional (strongly recommended for native VSA kernel):**
```bat
python -m pip install comfy_kitchen
```
> 若 `comfy_kitchen` 无 `sol_attn` 内核，节点自动降级为纯 PyTorch 块稀疏路径，功能不变、速度略慢；40 系显卡默认走 torch 路径。If `sol_attn` is unavailable the node auto-falls-back to the PyTorch block-sparse path (same function, a bit slower); RTX 40-series uses the torch path by default.

---

## 4. 模型下载 / Model Weights

| 文件 File | 放置目录 Folder | 大小 Size |
|---|---|---|
| `minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors` | `models/diffusion_models` | ~22.9 GB |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders` | — |
| `minimax_h3_video_vae_int8_convrot.safetensors` | `models/vae` | ~3.17 GB |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae` | — |

来源 / Sources：
- FastH3 模型卡 / Model card：<https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree>
- KJ 文件仓库 / File repo：<https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main>
- 国内镜像 / China mirror：`set HF_ENDPOINT=https://hf-mirror.com`（BSAI Pro 启动脚本已默认设置）

> ⚠️ FastH3 是 **4 步蒸馏**权重，文件名带 `fastvideo ... _4step`。**不要**误用 50 步基座
> `minimax_h3_f12va_pruned_w4a8_mixed.safetensors`。加载器默认 `strict_fast_h3_check=True` 防呆；
> 确需加载其它 H3 权重可关闭该校验。
> ⚠️ FastH3 is the **4-step distilled** weights (filename contains `fastvideo ... _4step`). **Do not**
> use the 50-step base. The loader validates the filename by default; disable `strict_fast_h3_check` only if you really need another H3 weight.

---

## 5. 示例工作流 1：文生视频+音频（T2VA）/ Workflow 1: Text-to-Video+Audio

**文件 / File**：`example_workflows/BSAI_FastH3_T2VA_4step_VSA.json`

最小可跑链路，文生视频 + 同步音轨，4 步出片。/ Minimal runnable chain: text-to-video + synced audio in 4 steps.

```
BSAIFastH3Loader ─► BSAIFastH3NativeVSA ─┬─► BasicGuider ─┐
                                          ├─► BSAIFastH3Timesteps ─► SamplerCustomAdvanced
                                          └─► BSAIFastH3VSAStats     ▲ ▲ ▲
CLIPLoader(qwen3vl32b) ─┐                                          │ │ │
MiniMaxH3ImageToVideo ◄─┴─ VAELoader(video)  ──► positive/latent ──┘ │ │
RandomNoise ────────────────────────────────────────────────────────┘ │
BSAIFastH3EulerSampler ───────────────────────────────────────────────┘
SamplerCustomAdvanced ─┬─► VAEDecode ─► CreateVideo(24fps) ─► SaveVideo
                       └─► VAEDecodeAudio ─► SaveAudio / CreateVideo 音轨
```

**使用步骤 / How to use：**
1. 确认第 4 节 4 个权重文件已就位，`BSAIFastH3Loader` 下拉选中 FastH3 4 步模型。Make sure the 4 weights in §4 are in place; select the FastH3 4-step model in `BSAIFastH3Loader`.
2. 在 `MiniMaxH3ImageToVideo.prompt` 填提示词（示例已含一段电影感描述）。Fill the prompt (a cinematic sample is included).
3. 点运行 / Queue。默认 **864×480 (0.4MP 16:9)、124 帧 ≈ 5s、4 步、seed 123456662**。
4. 跑完看 `BSAIFastH3VSAStats`：`sparse` 计数 > 0 即证明 VSA 真实参与。After the run check `BSAIFastH3VSAStats`: `sparse` count > 0 proves VSA was active.

**参数调整 / Tuning：**
- 时长 / Duration：改 `MiniMaxH3ImageToVideo.length`（124≈5s；训练区间约 124–362，再长未充分测试）。
- 分辨率 / Resolution：改 `width/height`（32 的倍数；0.4MP 起步，显存不够先降）。
- 随机种子 / Seed：`RandomNoise.noise_seed`。

---

## 6. 示例工作流 2：文生+图生+多参+4K 超分 / Workflow 2: T2V+I2V+Multi-ref+4K

**文件 / File**：`example_workflows/BSAI_FastH3_4step_VSA_文生+图生+多参+4K超分 v1.0.json`

完整商业链路：**文生视频（T2V）＋ 图生/多参考（I2V/Ref2V）＋ 音频 ＋ BSAI-H3-upscale-4K 超分放大**，
用 `easy ifElse` 一键切换模式，最后经 `VHS_VideoCombine` 导出。33 节点 / 46 连线。
Full pipeline: **T2V + I2V/Multi-ref + audio + BSAI-H3-upscale-4K**, mode switched by `easy ifElse`, exported via `VHS_VideoCombine`. 33 nodes / 46 links.

**前置依赖（除本插件外还需安装）/ Extra plugins required:**
| 依赖 Plugin | 用途 Purpose | 安装 / Install |
|---|---|---|
| [BSAI-MiniMAX-H3-Prompt](https://github.com/xm6018924/BSAI-MiniMAX-H3-Prompt) | `BSAI_H3_PromptTemplate` 提示词模板 / prompt template | Manager 或 git clone |
| [BSAI-H3-upscale-4K](https://github.com/xm6018924/BSAI-H3-upscale-4K) | `BSAI_H3_Upscale4K` 4K 超分放大 / 4K upscale | Manager 或 git clone |
| KJ `MiniMax-H3-experimental` 节点包 | `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` | 见 KJ 仓库 |
| `ComfyUI-Easy-Use` | `easy ifElse` 模式切换 / mode switch | Manager |
| `VideoHelperSuite` (VHS) | `VHS_VideoCombine` 导出 / export | Manager |
| `ComfyMath` | `ResolutionSelector` / `ComfyMathExpression` | Manager |
| `pysssss`（Custom-Scripts） | `ShowText\|pysssss` 输出 / text preview | Manager |
| `Muye` 系节点 | `MuyeTextEditOutput` | 按需 / optional |

**使用步骤 / How to use：**
1. 装齐上表依赖并重启 ComfyUI。Install deps above and restart.
2. 准备素材：3 个 `LoadImage` 放入参考图（角色/服装/场景）；用 `BSAI_H3_PromptTemplate` 选模板或直接手填提示词。Prepare 3 ref images; use the prompt template or write your own.
3. 模式切换 / Mode switch：`easy ifElse` 布尔开关（`PrimitiveBoolean`）在 **文生 T2V（`MiniMaxH3ImageToVideo`，864×480）** 与 **图生/多参 Ref2V（`MiniMaxH3ReferenceToVideo`，544×960 竖屏，`ref_image_size=max`）** 之间切换。
4. 时长由 `ComfyMathExpression` 按秒数自动换算帧数（`Float (duration)` 输入，默认约 5s）。Duration is auto-converted by `ComfyMathExpression` from a seconds float (≈5s default).
5. 采样链路与本插件示例 1 相同：Loader → NativeVSA → Timesteps → EulerSampler → SamplerCustomAdvanced。
6. 成片先 `VAEDecode/VAEDecodeAudio`，再进 `BSAI_H3_Upscale4K`（默认 `NVIDIA RTX Video Super Res` 4 倍），最后 `VHS_VideoCombine` 导出 mp4（24fps）。Frames → upscale 4K → export mp4.

> 提示 / Tip：`BSAI_H3_Upscale4K` 与 `BSAI_H3_PromptTemplate` 属独立 BSAI 仓库，若只想要最简 4 步出片，用示例工作流 1 即可。The two BSAI nodes are separate repos; use Workflow 1 if you only need minimal 4-step output.

---

## 7. 与官方/其它方案对比 / Comparison

| 方案 Solution | 去噪步数 Steps | 注意力 Attention | 一句话 Summary |
|---|---|---|---|
| MiniMax-H3 基座 / base | 50 | Dense | 慢、显存高，参考用 / slow, heavy VRAM |
| MiniMax-H3 4步加速 LoRA | 4 | Dense | 快但仍全注意力 / fast but full attention |
| **FastH3（本插件 / this pack）** | **4** | **VSA ~90% 稀疏** | 更快、更省显存 / faster & lighter VRAM |

> 官方基准 / Official baseline（FastVideo 模型卡）：1344×768@24FPS 单 Blackwell 5s≈16s / 10s≈31s / 15s≈47s；8×B200 5s≈6.8s。
> 第三方评测（aigc.douyoubuy.cn）：RTX 4060 Ti 8GB 实测 FastH3 0.4MP 5s ≈ 360s，比 4 步加速 LoRA 快约 1 分钟。

---

## 8. 原理速览 / How It Works

1. **FastH3 = 无数据 DMD2 蒸馏**：把 50 步 H3 蒸馏到 4 步，按 **shift-12 整流调度**在训练好的跳跃点 `999→749→500→250` 行走，一次 pipeline 同时生成同步视频+音频。
   **Data-free DMD2 distillation**: 50→4 steps on the shift-12 rectified schedule (`999→749→500→250`), producing synced video+audio in one pass.
2. **VSA（Video Sparse Attention）**：按 64-token 分块，用学习到的门控/打分选出重要视频块，仅对它们做精确注意力；文本/音频/参考等条件行始终精确。Blackwell 由专用内核加速。
   **VSA**: 64-token tiled scoring, exact attention only on important video blocks; conditioning rows (text/audio/ref) always exact. Hardware kernel on Blackwell.
3. **本插件 torch 后端 / torch backend**：块质心点积打分门控，每视频 query 块保留 `video_keep_percent` 的视频块 + 全部条件 KV，显式 matmul+softmax（兼容任意长度）。native 后端走 `comfy_kitchen.sol_attn`（tau 由 keep_percent 换算）。
   **native backend**: `comfy_kitchen.sol_attn` (tau derived from keep_percent).

---

## 9. FAQ / 排障 / FAQ & Troubleshooting

- **节点红叉 / "expects a MiniMax-H3 diffusion model"**：`BSAIFastH3NativeVSA` 只能接 H3 模型（`MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` 的 conditioning 链路、原生 H3 加载的模型）。
- **`strict_fast_h3_check` 报错 / error**：加载的文件名不含 `fastvideo/fasth3/_4step` —— 确认放的是 FastH3 4 步权重；确需强载则关闭该校验。
- **native 不可用 / unavailable**：装了 `comfy_kitchen` 但没 `sol_attn` 内核 → 自动用 torch 后端（40 系默认）。40-series defaults to torch.
- **和 Sage/其它注意力 patch 冲突 / conflicts with other attention patches**：本插件通过 `optimized_attention_override` 与既有补丁串接（`compose_with_foreign_patches` 默认开启）；仍异常先移除其它注意力 patch 复测。
- **视频与音频不同步 / audio out of sync**：确认 `schedule_mode=auto`（ComfyUI 0.31+ 原生 `ModelSamplingAV` 走单调度）；老版本自动走视频 shift 12 / 音频 shift 3 双调度。
- **节点爆红 / nodes red after install**：新装的节点需**重启 ComfyUI** 才会注册；FastH3 权重未下载完成（`.xltd` 临时后缀）时加载器会报找不到模型。New nodes need a **ComfyUI restart** to register; an unfinished download (`.xltd` suffix) makes the loader report "model not found".

---

## 10. 测试记录 / Test Records

- Python 3.13.12 · PyTorch 2.11.0+cu130 · CUDA 13.0 · ComfyUI 0.34.0（BSAI Pro v38）
- 5 个节点在真实 ComfyUI 加载机制下全部注册成功 / all 5 nodes registered under the real loader.
- 单元测试通过 / unit tests pass：时间步 `[999,749,500,250] → sigmas [0.9999,0.9728,0.9231,0.8,0.0]`；torch 稀疏注意力条件行与 Dense 全等、形状/有限性正确；BTHD/BHND 两种形态与短序列降级 Dense 均通过。
- native 后端：`comfy_kitchen.sol_attn` 在 GPU（bf16, head_dim 128, 8192 tokens）实测输出正确。
- ⚠️ 因 FastH3 蒸馏权重（~22.9GB）未完成下载，尚未做端到端出图验证；权重补齐后按示例工作流一键出片即可。End-to-end render not yet verified because the ~22.9GB weight was still downloading; once in place, run the example workflows directly.

---

## 11. 许可证 / License

[MIT](LICENSE) · © BSAI (xm6018924)。代码供学习与商用自由使用，请保留版权声明。
Code is freely usable for study and commercial work; please keep the copyright notice.
