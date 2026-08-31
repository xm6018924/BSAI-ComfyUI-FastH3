# -*- coding: utf-8 -*-
"""生成 BSAI FastH3 示例工作流 JSON（ComfyUI 0.34+ 格式）。"""
import json
import os

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "BSAI_FastH3_T2VA_4step_VSA.json")


def N(nid, ntype, pos, size, widgets, title=None, inputs=None, outputs=None):
    node = {
        "id": nid, "type": ntype, "pos": pos, "size": size, "flags": {},
        "order": nid - 1, "mode": 0,
        "inputs": inputs if inputs is not None else [],
        "outputs": outputs if outputs is not None else [],
        "properties": {"Node name for S&R": ntype},
        "widgets_values": widgets,
    }
    if title:
        node["title"] = title
    return node


def INP(name, typ, link):
    return {"name": name, "type": typ, "link": link}


def OUT(name, typ, links):
    return {"name": name, "type": typ, "links": links, "slot_index": 0}


FASTH3_MODEL = "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors"
TEXT_ENC = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_int8_convrot.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
PROMPT = ("cinematic aerial shot, a lone astronaut walking across a vast "
          "crimson desert under twin moons, volumetric god rays, slow dolly "
          "in, 35mm film grain, dramatic orchestral score building")

nodes = []
nodes.append(N(1, "BSAIFastH3Loader", [0, 0], [330, 100],
               [FASTH3_MODEL, "default", True], "BSAI① FastH3 专用加载器 (Dense兼容)",
               outputs=[OUT("MODEL", "MODEL", [1])]))
nodes.append(N(2, "BSAIFastH3NativeVSA", [380, 0], [340, 220],
               [True, 10.0, 0.0, 1.0, 8192, "exact_kv", "auto", True, True],
               "BSAI② FastH3 Native VSA 稀疏注意力",
               inputs=[INP("model", "MODEL", 1)],
               outputs=[OUT("MODEL", "MODEL", [2, 4, 20])]))
nodes.append(N(3, "BSAIFastH3Timesteps", [0, 160], [340, 90],
               ["999,749,500,250"], "BSAI③ FastH3 精确时间步 [999,749,500,250]",
               inputs=[INP("model", "MODEL", 2)],
               outputs=[OUT("SIGMAS", "SIGMAS", [7])]))
nodes.append(N(4, "BSAIFastH3EulerSampler", [380, 260], [300, 110],
               [12.0, 3.0, "auto"], "BSAI④ FastH3 Euler 4步采样器",
               outputs=[OUT("SAMPLER", "SAMPLER", [8])]))
nodes.append(N(5, "CLIPLoader", [0, 300], [330, 100],
               [TEXT_ENC, "minimax", "default"], "CLIP 文本编码器 (qwen3vl 32B)",
               outputs=[OUT("CLIP", "CLIP", [3])]))
nodes.append(N(6, "VAELoader", [0, 430], [330, 70], [VAE_VIDEO],
               "视频 VAE (int8_convrot)", outputs=[OUT("VAE", "VAE", [5, 10])]))
nodes.append(N(7, "VAELoader", [0, 530], [330, 70], [VAE_AUDIO],
               "音频 VAE (fp32)", outputs=[OUT("VAE", "VAE", [6, 11])]))
nodes.append(N(8, "MiniMaxH3ImageToVideo", [770, 0], [420, 160],
               [PROMPT, 864, 480, 124], "FastH3 T2VA · 文生视频+音频（首尾帧不接）",
               inputs=[INP("clip", "CLIP", 3), INP("vae", "VAE", 5)],
               outputs=[OUT("positive", "CONDITIONING", [6]),
                        OUT("latent", "LATENT", [9])]))
nodes.append(N(9, "RandomNoise", [770, 200], [280, 60], [123456662],
               "随机噪波", outputs=[OUT("NOISE", "NOISE", [12])]))
nodes.append(N(10, "BasicGuider", [770, 300], [300, 70], [],
               "BasicGuider",
               inputs=[INP("model", "MODEL", 4), INP("conditioning", "CONDITIONING", 6)],
               outputs=[OUT("GUIDER", "GUIDER", [13])]))
nodes.append(N(11, "SamplerCustomAdvanced", [770, 400], [300, 110], [],
               "SamplerCustomAdvanced",
               inputs=[INP("noise", "NOISE", 12), INP("guider", "GUIDER", 13),
                       INP("sampler", "SAMPLER", 8), INP("sigmas", "SIGMAS", 7),
                       INP("latent_image", "LATENT", 9)],
               outputs=[OUT("output", "LATENT", [14, 15])]))
nodes.append(N(12, "VAEDecode", [1120, 0], [300, 70], [], "视频解码 VAEDecode",
               inputs=[INP("samples", "LATENT", 14), INP("vae", "VAE", 10)],
               outputs=[OUT("IMAGE", "IMAGE", [16])]))
nodes.append(N(13, "VAEDecodeAudio", [1120, 110], [300, 70], [], "音频解码 VAEDecodeAudio",
               inputs=[INP("samples", "LATENT", 15), INP("vae", "VAE", 11)],
               outputs=[OUT("AUDIO", "AUDIO", [17, 18])]))
nodes.append(N(14, "CreateVideo", [1470, 0], [300, 90], [24.0, "auto"],
               "合成视频 24fps+音轨",
               inputs=[INP("images", "IMAGE", 16), INP("audio", "AUDIO", 17)],
               outputs=[OUT("VIDEO", "VIDEO", [19])]))
nodes.append(N(15, "SaveVideo", [1470, 140], [300, 90], ["BSAI_FastH3", "auto"],
               "保存视频",
               inputs=[INP("video", "VIDEO", 19)], outputs=[]))
nodes.append(N(16, "SaveAudio", [1470, 260], [300, 70], ["BSAI_FastH3"],
               "保存音频 (FLAC)",
               inputs=[INP("audio", "AUDIO", 18)], outputs=[]))
nodes.append(N(17, "BSAIFastH3VSAStats", [380, 420], [340, 70], [],
               "BSAI VSA 命中统计（只读）",
               inputs=[INP("model", "MODEL", 20)],
               outputs=[OUT("stats", "STRING", [])]))

links = [
    [1, 1, 0, 2, 0, "MODEL"],
    [2, 2, 0, 3, 0, "MODEL"],
    [3, 5, 0, 8, 0, "CLIP"],
    [4, 2, 0, 10, 0, "MODEL"],
    [5, 6, 0, 8, 1, "VAE"],
    [6, 8, 0, 10, 1, "CONDITIONING"],
    [7, 3, 0, 11, 3, "SIGMAS"],
    [8, 4, 0, 11, 2, "SAMPLER"],
    [9, 8, 1, 11, 4, "LATENT"],
    [10, 6, 0, 12, 1, "VAE"],
    [11, 7, 0, 13, 1, "VAE"],
    [12, 9, 0, 11, 0, "NOISE"],
    [13, 10, 0, 11, 1, "GUIDER"],
    [14, 11, 0, 12, 0, "LATENT"],
    [15, 11, 0, 13, 0, "LATENT"],
    [16, 12, 0, 14, 0, "IMAGE"],
    [17, 13, 0, 14, 1, "AUDIO"],
    [18, 13, 0, 16, 0, "AUDIO"],
    [19, 14, 0, 15, 0, "VIDEO"],
    [20, 2, 0, 17, 0, "MODEL"],
]

wf = {
    "last_node_id": 17,
    "last_link_id": 20,
    "nodes": nodes,
    "links": links,
    "groups": [],
    "config": {},
    "extra": {},
    "version": 0.4,
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(wf, f, ensure_ascii=False, indent=2)
print("written:", OUT_PATH)

# ---- 校验 ----
data = json.load(open(OUT_PATH, encoding="utf-8"))
assert data["last_node_id"] == 17 and data["last_link_id"] == 20
ids = {n["id"] for n in data["nodes"]}
assert len(ids) == len(data["nodes"]), "duplicate node ids"
for link in data["links"]:
    lid, oid, osl, tid, tsl, typ = link
    assert oid in ids and tid in ids, f"link {lid} dangling endpoint"
declared = set()
for n in data["nodes"]:
    for o in n.get("outputs", []):
        declared.update(o.get("links") or [])
declared.discard(None)
table = {l[0] for l in data["links"]}
assert declared <= table, f"declared not in table: {declared - table}"
for l in data["links"]:
    assert l[0] in declared, f"link {l[0]} in table but not declared"
print("validation OK: nodes", len(data["nodes"]), "links", len(data["links"]))
