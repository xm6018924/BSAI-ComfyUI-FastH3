"""BSAI FastH3 Native VSA (Video Sparse Attention) engine for MiniMax-H3.

What it does
------------
FastVideo's FastH3 4-step preview cuts the video DiT cost in half through two
levers: 4-step DMD2 distillation and VSA (Video Sparse Attention). VSA does not
make every video token attend to every other video token; it scores 64-token
tiles and keeps only the informative ones exact, so attention cost drops by
roughly an order of magnitude while the conditioning rows (text / audio /
reference) stay exact.

This module installs that behaviour as a per-model patch through the same seam
ComfyUI's H3 attention reads: ``transformer_options["optimized_attention_override"]``.
It mirrors the approach of the reference Sol-Attn pack (arXiv 2607.24027) that
closerAI uses for "FastH3 Native VSA", minus the Morton reorder:

* H3's packed ``[text][cond][ref][audio][video]`` segment layout is published by
  patching ``PackedLayout`` + ``rope_freqs`` (idempotent, no mutation of the
  layout object).
* Conditioning query rows always run dense.
* Video query rows attend to every conditioning KV plus the top-``keep_frac``
  video tiles.

Backends
--------
* ``native`` : ``comfy_kitchen.sol_attn`` (BF16, head_dim 128, sm_80+, CUDA).
  Ineligible calls fall back to dense automatically. This is the recommended,
  fastest path.
* ``torch``  : pure-PyTorch block-sparse SDPA over 64-token tiles. Works on any
  GPU / dtype (incl. 40-series, fp32/bf16). Slower than the native kernel but
  still removes most of the quadratic video self-attention.

Every failure is caught and downgraded to the dense path, never a crash.
"""

import logging
import sys
from functools import partial

import torch
import torch.nn.functional as F

try:
    import comfy_kitchen as _ck
    _CK_IMPORT_ERROR = None
except Exception as exc:                                   # pragma: no cover
    _ck = None
    _CK_IMPORT_ERROR = exc

BLOCK = 64                       # VSA tile size (tiled 64-token, as in FastVideo)
_STATS = {"sparse": 0, "dense": 0, "native": 0, "torch": 0, "errors": 0}
_SEEN = set()
_SPAN_INSTALLED = set()
_PATCHED_LAYOUTS = set()
_SPANS = {}                      # id(position_ids) -> (layout, video_span, audio_span)


# ---------------------------------------------------------------------------
# H3 packed-layout span publishing (compact, no Morton)
# ---------------------------------------------------------------------------

def _video_span(layout):
    segments = getattr(layout, "segments", None)
    if not segments:
        return None
    return next(((a, b) for a, b, kind in segments if kind == "video"), None)


def _audio_span(layout):
    segments = getattr(layout, "segments", None)
    if not segments:
        return None
    return next(((a, b) for a, b, kind in segments if kind == "audio"), None)


def _patch_packed_layout(module):
    """Record the video span of every PackedLayout built, without mutating it."""
    layout_cls = getattr(module, "PackedLayout", None)
    if layout_cls is None:
        raise RuntimeError(f"{module.__name__} has no PackedLayout")
    if id(layout_cls) in _PATCHED_LAYOUTS:
        return
    original_init = layout_cls.__init__

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, *args, **kwargs):
        original_init(self, text_len, latent_t, latent_h, latent_w, audio_t, *args, **kwargs)
        try:
            span = _video_span(self)
        except Exception:                                   # never break model construction
            span = None
        if torch.is_tensor(getattr(self, "position_ids", None)) and span is not None:
            _SPANS[id(self.position_ids)] = (self, span, _audio_span(self))

    layout_cls.__init__ = __init__
    _PATCHED_LAYOUTS.add(id(layout_cls))


def install_h3_span(model):
    """Idempotently publish the H3 video/audio spans into transformer_options.

    Inert unless the model is a MiniMax-H3 diffusion model (has .blocks /
    .rope_freqs / ._forward). Safe to call multiple times on the same object.
    """
    if id(model) in _SPAN_INSTALLED:
        return
    for attr in ("rope_freqs", "_forward", "blocks"):
        if not hasattr(model, attr):
            raise RuntimeError(
                "BSAI FastH3 Native VSA expects a MiniMax-H3 diffusion model "
                f"(.{attr} missing on {type(model).__name__}).")

    _patch_packed_layout(sys.modules[type(model).__module__])
    original_forward = model._forward
    original_rope = model.rope_freqs

    def _forward(x, timestep, context, transformer_options={}, **kwargs):
        model._vsa_options = transformer_options
        try:
            return original_forward(x, timestep, context,
                                    transformer_options=transformer_options, **kwargs)
        finally:
            model._vsa_options = None
            transformer_options.pop("h3_video_span", None)
            transformer_options.pop("h3_audio_span", None)

    def rope_freqs(position_ids, device):
        entry = _SPANS.get(id(position_ids))
        if entry is not None:
            options = getattr(model, "_vsa_options", None)
            if options is not None:
                options["h3_video_span"] = entry[1]
                options["h3_audio_span"] = entry[2]
        return original_rope(position_ids, device)

    model._forward = _forward
    model.rope_freqs = rope_freqs
    _SPAN_INSTALLED.add(id(model))


def vsa_stats():
    """Dispatch counters since process start (or last reset)."""
    return dict(_STATS)


def reset_vsa_stats():
    for key in _STATS:
        _STATS[key] = 0
    _SEEN.clear()


def _log_once(key, message):
    if key not in _SEEN:
        _SEEN.add(key)
        logging.info(f"[BSAI FastH3 VSA] {message}")


# ---------------------------------------------------------------------------
# torch block-sparse backend
# ---------------------------------------------------------------------------

def _torch_vsa(qs, ks, vs, scale, keep_frac, video_start, video_end,
               sink_conditioning, verbose):
    """Block-sparse attention over the video span (pure PyTorch).

    qs/ks/vs are [B, N, H, D]. Video query rows attend to the conditioning KV
    (rows before video_start) plus the top-``keep_frac`` video tiles; every
    other row runs dense. Returns the output [B, N, H, D].

    Attention is computed with explicit matmul+softmax (not SDPA): the stock
    SDPA in this PyTorch build requires equal query/key lengths, while our
    conditioning rows and per-tile sparse groups have different lengths.
    """
    B, N, H, D = qs.shape
    cond_end = int(video_start)
    vn = int(video_end) - cond_end
    out = torch.empty_like(qs)

    def _attn(qr, kr, vr):
        """Softmax attention for arbitrary query/key lengths.
        qr [B, Lq, H, D], kr/vr [B, Lk, H, D] -> [B, Lq, H, D].

        Heads are folded into the batch axis via transpose (NOT plain reshape,
        which would scramble the (token, head) grid).
        """
        b, lq, h, d = qr.shape
        lk = kr.shape[1]
        q2 = qr.transpose(1, 2).reshape(b * h, lq, d)           # [B*H, Lq, D]
        k2 = kr.transpose(1, 2).reshape(b * h, lk, d).transpose(-2, -1)
        att = torch.bmm(q2, k2) * scale
        att = att.softmax(dim=-1)
        out = torch.bmm(att, vr.transpose(1, 2).reshape(b * h, lk, d))
        return out.reshape(b, h, lq, d).transpose(1, 2)         # [B, Lq, H, D]

    # Conditioning / audio / reference query rows -> dense over the whole
    # sequence (they steer prompt & audio). Chunked to bound peak memory.
    if cond_end > 0:
        for i in range(0, cond_end, 256):
            j = min(i + 256, cond_end)
            out[:, i:j] = _attn(qs[:, i:j], ks, vs)
    if vn <= 0:
        out[:, cond_end:] = _attn(qs[:, cond_end:], ks, vs)
        return out

    # --- video span: split into 64-token tiles -------------------------------
    nvb = (vn + BLOCK - 1) // BLOCK
    pad = nvb * BLOCK - vn

    def _slice(t):
        s = t[:, cond_end:video_end]
        if not pad:
            return s
        # F.pad fills from the last dim backwards: (0,0)=D, (0,0)=H,
        # (0,pad)=L(sequence), (0,0)=B -> pad the video span at the end.
        return F.pad(s, (0, 0, 0, 0, 0, pad, 0, 0))

    qv = _slice(qs).view(B, nvb, BLOCK, H, D)
    kv = _slice(ks).view(B, nvb, BLOCK, H, D)
    vv = _slice(vs).view(B, nvb, BLOCK, H, D)

    # Gate: centroid dot-product score per tile (learned-gate style proxy)
    qc = qv.mean(dim=2)                                     # [B, nvb, H, D]
    kc = kv.mean(dim=2)
    scores = torch.einsum("bihd,bjhd->bij", qc, kc) * scale
    topk = max(1, round(nvb * keep_frac))
    topk_idx = scores.topk(topk, dim=-1).indices            # [B, nvb, topk]

    tok_idx = (topk_idx * BLOCK).unsqueeze(-1) + torch.arange(BLOCK, device=qs.device)
    tok_flat = tok_idx.reshape(B, nvb, topk * BLOCK)        # token idx in 0..nvb*BLOCK-1
    kvf = kv.reshape(B, nvb * BLOCK, H * D)
    vvf = vv.reshape(B, nvb * BLOCK, H * D)

    # Conditioning KV is shared across query tiles; process in chunks to bound
    # peak memory instead of expanding it once for every tile.
    use_cond = sink_conditioning != "off" and cond_end > 0
    CHUNK = 2
    pieces = []
    for i in range(0, nvb, CHUNK):
        j = min(i + CHUNK, nvb)
        nb = j - i
        q2 = qv[:, i:j].reshape(B * nb, BLOCK, H, D)        # [B*nb, BLOCK, H, D]
        # per-chunk expanded video KV: [B*nb, nvb*BLOCK, HD]
        kexp = kvf.unsqueeze(1).expand(B, nb, nvb * BLOCK, H * D).reshape(B * nb, nvb * BLOCK, H * D)
        vexp = vvf.unsqueeze(1).expand(B, nb, nvb * BLOCK, H * D).reshape(B * nb, nvb * BLOCK, H * D)
        idx = tok_flat[:, i:j].reshape(B * nb, topk * BLOCK)
        gk = torch.gather(kexp, 1, idx.unsqueeze(-1).expand(B * nb, topk * BLOCK, H * D))
        gv = torch.gather(vexp, 1, idx.unsqueeze(-1).expand(B * nb, topk * BLOCK, H * D))
        if use_cond:
            ck = ks[:, :cond_end].reshape(B, cond_end, H, D) \
                   .unsqueeze(1).expand(B, nb, cond_end, H, D).reshape(B * nb, cond_end, H, D)
            cv = vs[:, :cond_end].reshape(B, cond_end, H, D) \
                   .unsqueeze(1).expand(B, nb, cond_end, H, D).reshape(B * nb, cond_end, H, D)
            k2 = torch.cat([ck, gk.reshape(B * nb, topk * BLOCK, H, D)], dim=1)
            v2 = torch.cat([cv, gv.reshape(B * nb, topk * BLOCK, H, D)], dim=1)
        else:
            k2 = gk.reshape(B * nb, topk * BLOCK, H, D)
            v2 = gv.reshape(B * nb, topk * BLOCK, H, D)
        pieces.append(_attn(q2, k2, v2))

    ov = torch.cat(pieces, dim=0).reshape(B, nvb * BLOCK, H, D)[:, :vn]
    out[:, cond_end:video_end] = ov
    _STATS["torch"] += 1
    if verbose:
        _log_once(("torch", nvb, topk, cond_end),
                  f"torch sparse: {B}×{N} tokens, video tiles {nvb}, "
                  f"top-{topk} kept ({keep_frac * 100:.1f}%), conditioning rows {cond_end} exact")
    return out


# ---------------------------------------------------------------------------
# native comfy_kitchen.sol_attn backend
# ---------------------------------------------------------------------------

def _native_eligible(qs, ks, min_tokens):
    if _ck is None or not hasattr(_ck, "sol_attn"):
        return "comfy_kitchen sol_attn unavailable"
    if qs.device.type != "cuda":
        return "not cuda"
    if qs.dtype != torch.bfloat16:
        return f"dtype {qs.dtype} (kernel is bf16-only)"
    if qs.shape[-1] != 128:
        return f"head_dim {qs.shape[-1]} != 128"
    if qs.shape[1] != ks.shape[1] or qs.shape[2] != ks.shape[2]:
        return "q/k shape mismatch"
    if qs.shape[1] < min_tokens:
        return f"seq {qs.shape[1]} < {min_tokens}"
    return None


def _run_native(qs, ks, vs, scale, tau, video_start, sink_conditioning, verbose):
    reason = _native_eligible(qs, ks, 0)
    if reason is not None:
        if verbose:
            _log_once((tuple(qs.shape), reason), f"native declined ({reason}); dense")
        return None
    sink_blocks = (0, (int(video_start) + BLOCK - 1) // BLOCK)
    sink_q = (0, 0) if sink_conditioning != "exact_kv_and_rows" else sink_blocks
    out = _ck.sol_attn(
        qs, ks, vs, tau=tau, scale=scale,
        sink_blocks=list(sink_blocks), sink_q=list(sink_q), max_blocks=0,
        centroid_tail=True,
    )                                                            # BTHD
    _STATS["native"] += 1
    if verbose:
        _log_once((tuple(qs.shape), "native"),
                  f"native sparse {tuple(qs.shape)} tau={tau:.2f} sink={sink_blocks}")
    return out


def _tau_from_keep(keep_frac):
    """Sol-Attn's tau that keeps roughly ``keep_frac`` of blocks exact.
    tau=1.0 ~ 16%, 1.5 ~ 7%, 2.0 ~ 2.7% -> linear interpolation in between."""
    return max(0.05, 1.0 + 3.0 * (0.16 - keep_frac))


# ---------------------------------------------------------------------------
# override builder (installed into transformer_options["optimized_attention_override"])
# ---------------------------------------------------------------------------

def make_vsa_override(*, backend, keep_frac, tau, min_tokens, sigma_start, sigma_end,
                      sink_conditioning, verbose, previous=None):
    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        def dense():
            target = func if previous is None else partial(previous, func)
            return target(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape,
                          skip_output_reshape=skip_output_reshape, **kwargs)

        if mask is not None:
            _STATS["dense"] += 1
            return dense()

        if skip_reshape:
            b, _, _, dim_head = q.shape                        # BHND
            qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
        else:
            b, _, dim_head = q.shape                           # B, N, heads*dim_head
            dim_head //= heads
            qs, ks, vs = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        # sigma window (dense warm-up steps)
        if sigma_start is not None or sigma_end is not None:
            sigmas = kwargs.get("transformer_options", {}).get("sigmas")
            if sigmas is not None:
                sigma = float(sigmas[0])
                if (sigma_start is not None and sigma > sigma_start) or \
                   (sigma_end is not None and sigma < sigma_end):
                    _STATS["dense"] += 1
                    return dense()

        options = kwargs.get("transformer_options") or {}
        span = options.get("h3_video_span")
        if span is None:
            _STATS["dense"] += 1
            if verbose:
                _log_once("nospan", "no H3 video span published; dense")
            return dense()
        video_start, video_end = span
        tokens = qs.shape[1]
        if tokens < min_tokens or tokens < int(video_end):
            _STATS["dense"] += 1
            return dense()
        if qs.shape[1] != ks.shape[1]:                         # cross-attention
            _STATS["dense"] += 1
            return dense()

        scale = kwargs.get("scale", None)
        try:
            if backend == "native":
                out = _run_native(qs, ks, vs, scale, tau, video_start,
                                  sink_conditioning, verbose)
                if out is None:
                    _STATS["dense"] += 1
                    return dense()
                if skip_output_reshape:
                    return out.transpose(1, 2)                 # BHND
                return out.reshape(b, -1, heads * dim_head)
            out = _torch_vsa(qs, ks, vs, scale, keep_frac, video_start, video_end,
                             sink_conditioning, verbose)
        except Exception as exc:
            _STATS["errors"] += 1
            logging.error(f"[BSAI FastH3 VSA] backend failed ({exc}); dense fallback",
                          exc_info=verbose)
            return dense()
        _STATS["sparse"] += 1
        if skip_output_reshape:
            return out.transpose(1, 2)
        return out.reshape(b, -1, heads * dim_head)

    return override


# ---------------------------------------------------------------------------
# top-level apply (called by the node)
# ---------------------------------------------------------------------------

def apply_vsa(model, *, enabled, keep_percent, tau, start_percent, end_percent,
              min_tokens, sink_conditioning, backend, strict_native_backend,
              verbose, compose_with_foreign_patches=True):
    if not enabled:
        logging.info("[BSAI FastH3 VSA] disabled -> passthrough")
        return model

    m = model.clone()
    diffusion_model = m.get_model_object("diffusion_model")
    if not (hasattr(diffusion_model, "rope_freqs") and hasattr(diffusion_model, "_forward")):
        raise RuntimeError(
            "BSAI FastH3 Native VSA expects a MiniMax-H3 diffusion model; got "
            f"{type(diffusion_model).__name__}.")

    if backend == "native" and (_ck is None or not hasattr(_ck, "sol_attn")):
        if strict_native_backend:
            raise RuntimeError(
                "BSAI FastH3 Native VSA: comfy_kitchen.sol_attn is unavailable "
                f"({_CK_IMPORT_ERROR}); choose backend=torch or set "
                "strict_native_backend=False to use the pure-PyTorch sparse path.")
        logging.warning("[BSAI FastH3 VSA] native backend unavailable -> torch fallback")
        backend = "torch"

    install_h3_span(diffusion_model)
    ms = m.get_model_object("model_sampling")
    sigma_start = float(ms.percent_to_sigma(start_percent))
    sigma_end = float(ms.percent_to_sigma(end_percent))
    keep_frac = max(0.005, min(1.0, keep_percent / 100.0))
    tau = tau if (tau is not None and tau > 0) else _tau_from_keep(keep_frac)

    previous = m.model_options["transformer_options"].get("optimized_attention_override")
    if previous is not None:
        logging.info("[BSAI FastH3 VSA] chaining onto an existing attention override")

    if compose_with_foreign_patches:
        _install_compose_hooks(diffusion_model, "attn")

    m.model_options["transformer_options"]["optimized_attention_override"] = \
        make_vsa_override(backend=backend, keep_frac=keep_frac, tau=tau,
                          min_tokens=min_tokens, sigma_start=sigma_start,
                          sigma_end=sigma_end, sink_conditioning=sink_conditioning,
                          verbose=verbose, previous=previous)
    m.model_options["transformer_options"]["bsai_fasth3_vsa"] = {
        "backend": backend, "keep_percent": keep_percent, "tau": round(tau, 3),
        "min_tokens": min_tokens, "sink": sink_conditioning,
        "sigma_start": sigma_start, "sigma_end": sigma_end}
    reset_vsa_stats()
    logging.info(f"[BSAI FastH3 VSA] applied: backend={backend} "
                 f"keep={keep_percent}% tau={tau:.2f} min_tokens={min_tokens} "
                 f"sink={sink_conditioning} window=[{start_percent},{end_percent}]")
    return m


# ---------------------------------------------------------------------------
# composition with foreign attention object-patches (e.g. Sage low-VRAM patches)
# ---------------------------------------------------------------------------

_COMPOSE_HOOKED = set()


def _compose_module_patch(module, patched_forward):
    """Gate an object-patched attention forward: when the VSA gate says take it,
    run the stock forward (which reaches our optimized_attention_override); else
    keep the patch's own forward."""
    stock = type(module).forward

    def forward(*args, **kwargs):
        options = kwargs.get("transformer_options")
        if not isinstance(options, dict):
            options = next((a for a in args if isinstance(a, dict) and "bsai_fasth3_vsa" in a), {})
        gate = options.get("bsai_fasth3_vsa")
        x = args[0] if args else None
        tensor = x[0] if isinstance(x, list) and len(x) == 1 and torch.is_tensor(x[0]) else x
        take = gate is not None and torch.is_tensor(tensor) and tensor.device.type == "cuda"
        if take:
            tokens = tensor.shape[0] if tensor.ndim == 2 else tensor.shape[1]
            take = tokens >= gate.get("min_tokens", 0)
        if take:
            sigmas = options.get("sigmas")
            if sigmas is not None:
                sigma = float(sigmas[0])
                # dense warm-up window mirrors the override's percent gate
                take = not (gate.get("sigma_start") is not None and sigma > gate["sigma_start"]) \
                       and not (gate.get("sigma_end") is not None and sigma < gate["sigma_end"])
        if take:
            if tensor is not x:
                x.clear()                                      # consume the hand-off list
                args = (tensor,) + args[1:]
            return stock(module, *args, **kwargs)
        return patched_forward(*args, **kwargs)

    forward._bsai_composed = True
    return forward


def _install_compose_hooks(model, attn_attr):
    """Re-wrap any foreign attn forward before each block runs, so the VSA
    override composes with object patches applied by other nodes."""
    if id(model) in _COMPOSE_HOOKED:
        return

    def pre_hook(block, args):
        attn = getattr(block, attn_attr, None)
        if attn is None:
            return None
        fwd = attn.__dict__.get("forward")
        if fwd is None or getattr(fwd, "_bsai_composed", False):
            return None
        if getattr(fwd, "_uses_optimized_attention", False):
            return None
        if getattr(fwd, "__func__", None) is type(attn).forward:
            return None
        attn.forward = _compose_module_patch(attn, fwd)
        _log_once(("composed", attn_attr),
                  f"composing with a patched {attn_attr}.forward; VSA takes "
                  "eligible self-attention calls, the patch keeps the rest")
        return None

    for block in model.blocks:
        block.register_forward_pre_hook(pre_hook)
    _COMPOSE_HOOKED.add(id(model))
