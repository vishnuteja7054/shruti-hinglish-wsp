"""Pure-NumPy I/O, quantization, and export utilities for Udgam Labs Shruti (.safetensors <-> .cact).

Matches Cactus-Compute/needle export.py & quantize.py bit-for-bit without requiring JAX/Flax.
"""
import functools
import hashlib
import json
import math
import struct
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# 1. Safetensors Reader / Writer (Pure NumPy, zero external dependencies)
# ============================================================================

_ST_DTYPE_TO_NP = {
    "F32": np.float32,
    "F16": np.float16,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
}
_NP_TO_ST_DTYPE = {v: k for k, v in _ST_DTYPE_TO_NP.items()}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_safetensors(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load all tensors and __metadata__ from a .safetensors file."""
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hdr_len).decode("utf-8"))
        data_start = 8 + hdr_len
        raw = f.read()

    meta = header.pop("__metadata__", {})
    tensors: Dict[str, np.ndarray] = {}
    for name, info in header.items():
        dtype = _ST_DTYPE_TO_NP[info["dtype"]]
        shape = tuple(info["shape"])
        start, end = info["data_offsets"]
        arr = np.frombuffer(raw, dtype=dtype, count=(end - start) // np.dtype(dtype).itemsize, offset=start)
        tensors[name] = arr.reshape(shape).copy()
    return tensors, meta


def save_safetensors(path: str, tensors: Dict[str, np.ndarray], metadata: Optional[Dict[str, str]] = None) -> str:
    """Save tensors and __metadata__ to a .safetensors file and return its SHA-256."""
    header: Dict[str, Any] = {}
    if metadata is not None:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    offset = 0
    blobs: List[bytes] = []
    for name in sorted(tensors.keys()):
        arr = np.ascontiguousarray(tensors[name])
        dt_str = _NP_TO_ST_DTYPE.get(arr.dtype.type)
        if dt_str is None:
            arr = arr.astype(np.float32)
            dt_str = "F32"
        b = arr.tobytes()
        header[name] = {
            "dtype": dt_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(b)],
        }
        blobs.append(b)
        offset += len(b)

    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    hdr_bytes += b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for b in blobs:
            f.write(b)
    return sha256_file(path)


def load_safetensors_flat(path: str) -> Dict[str, np.ndarray]:
    """Load all tensors as a flat dict from a .safetensors file."""
    tensors, _ = load_safetensors(path)
    return tensors


save_safetensors_flat = save_safetensors




# Mapping between flat safetensors keys and Flax nested dict keys
_BLOCK_RENAME = {
    "attn_norm": ("ZCRMSNorm_0",),
    "q_proj": ("self_attn", "q_proj"),
    "k_proj": ("self_attn", "k_proj"),
    "v_proj": ("self_attn", "v_proj"),
    "q_norm": ("self_attn", "q_norm"),
    "k_norm": ("self_attn", "k_norm"),
    "gate_proj": ("self_attn", "gate_proj"),
    "out_proj": ("self_attn", "out_proj"),
    "q_taps": ("self_attn", "q_taps"),
    "k_taps": ("self_attn", "k_taps"),
    "v_taps": ("self_attn", "v_taps"),
    "cross_q": ("cross_attn", "q_proj"),
    "cross_k": ("cross_attn", "k_proj"),
    "cross_v": ("cross_attn", "v_proj"),
    "cross_q_norm": ("cross_attn", "q_norm"),
    "cross_k_norm": ("cross_attn", "k_norm"),
    "cross_gate_proj": ("cross_attn", "gate_proj"),
    "cross_out": ("cross_attn", "out_proj"),
}


def unflatten_safetensors(flat: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Convert flat safetensors dictionary into nested Flax param tree expected by export."""
    tree: Dict[str, Any] = {}
    for key, arr in flat.items():
        parts = tuple(key.split("/"))
        if len(parts) >= 2 and parts[1] == "blocks":
            rest = parts[2:]
            if len(rest) >= 2 and rest[0] in ("hadamard_mlp", "hadamard_mlp_0") and rest[1] == "norm":
                norm_name = "pre_hada_norm_0" if rest[0] == "hadamard_mlp_0" else "pre_hada_norm"
                parts = (parts[0], "layers", "block", norm_name) + rest[2:]
            elif rest[0] in _BLOCK_RENAME:
                parts = (parts[0], "layers", "block") + _BLOCK_RENAME[rest[0]] + rest[1:]
            else:
                parts = (parts[0], "layers", "block") + rest
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = arr
    # Add zero-initialized post norms if not saved in safetensors
    enc_block = tree["encoder"]["layers"]["block"]
    dec_block = tree["stack"]["layers"]["block"]
    enc_L, dim = enc_block["ZCRMSNorm_0"]["scale"].shape
    dec_L = dec_block["ZCRMSNorm_0"]["scale"].shape[0]
    enc_block.setdefault("post_attn_norm", {"scale": np.zeros((enc_L, dim), dtype=np.float32)})
    dec_block.setdefault("post_attn_norm", {"scale": np.zeros((dec_L, dim), dtype=np.float32)})
    dec_block.setdefault("post_cross_norm", {"scale": np.zeros((dec_L, dim), dtype=np.float32)})
    return tree


# ============================================================================
# 2. Cactus Quants (CQ-W4) & Hadamard Utilities
# ============================================================================

WEIGHT_BITS = 4
TERNARY_BITS = 1.58
_TERNARY_CB = np.array([-1.2240064, 0.0, 1.2240064])


@functools.lru_cache(maxsize=None)
def _lloyd_max_gaussian(bits: int, iters: int = 200, samples: int = 400000, seed: int = 0) -> np.ndarray:
    if bits == 1:
        c = math.sqrt(2.0 / math.pi)
        return np.array([-c, c])
    levels = 1 << bits
    x = np.sort(np.random.RandomState(seed).randn(samples))
    c = x[((np.arange(levels) + 0.5) / levels * samples).astype(int)].astype(np.float64)
    for _ in range(iters):
        bnd = (c[:-1] + c[1:]) / 2.0
        idx = np.searchsorted(bnd, x)
        for k in range(levels):
            m = idx == k
            if m.any():
                c[k] = x[m].mean()
    return np.sort(c)


@functools.lru_cache(maxsize=None)
def _cq_codebook_np(bits: float, group_size: int) -> np.ndarray:
    cb = _TERNARY_CB if bits == TERNARY_BITS else _lloyd_max_gaussian(int(bits))
    return (cb / np.sqrt(group_size)).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _cq_hadamard_np(group_size: int) -> np.ndarray:
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < group_size:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(group_size)).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _hada_perms(n: int, split: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if not split:
        p1 = np.random.RandomState(11).permutation(n).astype(np.int32)
        p2 = np.random.RandomState(13).permutation(n).astype(np.int32)
        return p1, p2
    half = n // 2
    p1_lo = np.random.RandomState(11).permutation(half)
    p1_hi = half + np.random.RandomState(12).permutation(half)
    p2_lo = np.random.RandomState(13).permutation(half)
    p2_hi = half + np.random.RandomState(14).permutation(half)
    return np.concatenate([p1_lo, p1_hi]).astype(np.int32), np.concatenate([p2_lo, p2_hi]).astype(np.int32)


def cq_quantize_dequantize_np(w: np.ndarray, bits: int = 4, group: int = 128) -> np.ndarray:
    """Simulate exact CQ-W4 pack + unpack (including FP16 group norms) on a 2D [out, in] matrix."""
    packed, norms = _cq_pack(w, bits, group)
    return _cq_unpack(packed, norms, w.shape[0], w.shape[1], bits, group)


# ============================================================================
# 3. CACT Container Pack / Unpack & RefTokenizer
# ============================================================================

TAG = 0x05E12A84
ALIGN = 64
FP16, FP32, CQ, RAW = 1, 2, 3, 4
CB_BITS = (2, 3, 4)
TERNARY_RECORD_BITS = 5

TK_NORMAL, TK_UNKNOWN, TK_CONTROL, TK_USER_DEFINED, TK_BYTE = 0, 1, 2, 3, 4
_TK_HDR = "<IIIIIBBH"
_TK_REC = "<fBH"

_HDR_FMT = "<48If"
_REC_FMT = "<BBHIIIIQQII"
REC_SIZE = struct.calcsize(_REC_FMT)
ENGRAM_SUB_DIM = 128
ENGRAM_CONV_TAPS = 4
AUDIO_MANIFEST_MAGIC = 3246.0
HADAMARD = ("d1", "d2", "b2", "d3", "d4", "w1a", "w1b", "w2a", "w2b", "w3a", "w3b", "cond_v", "cond_u")
PROJ = ("q_proj", "k_proj", "v_proj", "gate_proj", "out_proj")
_SP_META_SPACE = "▁"


@dataclass(frozen=True)
class WhistleConfig:
    text: int = 8906
    languages: tuple = ("en", "de", "fr", "es", "it", "nl", "pl", "hi")
    dim: int = 512
    kv_heads: int = 2
    qk_dim: int = 48
    v_dim: int = 64
    enc_layers: int = 8
    dec_layers: int = 8
    conv_kernel: int = 9
    taps: int = 3
    lanes: int = 4
    enc_lanes: int = 4
    sites: tuple = (3, 7)
    orders: tuple = (2, 3)
    slots: int = 18432
    hashes: int = 2
    stem: int = 128
    stem_stages: int = 3
    rope: float = 100000.0
    mels: int = 80
    rate: int = 16000
    fft: int = 512
    win: int = 400
    hop: int = 160
    max_tokens: int = 320
    pad: int = 0
    eos: int = 1
    bos: int = 2

    @classmethod
    def from_saved(cls, saved: Dict[str, Any]) -> "WhistleConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items() if k in known})

    @property
    def vocab(self) -> int:
        return self.text + len(self.languages)

    @property
    def heads(self) -> int:
        return self.dim // self.v_dim

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


WHISTLE_SMALL_CFG = WhistleConfig()



@dataclass
class ExportDecoderConfig:
    vocab_size: int
    d_model: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    qk_head_dim: int
    v_head_dim: int
    max_seq_len: int
    pad_token_id: int
    rope_theta: float
    mhc_lanes: int
    qkv_conv_taps: int
    engram_layers: tuple
    engram_slots: int
    engram_orders: tuple
    engram_heads: int
    kv_bits: int = 8


def make_decoder_export_config(cfg: WhistleConfig) -> ExportDecoderConfig:
    return ExportDecoderConfig(
        vocab_size=cfg.vocab,
        d_model=cfg.dim,
        num_heads=cfg.heads,
        num_kv_heads=cfg.kv_heads,
        num_layers=cfg.dec_layers,
        qk_head_dim=cfg.qk_dim,
        v_head_dim=cfg.v_dim,
        max_seq_len=cfg.max_tokens,
        pad_token_id=cfg.pad,
        rope_theta=cfg.rope,
        mhc_lanes=cfg.lanes,
        qkv_conv_taps=cfg.taps,
        engram_layers=cfg.sites,
        engram_slots=cfg.slots,
        engram_orders=cfg.orders,
        engram_heads=cfg.hashes,
        kv_bits=8,
    )


@functools.lru_cache(maxsize=8)
def mel_matrix(cfg: WhistleConfig) -> np.ndarray:
    lo = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
    hi = 2595.0 * np.log10(1.0 + (cfg.rate / 2.0) / 700.0)
    pts = 700.0 * (10.0 ** (np.linspace(lo, hi, cfg.mels + 2) / 2595.0) - 1.0)
    freqs = np.linspace(0.0, cfg.rate / 2.0, cfg.fft // 2 + 1)
    fb = np.zeros((cfg.fft // 2 + 1, cfg.mels), dtype=np.float32)
    for m in range(cfg.mels):
        l, c, r = pts[m], pts[m + 1], pts[m + 2]
        up = (freqs - l) / max(c - l, 1e-10)
        dn = (r - freqs) / max(r - c, 1e-10)
        fb[:, m] = np.maximum(0.0, np.minimum(up, dn)) * (2.0 / max(r - l, 1e-10))
    return fb


def _align(n: int) -> int:
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def _nearest_idx(x: np.ndarray, cb: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1)
    pos = np.clip(np.searchsorted(cb, flat), 1, len(cb) - 1)
    left, right = cb[pos - 1], cb[pos]
    idx = np.where(np.abs(flat - left) <= np.abs(flat - right), pos - 1, pos)
    return idx.reshape(x.shape).astype(np.uint8)


def _pack_lsb(idx: np.ndarray, bits: int) -> np.ndarray:
    out, in_pad = idx.shape
    g = idx.reshape(out, in_pad // 8, 8).astype(np.uint64)
    word = np.zeros(g.shape[:-1], np.uint64)
    for i in range(8):
        word |= g[..., i] << (i * bits)
    packed = np.empty(word.shape + (bits,), np.uint8)
    for b in range(bits):
        packed[..., b] = (word >> (8 * b)) & 0xFF
    return packed.reshape(out, in_pad * bits // 8)


def _unpack_lsb(packed: np.ndarray, bits: int, in_pad: int) -> np.ndarray:
    out = packed.shape[0]
    chunks = packed.reshape(out, in_pad // 8, bits).astype(np.uint64)
    word = np.zeros(chunks.shape[:-1], np.uint64)
    for b in range(bits):
        word |= chunks[..., b] << (8 * b)
    idx = np.empty((out, in_pad // 8, 8), np.uint8)
    mask = (1 << bits) - 1
    for i in range(8):
        idx[..., i] = (word >> (i * bits)) & mask
    return idx.reshape(out, in_pad)


def _packed_row_bytes(in_pad: int, bits: int, group: int) -> int:
    if bits == TERNARY_RECORD_BITS:
        return in_pad * 2 // 8
    return in_pad * bits // 8


def _cq_pack(w: np.ndarray, bits: int, group: int) -> Tuple[np.ndarray, np.ndarray]:
    cb = _cq_codebook_np(bits, group)
    H = _cq_hadamard_np(group)
    out, D = w.shape
    pad = (-D) % group
    wp = np.pad(w, ((0, 0), (0, pad))) if pad else w
    in_pad = wp.shape[1]
    g = wp.reshape(out, in_pad // group, group).astype(np.float32)
    rot = g @ H
    norm = np.sqrt((rot ** 2).sum(-1, keepdims=True))
    unit = rot / np.maximum(norm, 1e-12)
    idx = _nearest_idx(unit, cb).reshape(out, in_pad)
    return _pack_lsb(idx, bits), norm[:, :, 0].astype(np.float16)


def _cq_unpack(packed: np.ndarray, norms: np.ndarray, out: int, in_dim: int, bits: int, group: int) -> np.ndarray:
    record_bits = bits
    if bits == TERNARY_RECORD_BITS:
        bits = TERNARY_BITS
    cb = _cq_codebook_np(bits, group)
    H = _cq_hadamard_np(group)
    in_pad = (in_dim + group - 1) // group * group
    idx = _unpack_lsb(packed, int(bits), in_pad)
    unit = cb[idx].reshape(out, in_pad // group, group)
    rot = unit * norms.reshape(out, in_pad // group, 1).astype(np.float32)
    w = (rot @ H).reshape(out, in_pad)
    return w[:, :in_dim]


class _Tensor:
    __slots__ = ("name", "dtype", "shape", "blob", "group", "bits", "offset")

    def __init__(self, name, dtype, shape, blob, group=0, bits=0):
        self.name, self.dtype, self.shape = name, dtype, tuple(shape)
        self.blob, self.group, self.bits, self.offset = blob, group, bits, 0


def _fp16(name: str, arr: np.ndarray) -> _Tensor:
    arr = np.asarray(arr, np.float16)
    return _Tensor(name, FP16, arr.shape, arr.tobytes())


def _q(name: str, mat: np.ndarray, bits: int, group: int) -> _Tensor:
    packed, norms = _cq_pack(np.asarray(mat, np.float32), bits, group)
    return _Tensor(name, CQ, mat.shape, packed.tobytes() + norms.tobytes(), group, bits)


def _geometry(config: ExportDecoderConfig):
    qk_hd = config.qk_head_dim or (config.d_model // config.num_heads)
    v_hd = config.v_head_dim or (config.d_model // config.num_heads)
    orders = tuple(getattr(config, "engram_orders", (2, 3)))
    heads = getattr(config, "engram_heads", 0) or max(1, config.d_model // (len(orders) * ENGRAM_SUB_DIM))
    sub_dim = config.d_model // (len(orders) * heads)
    sites = tuple(getattr(config, "engram_layers", (3, 7)))
    return qk_hd, v_hd, orders, heads, sub_dim, sites


def _audio_tensors(params: Dict[str, Any], cfg: WhistleConfig, bits: int, group: int) -> List[_Tensor]:
    ts = [_Tensor("audio.manifest", FP32, (13,), np.asarray([
        AUDIO_MANIFEST_MAGIC, 1, cfg.text, cfg.enc_layers, cfg.enc_lanes, cfg.conv_kernel, cfg.stem,
        cfg.stem_stages, cfg.mels, cfg.rate, cfg.fft, cfg.win, cfg.hop], np.float32).tobytes())]

    def proj(prefix, p, i):
        return [_q(f"{prefix}.{n}", np.asarray(p[n]["kernel"][i]).T, bits, group) for n in PROJ]

    b = params["stack"]["layers"]["block"]
    ca = b["cross_attn"]
    for i in range(cfg.dec_layers):
        ts.append(_fp16(f"layer{i:02d}.cross_norm", b["cross_norm"]["scale"][i]))
        ts += proj(f"layer{i:02d}.cross", ca, i)
        ts += [
            _fp16(f"layer{i:02d}.cross_q_norm", ca["q_norm"]["scale"][i]),
            _fp16(f"layer{i:02d}.cross_k_norm", ca["k_norm"]["scale"][i]),
            _fp16(f"layer{i:02d}.post_cross_norm", b["post_cross_norm"]["scale"][i]),
            _fp16(f"layer{i:02d}.cross_gate", np.asarray(b["cross_gate"][i]).reshape(1)),
        ]
    ts.append(_fp16("pe_gate", np.asarray(params["pe_gate"]).reshape(1)))

    e = params["encoder"]["layers"]["block"]
    for i in range(cfg.enc_layers):
        ts += [_fp16(f"encoder{i:02d}.{n}", e[n]["scale"][i]) for n in (
            "pre_hada_norm_0", "ZCRMSNorm_0", "post_attn_norm", "conv_norm", "conv_out_norm", "pre_hada_norm")]
        ts += proj(f"encoder{i:02d}", e["self_attn"], i)
        ts += [
            _fp16(f"encoder{i:02d}.q_norm", e["self_attn"]["q_norm"]["scale"][i]),
            _fp16(f"encoder{i:02d}.k_norm", e["self_attn"]["k_norm"]["scale"][i]),
            _fp16(f"encoder{i:02d}.attn_gate", np.asarray(e["attn_gate"][i]).reshape(1)),
            _q(f"encoder{i:02d}.pw1", np.asarray(e["pw1"]["kernel"][i]).T, bits, group),
            _q(f"encoder{i:02d}.pw2", np.asarray(e["pw2"]["kernel"][i]).T, bits, group),
            _fp16(f"encoder{i:02d}.dw", np.asarray(e["dw"][i]).reshape(cfg.conv_kernel, -1)),
        ]
        ts += [_fp16(f"encoder{i:02d}.{mlp}.{k}", e[mlp][k][i])
               for mlp in ("hadamard_mlp_0", "hadamard_mlp") for k in HADAMARD]

    enc = params["encoder"]
    for name in ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post", "mhc_b_res"):
        ts.append(_fp16(f"encoder.{name}", np.asarray(enc[name])))
    for name in ("mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"):
        phi = np.asarray(enc[name])
        L, nC, lanes = phi.shape
        ts.append(_q(f"encoder.{name}", phi.transpose(0, 2, 1).reshape(L * lanes, nC), bits, group))
    ts.append(_fp16("encoder.final_norm", enc["final_norm"]["scale"]))

    stem = params["stem"]
    ts.append(_fp16("stem.w", np.asarray(stem["w"]).reshape(9, -1)))
    for i in range(1, cfg.stem_stages):
        ts += [_fp16(f"stem.dw_{i}", np.asarray(stem[f"dw_{i}"]).reshape(9, -1)),
               _q(f"stem.pw_{i}", np.asarray(stem[f"pw_{i}"]["kernel"]).T, bits, group)]
    ts.append(_q("stem.out", np.asarray(stem["out"]["kernel"]).T, bits, group))
    mel = mel_matrix(cfg)
    ts.append(_Tensor("mel", FP32, mel.shape, mel.tobytes()))
    return ts


def _tensors(params: Dict[str, Any], config: ExportDecoderConfig, bits: int, group: int, audio: WhistleConfig) -> List[_Tensor]:
    _, _, orders, heads, sub_dim, sites = _geometry(config)
    num_tables = len(orders) * heads
    embedding = np.asarray(params["embedding"]["embedding"])
    ts = [_q("embedding", embedding, bits, group)]

    taps_n = int(getattr(config, "qkv_conv_taps", 0))
    b = params["stack"]["layers"]["block"]
    sa, ha = b["self_attn"], b["hadamard_mlp"]
    for i in range(config.num_layers):
        ts += [
            _fp16(f"layer{i:02d}.norm_in", b["ZCRMSNorm_0"]["scale"][i]),
            _q(f"layer{i:02d}.q_proj", np.asarray(sa["q_proj"]["kernel"][i]).T, bits, group),
            _q(f"layer{i:02d}.k_proj", np.asarray(sa["k_proj"]["kernel"][i]).T, bits, group),
            _q(f"layer{i:02d}.v_proj", np.asarray(sa["v_proj"]["kernel"][i]).T, bits, group),
        ]
        if taps_n:
            ts += [
                _fp16(f"layer{i:02d}.q_taps", sa["q_taps"][i]),
                _fp16(f"layer{i:02d}.k_taps", sa["k_taps"][i]),
                _fp16(f"layer{i:02d}.v_taps", sa["v_taps"][i]),
            ]
        ts += [
            _fp16(f"layer{i:02d}.q_norm", sa["q_norm"]["scale"][i]),
            _fp16(f"layer{i:02d}.k_norm", sa["k_norm"]["scale"][i]),
            _q(f"layer{i:02d}.gate_proj", np.asarray(sa["gate_proj"]["kernel"][i]).T, bits, group),
            _q(f"layer{i:02d}.out_proj", np.asarray(sa["out_proj"]["kernel"][i]).T, bits, group),
            _fp16(f"layer{i:02d}.post_norm", b["post_attn_norm"]["scale"][i]),
            _fp16(f"layer{i:02d}.attn_gate", np.asarray(b["attn_gate"][i]).reshape(1)),
            _fp16(f"layer{i:02d}.pre_hada", b["pre_hada_norm"]["scale"][i]),
            _fp16(f"layer{i:02d}.d1", ha["d1"][i]),
            _fp16(f"layer{i:02d}.d2", ha["d2"][i]),
            _fp16(f"layer{i:02d}.b2", ha["b2"][i]),
            _fp16(f"layer{i:02d}.d3", ha["d3"][i]),
            _fp16(f"layer{i:02d}.d4", ha["d4"][i]),
            _fp16(f"layer{i:02d}.w1a", ha["w1a"][i]),
            _fp16(f"layer{i:02d}.w1b", ha["w1b"][i]),
            _fp16(f"layer{i:02d}.w2a", ha["w2a"][i]),
            _fp16(f"layer{i:02d}.w2b", ha["w2b"][i]),
            _fp16(f"layer{i:02d}.w3a", ha["w3a"][i]),
            _fp16(f"layer{i:02d}.w3b", ha["w3b"][i]),
            _fp16(f"layer{i:02d}.cond_v", ha["cond_v"][i]),
            _fp16(f"layer{i:02d}.cond_u", ha["cond_u"][i]),
        ]

    mhc = params["stack"]
    for name in ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post", "mhc_b_res"):
        ts.append(_fp16(name, np.asarray(mhc[name])))
    for name in ("mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"):
        phi = np.asarray(mhc[name])
        L, nC, lanes = phi.shape
        ts.append(_q(name, phi.transpose(0, 2, 1).reshape(L * lanes, nC), bits, group))

    hada_n = 1 << (config.d_model - 1).bit_length()
    p1, p2 = _hada_perms(hada_n, False)
    ts.append(_Tensor("hada_p1", FP32, (hada_n,), np.asarray(p1, np.float32).tobytes()))
    ts.append(_Tensor("hada_p2", FP32, (hada_n,), np.asarray(p2, np.float32).tobytes()))

    for s in range(len(sites)):
        eg = params[f"engrams_{s}"]
        tables = np.asarray(eg["embedding"]).reshape(num_tables * config.engram_slots, sub_dim)
        ts += [
            _q(f"engram{s}.tables", tables, bits, group),
            _q(f"engram{s}.key_proj", np.asarray(eg["key_proj"]["kernel"]).T, bits, group),
            _q(f"engram{s}.value_proj", np.asarray(eg["value_proj"]["kernel"]).T, bits, group),
            _fp16(f"engram{s}.taps", np.asarray(eg["taps"])),
        ]

    ts.append(_fp16("final_norm", params["stack"]["final_norm"]["scale"]))
    ts += _audio_tensors(params, audio, bits, group)
    return ts


def pack_cact(flat_safetensors: Dict[str, np.ndarray], whistle_cfg: WhistleConfig, tokenizer_blob: bytes,
              bits: int = WEIGHT_BITS, group: int = 128) -> bytes:
    """Pack a flat safetensors dict + WhistleConfig + raw tokenizer blob into a .cact binary buffer."""
    params = unflatten_safetensors(flat_safetensors)
    config = make_decoder_export_config(whistle_cfg)
    ts = _tensors(params, config, bits, group, whistle_cfg)
    ts.append(_Tensor("tokenizer", RAW, (), bytes(tokenizer_blob)))

    cb = np.concatenate([_cq_codebook_np(b, group) for b in CB_BITS]).astype(np.float32)
    kv_bits = 8
    qk_hd, v_hd, orders, heads, sub_dim, sites = _geometry(config)
    hada_n = 1 << (config.d_model - 1).bit_length()
    orders4 = (list(orders) + [0, 0, 0, 0])[:4]
    sites16 = (list(sites) + [0] * 16)[:16]
    header = struct.pack(
        _HDR_FMT, TAG, len(ts), len(cb), 0, kv_bits,
        config.vocab_size, 0,
        config.d_model, config.num_heads, config.num_kv_heads,
        config.num_layers, qk_hd, v_hd, config.max_seq_len, hada_n,
        config.mhc_lanes, 0,
        0, 0,
        config.qkv_conv_taps,
        config.engram_slots, sub_dim, len(orders) * heads,
        ENGRAM_CONV_TAPS, max(orders),
        0,
        len(orders), *orders4, len(sites), *sites16,
        float(config.rope_theta)
    ) + cb.tobytes()

    pos = len(header) + len(ts) * REC_SIZE
    for t in ts:
        pos = _align(pos)
        t.offset = pos
        pos += len(t.blob)

    directory = b"".join(
        struct.pack(_REC_FMT, t.dtype, len(t.shape), 0,
                    *(list(t.shape) + [0, 0, 0, 0])[:4], t.offset, len(t.blob),
                    t.group, t.bits)
        for t in ts
    )

    buf = bytearray(header + directory)
    for t in ts:
        buf.extend(b"\x00" * (t.offset - len(buf)))
        buf.extend(t.blob)
    return bytes(buf)


def read_tokenizer_blob(cact_path: str) -> bytes:
    with open(cact_path, "rb") as f:
        raw = f.read()
    tag, num_tensors, cb_n = struct.unpack_from(_HDR_FMT, raw, 0)[:3]
    if tag != TAG:
        raise ValueError(f"{cact_path} is not a Needle 3 .cact archive")
    off = struct.calcsize(_HDR_FMT) + cb_n * 4
    for _ in range(num_tensors):
        rec = struct.unpack(_REC_FMT, raw[off:off + REC_SIZE])
        off += REC_SIZE
        if rec[0] == RAW:
            return raw[rec[7]:rec[7] + rec[8]]
    raise ValueError(f"{cact_path} carries no tokenizer")


def parse_tokenizer_blob(blob: bytes) -> Dict[str, Any]:
    off = struct.calcsize(_TK_HDR)
    n, pad, eos, bos, unk, add_dummy, byte_fb, _ = struct.unpack_from(_TK_HDR, blob, 0)
    rec = struct.calcsize(_TK_REC)
    pieces, scores, types = [], [], []
    for _ in range(n):
        score, t, ln = struct.unpack_from(_TK_REC, blob, off)
        off += rec
        pieces.append(blob[off:off + ln].decode("utf-8"))
        scores.append(score)
        types.append(t)
        off += ln
    return {
        "pieces": pieces, "scores": scores, "types": types,
        "pad_id": pad, "eos_id": eos, "bos_id": bos, "unk_id": unk,
        "add_dummy_prefix": bool(add_dummy), "byte_fallback": bool(byte_fb)
    }


class RefTokenizer:
    def __init__(self, meta: Dict[str, Any]):
        self.pieces: List[str] = meta["pieces"]
        self.scores: List[float] = meta["scores"]
        self.types: List[int] = meta["types"]
        self.pad_id: int = meta["pad_id"]
        self.eos_id: int = meta["eos_id"]
        self.bos_id: int = meta["bos_id"]
        self.unk_id: int = meta["unk_id"]
        self.add_dummy: bool = meta["add_dummy_prefix"]
        self.byte_fallback: bool = meta["byte_fallback"]
        self.p2id = {p: i for i, p in enumerate(self.pieces)}
        self.byte_id = {
            int(p[3:5], 16): i
            for i, (p, t) in enumerate(zip(self.pieces, self.types)) if t == TK_BYTE
        }
        self.markers = sorted(
            (p for p, t in zip(self.pieces, self.types) if t == TK_USER_DEFINED),
            key=len, reverse=True
        )

    @classmethod
    def from_cact(cls, path: str) -> "RefTokenizer":
        return cls(parse_tokenizer_blob(read_tokenizer_blob(path)))

    def _bpe(self, seg: str) -> List[int]:
        syms = list(seg)
        while len(syms) > 1:
            best_score, best_j = None, -1
            for j in range(len(syms) - 1):
                idx = self.p2id.get(syms[j] + syms[j + 1])
                if idx is not None and (best_score is None or self.scores[idx] > best_score):
                    best_score, best_j = self.scores[idx], j
            if best_j < 0:
                break
            syms[best_j:best_j + 2] = [syms[best_j] + syms[best_j + 1]]
        ids: List[int] = []
        for s in syms:
            idx = self.p2id.get(s)
            if idx is not None:
                ids.append(idx)
            elif self.byte_fallback:
                ids.extend(self.byte_id[b] for b in s.encode("utf-8"))
            else:
                ids.append(self.unk_id)
        return ids

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        esc = text.replace(" ", _SP_META_SPACE)
        if self.add_dummy:
            esc = _SP_META_SPACE + esc
        ids, buf, i, n = [], [], 0, len(esc)
        while i < n:
            marker = next((m for m in self.markers if esc.startswith(m, i)), None)
            if marker is not None:
                ids += self._bpe("".join(buf))
                buf = []
                ids.append(self.p2id[marker])
                i += len(marker)
            else:
                buf.append(esc[i])
                i += 1
        ids += self._bpe("".join(buf))
        return ids

    def decode(self, ids: List[int]) -> str:
        buf = bytearray()
        for i in ids:
            t = self.types[i]
            if t == TK_BYTE:
                buf.append(int(self.pieces[i][3:5], 16))
            elif t in (TK_CONTROL, TK_UNKNOWN):
                continue
            else:
                buf += self.pieces[i].encode("utf-8")
        text = buf.decode("utf-8", "replace").replace(_SP_META_SPACE, " ")
        if self.add_dummy and text.startswith(" "):
            text = text[1:]
        return text


# ============================================================================
# 4. Surgical In-Place .cact Tensor & Row Patching (Zero Drift on Untouched Weights)
# ============================================================================

def get_cact_directory(cact_bytes: bytes, whistle_cfg: WhistleConfig,
                       dummy_flat: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Return the exact 529-entry directory mapping CACT tensor names to offsets and lengths."""
    params = unflatten_safetensors(dummy_flat)
    config = make_decoder_export_config(whistle_cfg)
    ts = _tensors(params, config, WEIGHT_BITS, 128, whistle_cfg)
    names = [t.name for t in ts] + ["tokenizer"]

    tag, num_tensors, cb_n = struct.unpack_from(_HDR_FMT, cact_bytes, 0)[:3]
    if tag != TAG or num_tensors != len(names):
        raise ValueError(f"CACT header mismatch: tag={hex(tag)}, num_tensors={num_tensors}, expected={len(names)}")

    off = struct.calcsize(_HDR_FMT) + cb_n * 4
    records = []
    for idx, name in enumerate(names):
        rec = struct.unpack(_REC_FMT, cact_bytes[off:off + REC_SIZE])
        off += REC_SIZE
        ndim = rec[1]
        shape = tuple(rec[3:3 + ndim])
        records.append({
            "index": idx,
            "name": name,
            "dtype": rec[0],
            "shape": shape,
            "offset": rec[7],
            "length": rec[8],
            "group": rec[9],
            "bits": rec[10],
        })
    return records


def surgical_patch_cact(
    orig_cact_bytes: bytes,
    new_flat: Dict[str, np.ndarray],
    orig_flat: Dict[str, np.ndarray],
    whistle_cfg: WhistleConfig,
    bits: int = WEIGHT_BITS,
    group: int = 128,
) -> Tuple[bytes, List[str]]:
    """Surgically patch only modified tensors (and only modified embedding rows) into a copy of orig_cact_bytes."""
    buf = bytearray(orig_cact_bytes)
    directory = get_cact_directory(orig_cact_bytes, whistle_cfg, orig_flat)
    rec_by_name = {r["name"]: r for r in directory}

    new_params = unflatten_safetensors(new_flat)
    orig_params = unflatten_safetensors(orig_flat)
    config = make_decoder_export_config(whistle_cfg)

    new_ts = {t.name: t for t in _tensors(new_params, config, bits, group, whistle_cfg)}
    orig_ts = {t.name: t for t in _tensors(orig_params, config, bits, group, whistle_cfg)}

    patched_names: List[str] = []

    # Check embedding row-by-row first so untouched vocabulary rows keep exact original bytes
    new_emb = np.asarray(new_flat["embedding/embedding"], dtype=np.float32)
    orig_emb = np.asarray(orig_flat["embedding/embedding"], dtype=np.float32)
    diff_rows = np.where(np.any(new_emb != orig_emb, axis=1))[0]
    if len(diff_rows) > 0:
        rec = rec_by_name["embedding"]
        V, D = rec["shape"]
        row_packed_bytes = (D * bits) // 8
        row_norm_bytes = (D // group) * 2
        packed_base = rec["offset"]
        norms_base = rec["offset"] + V * row_packed_bytes
        for r in diff_rows:
            p_row, n_row = _cq_pack(new_emb[r:r + 1], bits, group)
            p_bytes = p_row.tobytes()
            n_bytes = n_row.tobytes()
            buf[packed_base + r * row_packed_bytes: packed_base + (r + 1) * row_packed_bytes] = p_bytes
            buf[norms_base + r * row_norm_bytes: norms_base + (r + 1) * row_norm_bytes] = n_bytes
        patched_names.append(f"embedding(rows={len(diff_rows)})")

    # Check all other CACT records
    for name, nt in new_ts.items():
        if name == "embedding":
            continue
        ot = orig_ts[name]
        if nt.blob != ot.blob:
            rec = rec_by_name[name]
            if len(nt.blob) != rec["length"]:
                raise ValueError(f"Blob length mismatch for {name}: {len(nt.blob)} != {rec['length']}")
            buf[rec["offset"]:rec["offset"] + rec["length"]] = nt.blob
            patched_names.append(name)

    return bytes(buf), patched_names


_orig_surgical_patch_cact = surgical_patch_cact


def surgical_patch_cact(
    arg1: Any,
    arg2: Any,
    arg3: Any,
    arg4: Any,
    whistle_cfg: Any = WHISTLE_SMALL_CFG,
    bits: int = WEIGHT_BITS,
    group: int = 128,
) -> Any:
    """Supports both bytes-based and path-based surgical .cact patching."""
    if isinstance(whistle_cfg, dict):
        whistle_cfg = WhistleConfig.from_saved(whistle_cfg)
    if isinstance(arg1, (str, bytes)) and isinstance(arg2, str):
        # Called as (orig_cact_path, out_cact_path, orig_flat, new_flat, whistle_cfg)
        with open(arg1, "rb") as f:
            orig_bytes = f.read()
        orig_flat = arg3
        new_flat = arg4
        patched_bytes, patched_names = _orig_surgical_patch_cact(
            orig_bytes, new_flat, orig_flat, whistle_cfg, bits=bits, group=group
        )
        with open(arg2, "wb") as f:
            f.write(patched_bytes)
        return {
            "patched_tensors": patched_names,
            "num_patched": len(patched_names),
            "out_cact_bytes": len(patched_bytes),
        }
    return _orig_surgical_patch_cact(arg1, arg2, arg3, whistle_cfg, bits=bits, group=group)


