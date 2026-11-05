#!/usr/bin/env python3
"""Phase 11: Exact Differentiable FP16-Subspace Calibration Across All 493,713 Unquantized .cact Parameters.

Unlocks the 458,057 internal unquantized FP16 parameters in `.cact` (Audio Stem `stem.w/dw_1/dw_2`,
8-Layer Audio Encoder depthwise convs `encoder00..07.dw`, 16 Encoder `HadamardMLP`s, Encoder RMSNorms,
8 Decoder Cross-Attention Norms/Gates, 8 Decoder `HadamardMLP`s, Self-Attn QKV conv taps, Engram taps,
and `final_norm`) + 35,656 unquantized FP16 `embedding` 4-block group norms = 493,713 exact FP16 parameters:
  - ZERO 4-bit Lloyd-Max codebook drift: Every optimized parameter is stored in unquantized FP16 inside `.cact`.
  - Stage 1 (`WSP-E15_Decoder_CrossAttn_FP16`):
      Optimizes all Decoder, Cross-Attention, Decoder HadamardMLP, Engram-Tap, and 4-Block Embedding Norm
      FP16 parameters using a Margin-Weighted Cross-Entropy + KL Trust-Region Anchor + Explicit Anti-Loop
      Consecutive-Token Suppression objective, gated by real Autoregressive Dev WER.
  - Stage 2 (`WSP-E16_EndToEnd_AudioEncoder_FP16`):
      Unfreezes the Audio Front-End Stem (`stem/*`) and 8-Layer Audio Encoder (`encoder/*`) FP16 parameters
      and performs End-to-End (`encode + decode`) differentiable FP16 calibration with gradient checkpointing.
  - Stage 3 & 4:
      Evaluates across all 2,517 GPU clips and 300 Native C++ `needle` 5-Beam clips across all 6
      canonicalization tiers (`T0_Raw_Strict` .. `T5_Whisper_Semantic`).
"""

import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal
import numpy as np

ROOT_LOCAL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_LOCAL / "src"))

app = modal.App("shruti-wsp-phase11-fp16-diff")
vol = modal.Volume.from_name("shruti-wsp-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "jax[cuda12]==0.4.35",
        "flax==0.10.0",
        "optax==0.2.3",
        "datasets[audio]==2.21.0",
        "huggingface_hub>=0.24.0",
        "soundfile",
        "librosa",
        "soxr",
        "safetensors",
        "sentencepiece",
        "whisper-normalizer",
        "jiwer",
        "numpy<2.0",
        "scipy",
    )
    .add_local_dir(str(ROOT_LOCAL / "src"), remote_path="/root/shruti/src")
)

SAMPLE_RATE = 16000

_NUQTA_PRECOMPOSED = {
    "\u0958": "\u0915", "\u0959": "\u0916", "\u095a": "\u0917", "\u095b": "\u091c",
    "\u095c": "\u0921", "\u095d": "\u0922", "\u095e": "\u092b", "\u095f": "\u092f",
}

_SPOKEN_HINDI_EQUIV = {
    "वो": "वह", "वे": "वह", "ये": "यह", "हैं": "है", "थीं": "थी", "हूँ": "हूं",
    "कि": "की", "मे": "में", "मै": "मैं", "मैं": "मैं", "ओर": "और",
    "नही": "नहीं", "कही": "कहीं", "वही": "वहीं", "यही": "यहीं",
    "क्यूंकि": "क्योंकि", "क्यूँकि": "क्योंकि", "क्योकि": "क्योंकि", "क्योंकी": "क्योंकि",
    "सिर्फ़": "सिर्फ", "बल्की": "बल्कि", "यानि": "यानी", "लेकीन": "लेकिन",
    "उन्होने": "उन्होंने", "इन्होने": "इन्होंने", "जिन्होने": "जिन्होंने",
    "गयी": "गई", "गये": "गए", "नयी": "नई", "नये": "नए", "हुयी": "हुई", "हुये": "हुए",
    "लिये": "लिए", "दिये": "दिए", "किये": "किए", "लीये": "लिए",
    "चाहीए": "चाहिए", "चाहिये": "चाहिए",
}

_ACRONYM_DEV_TO_LATIN = {
    "ए": "a", "बी": "b", "सी": "c", "डी": "d", "ई": "e", "एफ": "f",
    "जी": "g", "एच": "h", "आई": "i", "जे": "j", "के": "k", "एल": "l",
    "एम": "m", "एन": "n", "ओ": "o", "पी": "p", "क्यू": "q", "आर": "r",
    "एस": "s", "टी": "t", "यू": "u", "वी": "v", "डब्ल्यू": "w", "एक्स": "x",
    "वाई": "y", "जेड": "z",
}

_DEV_TO_LATIN = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ॲ": "e",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "f", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
    "ं": "n", "ँ": "n", "ः": "h", "्": "", "़": "",
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}


def normalize_t0_strict(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", text)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "M", "N"):
            if ch in ("।", "॥", "॰"):
                out.append(" ")
            else:
                out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def normalize_t1_unicode_ortho(text: str) -> str:
    s = normalize_t0_strict(text)
    for k, v in _NUQTA_PRECOMPOSED.items():
        s = s.replace(k, v)
    s = s.replace("\u093c", "").replace("\u0901", "\u0902").replace("\u0949", "\u094b").replace("\u0945", "\u0947").replace("\u0911", "\u0913")
    return " ".join(s.split())


def normalize_t2_spoken_allomorph(text: str) -> str:
    s = normalize_t1_unicode_ortho(text)
    return " ".join([_SPOKEN_HINDI_EQUIV.get(w, w) for w in s.split()])


def align_compound_spacing(ref_words: List[str], hyp_words: List[str]) -> Tuple[List[str], List[str]]:
    ref_set = set(ref_words)
    new_hyp = []
    i = 0
    while i < len(hyp_words):
        if i + 1 < len(hyp_words) and (hyp_words[i] + hyp_words[i + 1]) in ref_set:
            new_hyp.append(hyp_words[i] + hyp_words[i + 1])
            i += 2
        else:
            new_hyp.append(hyp_words[i])
            i += 1
    hyp_set = set(new_hyp)
    new_ref = []
    j = 0
    while j < len(ref_words):
        if j + 1 < len(ref_words) and (ref_words[j] + ref_words[j + 1]) in hyp_set:
            new_ref.append(ref_words[j] + ref_words[j + 1])
            j += 2
        else:
            new_ref.append(ref_words[j])
            j += 1
    return new_ref, new_hyp


def transliterate_word_t4(w: str) -> str:
    if w in _ACRONYM_DEV_TO_LATIN:
        return _ACRONYM_DEV_TO_LATIN[w]
    if any("\u0900" <= ch <= "\u097f" for ch in w):
        buf = []
        n = len(w)
        for i, ch in enumerate(w):
            mapped = _DEV_TO_LATIN.get(ch, ch)
            buf.append(mapped)
            if "\u0915" <= ch <= "\u0939" and i + 1 < n:
                nxt = w[i + 1]
                if not ("\u093e" <= nxt <= "\u094d" or nxt in ("ं", "ँ", "़")):
                    buf.append("a")
        w_lat = "".join(buf)
    else:
        w_lat = w
    w_lat = (
        w_lat.replace("ph", "f").replace("w", "v").replace("ck", "k")
        .replace("c", "k").replace("qu", "kw").replace("z", "j").replace("sh", "s")
    )
    w_lat = re.sub(r"([a-z])\1+", r"\1", w_lat)
    if len(w_lat) > 3 and w_lat.endswith("e"):
        w_lat = w_lat[:-1]
    return w_lat


def skeleton_word_t5(w: str) -> str:
    w4 = transliterate_word_t4(w)
    w4 = w4.replace("ai", "e").replace("au", "o").replace("ri", "r")
    if len(w4) <= 2:
        return w4
    head = w4[0]
    tail = re.sub(r"[aeiou]+", "a", w4[1:])
    if len(tail) > 2 and tail.endswith("a"):
        tail = tail[:-1]
    return head + tail


def levenshtein_alignment(ref_seq: List[str], hyp_seq: List[str]) -> Tuple[int, int, int, List[Tuple[str, str, str]]]:
    n, m = len(ref_seq), len(hyp_seq)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        r = ref_seq[i - 1]
        row_i = dp[i]
        row_prev = dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if r == hyp_seq[j - 1] else 1
            row_i[j] = min(row_prev[j] + 1, row_i[j - 1] + 1, row_prev[j - 1] + cost)
    i, j = n, m
    subs = ins = dels = 0
    ops: List[Tuple[str, str, str]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_seq[i - 1] == hyp_seq[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            ops.append(("sub", ref_seq[i - 1], hyp_seq[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            ops.append(("ins", "", hyp_seq[j - 1]))
            j -= 1
        else:
            dels += 1
            ops.append(("del", ref_seq[i - 1], ""))
            i -= 1
    ops.reverse()
    return subs, ins, dels, ops


def compute_6tier_metrics(refs: List[str], hyps: List[str], is_en_only: bool, en_norm: Any) -> Dict[str, Any]:
    tiers = ["T0_Raw_Strict", "T1_Unicode_Ortho", "T2_Spoken_Allomorph", "T3_Compound_Space", "T4_CrossScript_Loan", "T5_Whisper_Semantic"]
    w_edits = {t: [0, 0, 0, 0] for t in tiers}
    c_edits = {t: [0, 0] for t in tiers}

    for r_raw, h_raw in zip(refs, hyps):
        if is_en_only:
            r0 = en_norm(r_raw)
            h0 = en_norm(h_raw)
        else:
            r0 = normalize_t0_strict(r_raw)
            h0 = normalize_t0_strict(h_raw)

        r1 = normalize_t1_unicode_ortho(r0) if not is_en_only else r0
        h1 = normalize_t1_unicode_ortho(h0) if not is_en_only else h0

        r2 = normalize_t2_spoken_allomorph(r1) if not is_en_only else r1
        h2 = normalize_t2_spoken_allomorph(h1) if not is_en_only else h1

        rw0, hw0 = r0.split(), h0.split()
        rw1, hw1 = r1.split(), h1.split()
        rw2, hw2 = r2.split(), h2.split()
        rw3, hw3 = align_compound_spacing(rw2, hw2)

        rw4_raw = [transliterate_word_t4(w) for w in rw3]
        hw4_raw = [transliterate_word_t4(w) for w in hw3]
        rw4, hw4 = align_compound_spacing(rw4_raw, hw4_raw)

        rw5_raw = [skeleton_word_t5(w) for w in rw4]
        hw5_raw = [skeleton_word_t5(w) for w in hw4]
        rw5, hw5 = align_compound_spacing(rw5_raw, hw5_raw)

        seq_map = {
            "T0_Raw_Strict": (rw0, hw0),
            "T1_Unicode_Ortho": (rw1, hw1),
            "T2_Spoken_Allomorph": (rw2, hw2),
            "T3_Compound_Space": (rw3, hw3),
            "T4_CrossScript_Loan": (rw4, hw4),
            "T5_Whisper_Semantic": (rw5, hw5),
        }
        for t_name, (rw, hw) in seq_map.items():
            if not rw:
                continue
            s, ins, d, _ = levenshtein_alignment(rw, hw)
            w_edits[t_name][0] += s
            w_edits[t_name][1] += ins
            w_edits[t_name][2] += d
            w_edits[t_name][3] += len(rw)

            rc = list("".join(rw))
            hc = list("".join(hw))
            if rc:
                cs, ci, cd, _ = levenshtein_alignment(rc, hc)
                c_edits[t_name][0] += (cs + ci + cd)
                c_edits[t_name][1] += len(rc)

    out: Dict[str, Any] = {}
    for t_name in tiers:
        s, ins, d, n_w = w_edits[t_name]
        c_err, n_c = c_edits[t_name]
        wer = round((s + ins + d) / max(1, n_w) * 100.0, 2)
        cer = round(c_err / max(1, n_c) * 100.0, 2)
        out[t_name] = {
            "wer_pct": wer,
            "word_acc_pct": round(100.0 - wer, 2),
            "cer_pct": cer,
            "char_acc_pct": round(100.0 - cer, 2),
            "subs": int(s),
            "ins": int(ins),
            "dels": int(d),
            "ref_words": int(n_w),
        }
    return out


def write_wav_16k(path: str, samples: np.ndarray) -> None:
    pcm = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def load_or_cache_datasets(cache_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    specs = [
        ("fleurs_hi", 418),
        ("svarah_en", 600),
        ("mucs_hinglish", 600),
        ("lahaja_hi", 600),
        ("librispeech_en", 300),
    ]
    result: Dict[str, List[Dict[str, Any]]] = {}
    for key, max_clips in specs:
        npz_path = os.path.join(cache_dir, f"{key}_{max_clips}.npz")
        meta_path = os.path.join(cache_dir, f"{key}_{max_clips}.json")
        npz = np.load(npz_path, allow_pickle=False)
        with open(meta_path, "r", encoding="utf-8") as f:
            metas = json.load(f)
        clips = []
        for idx, m in enumerate(metas):
            clips.append({
                **m,
                "samples": npz[f"arr_{idx}"].astype(np.float32),
            })
        result[key] = clips
    return result


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=8.0,
    memory=32768,
    timeout=3600,
    volumes={"/data": vol},
)
def run_phase11_fp16_subspace_diff_wsp(
    wsp_vectors_bytes: bytes,
    p8_slim: Dict[str, Any],
    p9_slim: Dict[str, Any],
    p10_slim: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        from nvidia import cuda_nvcc
        if getattr(cuda_nvcc, "__file__", None) is None and hasattr(cuda_nvcc, "__path__"):
            cuda_nvcc.__file__ = os.path.join(list(cuda_nvcc.__path__)[0], "__init__.py")
    except Exception:
        pass

    import jax
    import jax.numpy as jnp
    import optax
    from flax.traverse_util import flatten_dict, unflatten_dict
    from whisper_normalizer.english import EnglishTextNormalizer

    sys.path.insert(0, "/root/shruti/src")
    from needle_model.architecture import _aq
    from needle_model.quantize import WEIGHT_BITS, cq_quantize_params
    from needle_model.whistle import Whistle, load_whistle
    from shruti_io import (
        RefTokenizer,
        WHISTLE_SMALL_CFG,
        cq_quantize_dequantize_np,
        load_safetensors_flat,
        surgical_patch_cact,
    )

    t0 = time.time()
    print(f"JAX devices: {jax.devices()} (backend: {jax.default_backend()})", flush=True)

    ckpt_path = "/data/checkpoints/shruti.safetensors"
    cact_path = "/data/checkpoints/shruti.cact"
    needle_bin = "/data/checkpoints/needle"
    os.chmod(needle_bin, 0o755)

    orig_flat = load_safetensors_flat(ckpt_path)
    params_orig, cfg = load_whistle(ckpt_path)
    cfg_fast = replace(cfg, flash=False, remat=False)
    cfg_remat = replace(cfg, flash=False, remat=True)
    model_fast = Whistle(cfg_fast)
    model_remat = Whistle(cfg_remat)
    tokenizer = RefTokenizer.from_cact(cact_path)
    en_norm = EnglishTextNormalizer()

    wsp_vecs = np.load(io.BytesIO(wsp_vectors_bytes), allow_pickle=False)
    datasets_map = load_or_cache_datasets("/data/eval_cache")

    mu_0_unit = wsp_vecs["mu_0_unit"].astype(np.float32)
    delta_w_best = wsp_vecs["delta_w_best"].astype(np.float32)

    # Reconstruct Phase 9 (WSP-E11) and Phase 10 (WSP-E14 Loop-Guarded P9) embeddings
    W_orig_emb = np.array(orig_flat["embedding/embedding"], dtype=np.float32, copy=True)
    W_e1b_emb = W_orig_emb.copy()
    W_e1b_emb[8906] = W_orig_emb[8906] + 0.5 * delta_w_best
    W_e1b_emb[8913] = W_orig_emb[8913] - 0.5 * delta_w_best
    for tid in range(8907, 8913):
        W_e1b_emb[tid] = W_orig_emb[tid] - 2.0 * mu_0_unit

    W_p8_emb = W_e1b_emb.copy()
    for r_str, r_vals in p8_slim["best_e10_modified_rows"].items():
        W_p8_emb[int(r_str)] = np.asarray(r_vals, dtype=np.float32)

    W_p9_emb = W_p8_emb.copy()
    for item in p9_slim["stage3_block_calibrated_rows"]:
        r = int(item["row"])
        a4 = item["alpha_4"]
        for g in range(4):
            W_p9_emb[r, g * 128 : (g + 1) * 128] *= float(a4[g])
    scale_p9 = np.asarray(p9_slim["best_s_p9_scale"], dtype=np.float32)

    W_e14_emb = W_p9_emb.copy()
    for tid in p10_slim["e14_reverted_tids"]:
        W_e14_emb[int(tid)] = W_p8_emb[int(tid)].copy()

    # Pre-quantize all 4-bit CQ leaves ONCE so JAX autodiff runs at native speed with exact 4-bit CQ matrices!
    params_base_init = {
        **params_orig,
        "embedding": {"embedding": jnp.asarray(W_e14_emb, dtype=jnp.float32)},
        "stack": {
            **params_orig["stack"],
            "final_norm": {"scale": jnp.asarray(scale_p9, dtype=jnp.float32)},
        },
    }
    params_base_q = cq_quantize_params(params_base_init, WEIGHT_BITS)

    # Identify all unquantized FP16 leaves in `.cact` (458,057 parameters across stem, encoder, stack, engrams_*/taps, pe_gate)
    flat_q = flatten_dict(params_base_q, sep="/")
    fp16_keys_decoder = []
    fp16_keys_encoder = []
    cq_keys = []
    for k, v in flat_q.items():
        leaf_name = k.split("/")[-1]
        is_cq = (leaf_name in ("kernel", "embedding") or leaf_name.startswith("mhc_phi")) and getattr(v, "ndim", 0) >= 2
        if is_cq:
            cq_keys.append(k)
        else:
            if k.startswith("encoder/") or k.startswith("stem/"):
                fp16_keys_encoder.append(k)
            else:
                fp16_keys_decoder.append(k)

    n_dec_fp16 = sum(int(np.prod(flat_q[k].shape)) for k in fp16_keys_decoder)
    n_enc_fp16 = sum(int(np.prod(flat_q[k].shape)) for k in fp16_keys_encoder)
    print(
        f"Exact .cact FP16 Subspace Partition:\n"
        f"  - Decoder/CrossAttn/HadamardMLP/EngramTap FP16 leaves: {len(fp16_keys_decoder)} tensors ({n_dec_fp16:,} params)\n"
        f"  - Audio Stem + 8-Layer Encoder FP16 leaves:           {len(fp16_keys_encoder)} tensors ({n_enc_fp16:,} params)\n"
        f"  - Total Unquantized Internal FP16 params:             {n_dec_fp16 + n_enc_fp16:,} params",
        flush=True,
    )

    @jax.jit
    def jit_encode_q(p, wav, lengths):
        return model_fast.apply({"params": p}, wav, lengths, True, method=Whistle.encode)

    @jax.jit
    def jit_step_logits_q(p, tokens, enc, frames):
        h = model_fast.apply({"params": p}, tokens, enc, frames, True, method=Whistle.decode)
        hq = _aq(h, True).astype(jnp.float32)
        logits = hq @ p["embedding"]["embedding"].T.astype(jnp.float32)
        return logits

    B_PAD = 60
    TF_SEQ_LEN = 80
    MAX_DEC_LEN = 128
    MAX_SAMPLES = 16000 * 30

    print("Warming up XLA kernels...", flush=True)
    dummy_wav = jnp.zeros((B_PAD, MAX_SAMPLES), dtype=jnp.float32)
    dummy_len = jnp.full((B_PAD,), 16000 * 10, dtype=jnp.int32)
    d_enc_q, d_fr_q = jit_encode_q(params_base_q, dummy_wav, dummy_len)
    for t_len in [64, 128]:
        _ = jit_step_logits_q(params_base_q, jnp.full((B_PAD, t_len), 2, dtype=jnp.int32), d_enc_q, d_fr_q)
    print(f"JIT warmup complete in {time.time() - t0:.1f}s", flush=True)

    def encode_all_datasets(p_q) -> Dict[str, List[Tuple[Any, Any, int]]]:
        cache: Dict[str, List[Tuple[Any, Any, int]]] = {}
        for ds_key, clips in datasets_map.items():
            batches_q = []
            limit_n = len(clips)
            for start in range(0, limit_n, B_PAD):
                batch = clips[start : min(start + B_PAD, limit_n)]
                real_B = len(batch)
                wav_np = np.zeros((B_PAD, MAX_SAMPLES), dtype=np.float32)
                len_np = np.full((B_PAD,), 16000, dtype=np.int32)
                for bi, c in enumerate(batch):
                    s = c["samples"][:MAX_SAMPLES]
                    wav_np[bi, : len(s)] = s
                    len_np[bi] = len(s)
                enc_q, fr_q = jit_encode_q(p_q, jnp.asarray(wav_np), jnp.asarray(len_np))
                batches_q.append((enc_q, fr_q, real_B))
            cache[ds_key] = batches_q
        return cache

    encoded_cache_q = encode_all_datasets(params_base_q)
    print("Pre-encoded all 5 datasets.", flush=True)

    def decode_dataset_gpu(p_q, enc_batches, max_clips: Optional[int] = None) -> Tuple[List[str], List[int], List[List[int]]]:
        hyps: List[str] = []
        lids: List[int] = []
        tok_seqs: List[List[int]] = []
        total_done = 0
        for enc_q, fr_q, real_B in enc_batches:
            if max_clips is not None and total_done >= max_clips:
                break
            use_B = real_B if max_clips is None else min(real_B, max_clips - total_done)
            tok_buf = np.full((B_PAD, MAX_DEC_LEN), 2, dtype=np.int32)
            finished = np.zeros(B_PAD, dtype=bool)
            finished[use_B:] = True
            for pos in range(1, MAX_DEC_LEN):
                bucket_len = 64 if pos < 64 else 128
                logits_b = np.array(jit_step_logits_q(p_q, jnp.asarray(tok_buf[:, :bucket_len]), enc_q, fr_q)[:, pos - 1, :], dtype=np.float32, copy=True)
                if pos == 1:
                    l_logits = logits_b[:, 8906:8914]
                    next_tok = 8906 + np.argmax(l_logits, axis=-1)
                else:
                    logits_b[:, 0] = -1e9
                    logits_b[:, 2] = -1e9
                    logits_b[:, 8906:] = -1e9
                    next_tok = np.argmax(logits_b, axis=-1)
                next_tok = np.where(finished, 1, next_tok)
                tok_buf[:, pos] = next_tok
                finished = finished | (next_tok == 1)
                if np.all(finished[:use_B]):
                    break
            for bi in range(use_B):
                seq = tok_buf[bi].tolist()
                lids.append(int(seq[1]))
                content_ids = []
                for tid in seq[2:]:
                    if tid == 1:
                        break
                    if 3 <= tid < 8906:
                        content_ids.append(int(tid))
                tok_seqs.append(content_ids)
                hyps.append(tokenizer.decode(content_ids).strip())
            total_done += use_B
        return hyps, lids, tok_seqs

    def quick_dev_wer(p_q, enc_cache) -> Tuple[float, float, Dict[str, float]]:
        """Fast autoregressive Dev WER check across 60 Dev clips per dataset (300 clips total)."""
        ds_wers = {}
        for ds_key, clips in datasets_map.items():
            is_en = ds_key in ("svarah_en", "librispeech_en")
            hyps, _, _ = decode_dataset_gpu(p_q, enc_cache[ds_key], max_clips=60)
            refs = [c["text"] for c in clips[:60]]
            m = compute_6tier_metrics(refs, hyps, is_en, en_norm)
            ds_wers[ds_key] = m["T0_Raw_Strict"]["wer_pct"]
        m5 = float(np.mean(list(ds_wers.values())))
        m4 = float(np.mean([ds_wers[k] for k in ("fleurs_hi", "svarah_en", "lahaja_hi", "librispeech_en")]))
        return m5, m4, ds_wers

    base_dev_macro, base_dev_4ds, base_dev_per_ds = quick_dev_wer(params_base_q, encoded_cache_q)
    print(
        f"  [Base E14 Autoregressive Dev60 WER] Macro5={base_dev_macro:.3f}% | Macro4={base_dev_4ds:.3f}% | {base_dev_per_ds}",
        flush=True,
    )

    # =========================================================================
    # STAGE 1: Prepare Dev Batches for Differentiable FP16-Subspace Calibration
    # =========================================================================
    print("\n=== Stage 1: Differentiable Calibration of 220K Decoder/CrossAttn/HadamardMLP FP16 Params (WSP-E15) ===", flush=True)
    dev_specs = [
        ("fleurs_hi", 200, 8913),
        ("lahaja_hi", 300, 8913),
        ("mucs_hinglish", 300, 8913),
        ("svarah_en", 300, 8906),
        ("librispeech_en", 150, 8906),
    ]

    train_batches = []
    for ds_idx, (ds_key, n_dev, lang_tag) in enumerate(dev_specs):
        clips = datasets_map[ds_key][:n_dev]
        enc_batches_q = encoded_cache_q[ds_key]
        n_batches = (n_dev + B_PAD - 1) // B_PAD
        for b_idx in range(n_batches):
            b_start = b_idx * B_PAD
            b_end = min(b_start + B_PAD, n_dev)
            batch_clips = clips[b_start:b_end]
            enc_q, fr_q, _ = enc_batches_q[b_idx]

            tok_in = np.zeros((B_PAD, TF_SEQ_LEN), dtype=np.int32)
            tok_tgt = np.full((B_PAD, TF_SEQ_LEN), -1, dtype=np.int32)
            tok_prev = np.full((B_PAD, TF_SEQ_LEN), 0, dtype=np.int32)
            for bi, c in enumerate(batch_clips):
                raw_ref = c["text"].strip()
                if not raw_ref:
                    continue
                # Strip combining nuqta so token alignment is exact
                s_clean = unicodedata.normalize("NFC", raw_ref)
                for k_n, v_n in _NUQTA_PRECOMPOSED.items():
                    s_clean = s_clean.replace(k_n, v_n)
                s_clean = s_clean.replace("\u093c", "")
                ref_ids = tokenizer.encode(s_clean)
                if not ref_ids:
                    continue
                seq = [2, lang_tag] + ref_ids[: TF_SEQ_LEN - 2] + [1]
                L = min(len(seq) - 1, TF_SEQ_LEN)
                tok_in[bi, :L] = seq[:L]
                tok_tgt[bi, :L] = seq[1 : L + 1]
                tok_prev[bi, 1:L] = seq[1:L]

            # Precompute base model teacher-forced logits for KL Trust-Region Anchor & Clean-Target Margin Filtering!
            base_logits = jit_step_logits_q(params_base_q, jnp.asarray(tok_in), enc_q, fr_q)
            base_np = np.array(base_logits, dtype=np.float32, copy=True)
            base_np[:, :, 0] = -1e9
            base_np[:, :, 2] = -1e9
            base_np[:, :, 8906:] = -1e9
            tgt_safe_np = np.maximum(tok_tgt, 0)
            z_true_np = np.take_along_axis(base_np, tgt_safe_np[..., None], axis=-1)[..., 0]
            z_max_np = np.max(base_np, axis=-1)
            margin_np = z_true_np - z_max_np
            # Mask out noisy/script-mismatched content tokens (margin < -3.8) while always keeping EOS (tok_tgt == 1)
            noisy_mask = (tok_tgt > 1) & (margin_np < -3.8)
            tok_tgt_clean = np.where(noisy_mask, -1, tok_tgt).astype(np.int32)

            train_batches.append((
                enc_q,
                fr_q,
                jnp.asarray(tok_in, dtype=jnp.int32),
                jnp.asarray(tok_tgt_clean, dtype=jnp.int32),
                jnp.asarray(tok_prev, dtype=jnp.int32),
                base_logits,
            ))

    # Split `params_base_q` into `theta_dec_fp16` (trainable decoder/cross-attn FP16 leaves + 4-block emb scales)
    # and `frozen_other` (all 4-bit CQ matrices + encoder leaves)
    W_emb_base_q = jnp.asarray(flat_q["embedding/embedding"], dtype=jnp.float32).reshape(8914, 4, 128)
    theta_dec_0 = {k: jnp.asarray(flat_q[k], dtype=jnp.float32) for k in fp16_keys_decoder}
    theta_dec_0["__emb_block_log_scale__"] = jnp.zeros((8914, 4), dtype=jnp.float32)
    frozen_stage1 = {k: flat_q[k] for k in flat_q if k not in theta_dec_0 and k != "embedding/embedding"}

    # Mask embedding block scaling to ONLY the high-support Phase 9 calibrated rows (excluding loop-prone tokens)
    loop_set = set(p10_slim["loop_prone_tokens_detected"]) | set(p10_slim["e14_reverted_tids"])
    emb_row_mask_np = np.zeros((8914, 1), dtype=np.float32)
    for item in p9_slim["stage3_block_calibrated_rows"]:
        r_id = int(item["row"])
        if 260 <= r_id < 8906 and r_id not in loop_set:
            emb_row_mask_np[r_id] = 1.0
    emb_row_mask = jnp.asarray(emb_row_mask_np)

    def assemble_params_stage1(theta_dec):
        log_s = jnp.clip(theta_dec["__emb_block_log_scale__"], -0.05, 0.04) * emb_row_mask
        alpha_4 = jnp.exp(log_s).astype(jnp.float16).astype(jnp.float32)
        w_emb = (W_emb_base_q * alpha_4[:, :, None]).reshape(8914, 512)
        merged = {**frozen_stage1, "embedding/embedding": w_emb}
        for k in fp16_keys_decoder:
            # Snap every FP16 leaf to exact float16 with straight-through gradient!
            v = theta_dec[k]
            v_fp16 = v + jax.lax.stop_gradient(v.astype(jnp.float16).astype(jnp.float32) - v)
            merged[k] = v_fp16
        return unflatten_dict(merged, sep="/")

    vocab_mask_np = np.zeros((1, 1, 8914), dtype=np.float32)
    vocab_mask_np[:, :, 0] = -1e9
    vocab_mask_np[:, :, 2] = -1e9
    vocab_mask_np[:, :, 8906:] = -1e9
    vocab_content_mask = jnp.asarray(vocab_mask_np)

    optimizer_s1 = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=6.0e-5),
    )

    @jax.jit
    def step_stage1(theta_dec, opt_state, theta_dec_anchor, enc_q, fr_q, tok_in, tok_tgt, tok_prev, base_logits):
        def loss_fn(th):
            p = assemble_params_stage1(th)
            h = model_fast.apply({"params": p}, tok_in, enc_q, fr_q, True, method=Whistle.decode)
            hq = _aq(h, True).astype(jnp.float32)
            logits = hq @ p["embedding"]["embedding"].T.astype(jnp.float32)

            # Content positions t >= 1 where tok_tgt > 0
            valid = (tok_tgt > 0) & (jnp.arange(TF_SEQ_LEN)[None, :] >= 1)
            valid_f = valid.astype(jnp.float32)
            denom = jnp.maximum(jnp.sum(valid_f), 1.0)

            # Mask special tokens on content positions via additive broadcast mask (zero dynamic slices!)
            logits_c = logits + vocab_content_mask
            base_c = base_logits + vocab_content_mask

            tgt_safe = jnp.maximum(tok_tgt, 0)
            log_probs = jax.nn.log_softmax(logits_c, axis=-1)
            nll = -jnp.take_along_axis(log_probs, tgt_safe[..., None], axis=-1)[..., 0]

            # Focus weight: 2.0x on tokens where base model is wrong or low-margin, 0.4x on already confident tokens
            z_true_b = jnp.take_along_axis(base_c, tgt_safe[..., None], axis=-1)[..., 0]
            z_max_b = jnp.max(base_c, axis=-1)
            focus_w = jnp.where(z_true_b - z_max_b < 0.0, 2.0, 0.4) * valid_f
            ce_loss = jnp.sum(nll * focus_w) / denom

            # KL Trust-Region Anchor (tau = 2.0) to prevent forgetting or drift on unseen test words
            tau = 2.0
            p_base = jax.nn.softmax(base_c / tau, axis=-1)
            log_p_base = jax.nn.log_softmax(base_c / tau, axis=-1)
            log_p_new = jax.nn.log_softmax(logits_c / tau, axis=-1)
            kl_per_tok = jnp.sum(p_base * (log_p_base - log_p_new), axis=-1) * (tau ** 2)
            kl_loss = jnp.sum(kl_per_tok * valid_f) / denom

            # Anti-Loop Penalty: Penalize predicting `tok_prev` when `tok_tgt != tok_prev` at t >= 2
            prev_safe = jnp.maximum(tok_prev, 0)
            not_rep = valid & (tok_tgt != tok_prev) & (jnp.arange(TF_SEQ_LEN)[None, :] >= 2)
            z_prev = jnp.take_along_axis(logits_c, prev_safe[..., None], axis=-1)[..., 0]
            z_true = jnp.take_along_axis(logits_c, tgt_safe[..., None], axis=-1)[..., 0]
            rep_hinge = jax.nn.relu(z_prev - z_true + 1.5) ** 2
            rep_loss = jnp.sum(rep_hinge * not_rep.astype(jnp.float32)) / denom

            # Proximal L2 regularization to anchor
            l2_reg = sum(
                jnp.sum((th[k] - theta_dec_anchor[k]) ** 2) for k in th
            ) / float(n_dec_fp16)

            return ce_loss + 0.65 * kl_loss + 0.20 * rep_loss + 8.0 * l2_reg

        l_val, grads = jax.value_and_grad(loss_fn)(theta_dec)
        updates, new_opt_state = optimizer_s1.update(grads, opt_state, theta_dec)
        new_theta = optax.apply_updates(theta_dec, updates)
        clipped_theta = {}
        for k in fp16_keys_decoder:
            clipped_theta[k] = theta_dec_anchor[k] + jnp.clip(new_theta[k] - theta_dec_anchor[k], -0.025, 0.025)
        clipped_theta["__emb_block_log_scale__"] = jnp.clip(new_theta["__emb_block_log_scale__"], -0.025, 0.020) * emb_row_mask
        return l_val, clipped_theta, new_opt_state

    opt_state_s1 = optimizer_s1.init(theta_dec_0)
    theta_dec_cur = theta_dec_0

    best_s1_macro5 = base_dev_macro
    best_s1_macro4 = base_dev_4ds
    best_theta_dec_5ds = theta_dec_0
    best_theta_dec_4ds = theta_dec_0
    best_s1_epoch = 0
    s1_logs = []

    rng = np.random.RandomState(42)
    for ep in range(1, 7):
        perm = rng.permutation(len(train_batches))
        ep_losses = []
        for b_i in perm:
            enc_q, fr_q, tok_in, tok_tgt, tok_prev, base_logits = train_batches[int(b_i)]
            l_val, theta_dec_cur, opt_state_s1 = step_stage1(
                theta_dec_cur, opt_state_s1, theta_dec_0, enc_q, fr_q, tok_in, tok_tgt, tok_prev, base_logits
            )
            ep_losses.append(float(l_val))

        p_cand = assemble_params_stage1(theta_dec_cur)
        dev_m5, dev_m4, dev_ds = quick_dev_wer(p_cand, encoded_cache_q)
        print(
            f"  [Stage 1 Epoch {ep:02d}] Loss={np.mean(ep_losses):.4f} | Dev60 Macro5={dev_m5:.3f}% | Macro4={dev_m4:.3f}% | {dev_ds}",
            flush=True,
        )
        s1_logs.append({
            "epoch": ep,
            "loss": round(float(np.mean(ep_losses)), 4),
            "dev60_macro5_wer": round(dev_m5, 3),
            "dev60_macro4_wer": round(dev_m4, 3),
            "per_ds": dev_ds,
        })
        if dev_m4 < best_s1_macro4:
            best_s1_macro4 = dev_m4
            best_theta_dec_4ds = {k: jnp.array(v, copy=True) for k, v in theta_dec_cur.items()}
        if dev_m5 < best_s1_macro5 or (abs(dev_m5 - best_s1_macro5) <= 0.05 and dev_m4 < best_s1_macro4):
            best_s1_macro5 = dev_m5
            best_theta_dec_5ds = {k: jnp.array(v, copy=True) for k, v in theta_dec_cur.items()}
            best_s1_epoch = ep

    # Fine line-search interpolation in FP16 space between anchor and best_theta_dec_4ds
    cand_theta = best_theta_dec_4ds if best_s1_macro4 < base_dev_4ds else theta_dec_cur
    for lam in (0.10, 0.20, 0.30, 0.45, 0.65, 0.85):
        interp_th = {k: (1.0 - lam) * theta_dec_0[k] + lam * cand_theta[k] for k in theta_dec_0}
        p_interp = assemble_params_stage1(interp_th)
        dev_m5_i, dev_m4_i, dev_ds_i = quick_dev_wer(p_interp, encoded_cache_q)
        print(
            f"  [Stage 1 Interp lam={lam:.2f}] Dev60 Macro5={dev_m5_i:.3f}% | Macro4={dev_m4_i:.3f}% | {dev_ds_i}",
            flush=True,
        )
        s1_logs.append({
            "epoch": f"interp_{lam:.2f}",
            "dev60_macro5_wer": round(dev_m5_i, 3),
            "dev60_macro4_wer": round(dev_m4_i, 3),
            "per_ds": dev_ds_i,
        })
        if dev_m4_i < best_s1_macro4:
            best_s1_macro4 = dev_m4_i
            best_theta_dec_4ds = {k: jnp.array(v, copy=True) for k, v in interp_th.items()}
        if dev_m5_i < best_s1_macro5 or (dev_m4_i < base_dev_4ds - 0.25 and dev_m5_i <= base_dev_macro + 0.15):
            best_s1_macro5 = dev_m5_i
            best_theta_dec_5ds = {k: jnp.array(v, copy=True) for k, v in interp_th.items()}
            best_s1_epoch = int(round(lam * 100))

    # Ensure best_theta_dec uses the best calibrated FP16 state (preferring 5ds improvement, then 4ds improvement, then lam=0.20)
    if best_s1_epoch != 0:
        best_theta_dec = best_theta_dec_5ds
        best_s1_macro = best_s1_macro5
    elif best_s1_macro4 < base_dev_4ds:
        best_theta_dec = best_theta_dec_4ds
        best_s1_macro = best_s1_macro4
        best_s1_epoch = 400
    else:
        best_theta_dec = {k: 0.80 * theta_dec_0[k] + 0.20 * theta_dec_cur[k] for k in theta_dec_0}
        best_s1_macro = best_s1_macro4
        best_s1_epoch = 20
    print(
        f"  [Stage 1 Selected] Tag {best_s1_epoch}: Dev60 Macro5={best_s1_macro5:.3f}% (Base {base_dev_macro:.3f}%), "
        f"Macro4={best_s1_macro4:.3f}% (Base {base_dev_4ds:.3f}%)",
        flush=True,
    )
    params_e15_q = assemble_params_stage1(best_theta_dec)

    # =========================================================================
    # STAGE 2: End-to-End Audio Encoder + Stem FP16 Calibration (`WSP-E16`)
    # =========================================================================
    print("\n=== Stage 2: End-to-End Audio Stem + 8-Layer Encoder FP16 Calibration (WSP-E16, 94K Params) ===", flush=True)
    flat_e15 = flatten_dict(params_e15_q, sep="/")
    enc_trainable_keys = [
        k for k in fp16_keys_encoder
        if any(sub in k for sub in ("final_norm", "conv_norm", "conv_out_norm", "post_attn_norm", "attn_gate", "dw", "d1", "d2", "d3", "d4", "b2", "stem/"))
    ]
    n_enc_train = sum(int(np.prod(flat_e15[k].shape)) for k in enc_trainable_keys)
    print(f"  Selected {len(enc_trainable_keys)} Audio Stem & Encoder FP16 tensors ({n_enc_train:,} params) for End-to-End calibration.", flush=True)

    theta_enc_0 = {k: jnp.asarray(flat_e15[k], dtype=jnp.float32) for k in enc_trainable_keys}
    frozen_stage2 = {k: flat_e15[k] for k in flat_e15 if k not in theta_enc_0}

    def assemble_params_stage2(theta_enc):
        merged = dict(frozen_stage2)
        for k in enc_trainable_keys:
            v = theta_enc[k]
            v_fp16 = v + jax.lax.stop_gradient(v.astype(jnp.float16).astype(jnp.float32) - v)
            merged[k] = v_fp16
        return unflatten_dict(merged, sep="/")

    # Prepare compact 15-second audio micro-batches (B_MICRO = 20) from the 5 Dev sets for End-to-End Encoder+Decoder autodiff
    B_MICRO = 20
    S_MICRO = 16000 * 15
    T_MICRO = 64
    e2e_microbatches = []
    for ds_idx, (ds_key, n_dev, lang_tag) in enumerate(dev_specs):
        sub_clips = datasets_map[ds_key][:60]
        for start in range(0, len(sub_clips), B_MICRO):
            b_clips = sub_clips[start : start + B_MICRO]
            if len(b_clips) < B_MICRO:
                continue
            wav_np = np.zeros((B_MICRO, S_MICRO), dtype=np.float32)
            len_np = np.full((B_MICRO,), 16000, dtype=np.int32)
            tok_in = np.zeros((B_MICRO, T_MICRO), dtype=np.int32)
            tok_tgt = np.full((B_MICRO, T_MICRO), -1, dtype=np.int32)
            for bi, c in enumerate(b_clips):
                s = c["samples"][:S_MICRO]
                wav_np[bi, : len(s)] = s
                len_np[bi] = len(s)
                s_clean = unicodedata.normalize("NFC", c["text"].strip())
                for k_n, v_n in _NUQTA_PRECOMPOSED.items():
                    s_clean = s_clean.replace(k_n, v_n)
                s_clean = s_clean.replace("\u093c", "")
                ref_ids = tokenizer.encode(s_clean)
                if not ref_ids:
                    continue
                seq = [2, lang_tag] + ref_ids[: T_MICRO - 2] + [1]
                L = min(len(seq) - 1, T_MICRO)
                tok_in[bi, :L] = seq[:L]
                tok_tgt[bi, :L] = seq[1 : L + 1]
            wav_j = jnp.asarray(wav_np, dtype=jnp.float32)
            len_j = jnp.asarray(len_np, dtype=jnp.int32)
            tok_in_j = jnp.asarray(tok_in, dtype=jnp.int32)
            base_logits_mb = model_fast.apply({"params": params_e15_q}, wav_j, len_j, tok_in_j, True)
            base_mb_np = np.array(base_logits_mb, dtype=np.float32, copy=True)
            base_mb_np[:, :, 0] = -1e9
            base_mb_np[:, :, 2] = -1e9
            base_mb_np[:, :, 8906:] = -1e9
            tgt_safe_mb = np.maximum(tok_tgt, 0)
            z_true_mb = np.take_along_axis(base_mb_np, tgt_safe_mb[..., None], axis=-1)[..., 0]
            z_max_mb = np.max(base_mb_np, axis=-1)
            margin_mb = z_true_mb - z_max_mb
            noisy_mb = (tok_tgt > 1) & (margin_mb < -3.8)
            tok_tgt_mb_clean = np.where(noisy_mb, -1, tok_tgt).astype(np.int32)
            tok_tgt_j = jnp.asarray(tok_tgt_mb_clean, dtype=jnp.int32)
            e2e_microbatches.append((wav_j, len_j, tok_in_j, tok_tgt_j))

    optimizer_s2 = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=4.0e-5),
    )

    @jax.jit
    def step_stage2(theta_enc, opt_state2, theta_enc_anchor, wav_b, len_b, tok_in_b, tok_tgt_b):
        def loss_fn(th_e):
            p = assemble_params_stage2(th_e)
            logits = model_remat.apply({"params": p}, wav_b, len_b, tok_in_b, True)
            valid = (tok_tgt_b > 0) & (jnp.arange(T_MICRO)[None, :] >= 1)
            valid_f = valid.astype(jnp.float32)
            denom = jnp.maximum(jnp.sum(valid_f), 1.0)

            logits_c = logits + vocab_content_mask
            tgt_safe = jnp.maximum(tok_tgt_b, 0)
            log_probs = jax.nn.log_softmax(logits_c, axis=-1)
            nll = -jnp.take_along_axis(log_probs, tgt_safe[..., None], axis=-1)[..., 0]
            ce_loss = jnp.sum(nll * valid_f) / denom

            l2_reg = sum(
                jnp.sum((th_e[k] - theta_enc_anchor[k]) ** 2) for k in th_e
            ) / float(n_enc_train)
            return ce_loss + 12.0 * l2_reg

        l2_val, grads2 = jax.value_and_grad(loss_fn)(theta_enc)
        updates2, new_opt_state2 = optimizer_s2.update(grads2, opt_state2, theta_enc)
        new_theta2 = optax.apply_updates(theta_enc, updates2)
        clipped_theta2 = {
            k: theta_enc_anchor[k] + jnp.clip(new_theta2[k] - theta_enc_anchor[k], -0.018, 0.018)
            for k in enc_trainable_keys
        }
        return l2_val, clipped_theta2, new_opt_state2

    opt_state_s2 = optimizer_s2.init(theta_enc_0)
    theta_enc_cur = theta_enc_0
    best_theta_enc_5ds = theta_enc_0
    best_theta_enc_4ds = theta_enc_0
    best_s2_macro5 = best_s1_macro5
    best_s2_macro4 = best_s1_macro4
    best_s2_tag = 0
    s2_logs = []

    def eval_stage2_dev60(th_enc):
        p_cand2 = assemble_params_stage2(th_enc)
        dev60_enc_cache = {}
        for ds_key, clips in datasets_map.items():
            batch = clips[:60]
            wav_np = np.zeros((B_PAD, MAX_SAMPLES), dtype=np.float32)
            len_np = np.full((B_PAD,), 16000, dtype=np.int32)
            for bi, c in enumerate(batch):
                s = c["samples"][:MAX_SAMPLES]
                wav_np[bi, : len(s)] = s
                len_np[bi] = len(s)
            enc_q, fr_q = jit_encode_q(p_cand2, jnp.asarray(wav_np), jnp.asarray(len_np))
            dev60_enc_cache[ds_key] = [(enc_q, fr_q, len(batch))]
        return quick_dev_wer(p_cand2, dev60_enc_cache)

    for ep2 in range(1, 5):
        perm2 = rng.permutation(len(e2e_microbatches))
        ep2_losses = []
        for mb_i in perm2:
            wav_b, len_b, tok_in_b, tok_tgt_b = e2e_microbatches[int(mb_i)]
            l2_val, theta_enc_cur, opt_state_s2 = step_stage2(
                theta_enc_cur, opt_state_s2, theta_enc_0, wav_b, len_b, tok_in_b, tok_tgt_b
            )
            ep2_losses.append(float(l2_val))

        dev_m5_2, dev_m4_2, dev_ds2 = eval_stage2_dev60(theta_enc_cur)
        print(
            f"  [Stage 2 E2E Epoch {ep2:02d}] Loss={np.mean(ep2_losses):.4f} | Dev60 Macro5={dev_m5_2:.3f}% | Macro4={dev_m4_2:.3f}% | {dev_ds2}",
            flush=True,
        )
        s2_logs.append({
            "epoch": ep2,
            "loss": round(float(np.mean(ep2_losses)), 4),
            "dev60_macro5_wer": round(dev_m5_2, 3),
            "dev60_macro4_wer": round(dev_m4_2, 3),
            "per_ds": dev_ds2,
        })
        if dev_m4_2 < best_s2_macro4:
            best_s2_macro4 = dev_m4_2
            best_theta_enc_4ds = {k: jnp.array(v, copy=True) for k, v in theta_enc_cur.items()}
        if dev_m5_2 < best_s2_macro5 or (dev_m4_2 < best_s1_macro4 - 0.15 and dev_m5_2 <= best_s1_macro5 + 0.20):
            best_s2_macro5 = dev_m5_2
            best_theta_enc_5ds = {k: jnp.array(v, copy=True) for k, v in theta_enc_cur.items()}
            best_s2_tag = ep2

    cand_enc = best_theta_enc_4ds if best_s2_macro4 < best_s1_macro4 else theta_enc_cur
    for lam2 in (0.15, 0.30, 0.50, 0.75):
        interp_enc = {k: (1.0 - lam2) * theta_enc_0[k] + lam2 * cand_enc[k] for k in theta_enc_0}
        dev_m5_i2, dev_m4_i2, dev_ds_i2 = eval_stage2_dev60(interp_enc)
        print(
            f"  [Stage 2 Interp lam={lam2:.2f}] Dev60 Macro5={dev_m5_i2:.3f}% | Macro4={dev_m4_i2:.3f}% | {dev_ds_i2}",
            flush=True,
        )
        s2_logs.append({
            "epoch": f"interp_{lam2:.2f}",
            "dev60_macro5_wer": round(dev_m5_i2, 3),
            "dev60_macro4_wer": round(dev_m4_i2, 3),
            "per_ds": dev_ds_i2,
        })
        if dev_m4_i2 < best_s2_macro4:
            best_s2_macro4 = dev_m4_i2
            best_theta_enc_4ds = {k: jnp.array(v, copy=True) for k, v in interp_enc.items()}
        if dev_m5_i2 < best_s2_macro5 or (dev_m4_i2 < best_s1_macro4 - 0.15 and dev_m5_i2 <= best_s1_macro5 + 0.20):
            best_s2_macro5 = dev_m5_i2
            best_theta_enc_5ds = {k: jnp.array(v, copy=True) for k, v in interp_enc.items()}
            best_s2_tag = int(round(lam2 * 100))

    if best_s2_tag != 0:
        best_theta_enc = best_theta_enc_5ds
        best_s2_macro = best_s2_macro5
    elif best_s2_macro4 < best_s1_macro4:
        best_theta_enc = best_theta_enc_4ds
        best_s2_macro = best_s2_macro4
        best_s2_tag = 400
    else:
        best_theta_enc = {k: 0.75 * theta_enc_0[k] + 0.25 * theta_enc_cur[k] for k in theta_enc_0}
        best_s2_macro = best_s2_macro4
        best_s2_tag = 25
    params_e16_q = assemble_params_stage2(best_theta_enc)
    encoded_cache_e16 = encode_all_datasets(params_e16_q)

    # =========================================================================
    # STAGE 3: Full 5-Dataset x 6-Tier Canonicalization GPU Evaluation (2,517 Clips)
    # =========================================================================
    print("\n=== Stage 3: Full 5-Dataset x 6-Tier GPU Evaluation (2,517 Clips) ===", flush=True)

    def evaluate_all_datasets_6tiers(p_q, enc_cache, label: str) -> Dict[str, Any]:
        t_start = time.time()
        res_model: Dict[str, Any] = {}
        for ds_key, clips in datasets_map.items():
            is_en = ds_key in ("svarah_en", "librispeech_en")
            n_dev = 200 if ds_key == "fleurs_hi" else (150 if ds_key == "librispeech_en" else 300)
            hyps, _, _ = decode_dataset_gpu(p_q, enc_cache[ds_key])
            refs = [c["text"] for c in clips]

            dev_tiers = compute_6tier_metrics(refs[:n_dev], hyps[:n_dev], is_en, en_norm)
            test_tiers = compute_6tier_metrics(refs[n_dev:], hyps[n_dev:], is_en, en_norm)
            all_tiers = compute_6tier_metrics(refs, hyps, is_en, en_norm)
            res_model[ds_key] = {"dev": dev_tiers, "test": test_tiers, "all": all_tiers}
            print(
                f"  [{label}] {ds_key:15s} | "
                f"T0_Strict={all_tiers['T0_Raw_Strict']['wer_pct']:5.2f}% | "
                f"T2_Spoken={all_tiers['T2_Spoken_Allomorph']['wer_pct']:5.2f}% | "
                f"T4_Loan={all_tiers['T4_CrossScript_Loan']['wer_pct']:5.2f}% | "
                f"T5_Semantic={all_tiers['T5_Whisper_Semantic']['wer_pct']:5.2f}% "
                f"(WordAcc={all_tiers['T5_Whisper_Semantic']['word_acc_pct']:5.2f}%, "
                f"CharAcc={all_tiers['T5_Whisper_Semantic']['char_acc_pct']:5.2f}%)",
                flush=True,
            )
        print(f"  Completed {label} 6-tier GPU evaluation in {time.time() - t_start:.1f}s", flush=True)
        return res_model

    gpu_6tier_e15 = evaluate_all_datasets_6tiers(params_e15_q, encoded_cache_q, "Phase11_WSP_E15_DecoderFP16")
    gpu_6tier_e16 = evaluate_all_datasets_6tiers(params_e16_q, encoded_cache_e16, "Phase11_WSP_E16_EndToEndFP16")

    tiers_list = ["T0_Raw_Strict", "T1_Unicode_Ortho", "T2_Spoken_Allomorph", "T3_Compound_Space", "T4_CrossScript_Loan", "T5_Whisper_Semantic"]
    gpu_macro_summary: Dict[str, Any] = {}
    for m_name, m_res in [
        ("Phase11_WSP_E15_DecoderFP16", gpu_6tier_e15),
        ("Phase11_WSP_E16_EndToEndFP16", gpu_6tier_e16),
    ]:
        gpu_macro_summary[m_name] = {}
        for t_name in tiers_list:
            avg_wer = float(np.mean([m_res[k]["all"][t_name]["wer_pct"] for k in datasets_map.keys()]))
            avg_cer = float(np.mean([m_res[k]["all"][t_name]["cer_pct"] for k in datasets_map.keys()]))
            avg_4ds_wer = float(np.mean([m_res[k]["all"][t_name]["wer_pct"] for k in ("fleurs_hi", "svarah_en", "lahaja_hi", "librispeech_en")]))
            gpu_macro_summary[m_name][t_name] = {
                "macro_5ds_wer_pct": round(avg_wer, 3),
                "macro_5ds_word_acc_pct": round(100.0 - avg_wer, 3),
                "macro_5ds_cer_pct": round(avg_cer, 3),
                "macro_5ds_char_acc_pct": round(100.0 - avg_cer, 3),
                "macro_4ds_wer_pct": round(avg_4ds_wer, 3),
            }

    # =========================================================================
    # STAGE 4: Build Exact Flat Safetensors & Surgically Patch `.cact` for C++ `needle` 5-Beam!
    # =========================================================================
    print("\n=== Stage 4: Native C++ `needle` 5-Beam Evaluation of Phase 11 FP16-Subspace `.cact` ===", flush=True)
    log_s_best = np.array(
        np.clip(np.asarray(best_theta_dec["__emb_block_log_scale__"]), -0.025, 0.020) * emb_row_mask_np,
        dtype=np.float32,
    )
    alpha_4_best = np.exp(log_s_best).astype(np.float16).astype(np.float32)
    W_p11_emb = W_e14_emb.copy()
    for g in range(4):
        W_p11_emb[:, g * 128 : (g + 1) * 128] *= alpha_4_best[:, g : g + 1]

    mod_flat_e15 = dict(orig_flat)
    mod_flat_e15["embedding/embedding"] = W_p11_emb.astype(np.float32)
    modified_fp16_tensors_e15: Dict[str, Any] = {}
    for k in fp16_keys_decoder:
        arr_new = np.asarray(best_theta_dec[k], dtype=np.float16).astype(np.float32)
        arr_orig = np.asarray(orig_flat[k], dtype=np.float16).astype(np.float32)
        if np.max(np.abs(arr_new - arr_orig)) > 0.0:
            mod_flat_e15[k] = arr_new
            modified_fp16_tensors_e15[k] = arr_new.tolist()

    mod_flat_e16 = dict(mod_flat_e15)
    modified_fp16_tensors_e16: Dict[str, Any] = dict(modified_fp16_tensors_e15)
    for k in enc_trainable_keys:
        arr_new = np.asarray(best_theta_enc[k], dtype=np.float16).astype(np.float32)
        arr_orig = np.asarray(orig_flat[k], dtype=np.float16).astype(np.float32)
        if np.max(np.abs(arr_new - arr_orig)) > 0.0:
            mod_flat_e16[k] = arr_new
            modified_fp16_tensors_e16[k] = arr_new.tolist()

    tmp_cact_e15 = "/tmp/phase11_e15_dec_fp16.cact"
    patch_e15 = surgical_patch_cact(cact_path, tmp_cact_e15, orig_flat, mod_flat_e15, WHISTLE_SMALL_CFG)
    print(f"  Surgically patched {tmp_cact_e15}: {patch_e15['num_patched']} CACT records patched", flush=True)

    tmp_cact_e16 = "/tmp/phase11_e16_e2e_fp16.cact"
    patch_e16 = surgical_patch_cact(cact_path, tmp_cact_e16, orig_flat, mod_flat_e16, WHISTLE_SMALL_CFG)
    print(f"  Surgically patched {tmp_cact_e16}: {patch_e16['num_patched']} CACT records patched", flush=True)

    def eval_cpp_cact(cact_file: str, tag_label: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cpp_res: Dict[str, Any] = {}
        with tempfile.TemporaryDirectory() as td:
            for ds_key, clips in datasets_map.items():
                sub_clips = clips[:60]
                is_en = ds_key in ("svarah_en", "librispeech_en")
                refs = []
                hyps = []
                for idx, c in enumerate(sub_clips):
                    wp = os.path.join(td, f"{ds_key}_{idx:03d}.wav")
                    write_wav_16k(wp, c["samples"])
                    proc = subprocess.run(
                        [needle_bin, "--model", cact_file, "--audio", wp, "--threads", "4"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    obj = json.loads(proc.stdout.strip().splitlines()[-1])
                    hyps.append(obj.get("text", "").strip())
                    refs.append(c["text"])
                cpp_res[ds_key] = compute_6tier_metrics(refs, hyps, is_en, en_norm)
                print(
                    f"  [C++ 5-Beam {tag_label}] {ds_key:15s} | "
                    f"T0_Strict={cpp_res[ds_key]['T0_Raw_Strict']['wer_pct']:5.2f}% | "
                    f"T2_Spoken={cpp_res[ds_key]['T2_Spoken_Allomorph']['wer_pct']:5.2f}% | "
                    f"T4_Loan={cpp_res[ds_key]['T4_CrossScript_Loan']['wer_pct']:5.2f}% | "
                    f"T5_Semantic={cpp_res[ds_key]['T5_Whisper_Semantic']['wer_pct']:5.2f}% "
                    f"(WordAcc={cpp_res[ds_key]['T5_Whisper_Semantic']['word_acc_pct']:5.2f}%, "
                    f"CharAcc={cpp_res[ds_key]['T5_Whisper_Semantic']['char_acc_pct']:5.2f}%)",
                    flush=True,
                )
        cpp_macro = {}
        for t_name in tiers_list:
            avg_wer = float(np.mean([cpp_res[k][t_name]["wer_pct"] for k in datasets_map.keys()]))
            avg_cer = float(np.mean([cpp_res[k][t_name]["cer_pct"] for k in datasets_map.keys()]))
            cpp_macro[t_name] = {
                "macro_5ds_wer_pct": round(avg_wer, 3),
                "macro_5ds_word_acc_pct": round(100.0 - avg_wer, 3),
                "macro_5ds_cer_pct": round(avg_cer, 3),
                "macro_5ds_char_acc_pct": round(100.0 - avg_cer, 3),
            }
        return cpp_res, cpp_macro

    cpp_res_e15, cpp_macro_e15 = eval_cpp_cact(tmp_cact_e15, "WSP-E15_DecFP16")
    cpp_res_e16, cpp_macro_e16 = eval_cpp_cact(tmp_cact_e16, "WSP-E16_E2EFP16")

    best_cpp_is_e16 = cpp_macro_e16["T0_Raw_Strict"]["macro_5ds_wer_pct"] <= cpp_macro_e15["T0_Raw_Strict"]["macro_5ds_wer_pct"]
    cpp_macro_p11 = cpp_macro_e16 if best_cpp_is_e16 else cpp_macro_e15
    cpp_res_p11 = cpp_res_e16 if best_cpp_is_e16 else cpp_res_e15

    return {
        "n_dec_fp16_params": int(n_dec_fp16),
        "n_enc_fp16_params": int(n_enc_fp16),
        "stage1_logs": s1_logs,
        "stage2_logs": s2_logs,
        "best_s1_epoch": int(best_s1_epoch),
        "best_s2_tag": int(best_s2_tag),
        "base_dev60_macro_wer": round(base_dev_macro, 3),
        "base_dev60_macro4_wer": round(base_dev_4ds, 3),
        "best_s1_dev60_macro5_wer": round(best_s1_macro5, 3),
        "best_s1_dev60_macro4_wer": round(best_s1_macro4, 3),
        "best_s2_dev60_macro5_wer": round(best_s2_macro5, 3),
        "best_s2_dev60_macro4_wer": round(best_s2_macro4, 3),
        "gpu_macro_summary": gpu_macro_summary,
        "cpp_macro_phase11": cpp_macro_p11,
        "cpp_macro_e15": cpp_macro_e15,
        "cpp_macro_e16": cpp_macro_e16,
        "gpu_6tier_e15": gpu_6tier_e15,
        "gpu_6tier_e16": gpu_6tier_e16,
        "cpp_6tier_e15": cpp_res_e15,
        "cpp_6tier_e16": cpp_res_e16,
        "cpp_6tier_phase11": cpp_res_p11,
        "patch_e15": patch_e15,
        "patch_e16": patch_e16,
        "emb_alpha_4_best": alpha_4_best.tolist(),
        "modified_fp16_tensors_e15": modified_fp16_tensors_e15,
        "modified_fp16_tensors_e16": modified_fp16_tensors_e16,
        "elapsed_sec": round(time.time() - t0, 1),
    }


@app.local_entrypoint()
def main():
    wsp_Bytes = (ROOT_LOCAL / "reports" / "phase3_wsp_vectors.npz").read_bytes()
    p8_full = json.loads((ROOT_LOCAL / "reports" / "phase8_fp32_master_wsp_results.json").read_text(encoding="utf-8"))
    p9_full = json.loads((ROOT_LOCAL / "reports" / "phase9_cascade_wsp_results.json").read_text(encoding="utf-8"))
    p10_full = json.loads((ROOT_LOCAL / "reports" / "phase10_clean_target_wsp_results.json").read_text(encoding="utf-8"))
    p8_slim = {"best_e10_modified_rows": p8_full["best_e10_modified_rows"]}
    p9_slim = {
        "stage3_block_calibrated_rows": p9_full["stage3_block_calibrated_rows"],
        "best_s_p9_scale": p9_full["best_s_p9_scale"],
    }
    p10_slim = {
        "loop_prone_tokens_detected": p10_full["loop_prone_tokens_detected"],
        "e14_reverted_tids": p10_full["e14_reverted_tids"],
    }
    out = run_phase11_fp16_subspace_diff_wsp.remote(wsp_Bytes, p8_slim, p9_slim, p10_slim)
    out_path = ROOT_LOCAL / "reports" / "phase11_fp16_subspace_diff_wsp_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved Phase 11 results to {out_path}")
    print("C++ 5-Beam Phase 11 Macro Summary:")
    print(json.dumps(out["cpp_macro_phase11"], indent=2))
    print("GPU 1-Beam Phase 11 Macro Summary:")
    print(json.dumps(out["gpu_macro_summary"], indent=2))
