#!/usr/bin/env python3
"""Phase 8: Unquantized FP32 Master Model Weight-Space Programming -> Quantization-Compensated .cact Pipeline.

Implements the 3-Stage Master-to-Quantized WSP Architecture:
  Stage 1 (Pure Unquantized FP32 Master Compilation, `quant=False`):
    - Evaluates Official Unquantized FP32 Baseline (`params_orig`, `quant=False`) and FP32 Step-0 Subspace (`WSP-E1b-FP32`).
    - Performs Direct Logit Attribution on FP32 Dev hidden states (`stack/final_norm`, `layers_7/hadamard_mlp`, `engrams_1`).
    - Solves English-Nullspace Weighted Contrastive Ridge Compilation (`WSP-E9_FP32_Master`) across the full active vocabulary
      in continuous 32-bit floating point without 4-bit CQ-W4 rounding truncation.
    - Solves continuous FP32 Channel-Margin Ridge Compilation on `stack/final_norm/scale` ([512]).
    - Evaluates `WSP-E9_FP32_Master` in pure unquantized FP32 (`quant=False`) across all 5 datasets (Dev & Held-Out Test).

  Stage 2 (Codebook-Snap Lattice Projection + Unquantized FP16 Residual Compensation, `quant=True`):
    - Maps the dense FP32 master embedding updates `Delta_W_fp32` onto the 4-bit Hadamard CQ-W4 lattice by optimizing
      per-row lattice projection scale `alpha_r` to preserve the FP32 logit improvements after 4-bit quantization.
    - Absorbs residual CQ-W4 quantization error into the adjacent unquantized FP16 `stack/final_norm/scale` layer (`WSP-E10_QuantComp`).
    - Evaluates `WSP-E10_QuantComp` under full CQ-W4A8 STE (`quant=True`) across all 5 datasets (Dev & Held-Out Test).

  Stage 3 (Native C++ `needle` 5-Beam Engine Verification):
    - Exports `shruti_wsp_phase8_quantcomp.cact` via `surgical_patch_cact` and benchmarks on the C++ 5-beam engine.
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

app = modal.App("shruti-wsp-phase8-fp32-master")
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


def normalize_indic_hinglish(text: str) -> str:
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
def run_phase8_fp32_master_wsp(
    wsp_vectors_bytes: bytes,
    p7_res: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        from nvidia import cuda_nvcc
        if getattr(cuda_nvcc, "__file__", None) is None and hasattr(cuda_nvcc, "__path__"):
            cuda_nvcc.__file__ = os.path.join(list(cuda_nvcc.__path__)[0], "__init__.py")
    except Exception:
        pass

    import jax
    import jax.numpy as jnp
    from whisper_normalizer.english import EnglishTextNormalizer

    sys.path.insert(0, "/root/shruti/src")
    from needle_model.architecture import _aq, _rms_unit
    from needle_model.quantize import WEIGHT_BITS, cq_ste_params
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
    cfg = replace(cfg, flash=False, remat=False)
    model = Whistle(cfg)
    tokenizer = RefTokenizer.from_cact(cact_path)
    lang_names = list(cfg.languages)
    en_norm = EnglishTextNormalizer()

    wsp_vecs = np.load(io.BytesIO(wsp_vectors_bytes), allow_pickle=False)
    datasets_map = load_or_cache_datasets("/data/eval_cache")

    # Step-0 language subspace vectors
    mu_0_unit = wsp_vecs["mu_0_unit"].astype(np.float32)
    delta_w_best = wsp_vecs["delta_w_best"].astype(np.float32)

    W_orig_emb = np.array(orig_flat["embedding/embedding"], dtype=np.float32, copy=True)
    W_e1b_emb = W_orig_emb.copy()
    W_e1b_emb[8906] = W_orig_emb[8906] + 0.5 * delta_w_best
    W_e1b_emb[8913] = W_orig_emb[8913] - 0.5 * delta_w_best
    for tid in range(8907, 8913):
        W_e1b_emb[tid] = W_orig_emb[tid] - 2.0 * mu_0_unit

    # Pure Unquantized FP32 parameter trees (`quant=False`)
    params_e1b_fp32 = {
        **params_orig,
        "embedding": {"embedding": jnp.asarray(W_e1b_emb, dtype=jnp.float32)},
    }
    # Quantized CQ-W4 parameter tree (`quant=True`)
    params_e1b_q = cq_ste_params(params_e1b_fp32, WEIGHT_BITS)

    # Pure FP32 JIT functions (`quant=False` -> zero weight quantization, zero 8-bit activation quantization)
    @jax.jit
    def jit_encode_fp32(p, wav, lengths):
        return model.apply({"params": p}, wav, lengths, False, method=Whistle.encode)

    @jax.jit
    def jit_decode_hidden_fp32(p, tokens, enc, frames):
        h = model.apply({"params": p}, tokens, enc, frames, False, method=Whistle.decode)
        h_fp32 = h.astype(jnp.float32)
        scale = 1.0 + p["stack"]["final_norm"]["scale"].astype(jnp.float32)
        u_unit = h_fp32 / jnp.where(jnp.abs(scale) > 1e-6, scale, 1.0)
        return h_fp32, u_unit

    @jax.jit
    def jit_step_logits_fp32(p, tokens, enc, frames):
        h = model.apply({"params": p}, tokens, enc, frames, False, method=Whistle.decode)
        logits = h.astype(jnp.float32) @ p["embedding"]["embedding"].T.astype(jnp.float32)
        return logits

    # Quantized CQ-W4A8 JIT functions (`quant=True`)
    @jax.jit
    def jit_encode_q(p, wav, lengths):
        return model.apply({"params": p}, wav, lengths, True, method=Whistle.encode)

    @jax.jit
    def jit_decode_hidden_q(p, tokens, enc, frames):
        h = model.apply({"params": p}, tokens, enc, frames, True, method=Whistle.decode)
        hq = _aq(h, True).astype(jnp.float32)
        scale = 1.0 + p["stack"]["final_norm"]["scale"].astype(jnp.float32)
        u_unit = h.astype(jnp.float32) / jnp.where(jnp.abs(scale) > 1e-6, scale, 1.0)
        return hq, u_unit

    @jax.jit
    def jit_step_logits_q(p, tokens, enc, frames):
        h = model.apply({"params": p}, tokens, enc, frames, True, method=Whistle.decode)
        hq = _aq(h, True).astype(jnp.float32)
        logits = hq @ p["embedding"]["embedding"].T.astype(jnp.float32)
        return logits

    B_PAD = 60
    TF_SEQ_LEN = 80
    MAX_DEC_LEN = 128
    MAX_SAMPLES = 16000 * 30

    print("Warming up XLA kernels (both Unquantized FP32 and Quantized CQ-W4A8)...", flush=True)
    dummy_wav = jnp.zeros((B_PAD, MAX_SAMPLES), dtype=jnp.float32)
    dummy_len = jnp.full((B_PAD,), 16000 * 10, dtype=jnp.int32)
    d_enc_fp, d_fr_fp = jit_encode_fp32(params_e1b_fp32, dummy_wav, dummy_len)
    _ = jit_decode_hidden_fp32(params_e1b_fp32, jnp.full((B_PAD, TF_SEQ_LEN), 2, dtype=jnp.int32), d_enc_fp, d_fr_fp)
    d_enc_q, d_fr_q = jit_encode_q(params_e1b_q, dummy_wav, dummy_len)
    _ = jit_decode_hidden_q(params_e1b_q, jnp.full((B_PAD, TF_SEQ_LEN), 2, dtype=jnp.int32), d_enc_q, d_fr_q)
    for t_len in [64, 128]:
        _ = jit_step_logits_fp32(params_e1b_fp32, jnp.full((B_PAD, t_len), 2, dtype=jnp.int32), d_enc_fp, d_fr_fp)
        _ = jit_step_logits_q(params_e1b_q, jnp.full((B_PAD, t_len), 2, dtype=jnp.int32), d_enc_q, d_fr_q)
    print(f"JIT warmup complete in {time.time() - t0:.1f}s", flush=True)

    # Pre-encode all 5 datasets in BOTH Unquantized FP32 (`encoded_cache_fp32`) and Quantized CQ-W4A8 (`encoded_cache_q`)
    encoded_cache_fp32: Dict[str, List[Tuple[Any, Any, int]]] = {}
    encoded_cache_q: Dict[str, List[Tuple[Any, Any, int]]] = {}
    for ds_key, clips in datasets_map.items():
        batches_fp32 = []
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
            wav_j = jnp.asarray(wav_np)
            len_j = jnp.asarray(len_np)
            enc_fp, fr_fp = jit_encode_fp32(params_e1b_fp32, wav_j, len_j)
            enc_q, fr_q = jit_encode_q(params_e1b_q, wav_j, len_j)
            batches_fp32.append((enc_fp, fr_fp, real_B))
            batches_q.append((enc_q, fr_q, real_B))
        encoded_cache_fp32[ds_key] = batches_fp32
        encoded_cache_q[ds_key] = batches_q
        print(f"Pre-encoded {ds_key} ({limit_n} clips) in both FP32 and CQ-W4A8.", flush=True)

    # =========================================================================
    # STAGE 1: Extract Teacher-Forced Hidden States in Pure Unquantized FP32
    # =========================================================================
    print("\n=== Stage 1: Extracting Pure Unquantized FP32 Teacher-Forced States on DEV Splits ===", flush=True)
    dev_specs = [
        ("fleurs_hi", 200, 8913),
        ("lahaja_hi", 300, 8913),
        ("mucs_hinglish", 300, 8913),
        ("svarah_en", 300, 8906),
        ("librispeech_en", 150, 8906),
    ]

    all_h_fp32_list = []
    all_u_fp32_list = []
    all_hq_q_list = []
    all_u_q_list = []
    all_target_list = []
    all_ds_id_list = []

    for ds_idx, (ds_key, n_dev, lang_tag) in enumerate(dev_specs):
        clips = datasets_map[ds_key][:n_dev]
        enc_batches_fp = encoded_cache_fp32[ds_key]
        enc_batches_q = encoded_cache_q[ds_key]
        n_batches = (n_dev + B_PAD - 1) // B_PAD
        collected_tokens = 0
        for b_idx in range(n_batches):
            b_start = b_idx * B_PAD
            b_end = min(b_start + B_PAD, n_dev)
            batch_clips = clips[b_start:b_end]
            enc_fp, fr_fp, _ = enc_batches_fp[b_idx]
            enc_q, fr_q, _ = enc_batches_q[b_idx]

            tok_in = np.zeros((B_PAD, TF_SEQ_LEN), dtype=np.int32)
            tok_tgt = np.full((B_PAD, TF_SEQ_LEN), -1, dtype=np.int32)
            for bi, c in enumerate(batch_clips):
                ref_text = c["text"].strip()
                if not ref_text:
                    continue
                ref_ids = tokenizer.encode(ref_text)
                if not ref_ids:
                    continue
                seq = [2, lang_tag] + ref_ids[: TF_SEQ_LEN - 2] + [1]
                L = min(len(seq) - 1, TF_SEQ_LEN)
                tok_in[bi, :L] = seq[:L]
                tok_tgt[bi, :L] = seq[1 : L + 1]

            tok_j = jnp.asarray(tok_in)
            h_fp_b, u_fp_b = jit_decode_hidden_fp32(params_e1b_fp32, tok_j, enc_fp, fr_fp)
            hq_q_b, u_q_b = jit_decode_hidden_q(params_e1b_q, tok_j, enc_q, fr_q)

            h_fp_np = np.asarray(h_fp_b, dtype=np.float32)
            u_fp_np = np.asarray(u_fp_b, dtype=np.float32)
            hq_q_np = np.asarray(hq_q_b, dtype=np.float32)
            u_q_np = np.asarray(u_q_b, dtype=np.float32)

            for bi in range(len(batch_clips)):
                for t in range(1, TF_SEQ_LEN):
                    tid = int(tok_tgt[bi, t])
                    if tid <= 0:
                        break
                    all_h_fp32_list.append(h_fp_np[bi, t])
                    all_u_fp32_list.append(u_fp_np[bi, t])
                    all_hq_q_list.append(hq_q_np[bi, t])
                    all_u_q_list.append(u_q_np[bi, t])
                    all_target_list.append(tid)
                    all_ds_id_list.append(ds_idx)
                    collected_tokens += 1
        print(f"  [{ds_key} dev] Collected {collected_tokens} FP32 & CQ-W4 teacher-forced token states.", flush=True)

    H_fp32 = np.stack(all_h_fp32_list, axis=0)   # [N_tok, 512]
    U_fp32 = np.stack(all_u_fp32_list, axis=0)   # [N_tok, 512]
    H_q = np.stack(all_hq_q_list, axis=0)        # [N_tok, 512]
    U_q = np.stack(all_u_q_list, axis=0)          # [N_tok, 512]
    Y_dev = np.array(all_target_list, dtype=np.int32)
    DS_dev = np.array(all_ds_id_list, dtype=np.int32)
    N_tok = H_fp32.shape[0]
    row_idx = np.arange(N_tok)

    # Compare FP32 vs CQ-W4 baseline teacher-forced accuracy & quantization distortion
    W_q0 = np.asarray(params_e1b_q["embedding"]["embedding"], dtype=np.float32)

    Z_fp0 = H_fp32 @ W_e1b_emb.T
    Z_fp0[:, 0] = -1e9
    Z_fp0[:, 2] = -1e9
    Z_fp0[:, 8906:] = -1e9
    z_true_fp0 = Z_fp0[row_idx, Y_dev]
    Z_fp0_comp = Z_fp0.copy()
    Z_fp0_comp[row_idx, Y_dev] = -1e9
    Y_hat_fp0 = np.argmax(Z_fp0_comp, axis=1)
    z_comp_fp0 = Z_fp0_comp[row_idx, Y_hat_fp0]
    margins_fp0 = z_true_fp0 - z_comp_fp0

    Z_q0 = H_q @ W_q0.T
    Z_q0[:, 0] = -1e9
    Z_q0[:, 2] = -1e9
    Z_q0[:, 8906:] = -1e9
    z_true_q0 = Z_q0[row_idx, Y_dev]
    Z_q0_comp = Z_q0.copy()
    Z_q0_comp[row_idx, Y_dev] = -1e9
    Y_hat_q0 = np.argmax(Z_q0_comp, axis=1)
    z_comp_q0 = Z_q0_comp[row_idx, Y_hat_q0]
    margins_q0 = z_true_q0 - z_comp_q0

    hi_mask = np.isin(DS_dev, [0, 1])
    hinglish_mask = (DS_dev == 2)
    en_mask = np.isin(DS_dev, [3, 4])

    tf_fp0_all = float(np.mean(margins_fp0 > 0.0) * 100.0)
    tf_fp0_hi = float(np.mean(margins_fp0[hi_mask] > 0.0) * 100.0)
    tf_fp0_hing = float(np.mean(margins_fp0[hinglish_mask] > 0.0) * 100.0)
    tf_q0_all = float(np.mean(margins_q0 > 0.0) * 100.0)
    tf_q0_hi = float(np.mean(margins_q0[hi_mask] > 0.0) * 100.0)
    tf_q0_hing = float(np.mean(margins_q0[hinglish_mask] > 0.0) * 100.0)

    # Cosine similarity & L2 drift between FP32 hidden states and CQ-W4A8 hidden states
    h_cos_sim = float(np.mean(np.sum(H_fp32 * H_q, axis=1) / (np.linalg.norm(H_fp32, axis=1) * np.linalg.norm(H_q, axis=1) + 1e-8)))
    w_emb_quant_rmse = float(np.sqrt(np.mean((W_e1b_emb - W_q0) ** 2)))

    print(
        f"Teacher-Forced Top-1 Acc (FP32 vs CQ-W4A8): "
        f"All={tf_fp0_all:.2f}% vs {tf_q0_all:.2f}% | "
        f"Hindi={tf_fp0_hi:.2f}% vs {tf_q0_hi:.2f}% | "
        f"Hinglish={tf_fp0_hing:.2f}% vs {tf_q0_hing:.2f}% | "
        f"Hidden CosSim={h_cos_sim:.4f}",
        flush=True,
    )

    # =========================================================================
    # STAGE 1B: Direct Logit Attribution & Subspace Leverage Diagnostic (FP32)
    # =========================================================================
    print("\n=== Stage 1B: Direct Logit Attribution & Error Subspace Leverage Diagnostic (FP32) ===", flush=True)
    s0 = np.array(orig_flat["stack/final_norm/scale"], dtype=np.float32, copy=True)
    # Decompose H_fp32 = U_fp32 (unit RMS-normalized decoder output) + s0 * U_fp32 (final_norm learned scale contribution)
    clean_state_mask = (DS_dev != 2) | (Y_dev >= 8192)
    H_fp_clean = H_fp32[clean_state_mask]
    U_fp_clean = U_fp32[clean_state_mask]
    Y_clean = Y_dev[clean_state_mask]
    Y_hat_fp_clean = Y_hat_fp0[clean_state_mask]
    m_fp_clean = margins_fp0[clean_state_mask]
    DS_clean = DS_dev[clean_state_mask]
    N_clean = len(Y_clean)

    err_idx_fp = np.where((m_fp_clean < 0.0) & (Y_clean != 1))[0]
    ok_idx_fp = np.where((m_fp_clean > 1.0) & (Y_clean != 1))[0]
    w_diff_err = W_e1b_emb[Y_clean[err_idx_fp]] - W_e1b_emb[Y_hat_fp_clean[err_idx_fp]]
    u_contrib_err = float(np.mean(np.sum(U_fp_clean[err_idx_fp] * w_diff_err, axis=1)))
    s_contrib_err = float(np.mean(np.sum((s0[None, :] * U_fp_clean[err_idx_fp]) * w_diff_err, axis=1)))

    # Compute Signal-to-Interference Generalized Eigenvalue Spectrum of C_err vs C_ok
    C_err_fp = (H_fp_clean[err_idx_fp].T @ H_fp_clean[err_idx_fp]) / float(len(err_idx_fp))
    C_ok_fp = (H_fp_clean[ok_idx_fp].T @ H_fp_clean[ok_idx_fp]) / float(len(ok_idx_fp))
    C_ok_inv = np.linalg.inv(C_ok_fp + 0.5 * np.eye(512, dtype=np.float32))
    sir_eigs = np.linalg.eigvalsh(C_ok_inv @ C_err_fp)
    top5_sir_eigs = [round(float(x), 3) for x in sir_eigs[-5:][::-1]]
    print(
        f"  [FP32 Attribution on Dev Errors] Base-Decoder Margin={u_contrib_err:.3f}, "
        f"FinalNorm-Scale Margin={s_contrib_err:.3f}, Top-5 SIR Eigenvalues={top5_sir_eigs}",
        flush=True,
    )

    # =========================================================================
    # STAGE 1C: WSP-E9_FP32_Master — Pure Unquantized FP32 Multi-Row Compilation
    # =========================================================================
    print("\n=== Stage 1C: WSP-E9_FP32_Master — Unquantized FP32 Multi-Row Ridge Compiler ===", flush=True)
    # Weight clean English states (DS_dev in {3, 4}) by 2.0 in the preservation covariance
    # so that updates automatically project into the null-space of English speech!
    weights_cov = np.where(np.isin(DS_dev, [3, 4]), 2.0, 1.0).astype(np.float32)
    Cov_H_fp32 = ((H_fp32 * weights_cov[:, None]).T @ H_fp32) / float(np.sum(weights_cov))

    e9_fp32_logs = []
    best_e9_cfg = None
    best_W_fp32_master = W_e1b_emb.copy()
    best_e9_hits_gain = 0

    for lam_fp in [0.5, 1.5, 3.0]:
        Sigma_fp_inv = np.linalg.inv(Cov_H_fp32 + lam_fp * np.eye(512, dtype=np.float32))
        for tau_fp in [1.0, 2.0]:
            err_mask_fp = (m_fp_clean < tau_fp) & (m_fp_clean > -15.0) & (Y_clean != 1)
            deficit_fp = np.clip(tau_fp - m_fp_clean[err_mask_fp], 0.0, 6.0)
            H_err_fp = H_fp_clean[err_mask_fp]
            Y_err_true = Y_clean[err_mask_fp]
            Y_err_comp = Y_hat_fp_clean[err_mask_fp]

            B_fp = np.zeros((8914, 512), dtype=np.float32)
            pos_cnt = np.zeros(8914, dtype=np.int32)
            neg_cnt = np.zeros(8914, dtype=np.int32)
            for idx_e in range(len(Y_err_true)):
                vt = int(Y_err_true[idx_e])
                vc = int(Y_err_comp[idx_e])
                w_d = float(deficit_fp[idx_e])
                h_e = H_err_fp[idx_e]
                if 260 <= vt < 8906:
                    B_fp[vt] += w_d * h_e
                    pos_cnt[vt] += 1
                if 260 <= vc < 8906:
                    B_fp[vc] -= 0.75 * w_d * h_e
                    neg_cnt[vc] += 1

            B_fp /= float(N_clean)
            Delta_W_fp_unit = B_fp @ Sigma_fp_inv  # [8914, 512]

            active_rows_fp = np.where(
                (pos_cnt + neg_cnt >= 2) & (np.arange(8914) >= 260) & (np.arange(8914) < 8906)
            )[0]
            # Precompute unit score deltas for all active_rows_fp in a single BLAS SGEMM!
            Delta_Z_active_unit = H_fp_clean @ Delta_W_fp_unit[active_rows_fp].T  # [N_clean, N_active]
            orig_norms_active = np.linalg.norm(W_e1b_emb[active_rows_fp], axis=1)
            dw_unit_norms_active = np.linalg.norm(Delta_W_fp_unit[active_rows_fp], axis=1)

            for gain_fp in [8.0, 16.0, 28.0, 40.0]:
                W_cand_fp = W_e1b_emb.copy()
                accepted_fp_rows = []
                for idx_a, r in enumerate(active_rows_fp):
                    orig_norm = float(orig_norms_active[idx_a])
                    dw_u_norm = float(dw_unit_norms_active[idx_a])
                    if dw_u_norm < 1e-12:
                        continue
                    max_frac = 0.08 if r >= 8192 else 0.06
                    eff_scale = min(gain_fp, (max_frac * orig_norm) / dw_u_norm)
                    w_new_fp = W_e1b_emb[r] + eff_scale * Delta_W_fp_unit[r]
                    new_scores_fp = Z_fp0[clean_state_mask, r] + eff_scale * Delta_Z_active_unit[:, idx_a]

                    # Pure FP32 verification gate on H_fp_clean (zero quantization distortion!)
                    is_target = (Y_clean == r)
                    is_false_comp = (Y_hat_fp_clean == r) & (m_fp_clean < 0.0)
                    fn_fixed = int(np.sum(is_target & (m_fp_clean <= 0.0) & ((new_scores_fp - z_comp_fp0[clean_state_mask]) > 0.0)))
                    fn_broken = int(np.sum(is_target & (m_fp_clean > 0.0) & ((new_scores_fp - z_comp_fp0[clean_state_mask]) <= 0.0)))
                    fp_fixed = int(np.sum(is_false_comp & ((z_true_fp0[clean_state_mask] - new_scores_fp) > 0.0)))
                    was_right = (m_fp_clean > 0.0) & (~is_target)
                    new_fp = int(np.sum(was_right & (new_scores_fp > z_true_fp0[clean_state_mask])))

                    net_fp_gain = (fn_fixed + fp_fixed) - 2 * (fn_broken + new_fp)
                    if net_fp_gain >= 2:
                        W_cand_fp[r] = w_new_fp
                        accepted_fp_rows.append((int(r), int(net_fp_gain)))

                Z_cand_fp = H_fp_clean @ W_cand_fp.T
                Z_cand_fp[:, 0] = -1e9
                Z_cand_fp[:, 2] = -1e9
                Z_cand_fp[:, 8906:] = -1e9
                pred_cand_fp = np.argmax(Z_cand_fp, axis=1)
                hits_fp0 = int(np.sum(m_fp_clean > 0.0))
                hits_cand_fp = int(np.sum(pred_cand_fp == Y_clean))
                net_gain_fp = hits_cand_fp - hits_fp0

                rec_fp = {
                    "lam_fp": lam_fp,
                    "tau_fp": tau_fp,
                    "gain_fp": gain_fp,
                    "num_accepted_fp32_rows": len(accepted_fp_rows),
                    "net_fp32_token_hits_gain": net_gain_fp,
                    "tf_acc_fp32_clean_pct": round(hits_cand_fp / float(N_clean) * 100.0, 3),
                }
                e9_fp32_logs.append(rec_fp)
                print(f"  [WSP-E9 FP32 Grid] {rec_fp}", flush=True)
                if net_gain_fp > best_e9_hits_gain:
                    best_e9_hits_gain = net_gain_fp
                    best_e9_cfg = {**rec_fp, "accepted_rows": accepted_fp_rows}
                    best_W_fp32_master = W_cand_fp.copy()

    print(
        f"Best FP32 Master Embedding Config: {best_e9_cfg['num_accepted_fp32_rows']} rows updated, "
        f"+{best_e9_hits_gain} net FP32 token hits!",
        flush=True,
    )

    # Also solve continuous FP32 `stack/final_norm/scale` refinement on top of `best_W_fp32_master`
    Z_fp_after_emb = H_fp_clean @ best_W_fp32_master.T
    Z_fp_after_emb[:, 0] = -1e9
    Z_fp_after_emb[:, 2] = -1e9
    Z_fp_after_emb[:, 8906:] = -1e9
    r_idx_c = np.arange(N_clean)
    z_t_after = Z_fp_after_emb[r_idx_c, Y_clean]
    Z_c_after = Z_fp_after_emb.copy()
    Z_c_after[r_idx_c, Y_clean] = -1e9
    Y_hat_after = np.argmax(Z_c_after, axis=1)
    m_after = z_t_after - Z_c_after[r_idx_c, Y_hat_after]

    G_fp = U_fp_clean * (best_W_fp32_master[Y_clean] - best_W_fp32_master[Y_hat_after])
    Cov_G_fp = (G_fp.T @ G_fp) / float(N_clean)
    err_m_fp = (m_after < 1.0) & (m_after > -10.0) & (Y_clean != 1)
    r_tgt_fp = np.zeros(N_clean, dtype=np.float32)
    r_tgt_fp[err_m_fp] = np.clip(1.0 - m_after[err_m_fp], 0.0, 3.0)
    delta_s_fp = np.linalg.solve(Cov_G_fp + 50.0 * np.eye(512, dtype=np.float32), (G_fp.T @ r_tgt_fp) / float(N_clean))

    best_s_fp32_master = s0.copy()
    best_s_fp_hits = int(np.sum(m_after > 0.0))
    for step_fp_s in [0.0, 0.15, 0.3, 0.5]:
        s_try = s0 + np.clip(step_fp_s * delta_s_fp, -0.05, 0.05)
        h_try = U_fp_clean * (1.0 + s_try)[None, :]
        Z_try = h_try @ best_W_fp32_master.T
        Z_try[:, 0] = -1e9
        Z_try[:, 2] = -1e9
        Z_try[:, 8906:] = -1e9
        hits_try = int(np.sum(np.argmax(Z_try, axis=1) == Y_clean))
        if hits_try > best_s_fp_hits:
            best_s_fp_hits = hits_try
            best_s_fp32_master = s_try.copy()

    # =========================================================================
    # STAGE 2: Codebook-Snap Lattice Projection + FP16 Residual Compensation
    # =========================================================================
    print("\n=== Stage 2: WSP-E10_QuantComp — Codebook-Snap Lattice Projection & FP16 Residual Compensation ===", flush=True)
    # For every row r modified in our Unquantized FP32 Master model (`best_W_fp32_master`),
    # Delta_w_fp32[r] = best_W_fp32_master[r] - W_e1b_emb[r] has the exact FP32 direction.
    # When quantizing to 4-bit Hadamard CQ-W4 (`group=128`), small updates can fall below the
    # codebook step threshold. We search over lattice projection scales alpha_r in [0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
    # so the dequantized CQ-W4 row maximizes net hits on the quantized hidden states H_q_clean!
    H_q_clean = H_q[clean_state_mask]
    U_q_clean = U_q[clean_state_mask]
    Y_hat_q_clean = Y_hat_q0[clean_state_mask]
    m_q_clean = margins_q0[clean_state_mask]

    W_e10_raw = W_e1b_emb.copy()
    W_e10_q = W_q0.copy()
    lattice_snapped_rows = []

    fp32_accepted_row_ids = [r for r, _ in best_e9_cfg["accepted_rows"]]
    alpha_candidates = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]

    if fp32_accepted_row_ids:
        n_acc = len(fp32_accepted_row_ids)
        n_alp = len(alpha_candidates)
        W_tries_raw = np.zeros((n_acc * n_alp, 512), dtype=np.float32)
        for i_r, r in enumerate(fp32_accepted_row_ids):
            dw_fp32 = best_W_fp32_master[r] - W_e1b_emb[r]
            orig_norm = float(np.linalg.norm(W_e1b_emb[r]))
            dw_fp_norm = float(np.linalg.norm(dw_fp32))
            for i_a, alpha in enumerate(alpha_candidates):
                dw_scaled = alpha * dw_fp32
                if dw_fp_norm * alpha > 0.095 * orig_norm and dw_fp_norm > 1e-12:
                    dw_scaled = dw_fp32 * (0.095 * orig_norm / dw_fp_norm)
                W_tries_raw[i_r * n_alp + i_a] = W_e1b_emb[r] + dw_scaled

        W_tries_q = cq_quantize_dequantize_np(W_tries_raw, bits=4, group=128)
        Scores_tries_q = H_q_clean @ W_tries_q.T  # [N_clean, n_acc * n_alp]

        for i_r, r in enumerate(fp32_accepted_row_ids):
            is_target = (Y_clean == r)
            is_false_comp = (Y_hat_q_clean == r) & (m_q_clean < 0.0)
            was_right = (m_q_clean > 0.0) & (~is_target)

            best_r_gain = 1  # Require >= 2 net gain to accept into CQ-W4
            best_idx = -1
            best_alpha = 1.0
            for i_a, alpha in enumerate(alpha_candidates):
                col_idx = i_r * n_alp + i_a
                new_scores_q = Scores_tries_q[:, col_idx]
                fn_fixed = int(np.sum(is_target & (m_q_clean <= 0.0) & ((new_scores_q - z_comp_q0[clean_state_mask]) > 0.0)))
                fn_broken = int(np.sum(is_target & (m_q_clean > 0.0) & ((new_scores_q - z_comp_q0[clean_state_mask]) <= 0.0)))
                fp_fixed = int(np.sum(is_false_comp & ((z_true_q0[clean_state_mask] - new_scores_q) > 0.0)))
                new_fp = int(np.sum(was_right & (new_scores_q > z_true_q0[clean_state_mask])))

                net_q_gain = (fn_fixed + fp_fixed) - 2 * (fn_broken + new_fp)
                if net_q_gain > best_r_gain:
                    best_r_gain = net_q_gain
                    best_idx = col_idx
                    best_alpha = alpha

            if best_idx >= 0:
                W_e10_raw[r] = W_tries_raw[best_idx]
                W_e10_q[r] = W_tries_q[best_idx]
                lattice_snapped_rows.append({"row": int(r), "net_q_gain": int(best_r_gain), "alpha": best_alpha})

    Z_e10_emb = H_q_clean @ W_e10_q.T
    Z_e10_emb[:, 0] = -1e9
    Z_e10_emb[:, 2] = -1e9
    Z_e10_emb[:, 8906:] = -1e9
    hits_q0 = int(np.sum(m_q_clean > 0.0))
    hits_e10_emb = int(np.sum(np.argmax(Z_e10_emb, axis=1) == Y_clean))
    print(
        f"  [Stage 2A: Lattice-Snapped CQ-W4 Embedding] Snapped {len(lattice_snapped_rows)} / {len(fp32_accepted_row_ids)} "
        f"FP32 rows -> +{hits_e10_emb - hits_q0} net CQ-W4 teacher-forced hits!",
        flush=True,
    )

    # Stage 2B: Absorb remaining CQ-W4 quantization residual (W_fp32_master - W_e10_q)
    # into the adjacent UNQUANTIZED FP16 `stack/final_norm/scale` ([512])!
    # Notice: In FP32, target margin was <U_fp_clean * (1 + s_fp32), W_fp32[y*] - W_fp32[yhat]>.
    # In CQ-W4, current margin is <U_q_clean * (1 + s0), W_e10_q[y*] - W_e10_q[yhat]>.
    # The exact quantization margin loss on token t is:
    #   r_quant_t = m_fp32_master[t] - m_e10_q[t]
    z_t_e10 = Z_e10_emb[r_idx_c, Y_clean]
    Z_c_e10 = Z_e10_emb.copy()
    Z_c_e10[r_idx_c, Y_clean] = -1e9
    Y_hat_e10 = np.argmax(Z_c_e10, axis=1)
    m_e10_q = z_t_e10 - Z_c_e10[r_idx_c, Y_hat_e10]

    G_q_comp = U_q_clean * (W_e10_q[Y_clean] - W_e10_q[Y_hat_e10])
    Cov_Gq = (G_q_comp.T @ G_q_comp) / float(N_clean)

    # Target: recover the positive FP32 margin advantage where CQ-W4 degraded or remained below margin 1.0
    quant_loss_mask = (m_e10_q < 1.0) & (m_after > m_e10_q) & (Y_clean != 1)
    r_quant_loss = np.zeros(N_clean, dtype=np.float32)
    r_quant_loss[quant_loss_mask] = np.clip(m_after[quant_loss_mask] - m_e10_q[quant_loss_mask], 0.0, 3.0)

    best_s_e10 = s0.copy()
    best_e10_total_hits = hits_e10_emb
    best_s_comp_cfg = {"lam": 0.0, "step": 0.0, "hits_gain_over_emb": 0}

    for lam_qc in [25.0, 75.0, 200.0]:
        ds_qc_base = np.linalg.solve(
            Cov_Gq + lam_qc * np.eye(512, dtype=np.float32),
            (G_q_comp.T @ r_quant_loss) / float(N_clean),
        )
        for step_qc in [0.15, 0.3, 0.5, 0.75]:
            ds_qc = np.clip(step_qc * ds_qc_base, -0.04, 0.04)
            s_qc_cand = (s0 + ds_qc).astype(np.float16).astype(np.float32)
            h_qc_raw = U_q_clean * (1.0 + s_qc_cand)[None, :]
            absmax = np.max(np.abs(h_qc_raw), axis=-1, keepdims=True)
            scale_a8 = np.where(absmax > 0, absmax / 127.0, 1.0)
            h_qc_q = np.clip(np.round(h_qc_raw / scale_a8), -128, 127) * scale_a8

            Z_qc = h_qc_q @ W_e10_q.T
            Z_qc[:, 0] = -1e9
            Z_qc[:, 2] = -1e9
            Z_qc[:, 8906:] = -1e9
            hits_qc = int(np.sum(np.argmax(Z_qc, axis=1) == Y_clean))
            if hits_qc > best_e10_total_hits:
                best_e10_total_hits = hits_qc
                best_s_e10 = s_qc_cand.copy()
                best_s_comp_cfg = {
                    "lam": lam_qc,
                    "step": step_qc,
                    "hits_gain_over_emb": hits_qc - hits_e10_emb,
                    "total_hits_gain_over_q0": hits_qc - hits_q0,
                }

    print(f"  [Stage 2B: FP16 FinalNorm Quantization Residual Compensation] {best_s_comp_cfg}", flush=True)

    # =========================================================================
    # STAGE 3: Autoregressive Evaluations (Pure FP32 vs Quantized CQ-W4A8)
    # =========================================================================
    def evaluate_autoregressive(
        p_tree: Any,
        exp_name: str,
        quant_mode: bool,
    ) -> Dict[str, Any]:
        out_metrics: Dict[str, Any] = {}
        dev_splits = {"fleurs_hi": 200, "svarah_en": 300, "mucs_hinglish": 300, "lahaja_hi": 300, "librispeech_en": 150}
        enc_cache = encoded_cache_q if quant_mode else encoded_cache_fp32
        step_fn = jit_step_logits_q if quant_mode else jit_step_logits_fp32

        for ds_key, clips in datasets_map.items():
            n_dev = dev_splits[ds_key]
            enc_batches = enc_cache[ds_key]
            hyps_all: List[str] = []
            langs_all: List[str] = []

            for b_idx, (enc, frames, real_B) in enumerate(enc_batches):
                tokens = np.full((B_PAD, MAX_DEC_LEN), cfg.pad, dtype=np.int32)
                tokens[:, 0] = cfg.bos
                done = np.zeros(B_PAD, dtype=bool)
                done[real_B:] = True
                tags_mask = np.arange(cfg.vocab) >= cfg.text

                for pos in range(MAX_DEC_LEN - 1):
                    if np.all(done):
                        break
                    bucket_size = 64 if pos < 64 else 128
                    logits_all = np.asarray(step_fn(p_tree, jnp.asarray(tokens[:, :bucket_size]), enc, frames))
                    step_logits = logits_all[:, pos].copy()
                    if pos == 0:
                        step_logits[:, ~tags_mask] = -np.inf
                        pick = step_logits.argmax(axis=-1).astype(np.int32)
                    else:
                        step_logits[:, tags_mask] = -np.inf
                        step_logits[:, cfg.pad] = -np.inf
                        step_logits[:, cfg.bos] = -np.inf
                        pick = step_logits.argmax(axis=-1).astype(np.int32)
                    tokens[:, pos + 1] = np.where(done, cfg.pad, pick)
                    if pos > 0:
                        done |= (pick == cfg.eos)

                for bi in range(real_B):
                    seq = tokens[bi].tolist()
                    pred_tag_id = seq[1]
                    pred_lang = lang_names[pred_tag_id - cfg.text] if cfg.text <= pred_tag_id < cfg.vocab else "unk"
                    langs_all.append(pred_lang)
                    body = seq[2 : seq.index(cfg.eos)] if cfg.eos in seq[2:] else seq[2:]
                    body = [tid for tid in body if tid != cfg.pad and tid < cfg.text]
                    hyps_all.append(tokenizer.decode(body).strip())

            is_en_ds = ds_key in ("svarah_en", "librispeech_en")
            expected_lang = "en" if is_en_ds else "hi"

            def score_slice(s_start: int, s_end: int) -> Dict[str, Any]:
                w_sub = w_ins = w_del = w_ref = 0
                c_sub = c_ins = c_del = c_ref = 0
                lid_hits = 0
                count = 0
                for idx in range(s_start, min(s_end, len(hyps_all))):
                    ref_raw = clips[idx]["text"]
                    hyp_raw = hyps_all[idx]
                    pl = langs_all[idx]
                    if is_en_ds:
                        r_n = en_norm(ref_raw)
                        h_n = en_norm(hyp_raw) if pl == "en" else normalize_indic_hinglish(hyp_raw)
                    else:
                        r_n = normalize_indic_hinglish(ref_raw)
                        h_n = normalize_indic_hinglish(hyp_raw)
                    r_words = r_n.split()
                    h_words = h_n.split()
                    if not r_words:
                        continue
                    count += 1
                    if langs_all[idx] == expected_lang:
                        lid_hits += 1
                    ws, wi, wd, _ = levenshtein_alignment(r_words, h_words)
                    w_sub += ws
                    w_ins += wi
                    w_del += wd
                    w_ref += len(r_words)
                    r_chars = list(r_n.replace(" ", ""))
                    h_chars = list(h_n.replace(" ", ""))
                    cs, ci, cd, _ = levenshtein_alignment(r_chars, h_chars)
                    c_sub += cs
                    c_ins += ci
                    c_del += cd
                    c_ref += max(len(r_chars), 1)
                return {
                    "num_clips": count,
                    "wer_pct": round((w_sub + w_ins + w_del) / max(w_ref, 1) * 100.0, 2),
                    "cer_pct": round((c_sub + c_ins + c_del) / max(c_ref, 1) * 100.0, 2),
                    "lid_acc_pct": round(lid_hits / max(count, 1) * 100.0, 2),
                    "subs": w_sub,
                    "ins": w_ins,
                    "dels": w_del,
                }

            res_ds = {
                "dev": score_slice(0, n_dev),
                "test": score_slice(n_dev, len(clips)),
                "all": score_slice(0, len(clips)),
            }
            out_metrics[ds_key] = res_ds
            print(f"  [{exp_name} | {ds_key}] {res_ds}", flush=True)

        return out_metrics

    print("\n=== Stage 3A: Evaluating Pure Unquantized FP32 Models (`quant=False`) ===", flush=True)
    # 1. Official Unquantized FP32 Baseline (`params_orig`, `quant=False`)
    # Re-encode for `params_orig`? Note: `Whistle.encode` does NOT use `embedding/embedding`!
    # So `encoded_cache_fp32` is 100% identical for `params_orig`, `params_e1b_fp32`, and `params_e9_fp32`!
    res_orig_fp32 = evaluate_autoregressive(params_orig, "Original_Unquantized_FP32", quant_mode=False)
    res_e1b_fp32 = evaluate_autoregressive(params_e1b_fp32, "WSP_E1b_Unquantized_FP32", quant_mode=False)

    params_e9_fp32 = {
        **params_orig,
        "embedding": {"embedding": jnp.asarray(best_W_fp32_master, dtype=jnp.float32)},
        "stack": {
            **params_orig["stack"],
            "final_norm": {"scale": jnp.asarray(best_s_fp32_master, dtype=jnp.float32)},
        },
    }
    res_e9_fp32 = evaluate_autoregressive(params_e9_fp32, "WSP_E9_Master_Unquantized_FP32", quant_mode=False)

    print("\n=== Stage 3B: Evaluating Lattice-Snapped & Quantization-Compensated CQ-W4A8 Models (`quant=True`) ===", flush=True)
    # 1. Evaluate WSP-E10a (Lattice-Snapped CQ-W4 Embedding only, s0)
    params_e10a_raw = {
        **params_orig,
        "embedding": {"embedding": jnp.asarray(W_e10_raw, dtype=jnp.float32)},
    }
    params_e10a_q = cq_ste_params(params_e10a_raw, WEIGHT_BITS)
    res_e10a_q = evaluate_autoregressive(params_e10a_q, "WSP_E10a_LatticeSnap_CQ4", quant_mode=True)

    # 2. Evaluate WSP-E10b (Lattice-Snapped CQ-W4 Embedding + FP16 FinalNorm Quantization Compensation)
    params_e10b_raw = {
        **params_orig,
        "embedding": {"embedding": jnp.asarray(W_e10_raw, dtype=jnp.float32)},
        "stack": {
            **params_orig["stack"],
            "final_norm": {"scale": jnp.asarray(best_s_e10, dtype=jnp.float32)},
        },
    }
    params_e10b_q = cq_ste_params(params_e10b_raw, WEIGHT_BITS)
    res_e10b_q = evaluate_autoregressive(params_e10b_q, "WSP_E10b_QuantComp_CQ4", quant_mode=True)

    def summarize_eval(res_dict: Dict[str, Any]) -> Dict[str, float]:
        keys = list(datasets_map.keys())
        return {
            "dev_macro_wer": round(float(np.mean([res_dict[k]["dev"]["wer_pct"] for k in keys])), 3),
            "test_macro_wer": round(float(np.mean([res_dict[k]["test"]["wer_pct"] for k in keys])), 3),
            "all_macro_wer": round(float(np.mean([res_dict[k]["all"]["wer_pct"] for k in keys])), 3),
            "fleurs_hi_all_wer": res_dict["fleurs_hi"]["all"]["wer_pct"],
            "svarah_en_all_wer": res_dict["svarah_en"]["all"]["wer_pct"],
            "mucs_hinglish_all_wer": res_dict["mucs_hinglish"]["all"]["wer_pct"],
            "lahaja_hi_all_wer": res_dict["lahaja_hi"]["all"]["wer_pct"],
            "librispeech_en_all_wer": res_dict["librispeech_en"]["all"]["wer_pct"],
        }

    summary_table = {
        "Original_Unquantized_FP32": summarize_eval(res_orig_fp32),
        "WSP_E1b_Unquantized_FP32": summarize_eval(res_e1b_fp32),
        "WSP_E9_Master_Unquantized_FP32": summarize_eval(res_e9_fp32),
        "Phase7_WSP_E5_CQ4": p7_res["summary_macros"]["WSP_E5_Ridge_Embedding"],
        "WSP_E10a_LatticeSnap_CQ4": summarize_eval(res_e10a_q),
        "WSP_E10b_QuantComp_CQ4": summarize_eval(res_e10b_q),
    }
    for k_name, sm in summary_table.items():
        print(f"Summary [{k_name}]: {sm}", flush=True)

    # Pick the best quantized candidate between E10a and E10b by Dev Macro WER for C++ 5-beam verification
    best_e10_key = (
        "WSP_E10b_QuantComp_CQ4"
        if summary_table["WSP_E10b_QuantComp_CQ4"]["dev_macro_wer"] < summary_table["WSP_E10a_LatticeSnap_CQ4"]["dev_macro_wer"]
        else "WSP_E10a_LatticeSnap_CQ4"
    )
    best_e10_s_export = best_s_e10 if best_e10_key == "WSP_E10b_QuantComp_CQ4" else s0

    # =========================================================================
    # STAGE 4: Native C++ `needle` 5-Beam Engine Verification of Phase 8 `.cact`
    # =========================================================================
    print(f"\n=== Stage 4: C++ `needle` 5-Beam Verification of {best_e10_key} ===", flush=True)
    new_flat_p8 = {**orig_flat}
    new_flat_p8["embedding/embedding"] = W_e10_raw.astype(np.float32)
    new_flat_p8["stack/final_norm/scale"] = best_s_e10_export = best_e10_s_export.astype(np.float32)

    out_cact_p8 = "/tmp/shruti_wsp_phase8_quantcomp.cact"
    patch_info = surgical_patch_cact(
        cact_path,
        out_cact_p8,
        orig_flat,
        new_flat_p8,
        WHISTLE_SMALL_CFG,
    )
    print(f"Surgically patched {out_cact_p8}: {patch_info}", flush=True)

    cpp_results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for ds_key, clips in datasets_map.items():
            sub_clips = clips[:60]
            is_en_ds = ds_key in ("svarah_en", "librispeech_en")
            expected_lang = "en" if is_en_ds else "hi"
            w_sub = w_ins = w_del = w_ref = 0
            c_sub = c_ins = c_del = c_ref = 0
            lid_hits = 0
            count = 0
            for c in sub_clips:
                wp = os.path.join(tmpdir, f"{ds_key}_{c['idx']}.wav")
                write_wav_16k(wp, c["samples"])
                proc = subprocess.run(
                    [needle_bin, "--model", out_cact_p8, "--audio", wp, "--threads", "4"],
                    capture_output=True, text=True, check=True,
                )
                obj = json.loads(proc.stdout.strip().splitlines()[-1])
                hyp_raw = obj.get("text", "").strip()
                lang_p = obj.get("language", "hi")
                ref_raw = c["text"]
                if is_en_ds:
                    r_n = en_norm(ref_raw)
                    h_n = en_norm(hyp_raw) if lang_p == "en" else normalize_indic_hinglish(hyp_raw)
                else:
                    r_n = normalize_indic_hinglish(ref_raw)
                    h_n = normalize_indic_hinglish(hyp_raw)
                r_words = r_n.split()
                h_words = h_n.split()
                if not r_words:
                    continue
                count += 1
                if lang_p == expected_lang:
                    lid_hits += 1
                ws, wi, wd, _ = levenshtein_alignment(r_words, h_words)
                w_sub += ws
                w_ins += wi
                w_del += wd
                w_ref += len(r_words)
                r_chars = list(r_n.replace(" ", ""))
                h_chars = list(h_n.replace(" ", ""))
                cs, ci, cd, _ = levenshtein_alignment(r_chars, h_chars)
                c_sub += cs
                c_ins += ci
                c_del += cd
                c_ref += max(len(r_chars), 1)

            cpp_results[ds_key] = {
                "num_clips": count,
                "wer_pct": round((w_sub + w_ins + w_del) / max(w_ref, 1) * 100.0, 2),
                "cer_pct": round((c_sub + c_ins + c_del) / max(c_ref, 1) * 100.0, 2),
                "lid_acc_pct": round(lid_hits / max(count, 1) * 100.0, 2),
            }
            print(f"  [C++ 5-Beam | {ds_key}] {cpp_results[ds_key]}", flush=True)

    cpp_macro_wer = round(float(np.mean([cpp_results[k]["wer_pct"] for k in datasets_map.keys()])), 3)
    print(f"C++ 5-Beam Macro WER ({best_e10_key}): {cpp_macro_wer}%", flush=True)

    final_output = {
        "fp32_vs_cq4_diagnostics": {
            "total_dev_tokens": int(N_tok),
            "clean_dev_tokens": int(N_clean),
            "tf_acc_fp32_all_pct": round(tf_fp0_all, 3),
            "tf_acc_cq4_all_pct": round(tf_q0_all, 3),
            "tf_acc_fp32_hi_pct": round(tf_fp0_hi, 3),
            "tf_acc_cq4_hi_pct": round(tf_q0_hi, 3),
            "tf_acc_fp32_hinglish_pct": round(tf_fp0_hing, 3),
            "tf_acc_cq4_hinglish_pct": round(tf_q0_hing, 3),
            "hidden_fp32_vs_cq4_cos_sim": round(h_cos_sim, 5),
            "embedding_cq4_rmse": round(w_emb_quant_rmse, 5),
            "direct_logit_attribution_on_dev_errors": {
                "base_decoder_unit_margin": round(u_contrib_err, 4),
                "final_norm_scale_margin": round(s_contrib_err, 4),
                "top5_sir_eigenvalues": top5_sir_eigs,
            },
        },
        "e9_fp32_grid_logs": e9_fp32_logs,
        "best_e9_fp32_cfg": best_e9_cfg,
        "best_fp32_master_modified_rows": {
            str(r): best_W_fp32_master[r].tolist()
            for r, _ in (best_e9_cfg["accepted_rows"] if best_e9_cfg else [])
        },
        "best_fp32_master_s_scale": best_s_fp32_master.tolist(),
        "lattice_snapped_rows": lattice_snapped_rows,
        "best_e10_modified_rows": {
            str(item["row"]): W_e10_raw[item["row"]].tolist()
            for item in lattice_snapped_rows
        },
        "best_s_comp_cfg": best_s_comp_cfg,
        "best_s_e10_scale": best_s_e10.tolist(),
        "experiments": {
            "Original_Unquantized_FP32": res_orig_fp32,
            "WSP_E1b_Unquantized_FP32": res_e1b_fp32,
            "WSP_E9_Master_Unquantized_FP32": res_e9_fp32,
            "WSP_E10a_LatticeSnap_CQ4": res_e10a_q,
            "WSP_E10b_QuantComp_CQ4": res_e10b_q,
        },
        "summary_table": summary_table,
        "selected_best_quantized": best_e10_key,
        "cpp_needle_5beam_phase8": cpp_results,
        "cpp_needle_5beam_macro_wer": cpp_macro_wer,
        "patch_info": patch_info,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    with open("/data/eval_cache/phase8_fp32_master_wsp_results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    vol.commit()
    return final_output


@app.local_entrypoint()
def main():
    wsp_vec_path = ROOT_LOCAL / "reports" / "phase3_wsp_vectors.npz"
    p7_path = ROOT_LOCAL / "reports" / "phase7_analytic_wsp_results.json"
    out_json_path = ROOT_LOCAL / "reports" / "phase8_fp32_master_wsp_results.json"
    wsp_bytes = wsp_vec_path.read_bytes()
    p7_res = json.loads(p7_path.read_text(encoding="utf-8"))
    res = run_phase8_fp32_master_wsp.remote(wsp_bytes, p7_res)
    out_json_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Saved Phase 8 Unquantized FP32 Master -> Quantized WSP results to {out_json_path}")
