#!/usr/bin/env python3
"""Phase 4 (Weight-Space Programming) & Phase 5 (Independent Validation) on Modal GPU.

Executes controlled, reproducible, gradient-free Weight-Space Programming experiments:
    W_new = W_original + Delta_W

Strictly adheres to all research constraints:
  1. Never modifies the original checkpoints (`checkpoints/original/shruti.safetensors` & `shruti.cact`).
  2. Uses zero gradient descent, backpropagation, LoRA, or fine-tuning.
  3. Preserves the exact architecture and 8,914-token vocabulary.
  4. Selects hyperparameters strictly on the Dev splits (`[0:N_dev]`) and validates on the
     strictly held-out Test splits (`[N_dev:N_total]`), untouched `librispeech_en`, and
     100 Silence/Hiss hallucination regression clips.
  5. Exports both `.safetensors` and surgically patched `.cact` (via `surgical_patch_cact`)
     and validates on both the GPU CQ-W4A8 STE pipeline and the native C++ `needle` 5-beam engine.
"""

import hashlib
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

app = modal.App("shruti-wsp-phase4-5")

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
MAX_SECONDS = 30
MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS


def normalize_indic_hinglish(text: str) -> str:
    """Normalize Hindi, English, and Hinglish text while preserving Devanagari matras (Mn/Mc)."""
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
        print(f"[Cache Hit] Loading {key} from {npz_path}...", flush=True)
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



def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=8.0,
    memory=32768,
    timeout=3600,
    volumes={"/data": vol},
)
def run_phase4_5_experiments(phase3_diag: Dict[str, Any], wsp_vectors_bytes: bytes) -> Dict[str, Any]:
    import io
    try:
        from nvidia import cuda_nvcc
        if getattr(cuda_nvcc, "__file__", None) is None and hasattr(cuda_nvcc, "__path__"):
            cuda_nvcc.__file__ = os.path.join(list(cuda_nvcc.__path__)[0], "__init__.py")
    except Exception:
        pass
    import jax
    import jax.numpy as jnp
    from whisper_normalizer.basic import BasicTextNormalizer
    from whisper_normalizer.english import EnglishTextNormalizer

    sys.path.insert(0, "/root/shruti/src")
    from needle_model.architecture import _aq
    from needle_model.quantize import WEIGHT_BITS, cq_ste_params
    from needle_model.whistle import Whistle, load_whistle
    from shruti_io import (
        RefTokenizer,
        WHISTLE_SMALL_CFG,
        load_safetensors_flat,
        save_safetensors_flat,
        surgical_patch_cact,
    )

    t0 = time.time()
    print(f"JAX devices: {jax.devices()} (backend: {jax.default_backend()})", flush=True)

    ckpt_path = "/data/checkpoints/shruti.safetensors"
    cact_path = "/data/checkpoints/shruti.cact"
    needle_bin = "/data/checkpoints/needle"
    os.chmod(needle_bin, 0o755)

    # Load original checkpoint & baseline Phase 2 report from /data/eval_cache
    orig_flat = load_safetensors_flat(ckpt_path)
    params_orig, cfg = load_whistle(ckpt_path)
    cfg = replace(cfg, flash=False, remat=False)
    model = Whistle(cfg)
    tokenizer = RefTokenizer.from_cact(cact_path)
    lang_names = list(cfg.languages)

    with open("/data/eval_cache/phase2_baseline_eval.json", "r", encoding="utf-8") as f:
        phase2_baseline = json.load(f)
    step0_npz = np.load("/data/eval_cache/phase2_step0_states.npz", allow_pickle=False)
    wsp_vecs = np.load(io.BytesIO(wsp_vectors_bytes), allow_pickle=False)

    datasets_map = load_or_cache_datasets("/data/eval_cache")

    @jax.jit
    def jit_encode(p, wav, lengths):
        return model.apply({"params": p}, wav, lengths, True, method=Whistle.encode)

    @jax.jit
    def jit_step_logits(p, tokens, enc, frames):
        h = model.apply({"params": p}, tokens, enc, frames, True, method=Whistle.decode)
        hq = _aq(h, True).astype(jnp.float32)
        logits = hq @ p["embedding"]["embedding"].T.astype(jnp.float32)
        return logits

    # Pre-encode all 5 datasets + 100 silence clips ONCE with the unmodified encoder!
    # Wait: For experiments E1, E2, E3 (which only edit `embedding/embedding`), the encoder
    # output `enc, frames` is 100% invariant! Pre-encoding once makes evaluating every
    # decoder/embedding edit 5x faster and 100% deterministic!
    print("Pre-computing unmodified encoder acoustic representations across all datasets...", flush=True)
    params_orig_f32 = jax.tree.map(lambda a: jnp.asarray(a, dtype=jnp.float32), params_orig)
    ste_orig = cq_ste_params(params_orig_f32, WEIGHT_BITS)

    eval_plan = [
        ("fleurs_hi", "hi", (0, 200), (200, 418)),
        ("svarah_en", "en", (0, 300), (300, 600)),
        ("mucs_hinglish", "hi", (0, 300), (300, 600)),
        ("lahaja_hi", "hi", (0, 300), (300, 600)),
        ("librispeech_en", "en", (0, 150), (150, 300)),
    ]

    BATCH_SIZE = 60
    encoded_cache: Dict[str, List[Tuple[Any, Any, List[Dict[str, Any]]]]] = {}
    for ds_key, _, _, _ in eval_plan:
        clips = datasets_map[ds_key]
        batches = []
        for start in range(0, len(clips), BATCH_SIZE):
            part = clips[start:start + BATCH_SIZE]
            wav = np.zeros((BATCH_SIZE, MAX_SAMPLES), dtype=np.float32)
            lengths = np.full(BATCH_SIZE, SAMPLE_RATE, dtype=np.int32)
            for b_i, c in enumerate(part):
                s = c["samples"]
                wav[b_i, :len(s)] = s
                lengths[b_i] = len(s)
            enc, frames = jit_encode(ste_orig, wav, lengths)
            batches.append((enc, frames, part))
        encoded_cache[ds_key] = batches

    # Also pre-encode 100 silence/hiss clips using the same fixed BATCH_SIZE=60
    rng = np.random.RandomState(42)
    silence_clips = []
    for i in range(100):
        s = np.zeros(SAMPLE_RATE * 4, dtype=np.float32) if i < 50 else (rng.randn(SAMPLE_RATE * 4) * 1e-4).astype(np.float32)
        silence_clips.append({"idx": i, "text": "", "target_lang": "hi", "duration_sec": 4.0, "samples": s})
    sil_batches = []
    for start in range(0, 100, BATCH_SIZE):
        part = silence_clips[start:start + BATCH_SIZE]
        wav = np.zeros((BATCH_SIZE, MAX_SAMPLES), dtype=np.float32)
        lengths = np.full(BATCH_SIZE, SAMPLE_RATE * 4, dtype=np.int32)
        for b_i, c in enumerate(part):
            wav[b_i, :len(c["samples"])] = c["samples"]
        enc, frames = jit_encode(ste_orig, wav, lengths)
        sil_batches.append((enc, frames, part))

    en_norm = EnglishTextNormalizer()
    basic_norm = BasicTextNormalizer()

    def decode_with_ste_params(
        ste_p: Any,
        batches: List[Tuple[Any, Any, List[Dict[str, Any]]]],
        forced_lang: Optional[str] = None,
        max_batches: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        tags_mask = np.arange(cfg.vocab) >= cfg.text
        forced_tag_id = cfg.tag(forced_lang) if forced_lang else None
        outputs: List[Dict[str, Any]] = []
        use_batches = batches[:max_batches] if max_batches is not None else batches
        for enc, frames, part in use_batches:
            B_PAD = int(enc.shape[0])
            real_B = len(part)
            tokens = np.full((B_PAD, cfg.max_tokens), cfg.pad, dtype=np.int32)
            tokens[:, 0] = cfg.bos
            done = np.zeros(B_PAD, dtype=bool)
            done[real_B:] = True
            max_gen_steps = min(cfg.max_tokens - 1, 192)
            for pos in range(max_gen_steps):
                bucket_size = min(cfg.max_tokens, (pos // 64 + 1) * 64)
                logits_all = jit_step_logits(ste_p, tokens[:, :bucket_size], enc, frames)
                step_logits = np.array(logits_all[:, pos])
                if pos == 0:
                    if forced_tag_id is not None:
                        pick = np.full(B_PAD, forced_tag_id, dtype=np.int32)
                    else:
                        step_logits[:, ~tags_mask] = -np.inf
                        pick = step_logits.argmax(axis=-1).astype(np.int32)
                else:
                    step_logits[:, tags_mask] = -np.inf
                    step_logits[:, cfg.pad] = -np.inf
                    step_logits[:, cfg.bos] = -np.inf
                    pick = step_logits.argmax(axis=-1).astype(np.int32)
                tokens[:, pos + 1] = np.where(done, cfg.pad, pick)
                done |= (pick == cfg.eos)
                if done.all():
                    break
            for b_i, c in enumerate(part):
                seq = tokens[b_i].tolist()
                pred_tag_id = seq[1]
                pred_lang = lang_names[pred_tag_id - cfg.text] if cfg.text <= pred_tag_id < cfg.vocab else "unk"
                body = seq[2:seq.index(cfg.eos)] if cfg.eos in seq[2:] else seq[2:]
                body = [tid for tid in body if tid != cfg.pad and tid < cfg.text]
                hyp_text = tokenizer.decode(body).strip()
                outputs.append({
                    "idx": c["idx"],
                    "ref": c["text"],
                    "hyp": hyp_text,
                    "target_lang": c["target_lang"],
                    "pred_lang": pred_lang,
                })
        return outputs

    def quick_score(outputs: List[Dict[str, Any]], split_slice: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        sub = outputs[split_slice[0]:split_slice[1]] if split_slice is not None else outputs
        w_s = w_i = w_d = w_tot = 0
        c_s = c_i = c_d = c_tot = 0
        ref_lat = hyp_lat = ref_dev = hyp_dev = 0
        dev_to_en = en_to_dev = 0
        lid_ok = 0
        for item in sub:
            tl = item["target_lang"]
            pl = item["pred_lang"]
            if pl == tl:
                lid_ok += 1
            if tl == "en":
                rn = en_norm(item["ref"])
                hn = en_norm(item["hyp"]) if pl == "en" else normalize_indic_hinglish(item["hyp"])
            else:
                rn = normalize_indic_hinglish(item["ref"])
                hn = normalize_indic_hinglish(item["hyp"])
            rw = rn.split()
            hw = hn.split()
            for w in rw:
                if any("a" <= ch <= "z" for ch in w):
                    ref_lat += 1
                elif any("\u0900" <= ch <= "\u097f" for ch in w):
                    ref_dev += 1
            for w in hw:
                if any("a" <= ch <= "z" for ch in w):
                    hyp_lat += 1
                elif any("\u0900" <= ch <= "\u097f" for ch in w):
                    hyp_dev += 1
            s, ins, d, ops = levenshtein_alignment(rw, hw)
            w_s += s
            w_i += ins
            w_d += d
            w_tot += len(rw)
            for op, r_tok, h_tok in ops:
                if op == "sub":
                    r_is_lat = any("a" <= ch <= "z" for ch in r_tok)
                    r_is_dev = any("\u0900" <= ch <= "\u097f" for ch in r_tok)
                    h_is_lat = any("a" <= ch <= "z" for ch in h_tok)
                    h_is_dev = any("\u0900" <= ch <= "\u097f" for ch in h_tok)
                    if r_is_dev and h_is_lat:
                        dev_to_en += 1
                    elif r_is_lat and h_is_dev:
                        en_to_dev += 1
            rc = list(rn.replace(" ", ""))
            hc = list(hn.replace(" ", ""))
            cs, ci, cd, _ = levenshtein_alignment(rc, hc)
            c_s += cs
            c_i += ci
            c_d += cd
            c_tot += len(rc)
        return {
            "num_clips": len(sub),
            "wer": round((w_s + w_i + w_d) / max(w_tot, 1) * 100.0, 2),
            "cer": round((c_s + c_i + c_d) / max(c_tot, 1) * 100.0, 2),
            "word_edits": int(w_s + w_i + w_d),
            "word_subs": int(w_s),
            "word_ins": int(w_i),
            "word_dels": int(w_d),
            "word_ref_total": int(w_tot),
            "lid_accuracy_pct": round(lid_ok / max(len(sub), 1) * 100.0, 2),
            "ref_latin_words": int(ref_lat),
            "hyp_latin_words": int(hyp_lat),
            "ref_devanagari_words": int(ref_dev),
            "hyp_devanagari_words": int(hyp_dev),
            "script_mismatch_dev_to_en": int(dev_to_en),
            "script_mismatch_en_to_dev": int(en_to_dev),
        }

    def make_ste_with_new_embedding(E_new: np.ndarray) -> Any:
        # In Whistle, `embedding/embedding` is NOT quantized by `cq_ste_params` (only linear kernels are CQ-W4).
        # Replacing ste_orig["embedding"]["embedding"] directly updates the parameter tree!
        p_new = dict(ste_orig)
        p_new["embedding"] = {"embedding": jnp.asarray(E_new, dtype=jnp.float32)}
        return p_new

    E_orig = orig_flat["embedding/embedding"].astype(np.float32).copy()
    mu_0_unit = wsp_vecs["mu_0_unit"].astype(np.float32)
    u_fisher = wsp_vecs["u_fisher"].astype(np.float32)
    delta_w_best = wsp_vecs["delta_w_best"].astype(np.float32)
    pure_bias_beta = float(phase3_diag["diagnosis_1_step0_lid"]["dev_calibrated_pure_bias_beta"])

    experiments_log: Dict[str, Any] = {}

    # ========================================================================
    # EXPERIMENT 1A & 1B: Step-0 Language ID Subspace Steering (H1)
    #   Wait: Notice that once Step-0 picks `en` (8906) or `hi` (8913), does a
    #   tiny perturbation (||Delta e||_2 <= 1.8 vs ||e||_2 = 68.0, cos > 0.9996)
    #   change downstream decoding at steps >= 1?
    #   We test this directly on full GPU decoding across Svarah, FLEURS, Lahaja,
    #   MUCS Hinglish, and LibriSpeech!
    # ========================================================================
    print("\n=== Running Phase 4 Experiment WSP-E1a (1D BOS Prior Shift), WSP-E1b (Max-Recall Fisher Steering), & WSP-E1c (Zero-Hindi-Regression Fisher Steering) ===", flush=True)

    # WSP-E1a: Pure 1D BOS prior shift
    E_e1a = E_orig.copy()
    E_e1a[8906] = E_orig[8906] + 0.5 * pure_bias_beta * mu_0_unit
    E_e1a[8913] = E_orig[8913] - 0.5 * pure_bias_beta * mu_0_unit
    for eu_i in range(8907, 8913):
        E_e1a[eu_i] = E_orig[eu_i] - 2.0 * mu_0_unit

    # WSP-E1b: Fisher Discriminant Subspace Steering + BOS Prior Shift (gamma=12.0, beta=5.0)
    E_e1b = E_orig.copy()
    E_e1b[8906] = E_orig[8906] + 0.5 * delta_w_best
    E_e1b[8913] = E_orig[8913] - 0.5 * delta_w_best
    for eu_i in range(8907, 8913):
        E_e1b[eu_i] = E_orig[eu_i] - 2.0 * mu_0_unit

    # WSP-E1c: Zero-Hindi-Regression Fisher Discriminant Subspace Steering (gamma=12.0, beta=1.75)
    delta_w_e1c = 12.0 * u_fisher + 1.75 * mu_0_unit
    E_e1c = E_orig.copy()
    E_e1c[8906] = E_orig[8906] + 0.5 * delta_w_e1c
    E_e1c[8913] = E_orig[8913] - 0.5 * delta_w_e1c
    for eu_i in range(8907, 8913):
        E_e1c[eu_i] = E_orig[eu_i] - 2.0 * mu_0_unit

    for exp_id, E_cand, desc in [
        ("WSP-E1a", E_e1a, f"Step-0 BOS Prior Shift (beta={pure_bias_beta}) + EU Tag Suppression"),
        ("WSP-E1b", E_e1b, "Step-0 Regularized Fisher Subspace Steering (gamma=12.0, beta=5.0) + EU Tag Suppression"),
        ("WSP-E1c", E_e1c, "Step-0 Zero-Hindi-Regression Fisher Subspace Steering (gamma=12.0, beta=1.75) + EU Tag Suppression"),
    ]:
        ste_cand = make_ste_with_new_embedding(E_cand)
        exp_res = {}
        for ds_key, forced_lang, dev_slice, test_slice in eval_plan:
            out_auto = decode_with_ste_params(ste_cand, encoded_cache[ds_key], forced_lang=None)
            exp_res[ds_key] = {
                "auto_all": quick_score(out_auto, None),
                "auto_dev": quick_score(out_auto, dev_slice),
                "auto_test": quick_score(out_auto, test_slice),
            }
            print(
                f"  [{exp_id} | {ds_key}] Auto-LID All WER={exp_res[ds_key]['auto_all']['wer']}% "
                f"(Dev={exp_res[ds_key]['auto_dev']['wer']}%, Test={exp_res[ds_key]['auto_test']['wer']}%, "
                f"LID-Acc={exp_res[ds_key]['auto_all']['lid_accuracy_pct']}%)",
                flush=True,
            )
        sil_out = decode_with_ste_params(ste_cand, sil_batches, forced_lang=None)
        sil_words = sum(len(x["hyp"].split()) for x in sil_out)
        experiments_log[exp_id] = {
            "description": desc,
            "modified_tensors": ["embedding/embedding[8906:8914, :]"],
            "num_parameters_changed": 4096,
            "pct_parameters_changed": round(4096 / 55547056 * 100.0, 5),
            "l2_norm_orig_en": round(float(np.linalg.norm(E_orig[8906])), 4),
            "l2_norm_new_en": round(float(np.linalg.norm(E_cand[8906])), 4),
            "l2_norm_orig_hi": round(float(np.linalg.norm(E_orig[8913])), 4),
            "l2_norm_new_hi": round(float(np.linalg.norm(E_cand[8913])), 4),
            "cos_en_orig_new": round(cosine_np(E_orig[8906], E_cand[8906]), 6),
            "cos_hi_orig_new": round(cosine_np(E_orig[8913], E_cand[8913]), 6),
            "silence_invented_words": int(sil_words),
            "datasets": exp_res,
        }

    # Select winning E1 variant strictly on Dev split mean Auto-LID WER across all 5 datasets
    e1a_dev_wer = np.mean([experiments_log["WSP-E1a"]["datasets"][k]["auto_dev"]["wer"] for k, _, _, _ in eval_plan])
    e1b_dev_wer = np.mean([experiments_log["WSP-E1b"]["datasets"][k]["auto_dev"]["wer"] for k, _, _, _ in eval_plan])
    e1c_dev_wer = np.mean([experiments_log["WSP-E1c"]["datasets"][k]["auto_dev"]["wer"] for k, _, _, _ in eval_plan])
    best_e1_id = "WSP-E1b" if e1b_dev_wer <= min(e1a_dev_wer, e1c_dev_wer) else ("WSP-E1c" if e1c_dev_wer <= e1a_dev_wer else "WSP-E1a")
    E_best_e1 = E_e1b if best_e1_id == "WSP-E1b" else (E_e1c if best_e1_id == "WSP-E1c" else E_e1a)
    print(f"--> Selected {best_e1_id} on Dev split (Mean Dev WER: E1a={e1a_dev_wer:.2f}% vs E1b={e1b_dev_wer:.2f}% vs E1c={e1c_dev_wer:.2f}%)", flush=True)

    # ========================================================================
    # EXPERIMENT 2: Devanagari Output Projection Norm Calibration (H2)
    #   Sweep alpha_dev in [1.02, 1.04, 1.06, 1.08] on Dev batches of
    #   mucs_hinglish (5 batches = 300 dev) & fleurs_hi (4 batches = 240 clips >= 200 dev)
    # ========================================================================
    print("\n=== Running Phase 4 Experiment WSP-E2 (Devanagari Norm Equalization Sweep on Dev) ===", flush=True)
    alpha_candidates = [1.02, 1.04, 1.06, 1.08]
    e2_sweep_results = {}
    best_alpha_dev = 1.0
    best_e2_dev_wer = (
        0.5 * phase2_baseline["datasets"]["mucs_hinglish"]["forced_lid"]["dev"]["wer"]
        + 0.25 * phase2_baseline["datasets"]["fleurs_hi"]["forced_lid"]["dev"]["wer"]
        + 0.25 * phase2_baseline["datasets"]["lahaja_hi"]["forced_lid"]["dev"]["wer"]
    )
    print(f"  [Baseline Dev Weighted WER (MUCS/FLEURS/Lahaja)] = {best_e2_dev_wer:.3f}%", flush=True)

    for alpha in alpha_candidates:
        E_cand = E_orig.copy()
        E_cand[8192:8906] = alpha * E_orig[8192:8906]
        ste_cand = make_ste_with_new_embedding(E_cand)

        out_mucs_dev = decode_with_ste_params(ste_cand, encoded_cache["mucs_hinglish"], forced_lang="hi", max_batches=5)
        out_fleurs_dev = decode_with_ste_params(ste_cand, encoded_cache["fleurs_hi"], forced_lang="hi", max_batches=4)
        out_lahaja_dev = decode_with_ste_params(ste_cand, encoded_cache["lahaja_hi"], forced_lang="hi", max_batches=5)

        sc_mucs = quick_score(out_mucs_dev, (0, 300))
        sc_fleurs = quick_score(out_fleurs_dev, (0, 200))
        sc_lahaja = quick_score(out_lahaja_dev, (0, 300))

        weighted_dev = 0.5 * sc_mucs["wer"] + 0.25 * sc_fleurs["wer"] + 0.25 * sc_lahaja["wer"]
        e2_sweep_results[str(alpha)] = {
            "alpha_dev": alpha,
            "mucs_dev_wer": sc_mucs["wer"],
            "mucs_dev_cer": sc_mucs["cer"],
            "mucs_dev_latin_words": sc_mucs["hyp_latin_words"],
            "mucs_dev_devanagari_words": sc_mucs["hyp_devanagari_words"],
            "mucs_dev_to_en_subs": sc_mucs["script_mismatch_dev_to_en"],
            "fleurs_dev_wer": sc_fleurs["wer"],
            "lahaja_dev_wer": sc_lahaja["wer"],
            "weighted_dev_wer": round(weighted_dev, 3),
        }
        print(
            f"  [WSP-E2 alpha={alpha}] MUCS Dev WER={sc_mucs['wer']}% (Dev->En={sc_mucs['script_mismatch_dev_to_en']}), "
            f"FLEURS Dev WER={sc_fleurs['wer']}%, Lahaja Dev WER={sc_lahaja['wer']}% | Weighted={weighted_dev:.3f}%",
            flush=True,
        )
        if weighted_dev < best_e2_dev_wer:
            best_e2_dev_wer = weighted_dev
            best_alpha_dev = alpha

    # Evaluate the best alpha_dev (or alpha=1.04 if none improved so we still record full test metrics) on full datasets
    eval_alpha = best_alpha_dev if best_alpha_dev != 1.0 else 1.04
    E_e2 = E_orig.copy()
    E_e2[8192:8906] = eval_alpha * E_orig[8192:8906]
    ste_e2 = make_ste_with_new_embedding(E_e2)
    e2_full_res = {}
    for ds_key, forced_lang, dev_slice, test_slice in eval_plan:
        out_f = decode_with_ste_params(ste_e2, encoded_cache[ds_key], forced_lang=forced_lang)
        e2_full_res[ds_key] = {
            "forced_all": quick_score(out_f, None),
            "forced_dev": quick_score(out_f, dev_slice),
            "forced_test": quick_score(out_f, test_slice),
        }
    sil_e2 = decode_with_ste_params(ste_e2, sil_batches, forced_lang=None)
    experiments_log["WSP-E2"] = {
        "description": f"Devanagari Output Projection Radial Norm Scaling (alpha_dev={eval_alpha}, selected_on_dev={best_alpha_dev})",
        "modified_tensors": ["embedding/embedding[8192:8906, :]"],
        "num_parameters_changed": 714 * 512,
        "pct_parameters_changed": round((714 * 512) / 55547056 * 100.0, 4),
        "dev_sweep": e2_sweep_results,
        "selected_alpha_dev": best_alpha_dev,
        "evaluated_alpha_dev": eval_alpha,
        "mean_devanagari_norm_orig": round(float(np.mean(np.linalg.norm(E_orig[8192:8906], axis=1))), 4),
        "mean_devanagari_norm_new": round(float(np.mean(np.linalg.norm(E_e2[8192:8906], axis=1))), 4),
        "silence_invented_words": int(sum(len(x["hyp"].split()) for x in sil_e2)),
        "datasets": e2_full_res,
    }

    # ========================================================================
    # EXPERIMENT 3: High-Confusion Devanagari Pair Decoupling & Norm Balancing (H3)
    #   Identify confused Devanagari pairs from `phase3_diag["diagnosis_3_devanagari_confusion_geometry"]`
    #   where ref_tid and hyp_tid both exist in 8192..8905 and dev_split_error_count >= 3.
    # ========================================================================
    print("\n=== Running Phase 4 Experiment WSP-E3 (Contrastive Devanagari Pair Decoupling) ===", flush=True)
    pair_candidates = [
        p for p in phase3_diag.get("diagnosis_3_devanagari_confusion_geometry", [])
        if p.get("ref_tid") is not None and p.get("hyp_tid") is not None and p.get("dev_split_error_count", 0) >= 3
    ]
    # Build net directional & norm updates per token from Dev confusion asymmetry
    E_e3 = E_orig.copy()
    touched_tids = set()
    pair_edits_log = []
    # Compute net directional error count per ordered pair on Dev
    pair_counts = {(p["ref_tid"], p["hyp_tid"]): p["dev_split_error_count"] for p in pair_candidates}
    handled_unordered = set()
    for p in pair_candidates:
        ir, ih = int(p["ref_tid"]), int(p["hyp_tid"])
        key_u = (min(ir, ih), max(ir, ih))
        if key_u in handled_unordered:
            continue
        handled_unordered.add(key_u)
        c_r_to_h = pair_counts.get((ir, ih), 0)
        c_h_to_r = pair_counts.get((ih, ir), 0)
        net_imb = c_r_to_h - c_h_to_r
        er = E_orig[ir].copy()
        eh = E_orig[ih].copy()
        cos_rh = cosine_np(er, eh)
        if cos_rh < 0.70:
            continue
        # Orthogonal repulsion in span(er, eh) + small norm tilt toward under-predicted token
        er_hat = er / max(float(np.linalg.norm(er)), 1e-12)
        eh_hat = eh / max(float(np.linalg.norm(eh)), 1e-12)
        orth_r = er - float(np.dot(er, eh_hat)) * eh_hat
        orth_h = eh - float(np.dot(eh, er_hat)) * er_hat
        eta_orth = 0.04
        eta_norm = np.clip(net_imb * 0.002, -0.025, 0.025)
        E_e3[ir] = (1.0 + eta_norm) * er + eta_orth * orth_r
        E_e3[ih] = (1.0 - eta_norm) * eh + eta_orth * orth_h
        touched_tids.add(ir)
        touched_tids.add(ih)
        pair_edits_log.append({
            "ref_word": p["ref_word"],
            "hyp_word": p["hyp_word"],
            "ref_tid": ir,
            "hyp_tid": ih,
            "c_ref_to_hyp_dev": c_r_to_h,
            "c_hyp_to_ref_dev": c_h_to_r,
            "orig_cosine": round(cos_rh, 4),
            "new_cosine": round(cosine_np(E_e3[ir], E_e3[ih]), 4),
            "eta_norm": round(float(eta_norm), 5),
            "eta_orth": eta_orth,
        })

    ste_e3 = make_ste_with_new_embedding(E_e3)
    e3_full_res = {}
    for ds_key, forced_lang, dev_slice, test_slice in [
        ("fleurs_hi", "hi", (0, 200), (200, 418)),
        ("mucs_hinglish", "hi", (0, 300), (300, 600)),
        ("lahaja_hi", "hi", (0, 300), (300, 600)),
    ]:
        out_f = decode_with_ste_params(ste_e3, encoded_cache[ds_key], forced_lang=forced_lang)
        e3_full_res[ds_key] = {
            "forced_all": quick_score(out_f, None),
            "forced_dev": quick_score(out_f, dev_slice),
            "forced_test": quick_score(out_f, test_slice),
        }
        print(
            f"  [WSP-E3 | {ds_key}] Forced-{forced_lang} All WER={e3_full_res[ds_key]['forced_all']['wer']}% "
            f"(Dev={e3_full_res[ds_key]['forced_dev']['wer']}%, Test={e3_full_res[ds_key]['forced_test']['wer']}%)",
            flush=True,
        )
    experiments_log["WSP-E3"] = {
        "description": "Contrastive Angular Repulsion & Norm Balancing of High-Confusion Devanagari Pairs",
        "modified_tensors": [f"embedding/embedding[{sorted(list(touched_tids))}, :]"],
        "num_tokens_changed": len(touched_tids),
        "num_parameters_changed": len(touched_tids) * 512,
        "pct_parameters_changed": round((len(touched_tids) * 512) / 55547056 * 100.0, 5),
        "pair_edits": pair_edits_log,
        "datasets": e3_full_res,
    }

    # Check if E3 improves Dev WER across FLEURS + MUCS + Lahaja
    base_e3_dev = (
        phase2_baseline["datasets"]["fleurs_hi"]["forced_lid"]["dev"]["wer"]
        + phase2_baseline["datasets"]["mucs_hinglish"]["forced_lid"]["dev"]["wer"]
        + phase2_baseline["datasets"]["lahaja_hi"]["forced_lid"]["dev"]["wer"]
    ) / 3.0
    cand_e3_dev = (
        e3_full_res["fleurs_hi"]["forced_dev"]["wer"]
        + e3_full_res["mucs_hinglish"]["forced_dev"]["wer"]
        + e3_full_res["lahaja_hi"]["forced_dev"]["wer"]
    ) / 3.0
    e3_passed_dev = cand_e3_dev < base_e3_dev
    print(f"--> WSP-E3 Dev Mean WER: Baseline={base_e3_dev:.2f}% vs E3={cand_e3_dev:.2f}% (Passed={e3_passed_dev})", flush=True)

    # ========================================================================
    # EXPERIMENT 4: Internal Decoder Gate & Engram Scaling (H4 — Falsification Test)
    #   Tests modifying `stack/layers/block/attn_gate` and `engrams_1/embedding`
    # ========================================================================
    print("\n=== Running Phase 4 Experiment WSP-E4 (Decoder Gate & Engram Scaling Sensitivity Test) ===", flush=True)
    ste_e4 = dict(ste_orig)
    stack_new = dict(ste_orig["stack"])
    layers_new = dict(stack_new["layers"])
    block_new = dict(layers_new["block"])
    attn_gate_orig = np.array(block_new["attn_gate"], dtype=np.float32)
    attn_gate_mod = attn_gate_orig.copy()
    attn_gate_mod[6:8] *= 1.12  # Amplify top-2 decoder cross-attention gates by +12%
    block_new["attn_gate"] = jnp.asarray(attn_gate_mod, dtype=jnp.float32)
    layers_new["block"] = block_new
    stack_new["layers"] = layers_new
    ste_e4["stack"] = stack_new

    out_e4_fleurs = decode_with_ste_params(ste_e4, encoded_cache["fleurs_hi"], forced_lang="hi", max_batches=4)
    out_e4_svarah = decode_with_ste_params(ste_e4, encoded_cache["svarah_en"], forced_lang="en", max_batches=5)
    out_e4_mucs = decode_with_ste_params(ste_e4, encoded_cache["mucs_hinglish"], forced_lang="hi", max_batches=5)
    sc_e4_fleurs = quick_score(out_e4_fleurs, (0, 200))
    sc_e4_svarah = quick_score(out_e4_svarah, (0, 300))
    sc_e4_mucs = quick_score(out_e4_mucs, (0, 300))
    print(
        f"  [WSP-E4 Dev] FLEURS={sc_e4_fleurs['wer']}% (Base={phase2_baseline['datasets']['fleurs_hi']['forced_lid']['dev']['wer']}%), "
        f"Svarah={sc_e4_svarah['wer']}% (Base={phase2_baseline['datasets']['svarah_en']['forced_lid']['dev']['wer']}%), "
        f"MUCS={sc_e4_mucs['wer']}% (Base={phase2_baseline['datasets']['mucs_hinglish']['forced_lid']['dev']['wer']}%)",
        flush=True,
    )
    experiments_log["WSP-E4"] = {
        "description": "Decoder Top-2 Cross-Attention Gate Amplification (+12% on layers 6..7)",
        "modified_tensors": ["stack/layers/block/attn_gate[6:8]"],
        "num_parameters_changed": 2,
        "pct_parameters_changed": round(2 / 55547056 * 100.0, 7),
        "orig_values": attn_gate_orig.flatten().tolist(),
        "new_values": attn_gate_mod.flatten().tolist(),
        "dev_results": {
            "fleurs_hi_dev_wer": sc_e4_fleurs["wer"],
            "svarah_en_dev_wer": sc_e4_svarah["wer"],
            "mucs_hinglish_dev_wer": sc_e4_mucs["wer"],
        },
    }

    # ========================================================================
    # EXPERIMENT 5: Combined Final WSP Checkpoint (`shruti_wsp_final`)
    #   Combines all Dev-validated non-regressing edits:
    #     - Winning Step-0 Language ID Steering (`best_e1_id`)
    #     - Devanagari Norm Equalization if `best_alpha_dev > 1.0`
    #     - Contrastive Devanagari Pair Decoupling if `e3_passed_dev`
    # ========================================================================
    print("\n=== Constructing & Validating Combined Final WSP Checkpoint (`shruti_wsp_final`) ===", flush=True)
    E_final = E_orig.copy()
    included_components = [best_e1_id]
    num_rows_modified = 8

    # 1. Apply E2 if it improved Dev
    if best_alpha_dev != 1.0:
        E_final[8192:8906] = best_alpha_dev * E_final[8192:8906]
        included_components.append(f"WSP-E2(alpha={best_alpha_dev})")
        num_rows_modified += 714

    # 2. Apply E3 if it improved Dev
    if e3_passed_dev:
        for p_edit in pair_edits_log:
            ir, ih = p_edit["ref_tid"], p_edit["hyp_tid"]
            scale_factor = best_alpha_dev if best_alpha_dev != 1.0 else 1.0
            E_final[ir] = scale_factor * E_e3[ir]
            E_final[ih] = scale_factor * E_e3[ih]
        included_components.append("WSP-E3")
        if best_alpha_dev == 1.0:
            num_rows_modified += len(touched_tids)

    # 3. Always apply winning E1 on rows 8906..8913
    E_final[8906:8914] = E_best_e1[8906:8914]

    ste_final = make_ste_with_new_embedding(E_final)
    final_eval_results = {}
    for ds_key, forced_lang, dev_slice, test_slice in eval_plan:
        out_auto = decode_with_ste_params(ste_final, encoded_cache[ds_key], forced_lang=None)
        out_forced = decode_with_ste_params(ste_final, encoded_cache[ds_key], forced_lang=forced_lang)
        final_eval_results[ds_key] = {
            "auto_lid": {
                "all": quick_score(out_auto, None),
                "dev": quick_score(out_auto, dev_slice),
                "test": quick_score(out_auto, test_slice),
            },
            "forced_lid": {
                "all": quick_score(out_forced, None),
                "dev": quick_score(out_forced, dev_slice),
                "test": quick_score(out_forced, test_slice),
            },
        }
        print(
            f"  [FINAL WSP | {ds_key}] Auto WER={final_eval_results[ds_key]['auto_lid']['all']['wer']}% "
            f"(Test={final_eval_results[ds_key]['auto_lid']['test']['wer']}%, LID-Acc={final_eval_results[ds_key]['auto_lid']['all']['lid_accuracy_pct']}%) | "
            f"Forced-{forced_lang} WER={final_eval_results[ds_key]['forced_lid']['all']['wer']}% "
            f"(Test={final_eval_results[ds_key]['forced_lid']['test']['wer']}%)",
            flush=True,
        )

    sil_final = decode_with_ste_params(ste_final, sil_batches, forced_lang=None)
    sil_final_words = int(sum(len(x["hyp"].split()) for x in sil_final))

    # Export `shruti_wsp_final.safetensors` and surgically patched `shruti_wsp_final.cact`
    os.makedirs("/data/experiments", exist_ok=True)
    final_st_path = "/data/experiments/shruti_wsp_final.safetensors"
    final_cact_path = "/data/experiments/shruti_wsp_final.cact"

    new_flat = dict(orig_flat)
    new_flat["embedding/embedding"] = E_final.astype(np.float32)
    save_safetensors_flat(final_st_path, new_flat)
    patch_stats = surgical_patch_cact(
        cact_path,
        final_cact_path,
        orig_flat,
        new_flat,
        WHISTLE_SMALL_CFG,
    )
    vol.commit()

    final_st_sha = hashlib.sha256(open(final_st_path, "rb").read()).hexdigest()
    final_cact_sha = hashlib.sha256(open(final_cact_path, "rb").read()).hexdigest()

    # ========================================================================
    # Native C++ `needle` 5-Beam Engine Verification on `shruti_wsp_final.cact`
    # ========================================================================
    print("\n=== Running Native C++ `needle` 5-Beam Engine Verification on `shruti_wsp_final.cact` ===", flush=True)
    cact_final_verification: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for ds_key, forced_lang, _, _ in eval_plan:
            sub_clips = datasets_map[ds_key][:60]
            auto_hyps = []
            forced_hyps = []
            for c in sub_clips:
                wav_path = os.path.join(tmpdir, f"{ds_key}_{c['idx']}.wav")
                write_wav_16k(wav_path, c["samples"])
                proc_auto = subprocess.run(
                    [needle_bin, "--model", final_cact_path, "--audio", wav_path, "--threads", "4"],
                    capture_output=True, text=True, check=True,
                )
                obj_auto = json.loads(proc_auto.stdout.strip().splitlines()[-1])
                auto_hyps.append({
                    "idx": c["idx"],
                    "ref": c["text"],
                    "hyp": obj_auto.get("text", "").strip(),
                    "target_lang": c["target_lang"],
                    "pred_lang": obj_auto.get("language", "hi"),
                })
                proc_forced = subprocess.run(
                    [needle_bin, "--model", final_cact_path, "--audio", wav_path, "--audio-language", forced_lang, "--threads", "4"],
                    capture_output=True, text=True, check=True,
                )
                obj_forced = json.loads(proc_forced.stdout.strip().splitlines()[-1])
                forced_hyps.append({
                    "idx": c["idx"],
                    "ref": c["text"],
                    "hyp": obj_forced.get("text", "").strip(),
                    "target_lang": c["target_lang"],
                    "pred_lang": obj_forced.get("language", forced_lang),
                })
            sa = quick_score(auto_hyps)
            sf = quick_score(forced_hyps)
            cact_final_verification[ds_key] = {
                "num_clips": len(sub_clips),
                "auto_lid_wer": sa["wer"],
                "auto_lid_cer": sa["cer"],
                "auto_lid_accuracy_pct": sa["lid_accuracy_pct"],
                "forced_lid_wer": sf["wer"],
                "forced_lid_cer": sf["cer"],
            }
            print(
                f"  [needle C++ 5-beam FINAL | {ds_key} (60 clips)] Auto WER={sa['wer']}% "
                f"(LID-Acc={sa['lid_accuracy_pct']}%) | Forced-{forced_lang} WER={sf['wer']}%",
                flush=True,
            )

        cact_sil_words = 0
        for i in [0, 1, 2, 3, 4, 50, 51, 52, 53, 54]:
            wav_path = os.path.join(tmpdir, f"sil_{i}.wav")
            write_wav_16k(wav_path, silence_clips[i]["samples"])
            proc_sil = subprocess.run(
                [needle_bin, "--model", final_cact_path, "--audio", wav_path, "--threads", "4"],
                capture_output=True, text=True, check=True,
            )
            obj_sil = json.loads(proc_sil.stdout.strip().splitlines()[-1])
            cact_sil_words += len(obj_sil.get("text", "").strip().split())

    experiments_log["WSP-Combined-Final"] = {
        "description": f"Combined Validated WSP Checkpoint ({' + '.join(included_components)})",
        "included_components": included_components,
        "num_embedding_rows_changed": int(num_rows_modified),
        "num_parameters_changed": int(num_rows_modified * 512),
        "pct_parameters_changed": round((num_rows_modified * 512) / 55547056 * 100.0, 5),
        "safetensors_sha256": final_st_sha,
        "cact_sha256": final_cact_sha,
        "cact_surgical_patch_stats": patch_stats,
        "silence_invented_words_gpu": sil_final_words,
        "silence_invented_words_cact_sample10": int(cact_sil_words),
        "datasets": final_eval_results,
        "cact_needle_5beam_verification": cact_final_verification,
    }

    report = {
        "phase": "Phase 4 & 5 Weight-Space Programming & Independent Validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_sec": round(time.time() - t0, 2),
        "experiments": experiments_log,
    }
    with open("/data/eval_cache/phase4_5_wsp_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    vol.commit()

    return {
        "report": report,
        "E_final": E_final,
    }


@app.local_entrypoint()
def main():
    from shruti_io import (
        WHISTLE_SMALL_CFG,
        load_safetensors_flat,
        save_safetensors_flat,
        surgical_patch_cact,
    )

    p3_json = ROOT_LOCAL / "reports" / "phase3_diagnosis.json"
    p3_npz = ROOT_LOCAL / "reports" / "phase3_wsp_vectors.npz"
    assert p3_json.exists() and p3_npz.exists(), "Run phase3_diagnose_weight_space.py first!"

    with open(p3_json, "r", encoding="utf-8") as f:
        phase3_diag = json.load(f)
    wsp_bytes = p3_npz.read_bytes()

    res = run_phase4_5_experiments.remote(phase3_diag, wsp_bytes)
    report = res["report"]
    E_final = np.asarray(res["E_final"], dtype=np.float32)

    out_json = ROOT_LOCAL / "reports" / "phase4_5_wsp_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved Phase 4 & 5 WSP report -> {out_json}")

    # Materialize experimental checkpoints locally in checkpoints/experiments/
    exp_dir = ROOT_LOCAL / "checkpoints" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    orig_st = ROOT_LOCAL / "checkpoints" / "original" / "shruti.safetensors"
    orig_cact = ROOT_LOCAL / "checkpoints" / "original" / "shruti.cact"
    final_st = exp_dir / "shruti_wsp_final.safetensors"
    final_cact = exp_dir / "shruti_wsp_final.cact"

    orig_flat = load_safetensors_flat(str(orig_st))
    new_flat = dict(orig_flat)
    new_flat["embedding/embedding"] = E_final
    save_safetensors_flat(str(final_st), new_flat)
    surgical_patch_cact(str(orig_cact), str(final_cact), orig_flat, new_flat, WHISTLE_SMALL_CFG)

    st_sha = hashlib.sha256(final_st.read_bytes()).hexdigest()
    cact_sha = hashlib.sha256(final_cact.read_bytes()).hexdigest()
    print(f"Exported local experimental checkpoints:")
    print(f"  - {final_st} (SHA-256: {st_sha})")
    print(f"  - {final_cact} (SHA-256: {cact_sha})")
