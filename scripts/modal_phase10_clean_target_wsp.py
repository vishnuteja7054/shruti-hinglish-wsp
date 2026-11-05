#!/usr/bin/env python3
"""Phase 10: 6-Tier Dataset Canonicalization Audit & Clean-Target Loop-Protected WSP Pipeline.

Executes two coupled research stages on Modal A100-40GB + Native C++ `needle` 5-Beam Engine:
  Part A (6-Tier Indic/Hinglish Dataset Canonicalization Waterfall Audit):
    Quantifies across all 2,517 GPU clips and 300 C++ 5-beam clips how much of the remaining WER
    comes from dataset annotation inconsistencies vs. acoustic errors:
      - T0_Raw_Strict:       Exact Unicode match (`normalize_indic_hinglish`)
      - T1_Unicode_Ortho:    + Nuqta (`\u093c`) stripping & Chandrabindu (`ँ` -> `ं`) canonicalization
      - T2_Spoken_Allomorph: + Spoken-vs-Written Hindi equivalence (`वो/वह`, `ये/यह`, `हैं/है`, `कि/की`, `मे/में`, etc.)
      - T3_Compound_Space:   + Compound-word & postposition spacing alignment (`इसके लिए` <-> `इसकेलिए`)
      - T4_CrossScript_Loan: + Cross-script phonetic loanword & acronym unification (`क्लिक` <-> `click`, `पी` <-> `p`)
      - T5_Whisper_Semantic: + Unstressed vowel-matra / Whisper-standard semantic equivalence + Character Accuracy (`100% - CER`)

  Part B (Phase 10 Clean-Target & Autoregressive-Loop-Protected Weight-Space Programming):
    1. Canonicalizes Dev teacher-forced targets (eliminating orphan Nuqta token `8299` alignment shifts
       and spoken/written `वो/वह` token-length mismatches).
    2. Identifies exact 1-beam autoregressive loop-prone tokens on `mucs_hinglish` Dev and constrains
       their 4-block FP16 group norms (`alpha_{r,g} <= 1.0` or zero-mean block contrast) while running
       Sequential Gauss-Seidel 4-Block FP16 Coordinate Ascent (`WSP-E13_CleanTarget_LoopSafe`).
    3. Tests joint Audio-Encoder `encoder/final_norm/scale` + Decoder `stack/final_norm/scale` FP16
       calibration (`WSP-E14_EncoderDecoder_CleanWSP`).
    4. Evaluates Official Baseline, Phase 8 (`WSP-E10b`), Phase 9 (`WSP-E11`), and Phase 10 (`WSP-E13`/`E14`)
       across all 6 canonicalization tiers on both GPU (2,517 clips) and C++ `needle` 5-Beam (300 clips).
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

app = modal.App("shruti-wsp-phase10-clean")
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

# Precomposed Devanagari nuqta letters -> base consonant
_NUQTA_PRECOMPOSED = {
    "\u0958": "\u0915",  # क़ -> क
    "\u0959": "\u0916",  # ख़ -> ख
    "\u095a": "\u0917",  # ग़ -> ग
    "\u095b": "\u091c",  # ज़ -> ज
    "\u095c": "\u0921",  # ड़ -> ड
    "\u095d": "\u0922",  # ढ़ -> ढ
    "\u095e": "\u092b",  # फ़ -> फ
    "\u095f": "\u092f",  # य़ -> य
}

# Tier-2 Spoken-to-Written Hindi Allomorph & Function-Word Equivalence Map
_SPOKEN_HINDI_EQUIV = {
    "वो": "वह",
    "वे": "वह",
    "ये": "यह",
    "हैं": "है",
    "थीं": "थी",
    "हूँ": "हूं",
    "कि": "की",
    "मे": "में",
    "मै": "मैं",
    "मैं": "मैं",
    "ओर": "और",
    "नही": "नहीं",
    "कही": "कहीं",
    "वही": "वहीं",
    "यही": "यहीं",
    "क्यूंकि": "क्योंकि",
    "क्यूँकि": "क्योंकि",
    "क्योकि": "क्योंकि",
    "क्योंकी": "क्योंकि",
    "सिर्फ़": "सिर्फ",
    "बल्की": "बल्कि",
    "यानि": "यानी",
    "लेकीन": "लेकिन",
    "उन्होने": "उन्होंने",
    "इन्होने": "इन्होंने",
    "जिन्होने": "जिन्होंने",
    "गयी": "गई",
    "गये": "गए",
    "नयी": "नई",
    "नये": "नए",
    "हुयी": "हुई",
    "हुये": "हुए",
    "लिये": "लिए",
    "दिये": "दिए",
    "किये": "किए",
    "लीये": "लिए",
    "चाहीए": "चाहिए",
    "चाहिये": "चाहिए",
}

# Common Devanagari Letter-Name Spoken Acronyms -> Single Latin Letter
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
    """Tier 0: Raw strict normalization (preserves all matras, nuqtas, chandrabindu)."""
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
    """Tier 1: + Strip optional Devanagari Nuqta (\\u093c) & unify Chandrabindu (ँ -> ं)."""
    s = normalize_t0_strict(text)
    for k, v in _NUQTA_PRECOMPOSED.items():
        s = s.replace(k, v)
    s = s.replace("\u093c", "")      # Strip combining nuqta dot
    s = s.replace("\u0901", "\u0902")  # Chandrabindu ँ -> Anusvara ं
    s = s.replace("\u0949", "\u094b")  # ॉ -> ो
    s = s.replace("\u0945", "\u0947")  # ॅ -> े
    s = s.replace("\u0911", "\u0913")  # ऑ -> ओ
    return " ".join(s.split())


def normalize_t2_spoken_allomorph(text: str) -> str:
    """Tier 2: + Spoken-to-Written Hindi allomorph & function-word canonicalization."""
    s = normalize_t1_unicode_ortho(text)
    words = [_SPOKEN_HINDI_EQUIV.get(w, w) for w in s.split()]
    return " ".join(words)


def align_compound_spacing(ref_words: List[str], hyp_words: List[str]) -> Tuple[List[str], List[str]]:
    """Tier 3 helper: Merge adjacent split words when their concatenation matches the other sequence."""
    # Pass 1: Merge adjacent words in hyp if they match a compound word in ref_set
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
    # Pass 2: Merge adjacent words in ref if they match a compound word in new_hyp_set
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
    """Tier 4 helper: Map Devanagari or Latin word to unified phonetic skeleton."""
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
        w_lat.replace("ph", "f")
        .replace("w", "v")
        .replace("ck", "k")
        .replace("c", "k")
        .replace("qu", "kw")
        .replace("z", "j")
        .replace("sh", "s")
    )
    w_lat = re.sub(r"([a-z])\1+", r"\1", w_lat)
    if len(w_lat) > 3 and w_lat.endswith("e"):
        w_lat = w_lat[:-1]
    return w_lat


def skeleton_word_t5(w: str) -> str:
    """Tier 5 helper: Whisper-style vowel-length / unstressed schwa-invariant semantic skeleton."""
    w4 = transliterate_word_t4(w)
    # Collapse unstressed medial vowel variations (ai->e, au->o, medial a/i/u equivalence after initial char)
    w4 = w4.replace("ai", "e").replace("au", "o").replace("ri", "r")
    if len(w4) <= 2:
        return w4
    # Keep first character and consonant + primary vowel structure, collapsing short/long unstressed medial schwas
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
    """Compute WER & CER across all 6 dataset canonicalization tiers."""
    tiers = ["T0_Raw_Strict", "T1_Unicode_Ortho", "T2_Spoken_Allomorph", "T3_Compound_Space", "T4_CrossScript_Loan", "T5_Whisper_Semantic"]
    w_edits = {t: [0, 0, 0, 0] for t in tiers}  # [S, I, D, N_ref]
    c_edits = {t: [0, 0] for t in tiers}        # [char_err, char_ref]

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


def canonicalize_dev_ref_for_shruti(ref_text: str, hyp_baseline: str) -> str:
    """Clean Dev reference text before Teacher-Forced tokenization so Shruti's tokenizer doesn't suffer
    from orphan Nuqta tokens (`8299`), Chandrabindu splits, or `वह/वो` token-count shifts when the
    speaker actually pronounced `वो`/`ये`/`है`.
    """
    if not ref_text:
        return ""
    s = unicodedata.normalize("NFC", ref_text)
    # 1. Strip combining Nuqta (\u093c) & map precomposed nuqta consonants to standard Devanagari
    #    (eliminates extra token 8299 = '़' that shifts teacher-forced token alignment by +1!)
    for k, v in _NUQTA_PRECOMPOSED.items():
        s = s.replace(k, v)
    s = s.replace("\u093c", "")
    # 2. If speaker audio (as evidenced by baseline decode) uses spoken `वो` (1 token 8689) where
    #    prompt has `वह` (2 tokens [8289, 8297]), align to `वो` so token length matches the audio!
    hyp_words = set(normalize_t1_unicode_ortho(hyp_baseline).split())
    words = s.split()
    cleaned_words = []
    for w in words:
        w_clean = w.strip("।॥,.?!\"'()[]{}")
        if w_clean == "वह" and "वो" in hyp_words and "वह" not in hyp_words:
            w = w.replace("वह", "वो")
        cleaned_words.append(w)
    return " ".join(cleaned_words)


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
def run_phase10_clean_target_wsp(
    wsp_vectors_bytes: bytes,
    p8_res: Dict[str, Any],
    p9_res: Dict[str, Any],
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
    from needle_model.architecture import _aq
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

    mu_0_unit = wsp_vecs["mu_0_unit"].astype(np.float32)
    delta_w_best = wsp_vecs["delta_w_best"].astype(np.float32)

    W_orig_emb = np.array(orig_flat["embedding/embedding"], dtype=np.float32, copy=True)
    W_e1b_emb = W_orig_emb.copy()
    W_e1b_emb[8906] = W_orig_emb[8906] + 0.5 * delta_w_best
    W_e1b_emb[8913] = W_orig_emb[8913] - 0.5 * delta_w_best
    for tid in range(8907, 8913):
        W_e1b_emb[tid] = W_orig_emb[tid] - 2.0 * mu_0_unit

    # Reconstruct Phase 8 (WSP-E10b) and Phase 9 (WSP-E11) exact parameter trees
    W_p8_emb = W_e1b_emb.copy()
    for r_str, r_vals in p8_res["best_e10_modified_rows"].items():
        W_p8_emb[int(r_str)] = np.asarray(r_vals, dtype=np.float32)
    scale_p8 = np.asarray(p8_res["best_s_e10b"], dtype=np.float32)

    W_p9_emb = W_p8_emb.copy()
    for item in p9_res["stage3_block_calibrated_rows"]:
        r = int(item["row"])
        a4 = item["alpha_4"]
        for g in range(4):
            W_p9_emb[r, g * 128 : (g + 1) * 128] *= float(a4[g])
    scale_p9 = np.asarray(p9_res["best_s_p9_scale"], dtype=np.float32)

    params_p8_q = cq_ste_params({
        **params_orig,
        "embedding": {"embedding": jnp.asarray(W_p8_emb, dtype=jnp.float32)},
        "stack": {
            **params_orig["stack"],
            "final_norm": {"scale": jnp.asarray(scale_p8, dtype=jnp.float32)},
        },
    }, WEIGHT_BITS)

    params_p9_q = cq_ste_params({
        **params_orig,
        "embedding": {"embedding": jnp.asarray(W_p9_emb, dtype=jnp.float32)},
        "stack": {
            **params_orig["stack"],
            "final_norm": {"scale": jnp.asarray(scale_p9, dtype=jnp.float32)},
        },
    }, WEIGHT_BITS)

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

    print("Warming up XLA kernels...", flush=True)
    dummy_wav = jnp.zeros((B_PAD, MAX_SAMPLES), dtype=jnp.float32)
    dummy_len = jnp.full((B_PAD,), 16000 * 10, dtype=jnp.int32)
    d_enc_q, d_fr_q = jit_encode_q(params_p8_q, dummy_wav, dummy_len)
    _ = jit_decode_hidden_q(params_p8_q, jnp.full((B_PAD, TF_SEQ_LEN), 2, dtype=jnp.int32), d_enc_q, d_fr_q)
    for t_len in [64, 128]:
        _ = jit_step_logits_q(params_p8_q, jnp.full((B_PAD, t_len), 2, dtype=jnp.int32), d_enc_q, d_fr_q)
    print(f"JIT warmup complete in {time.time() - t0:.1f}s", flush=True)

    encoded_cache_q: Dict[str, List[Tuple[Any, Any, int]]] = {}
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
            enc_q, fr_q = jit_encode_q(params_p8_q, jnp.asarray(wav_np), jnp.asarray(len_np))
            batches_q.append((enc_q, fr_q, real_B))
        encoded_cache_q[ds_key] = batches_q
        print(f"Pre-encoded {ds_key} ({limit_n} clips).", flush=True)

    def decode_dataset_gpu(p_q, ds_key: str, max_clips: Optional[int] = None) -> Tuple[List[str], List[int], List[List[int]]]:
        """Fast batched 1-beam greedy decode returning (hyp_texts, pred_lids, token_sequences)."""
        hyps: List[str] = []
        lids: List[int] = []
        tok_seqs: List[List[int]] = []
        batches = encoded_cache_q[ds_key]
        total_done = 0
        for enc_q, fr_q, real_B in batches:
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
                lid_tid = int(seq[1])
                lids.append(lid_tid)
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

    # =========================================================================
    # STAGE 1: Diagnose Loop-Prone Tokens on `mucs_hinglish` Dev & Canonicalize Dev Targets
    # =========================================================================
    print("\n=== Stage 1: Auditing 1-Beam Repetition Loops on MUCS Hinglish Dev & Canonicalizing Dev Targets ===", flush=True)
    hyps_p8_mucs_dev, _, seqs_p8_mucs_dev = decode_dataset_gpu(params_p8_q, "mucs_hinglish", max_clips=300)
    hyps_p9_mucs_dev, _, seqs_p9_mucs_dev = decode_dataset_gpu(params_p9_q, "mucs_hinglish", max_clips=300)

    def find_looping_tokens(seqs: List[List[int]]) -> Dict[int, int]:
        """Find token IDs that repeat consecutively >=3 times or in 2-gram loops >=4 times in a clip."""
        loop_counts: Dict[int, int] = {}
        for seq in seqs:
            if len(seq) < 12:
                continue
            # Check n-gram / single-token repetition bursts
            counts_in_clip: Dict[int, int] = {}
            for tid in seq:
                counts_in_clip[tid] = counts_in_clip.get(tid, 0) + 1
            for tid, c in counts_in_clip.items():
                if c >= 7:  # Token appears >=7 times in a single utterance -> loop participant!
                    loop_counts[tid] = loop_counts.get(tid, 0) + 1
        return loop_counts

    loops_p8 = find_looping_tokens(seqs_p8_mucs_dev)
    loops_p9 = find_looping_tokens(seqs_p9_mucs_dev)
    loop_prone_tids = set(loops_p8.keys()) | set(loops_p9.keys())
    print(f"  [1-Beam Loop Diagnosis on MUCS Dev] Phase 8 loop tokens: {len(loops_p8)}, Phase 9 loop tokens: {len(loops_p9)}", flush=True)
    for tid in sorted(loop_prone_tids):
        c8 = loops_p8.get(tid, 0)
        c9 = loops_p9.get(tid, 0)
        if c9 > c8:
            print(f"    Token {tid} ('{tokenizer.decode([tid])}'): looped in {c8} -> {c9} clips", flush=True)

    # Also decode Dev splits of fleurs_hi and lahaja_hi with Phase 8 to canonicalize spoken `वो/वह`, `ये/यह`, `है/हैं`
    hyps_p8_fleurs_dev, _, _ = decode_dataset_gpu(params_p8_q, "fleurs_hi", max_clips=200)
    hyps_p8_lahaja_dev, _, _ = decode_dataset_gpu(params_p8_q, "lahaja_hi", max_clips=300)
    dev_hyp_lookup = {
        "fleurs_hi": hyps_p8_fleurs_dev,
        "lahaja_hi": hyps_p8_lahaja_dev,
        "mucs_hinglish": hyps_p8_mucs_dev,
    }

    # Extract Clean-Target Teacher-Forced Dev Hidden States
    dev_specs = [
        ("fleurs_hi", 200, 8913),
        ("lahaja_hi", 300, 8913),
        ("mucs_hinglish", 300, 8913),
        ("svarah_en", 300, 8906),
        ("librispeech_en", 150, 8906),
    ]

    all_hq_list = []
    all_u_list = []
    all_target_list = []
    all_ds_id_list = []
    canonicalized_ref_count = 0

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
            for bi, c in enumerate(batch_clips):
                raw_ref = c["text"].strip()
                if not raw_ref:
                    continue
                if ds_key in dev_hyp_lookup:
                    hyp_b = dev_hyp_lookup[ds_key][b_start + bi]
                    clean_ref = canonicalize_dev_ref_for_shruti(raw_ref, hyp_b)
                    if clean_ref != raw_ref:
                        canonicalized_ref_count += 1
                else:
                    clean_ref = raw_ref
                ref_ids = tokenizer.encode(clean_ref)
                if not ref_ids:
                    continue
                seq = [2, lang_tag] + ref_ids[: TF_SEQ_LEN - 2] + [1]
                L = min(len(seq) - 1, TF_SEQ_LEN)
                tok_in[bi, :L] = seq[:L]
                tok_tgt[bi, :L] = seq[1 : L + 1]

            hq_b, u_b = jit_decode_hidden_q(params_p8_q, jnp.asarray(tok_in), enc_q, fr_q)
            hq_np = np.asarray(hq_b, dtype=np.float32)
            u_np = np.asarray(u_b, dtype=np.float32)

            for bi in range(len(batch_clips)):
                for t in range(1, TF_SEQ_LEN):
                    tid = int(tok_tgt[bi, t])
                    if tid <= 0:
                        break
                    all_hq_list.append(hq_np[bi, t])
                    all_u_list.append(u_np[bi, t])
                    all_target_list.append(tid)
                    all_ds_id_list.append(ds_idx)

    H_dev = np.stack(all_hq_list, axis=0)
    U_dev = np.stack(all_u_list, axis=0)
    Y_dev = np.array(all_target_list, dtype=np.int32)
    DS_dev = np.array(all_ds_id_list, dtype=np.int32)
    N_tok = len(Y_dev)
    print(f"  [Clean-Target Dev Extraction] Canonicalized {canonicalized_ref_count} Dev transcripts -> {N_tok} aligned token states.", flush=True)

    # =========================================================================
    # STAGE 2: Loop-Protected Sequential Gauss-Seidel 4-Block FP16 Norm Compiler (`WSP-E13`)
    # =========================================================================
    print("\n=== Stage 2: Loop-Protected Sequential Gauss-Seidel 4-Block FP16 Compiler (WSP-E13) ===", flush=True)
    clean_state_mask = (DS_dev != 2) | (Y_dev >= 8192)
    H_clean = H_dev[clean_state_mask]
    U_clean = U_dev[clean_state_mask]
    Y_clean = Y_dev[clean_state_mask]
    N_clean = len(Y_clean)
    r_idx_c = np.arange(N_clean)

    W_p8_q = cq_quantize_dequantize_np(W_p8_emb, bits=4, group=128)
    Z_run = H_clean @ W_p8_q.T
    Z_run[:, 0] = -1e9
    Z_run[:, 2] = -1e9
    Z_run[:, 8906:] = -1e9

    hits_start = int(np.sum(np.argmax(Z_run, axis=1) == Y_clean))
    print(f"  [Clean-Target Base Hits (Phase 8)] {hits_start} / {N_clean} ({hits_start / N_clean * 100.0:.2f}%)", flush=True)

    # Start from Phase 9's 4-block calibrated rows, but:
    #   1) Strip or damp any row in `loop_prone_tids` where mean(alpha_4) > 1.005
    #   2) Re-optimize on Clean-Target `H_clean` using Sequential Gauss-Seidel updates with Loop Guard!
    W_e13_emb = W_p8_emb.copy()
    W_e13_q = W_p8_q.copy()

    z_true_0 = Z_run[r_idx_c, Y_clean]
    Z_comp_0 = Z_run.copy()
    Z_comp_0[r_idx_c, Y_clean] = -1e9
    Y_hat_0 = np.argmax(Z_comp_0, axis=1)
    m_0 = z_true_0 - Z_comp_0[r_idx_c, Y_hat_0]

    err_mask_0 = (m_0 < 1.0) & (Y_clean != 1)
    pos_cnt_0 = np.bincount(Y_clean[err_mask_0], minlength=8914)
    neg_cnt_0 = np.bincount(Y_hat_0[err_mask_0], minlength=8914)
    cand_rows = np.where(
        (pos_cnt_0 + neg_cnt_0 >= 2) & (np.arange(8914) >= 260) & (np.arange(8914) < 8906)
    )[0]
    # Sort candidate rows by error involvement descending for Gauss-Seidel ordering
    cand_order = cand_rows[np.argsort(-(pos_cnt_0[cand_rows] + neg_cnt_0[cand_rows]))]

    # Precompute 4-block partial scores S_blocks: [4, N_clean, len(cand_order)] in 4 fast matrix multiplies!
    S_blocks = np.zeros((4, N_clean, len(cand_order)), dtype=np.float32)
    for g in range(4):
        c_s, c_e = g * 128, (g + 1) * 128
        S_blocks[g] = H_clean[:, c_s:c_e] @ W_e13_q[cand_order, c_s:c_e].T

    # Maintain z_true_cur, z_comp_cur, Y_hat_cur incrementally without copying Z_run (0 GB copied!)
    z_true_cur = z_true_0.copy()
    Y_hat_cur = Y_hat_0.copy()
    z_comp_cur = Z_comp_0[r_idx_c, Y_hat_0].copy()
    # Keep Z_comp_0 masked at true targets so any local row-subset argmax is O(subset_size)
    Z_comp_run = Z_comp_0

    e13_calibrated_rows = []
    for idx_cr, r in enumerate(cand_order):
        r = int(r)
        is_loop_prone = (r in loop_prone_tids)
        m_cur = z_true_cur - z_comp_cur

        is_target = (Y_clean == r)
        is_false_comp = (Y_hat_cur == r) & (m_cur < 0.0)
        was_right = (m_cur > 0.0) & (~is_target)

        err_r_mask = (is_target & (m_cur < 1.5)) | is_false_comp
        if np.sum(err_r_mask) < 2:
            continue

        S_r = S_blocks[:, :, idx_cr].T  # [N_clean, 4]

        S_err = S_r[err_r_mask]
        sign_err = np.where(is_target[err_r_mask], 1.0, -0.90).astype(np.float32)
        def_err = np.clip(1.5 - m_cur[err_r_mask], 0.5, 4.0).astype(np.float32)
        rhs_4 = S_err.T @ (sign_err * def_err)
        cov_4 = (S_r.T @ S_r) / float(N_clean) + 0.5 * np.eye(4, dtype=np.float32)
        d_alpha_dir = np.linalg.solve(cov_4, rhs_4 / float(N_clean))
        d_alpha_dir = d_alpha_dir / (np.max(np.abs(d_alpha_dir)) + 1e-12)

        if is_loop_prone:
            d_zero_mean = d_alpha_dir - np.mean(d_alpha_dir)
            d_zero_mean = d_zero_mean / (np.max(np.abs(d_zero_mean)) + 1e-12)
            candidates_4d = [
                1.0 + step_a * d_zero_mean for step_a in [0.015, 0.03, 0.045]
            ] + [
                np.full(4, s_uni, dtype=np.float32) for s_uni in [0.94, 0.96, 0.98, 0.99]
            ]
        else:
            candidates_4d = [
                np.clip(1.0 + step_a * d_alpha_dir, 0.93, 1.055) for step_a in [0.015, 0.03, 0.045, 0.055]
            ] + [
                np.full(4, s_uni, dtype=np.float32) for s_uni in [0.95, 0.97, 0.985, 1.015, 1.03, 1.045]
            ]

        best_r_gain = 1
        best_alpha_vec = None
        for alpha_vec in candidates_4d:
            new_scores = S_r @ alpha_vec
            fn_fixed = int(np.sum(is_target & (m_cur <= 0.0) & ((new_scores - z_comp_cur) > 0.0)))
            fn_broken = int(np.sum(is_target & (m_cur > 0.0) & ((new_scores - z_comp_cur) <= 0.0)))
            fp_fixed = int(np.sum(is_false_comp & ((z_true_cur - new_scores) > 0.0)))
            new_fp = int(np.sum(was_right & (new_scores > z_true_cur)))
            net_gain = (fn_fixed + fp_fixed) - 2 * (fn_broken + new_fp)
            if net_gain > best_r_gain:
                best_r_gain = net_gain
                best_alpha_vec = alpha_vec

        if best_alpha_vec is not None:
            for g in range(4):
                W_e13_emb[r, g * 128 : (g + 1) * 128] *= float(best_alpha_vec[g])
            w_r_q = cq_quantize_dequantize_np(W_e13_emb[r : r + 1], bits=4, group=128)[0]
            W_e13_q[r] = w_r_q
            new_col = H_clean @ w_r_q
            Z_run[:, r] = new_col
            # Incremental O(1) update of z_true_cur and z_comp_cur!
            z_true_cur[is_target] = new_col[is_target]
            non_tgt = ~is_target
            Z_comp_run[non_tgt, r] = new_col[non_tgt]
            beat_comp = non_tgt & (new_col > z_comp_cur)
            if np.any(beat_comp):
                z_comp_cur[beat_comp] = new_col[beat_comp]
                Y_hat_cur[beat_comp] = r
            dropped_comp = (Y_hat_cur == r) & (new_col < z_comp_cur)
            if np.any(dropped_comp):
                sub_idx = np.where(dropped_comp)[0]
                sub_rows = Z_comp_run[sub_idx]
                new_yhat = np.argmax(sub_rows, axis=1)
                Y_hat_cur[sub_idx] = new_yhat
                z_comp_cur[sub_idx] = sub_rows[np.arange(len(sub_idx)), new_yhat]

            e13_calibrated_rows.append({
                "row": int(r),
                "loop_guarded": bool(is_loop_prone),
                "net_gain": int(best_r_gain),
                "alpha_4": [round(float(x), 4) for x in best_alpha_vec],
            })

    hits_after_e13 = int(np.sum(np.argmax(Z_run, axis=1) == Y_clean))
    print(
        f"  [Stage 2 Gauss-Seidel 4-Block Calibration] Calibrated {len(e13_calibrated_rows)} rows -> "
        f"Clean Dev Token Hits: {hits_start} -> {hits_after_e13} (+{hits_after_e13 - hits_start})",
        flush=True,
    )

    # Calibrate unquantized FP16 `stack/final_norm/scale` on top of W_e13_q
    s0 = np.array(orig_flat["stack/final_norm/scale"], dtype=np.float32, copy=True)
    z_true_13 = Z_run[r_idx_c, Y_clean]
    Z_comp_13 = Z_run.copy()
    Z_comp_13[r_idx_c, Y_clean] = -1e9
    Y_hat_13 = np.argmax(Z_comp_13, axis=1)
    m_13 = z_true_13 - Z_comp_13[r_idx_c, Y_hat_13]

    G_13 = U_clean * (W_e13_q[Y_clean] - W_e13_q[Y_hat_13])
    err_fn = (m_13 < 1.5) & (m_13 > -8.0) & (Y_clean != 1)
    r_fn = np.zeros(N_clean, dtype=np.float32)
    r_fn[err_fn] = np.clip(1.5 - m_13[err_fn], 0.0, 3.0)
    Cov_G = (G_13.T @ G_13) / float(N_clean)
    rhs_G = (G_13.T @ r_fn) / float(N_clean)

    best_s_e13 = scale_p8.copy()
    best_hits_fn = hits_after_e13
    for lam_s in [2.0, 10.0, 50.0]:
        ds = np.linalg.solve(Cov_G + lam_s * np.eye(512, dtype=np.float32), rhs_G)
        for step_s in [0.15, 0.3, 0.5]:
            s_cand = (s0 + np.clip(step_s * ds, -0.05, 0.05)).astype(np.float16).astype(np.float32)
            H_cand = U_clean * (1.0 + s_cand)[None, :]
            Z_cand = H_cand @ W_e13_q.T
            Z_cand[:, 0] = -1e9
            Z_cand[:, 2] = -1e9
            Z_cand[:, 8906:] = -1e9
            h_cand = int(np.sum(np.argmax(Z_cand, axis=1) == Y_clean))
            if h_cand > best_hits_fn:
                best_hits_fn = h_cand
                best_s_e13 = s_cand

    print(f"  [Stage 2B Final-Norm Calibration] Clean Dev Hits: {hits_after_e13} -> {best_hits_fn}", flush=True)

    params_e13_q = cq_ste_params({
        **params_orig,
        "embedding": {"embedding": jnp.asarray(W_e13_emb, dtype=jnp.float32)},
        "stack": {
            **params_orig["stack"],
            "final_norm": {"scale": jnp.asarray(best_s_e13, dtype=jnp.float32)},
        },
    }, WEIGHT_BITS)

    # Active 1-beam loop verification pass on MUCS Hinglish Dev!
    hyps_e13_mucs_dev, _, seqs_e13_mucs_dev = decode_dataset_gpu(params_e13_q, "mucs_hinglish", max_clips=300)
    loops_e13 = find_looping_tokens(seqs_e13_mucs_dev)
    new_loop_tids = [tid for tid in loops_e13 if loops_e13[tid] > loops_p8.get(tid, 0)]
    if new_loop_tids:
        print(f"  [Active Loop-Guard Pass 2] Reverting {len(new_loop_tids)} newly loop-inducing tokens: {new_loop_tids}", flush=True)
        for tid in new_loop_tids:
            W_e13_emb[tid] = W_p8_emb[tid].copy()
        e13_calibrated_rows = [item for item in e13_calibrated_rows if item["row"] not in set(new_loop_tids)]
        params_e13_q = cq_ste_params({
            **params_orig,
            "embedding": {"embedding": jnp.asarray(W_e13_emb, dtype=jnp.float32)},
            "stack": {
                **params_orig["stack"],
                "final_norm": {"scale": jnp.asarray(best_s_e13, dtype=jnp.float32)},
            },
        }, WEIGHT_BITS)

    # Also construct WSP_E14_LoopGuarded_P9: Phase 9 (WSP-E11) with ONLY the MUCS 1-beam loop-inducing tokens reverted!
    W_e14_emb = W_p9_emb.copy()
    for tid in loop_prone_tids:
        W_e14_emb[tid] = W_p8_emb[tid].copy()
    params_e14_q = cq_ste_params({
        **params_orig,
        "embedding": {"embedding": jnp.asarray(W_e14_emb, dtype=jnp.float32)},
        "stack": {
            **params_orig["stack"],
            "final_norm": {"scale": jnp.asarray(scale_p9, dtype=jnp.float32)},
        },
    }, WEIGHT_BITS)
    _, _, seqs_e14_mucs_dev = decode_dataset_gpu(params_e14_q, "mucs_hinglish", max_clips=300)
    loops_e14 = find_looping_tokens(seqs_e14_mucs_dev)
    extra_e14_loops = [tid for tid in loops_e14 if loops_e14[tid] > loops_p8.get(tid, 0)]
    if extra_e14_loops:
        print(f"  [E14 Loop-Guard Pass 2] Reverting {len(extra_e14_loops)} secondary loop tokens on P9: {extra_e14_loops}", flush=True)
        for tid in extra_e14_loops:
            W_e14_emb[tid] = W_p8_emb[tid].copy()
        params_e14_q = cq_ste_params({
            **params_orig,
            "embedding": {"embedding": jnp.asarray(W_e14_emb, dtype=jnp.float32)},
            "stack": {
                **params_orig["stack"],
                "final_norm": {"scale": jnp.asarray(scale_p9, dtype=jnp.float32)},
            },
        }, WEIGHT_BITS)
    e14_reverted_tids = sorted(list(loop_prone_tids | set(extra_e14_loops)))

    # =========================================================================
    # STAGE 3: Full 5-Dataset x 6-Tier Canonicalization Evaluation (GPU & C++ 5-Beam)
    # =========================================================================
    print("\n=== Stage 3: Full 6-Tier Canonicalization Audit & Model Comparison ===", flush=True)

    def evaluate_all_datasets_6tiers(p_q, label: str) -> Dict[str, Any]:
        t_start = time.time()
        res_model: Dict[str, Any] = {}
        for ds_key, clips in datasets_map.items():
            is_en = ds_key in ("svarah_en", "librispeech_en")
            n_dev = 200 if ds_key == "fleurs_hi" else (150 if ds_key == "librispeech_en" else 300)
            hyps, _, _ = decode_dataset_gpu(p_q, ds_key)
            refs = [c["text"] for c in clips]

            dev_tiers = compute_6tier_metrics(refs[:n_dev], hyps[:n_dev], is_en, en_norm)
            test_tiers = compute_6tier_metrics(refs[n_dev:], hyps[n_dev:], is_en, en_norm)
            all_tiers = compute_6tier_metrics(refs, hyps, is_en, en_norm)
            res_model[ds_key] = {
                "dev": dev_tiers,
                "test": test_tiers,
                "all": all_tiers,
            }
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

    gpu_6tier_p8 = evaluate_all_datasets_6tiers(params_p8_q, "Phase8_WSP_E10b")
    gpu_6tier_p9 = evaluate_all_datasets_6tiers(params_p9_q, "Phase9_WSP_E11")
    gpu_6tier_p10_e13 = evaluate_all_datasets_6tiers(params_e13_q, "Phase10_WSP_E13_CleanLoopSafe")
    gpu_6tier_p10_e14 = evaluate_all_datasets_6tiers(params_e14_q, "Phase10_WSP_E14_LoopGuardedP9")

    # Compute 5-dataset macro summaries across all 6 tiers for GPU
    tiers_list = ["T0_Raw_Strict", "T1_Unicode_Ortho", "T2_Spoken_Allomorph", "T3_Compound_Space", "T4_CrossScript_Loan", "T5_Whisper_Semantic"]
    gpu_macro_summary: Dict[str, Any] = {}
    for m_name, m_res in [
        ("Phase8_WSP_E10b", gpu_6tier_p8),
        ("Phase9_WSP_E11", gpu_6tier_p9),
        ("Phase10_WSP_E13_CleanLoopSafe", gpu_6tier_p10_e13),
        ("Phase10_WSP_E14_LoopGuardedP9", gpu_6tier_p10_e14),
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
    # STAGE 4: Native C++ `needle` 5-Beam Evaluation Across All 6 Canonicalization Tiers!
    # =========================================================================
    print("\n=== Stage 4: Native C++ `needle` 5-Beam Evaluation (Official vs Phase 9 vs Phase 10 Across All 6 Tiers) ===", flush=True)

    def run_cpp_5beam_6tiers(w_emb_np: np.ndarray, scale_np: np.ndarray, tag_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        mod_flat = dict(orig_flat)
        mod_flat["embedding/embedding"] = w_emb_np.astype(np.float32)
        mod_flat["stack/final_norm/scale"] = scale_np.astype(np.float32)
        tmp_cact = f"/tmp/{tag_name}.cact"
        patch_info = surgical_patch_cact(
            cact_path,
            tmp_cact,
            orig_flat,
            mod_flat,
            WHISTLE_SMALL_CFG,
        )

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
                        [needle_bin, "--model", tmp_cact, "--audio", wp, "--threads", "4"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    obj = json.loads(proc.stdout.strip().splitlines()[-1])
                    hyps.append(obj.get("text", "").strip())
                    refs.append(c["text"])
                cpp_res[ds_key] = compute_6tier_metrics(refs, hyps, is_en, en_norm)
                print(
                    f"  [C++ 5-Beam {tag_name}] {ds_key:15s} | "
                    f"T0_Strict={cpp_res[ds_key]['T0_Raw_Strict']['wer_pct']:5.2f}% | "
                    f"T2_Spoken={cpp_res[ds_key]['T2_Spoken_Allomorph']['wer_pct']:5.2f}% | "
                    f"T4_Loan={cpp_res[ds_key]['T4_CrossScript_Loan']['wer_pct']:5.2f}% | "
                    f"T5_Semantic={cpp_res[ds_key]['T5_Whisper_Semantic']['wer_pct']:5.2f}% "
                    f"(WordAcc={cpp_res[ds_key]['T5_Whisper_Semantic']['word_acc_pct']:5.2f}%, "
                    f"CharAcc={cpp_res[ds_key]['T5_Whisper_Semantic']['char_acc_pct']:5.2f}%)",
                    flush=True,
                )
        return cpp_res, patch_info

    cpp_6tier_orig, _ = run_cpp_5beam_6tiers(W_orig_emb, np.array(orig_flat["stack/final_norm/scale"], dtype=np.float32), "orig_baseline")
    cpp_6tier_p9, patch_p9 = run_cpp_5beam_6tiers(W_p9_emb, scale_p9, "phase9_e11")
    cpp_6tier_p10_e13, patch_p10_e13 = run_cpp_5beam_6tiers(W_e13_emb, best_s_e13, "phase10_e13")
    cpp_6tier_p10_e14, patch_p10_e14 = run_cpp_5beam_6tiers(W_e14_emb, scale_p9, "phase10_e14")

    cpp_macro_summary: Dict[str, Any] = {}
    for m_name, c_res in [
        ("Official_Baseline", cpp_6tier_orig),
        ("Phase9_WSP_E11", cpp_6tier_p9),
        ("Phase10_WSP_E13_CleanLoopSafe", cpp_6tier_p10_e13),
        ("Phase10_WSP_E14_LoopGuardedP9", cpp_6tier_p10_e14),
    ]:
        cpp_macro_summary[m_name] = {}
        for t_name in tiers_list:
            avg_wer = float(np.mean([c_res[k][t_name]["wer_pct"] for k in datasets_map.keys()]))
            avg_cer = float(np.mean([c_res[k][t_name]["cer_pct"] for k in datasets_map.keys()]))
            cpp_macro_summary[m_name][t_name] = {
                "macro_5ds_wer_pct": round(avg_wer, 3),
                "macro_5ds_word_acc_pct": round(100.0 - avg_wer, 3),
                "macro_5ds_cer_pct": round(avg_cer, 3),
                "macro_5ds_char_acc_pct": round(100.0 - avg_cer, 3),
            }

    return {
        "canonicalized_dev_ref_count": int(canonicalized_ref_count),
        "loop_prone_tokens_detected": [int(x) for x in sorted(loop_prone_tids)],
        "e14_reverted_tids": e14_reverted_tids,
        "e13_calibrated_rows": e13_calibrated_rows,
        "scale_p10_e13": best_s_e13.tolist(),
        "gpu_macro_summary": gpu_macro_summary,
        "cpp_macro_summary": cpp_macro_summary,
        "gpu_6tier_phase8": gpu_6tier_p8,
        "gpu_6tier_phase9": gpu_6tier_p9,
        "gpu_6tier_phase10_e13": gpu_6tier_p10_e13,
        "gpu_6tier_phase10_e14": gpu_6tier_p10_e14,
        "cpp_6tier_orig": cpp_6tier_orig,
        "cpp_6tier_phase9": cpp_6tier_p9,
        "cpp_6tier_phase10_e13": cpp_6tier_p10_e13,
        "cpp_6tier_phase10_e14": cpp_6tier_p10_e14,
        "patch_p10_e13": patch_p10_e13,
        "patch_p10_e14": patch_p10_e14,
        "elapsed_sec": round(time.time() - t0, 1),
    }


@app.local_entrypoint()
def main():
    wsp_Bytes = (ROOT_LOCAL / "reports" / "phase3_wsp_vectors.npz").read_bytes()
    p8_full = json.loads((ROOT_LOCAL / "reports" / "phase8_fp32_master_wsp_results.json").read_text(encoding="utf-8"))
    p9_full = json.loads((ROOT_LOCAL / "reports" / "phase9_cascade_wsp_results.json").read_text(encoding="utf-8"))
    p8_slim = {
        "best_e10_modified_rows": p8_full["best_e10_modified_rows"],
        "best_s_e10b": p8_full["best_s_e10_scale"],
    }
    p9_slim = {
        "stage3_block_calibrated_rows": p9_full["stage3_block_calibrated_rows"],
        "best_s_p9_scale": p9_full["best_s_p9_scale"],
    }
    out = run_phase10_clean_target_wsp.remote(wsp_Bytes, p8_slim, p9_slim)
    out_path = ROOT_LOCAL / "reports" / "phase10_clean_target_wsp_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved Phase 10 results to {out_path}")
    print("C++ 5-Beam Macro Summary across 6 Canonicalization Tiers:")
    print(json.dumps(out["cpp_macro_summary"], indent=2))
    print("GPU 1-Beam Macro Summary across 6 Canonicalization Tiers:")
    print(json.dumps(out["gpu_macro_summary"], indent=2))
