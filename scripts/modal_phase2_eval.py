#!/usr/bin/env python3
"""Phase 2 Modal GPU Baseline Evaluation & Activation/Logit Extraction Pipeline for Shruti WSP Research.

Runs on Modal GPU (L4 / A10G) with persistent dataset + activation caching on Modal Volume `shruti-wsp-vol`.
Evaluates the unmodified baseline checkpoint (`shruti.safetensors` under exact CQ-W4A8 STE and `shruti.cact`
via native `needle` 5-beam C++ engine) across:
  1. FLEURS Hindi (`google/fleurs`, `hi_in`, `test`, 418 clips) — Auto-LID & Forced `hi`
  2. Svarah Indian English (`tinisoft/svarah-with-asl-gloss` / `Bhargav0044/svarah1`, 600 clips: 300 dev + 300 test) — Auto-LID & Forced `en`
  3. MUCS Hinglish (`dianavdavidson/MUCS-Hinglish`, `test`, 600 clips: 300 dev + 300 test) — Auto-LID & Forced `hi`
  4. Lahaja Accented Hindi (`SayantanJoker/lahaja`, `test`, 600 clips: 300 dev + 300 test) — Auto-LID & Forced `hi`
  5. LibriSpeech Clean (`openslr/librispeech_asr`, `clean`, `test`, 300 clips: 150 dev + 150 test) — Auto-LID & Forced `en`
  6. Silence & Hiss Hallucination Test (100 clips: 50 digital silence + 50 faint Gaussian hiss)

Also extracts and saves:
  - Step-0 pre-logit decoder hidden representations h_0 in R^{512} and quantized aq(h_0) in R^{512} for every clip
  - Step-0 logits across all 8 language tags (`en, de, fr, es, it, nl, pl, hi`)
  - Token-level Latin vs. Devanagari logit competition & hidden-state norms on MUCS Hinglish
  - Word-level Levenshtein error taxonomy (substitutions, insertions, deletions, top confusion pairs, look-alikes)
"""

import io
import json
import math
import os
import re
import subprocess
import tempfile
import time
import unicodedata
import wave
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

ROOT_LOCAL = Path(__file__).resolve().parent.parent

app = modal.App("shruti-wsp-phase2")

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


# ============================================================================
# Text Normalization & Error Scoring (Indic-preserving + Whisper Normalizers)
# ============================================================================

def normalize_indic_hinglish(text: str) -> str:
    """Normalize Hindi, English, and Hinglish text while preserving Devanagari matras (Mn/Mc)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    # Remove bracketed annotations like [english], <unk>, etc.
    text = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", text)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        # Keep Letters (L*), Combining Marks (M* - critical for Devanagari matras/anusvara/chandrabindu/halant), Numbers (N*)
        if cat[0] in ("L", "M", "N"):
            # Exclude Devanagari danda/double-danda if categorized strangely
            if ch in ("।", "॥", "॰"):
                out.append(" ")
            else:
                out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def is_latin_word(w: str) -> bool:
    return any("a" <= c <= "z" for c in w.lower())


def is_devanagari_word(w: str) -> bool:
    return any("\u0900" <= c <= "\u097f" for c in w)


def levenshtein_alignment(ref_seq: List[str], hyp_seq: List[str]) -> Tuple[int, int, int, List[Tuple[str, str, str]]]:
    """Compute exact (substitutions, insertions, deletions, ops) between ref_seq and hyp_seq."""
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
            row_i[j] = min(
                row_prev[j] + 1,       # deletion
                row_i[j - 1] + 1,      # insertion
                row_prev[j - 1] + cost # match / substitution
            )
    # Backtrack
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


def write_wav_16k(path: str, samples) -> None:
    import numpy as np
    pcm = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


# ============================================================================
# Dataset Loading & Caching on Modal Volume
# ============================================================================

def load_or_cache_datasets(cache_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load evaluation splits from Hugging Face or cached NPZ/JSON shards on Modal Volume."""
    import sys
    import numpy as np
    import soxr
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyarrow==17.0.0"])
    from datasets import Audio, load_dataset

    os.makedirs(cache_dir, exist_ok=True)
    specs = [
        {
            "key": "fleurs_hi",
            "hf_id": "google/fleurs",
            "hf_cfg": "hi_in",
            "split": "test",
            "max_clips": 418,
            "text_cols": ["transcription", "raw_transcription", "text"],
            "target_lang": "hi",
            "meta_cols": ["gender", "id"],
        },
        {
            "key": "svarah_en",
            "hf_id": "tinisoft/svarah-with-asl-gloss",
            "fallback_hf_id": "Bhargav0044/svarah1",
            "hf_cfg": "default",
            "split": "test",
            "fallback_split": "train",
            "max_clips": 600,
            "text_cols": ["text", "reference_transcription", "sentence"],
            "target_lang": "en",
            "meta_cols": ["primary_language", "native_place_state", "gender"],
        },
        {
            "key": "mucs_hinglish",
            "hf_id": "dianavdavidson/MUCS-Hinglish",
            "hf_cfg": "default",
            "split": "test",
            "max_clips": 600,
            "text_cols": ["transcript", "text"],
            "target_lang": "hi",
            "meta_cols": ["segment_id", "speaker_id", "ratio_english_words"],
        },
        {
            "key": "lahaja_hi",
            "hf_id": "SayantanJoker/lahaja",
            "hf_cfg": "default",
            "split": "test",
            "max_clips": 600,
            "text_cols": ["text", "normalized", "verbatim"],
            "target_lang": "hi",
            "meta_cols": ["gender", "age_group"],
        },
        {
            "key": "librispeech_en",
            "hf_id": "openslr/librispeech_asr",
            "hf_cfg": "clean",
            "split": "test",
            "max_clips": 300,
            "text_cols": ["text"],
            "target_lang": "en",
            "meta_cols": ["speaker_id", "id"],
        },
    ]

    result: Dict[str, List[Dict[str, Any]]] = {}
    for spec in specs:
        key = spec["key"]
        npz_path = os.path.join(cache_dir, f"{key}_{spec['max_clips']}.npz")
        meta_path = os.path.join(cache_dir, f"{key}_{spec['max_clips']}.json")
        if os.path.exists(npz_path) and os.path.exists(meta_path):
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
            continue

        print(f"[Downloading] Fetching {key} from Hugging Face ({spec['hf_id']})...", flush=True)
        try:
            ds = load_dataset(
                spec["hf_id"],
                spec["hf_cfg"],
                split=spec["split"],
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"  Primary load failed for {spec['hf_id']}: {e}; trying fallback...", flush=True)
            ds = load_dataset(
                spec["fallback_hf_id"],
                spec["hf_cfg"],
                split=spec.get("fallback_split", "train"),
                streaming=True,
                trust_remote_code=True,
            )
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

        clips = []
        arrays = {}
        metas = []
        for row in ds:
            if len(clips) >= spec["max_clips"]:
                break
            audio_obj = row["audio"]
            samples = np.asarray(audio_obj["array"], dtype=np.float32)
            sr = int(audio_obj["sampling_rate"])
            if samples.ndim > 1:
                samples = samples.mean(axis=-1).astype(np.float32)
            if sr != SAMPLE_RATE:
                samples = soxr.resample(samples, sr, SAMPLE_RATE, quality="HQ").astype(np.float32)
            if len(samples) == 0 or len(samples) > MAX_SAMPLES:
                continue
            text = ""
            for tcol in spec["text_cols"]:
                if row.get(tcol):
                    text = str(row[tcol]).strip()
                    break
            if not text:
                continue
            meta = {
                "idx": len(clips),
                "text": text,
                "target_lang": spec["target_lang"],
                "duration_sec": round(len(samples) / SAMPLE_RATE, 3),
                "extra": {mc: str(row.get(mc, "")) for mc in spec["meta_cols"] if row.get(mc) is not None},
            }
            arrays[f"arr_{len(clips)}"] = samples
            metas.append(meta)
            clips.append({**meta, "samples": samples})

        np.savez_compressed(npz_path, **arrays)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metas, f, ensure_ascii=False, indent=1)
        vol.commit()
        print(f"[Cached & Committed] Saved {len(clips)} clips for {key} -> {npz_path}", flush=True)
        result[key] = clips

    vol.commit()
    return result


# ============================================================================
# Main Modal GPU Evaluation Function
# ============================================================================

@app.function(
    image=image,
    gpu=["A10G", "A100", "T4"],
    cpu=8.0,
    memory=32768,
    timeout=3600,
    volumes={"/data": vol},
)
def run_phase2_evaluation() -> Dict[str, Any]:
    import sys
    import numpy as np
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
    from needle_model.whistle import Whistle, load_whistle
    from needle_model.quantize import cq_ste_params, WEIGHT_BITS
    from needle_model.architecture import _aq, _rms_unit
    from shruti_io import RefTokenizer

    t0 = time.time()
    print(f"JAX devices: {jax.devices()} (default backend: {jax.default_backend()})", flush=True)

    import hashlib
    import urllib.request
    from huggingface_hub import hf_hub_download

    os.makedirs("/data/checkpoints", exist_ok=True)
    ckpt_path = "/data/checkpoints/shruti.safetensors"
    cact_path = "/data/checkpoints/shruti.cact"
    needle_bin = "/data/checkpoints/needle"

    if not os.path.exists(ckpt_path):
        print("Downloading shruti.safetensors from Hugging Face (udgamlabs/Shruti) into /data/checkpoints...", flush=True)
        hf_hub_download(repo_id="udgamlabs/Shruti", filename="shruti.safetensors", local_dir="/data/checkpoints")
    if not os.path.exists(cact_path):
        print("Downloading shruti.cact from Hugging Face (udgamlabs/Shruti) into /data/checkpoints...", flush=True)
        hf_hub_download(repo_id="udgamlabs/Shruti", filename="shruti.cact", local_dir="/data/checkpoints")
    if os.path.exists(needle_bin):
        os.chmod(needle_bin, 0o755)
    vol.commit()

    st_sha = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
    cact_sha = hashlib.sha256(open(cact_path, "rb").read()).hexdigest()
    print(f"Verified checkpoints on /data/checkpoints: safetensors={st_sha[:16]}..., cact={cact_sha[:16]}...", flush=True)
    assert st_sha == "5c43105db153a47d02c47131d37d066343f04f1c3ff82315ff5da1ce264fbfa7"
    assert cact_sha == "b189e954aaed4a9fd41acbad3ac05e486381e96325b22a4c695b9ea5153d7065"

    # 1. Load model, CQ-W4 STE params, and RefTokenizer from shruti.cact
    params, cfg = load_whistle(ckpt_path)
    cfg = replace(cfg, flash=False, remat=False)
    model = Whistle(cfg)
    params_f32 = jax.tree.map(lambda a: jnp.asarray(a, dtype=jnp.float32), params)
    ste_params = cq_ste_params(params_f32, WEIGHT_BITS)
    tokenizer = RefTokenizer.from_cact(cact_path)

    print(f"Loaded WhistleConfig: vocab={cfg.vocab}, text={cfg.text}, languages={cfg.languages}", flush=True)
    lang_names = list(cfg.languages)
    lang_ids = [cfg.text + i for i in range(len(lang_names))]
    en_tag_id = cfg.tag("en")  # 8906
    hi_tag_id = cfg.tag("hi")  # 8913

    # 2. Load or cache evaluation datasets
    datasets_map = load_or_cache_datasets("/data/eval_cache")

    # 3. Compile JIT functions for:
    #    (a) Acoustic encoding
    #    (b) Pre-logit decoder representation h_t = Whistle.decode(...)
    #    (c) Step logits = _aq(h_t, True) @ E^T
    @jax.jit
    def jit_encode(p, wav, lengths):
        return model.apply({"params": p}, wav, lengths, True, method=Whistle.encode)

    @jax.jit
    def jit_decode_hidden(p, tokens, enc, frames):
        return model.apply({"params": p}, tokens, enc, frames, True, method=Whistle.decode)

    @jax.jit
    def jit_step_logits(p, tokens, enc, frames):
        h = model.apply({"params": p}, tokens, enc, frames, True, method=Whistle.decode)
        hq = _aq(h, True).astype(jnp.float32)
        logits = hq @ p["embedding"]["embedding"].T.astype(jnp.float32)
        return logits, h, hq

    # Warmup JIT
    print("Warming up JAX JIT kernels...", flush=True)
    dummy_wav = jnp.zeros((4, MAX_SAMPLES), dtype=jnp.float32)
    dummy_len = jnp.full((4,), SAMPLE_RATE * 5, dtype=jnp.int32)
    d_enc, d_frames = jit_encode(ste_params, dummy_wav, dummy_len)
    dummy_tok = jnp.full((4, 64), cfg.pad, dtype=jnp.int32).at[:, 0].set(cfg.bos)
    _ = jit_step_logits(ste_params, dummy_tok, d_enc, d_frames)
    print(f"JIT warmup complete in {time.time() - t0:.1f}s", flush=True)

    en_norm = EnglishTextNormalizer()
    basic_norm = BasicTextNormalizer()

    def decode_dataset_gpu(
        clips: List[Dict[str, Any]],
        forced_lang: Optional[str] = None,
        batch_size: int = 24,
        collect_step0: bool = False,
        collect_token_script_stats: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
        """Run batched CQ-W4A8 STE GPU decoding on `clips`."""
        n_clips = len(clips)
        tags_mask = np.arange(cfg.vocab) >= cfg.text
        forced_tag_id = cfg.tag(forced_lang) if forced_lang else None

        outputs: List[Dict[str, Any]] = []
        h0_list = []
        h0q_list = []
        lid_logits_list = []

        # Token script competition accumulators (for MUCS Hinglish)
        latin_tok_mask = np.zeros(cfg.vocab, dtype=bool)
        dev_tok_mask = np.zeros(cfg.vocab, dtype=bool)
        for tid in range(260, cfg.text):
            piece = tokenizer.pieces[tid]
            if any("\u0900" <= c <= "\u097f" for c in piece):
                dev_tok_mask[tid] = True
            elif any("a" <= c.lower() <= "z" for c in piece):
                latin_tok_mask[tid] = True

        script_competition = {
            "picked_latin_count": 0,
            "picked_dev_count": 0,
            "picked_latin_margin_over_dev": [],
            "picked_dev_margin_over_latin": [],
            "picked_latin_h_norm": [],
            "picked_dev_h_norm": [],
        }

        for start in range(0, n_clips, batch_size):
            part = clips[start:start + batch_size]
            B = len(part)
            wav = np.zeros((B, MAX_SAMPLES), dtype=np.float32)
            lengths = np.zeros(B, dtype=np.int32)
            for b_i, c in enumerate(part):
                s = c["samples"]
                wav[b_i, :len(s)] = s
                lengths[b_i] = len(s)

            enc, frames = jit_encode(ste_params, wav, lengths)
            tokens = np.full((B, cfg.max_tokens), cfg.pad, dtype=np.int32)
            tokens[:, 0] = cfg.bos
            done = np.zeros(B, dtype=bool)

            max_gen_steps = min(cfg.max_tokens - 1, 192)
            for pos in range(max_gen_steps):
                bucket_size = min(cfg.max_tokens, (pos // 64 + 1) * 64)
                logits_all, h_all, hq_all = jit_step_logits(ste_params, tokens[:, :bucket_size], enc, frames)
                step_logits = np.array(logits_all[:, pos])

                if pos == 0:
                    if collect_step0:
                        h0_list.append(np.array(h_all[:, 0, :], dtype=np.float32))
                        h0q_list.append(np.array(hq_all[:, 0, :], dtype=np.float32))
                        lid_logits_list.append(step_logits[:, cfg.text:cfg.vocab].copy())
                    if forced_tag_id is not None:
                        pick = np.full(B, forced_tag_id, dtype=np.int32)
                    else:
                        step_logits[:, ~tags_mask] = -np.inf
                        pick = step_logits.argmax(axis=-1).astype(np.int32)
                else:
                    step_logits[:, tags_mask] = -np.inf
                    # Suppress <pad> and <s> during generation
                    step_logits[:, cfg.pad] = -np.inf
                    step_logits[:, cfg.bos] = -np.inf
                    pick = step_logits.argmax(axis=-1).astype(np.int32)

                    if collect_token_script_stats:
                        h_pos_norm = np.linalg.norm(np.array(h_all[:, pos, :], dtype=np.float32), axis=-1)
                        max_lat = step_logits[:, latin_tok_mask].max(axis=-1)
                        max_dev = step_logits[:, dev_tok_mask].max(axis=-1)
                        for b_i in range(B):
                            if done[b_i]:
                                continue
                            pid = int(pick[b_i])
                            if latin_tok_mask[pid]:
                                script_competition["picked_latin_count"] += 1
                                script_competition["picked_latin_margin_over_dev"].append(float(max_lat[b_i] - max_dev[b_i]))
                                script_competition["picked_latin_h_norm"].append(float(h_pos_norm[b_i]))
                            elif dev_tok_mask[pid]:
                                script_competition["picked_dev_count"] += 1
                                script_competition["picked_dev_margin_over_latin"].append(float(max_dev[b_i] - max_lat[b_i]))
                                script_competition["picked_dev_h_norm"].append(float(h_pos_norm[b_i]))

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
                    "pred_tag_id": int(pred_tag_id),
                    "duration_sec": c["duration_sec"],
                    "extra": c.get("extra", {}),
                })

        step0_data = None
        if collect_step0 and h0_list:
            step0_data = {
                "h0": np.concatenate(h0_list, axis=0),
                "h0q": np.concatenate(h0q_list, axis=0),
                "lid_logits": np.concatenate(lid_logits_list, axis=0),
            }

        comp_summary = {}
        if collect_token_script_stats:
            comp_summary = {
                "picked_latin_tokens": script_competition["picked_latin_count"],
                "picked_devanagari_tokens": script_competition["picked_dev_count"],
                "mean_latin_margin_over_dev": float(np.mean(script_competition["picked_latin_margin_over_dev"])) if script_competition["picked_latin_margin_over_dev"] else 0.0,
                "mean_dev_margin_over_latin": float(np.mean(script_competition["picked_dev_margin_over_latin"])) if script_competition["picked_dev_margin_over_latin"] else 0.0,
                "mean_latin_step_h_norm": float(np.mean(script_competition["picked_latin_h_norm"])) if script_competition["picked_latin_h_norm"] else 0.0,
                "mean_dev_step_h_norm": float(np.mean(script_competition["picked_dev_h_norm"])) if script_competition["picked_dev_h_norm"] else 0.0,
            }
        return outputs, step0_data, comp_summary

    def score_outputs(
        outputs: List[Dict[str, Any]],
        lid_logits: Optional[np.ndarray] = None,
        split_slice: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive WER, CER, LID accuracy, script balance, and failure taxonomy."""
        if split_slice is not None:
            lo, hi = split_slice
            sub_out = outputs[lo:hi]
            sub_lid = lid_logits[lo:hi] if lid_logits is not None else None
        else:
            sub_out = outputs
            sub_lid = lid_logits

        w_subs = w_ins = w_dels = w_ref_total = 0
        c_subs = c_ins = c_dels = c_ref_total = 0
        whisp_edits = whisp_ref_total = 0

        ref_latin_words = hyp_latin_words = 0
        ref_dev_words = hyp_dev_words = 0

        lid_counts = Counter()
        sub_pairs = Counter()
        del_words = Counter()
        ins_words = Counter()

        # Specific diagnostic failure counters
        failure_buckets = {
            "lid_misrouted_clips": 0,
            "script_mismatch_en_to_dev": 0,
            "script_mismatch_dev_to_en": 0,
            "hindi_lookalikes": Counter(),
            "matra_nasal_confusions": Counter(),
        }

        lookalike_set = {
            ("है", "हैं"), ("हैं", "है"),
            ("मैं", "में"), ("में", "मैं"),
            ("की", "के"), ("के", "की"),
            ("का", "को"), ("को", "का"),
            ("का", "के"), ("के", "का"),
            ("की", "को"), ("को", "की"),
            ("पे", "पर"), ("पर", "पे"),
            ("वो", "वह"), ("वह", "वो"),
            ("ओर", "और"), ("और", "ओर"),
        }

        worst_examples = []

        for idx_in_sub, item in enumerate(sub_out):
            t_lang = item["target_lang"]
            p_lang = item["pred_lang"]
            lid_counts[p_lang] += 1
            if p_lang != t_lang:
                failure_buckets["lid_misrouted_clips"] += 1

            # Primary Indic/English normalization (preserves Devanagari matras)
            if t_lang == "en":
                ref_norm = en_norm(item["ref"])
                hyp_norm = en_norm(item["hyp"]) if p_lang == "en" else normalize_indic_hinglish(item["hyp"])
            else:
                ref_norm = normalize_indic_hinglish(item["ref"])
                hyp_norm = normalize_indic_hinglish(item["hyp"])

            ref_words = ref_norm.split()
            hyp_words = hyp_norm.split()

            for w in ref_words:
                if is_latin_word(w):
                    ref_latin_words += 1
                elif is_devanagari_word(w):
                    ref_dev_words += 1
            for w in hyp_words:
                if is_latin_word(w):
                    hyp_latin_words += 1
                elif is_devanagari_word(w):
                    hyp_dev_words += 1

            s, ins, d, ops = levenshtein_alignment(ref_words, hyp_words)
            w_subs += s
            w_ins += ins
            w_dels += d
            w_ref_total += len(ref_words)

            for op, rw, hw in ops:
                if op == "sub":
                    sub_pairs[(rw, hw)] += 1
                    if is_latin_word(rw) and is_devanagari_word(hw):
                        failure_buckets["script_mismatch_en_to_dev"] += 1
                    elif is_devanagari_word(rw) and is_latin_word(hw):
                        failure_buckets["script_mismatch_dev_to_en"] += 1
                    if (rw, hw) in lookalike_set:
                        failure_buckets["hindi_lookalikes"][f"{rw}->{hw}"] += 1
                    elif is_devanagari_word(rw) and is_devanagari_word(hw):
                        # Check if consonant skeleton matches and only matras/anusvara differ
                        skel_r = "".join(c for c in rw if unicodedata.category(c).startswith("L"))
                        skel_h = "".join(c for c in hw if unicodedata.category(c).startswith("L"))
                        if skel_r == skel_h and rw != hw:
                            failure_buckets["matra_nasal_confusions"][f"{rw}->{hw}"] += 1
                elif op == "del":
                    del_words[rw] += 1
                elif op == "ins":
                    ins_words[hw] += 1

            # Character-level Levenshtein (excluding spaces)
            ref_chars = list(ref_norm.replace(" ", ""))
            hyp_chars = list(hyp_norm.replace(" ", ""))
            cs, cins, cd, _ = levenshtein_alignment(ref_chars, hyp_chars)
            c_subs += cs
            c_ins += cins
            c_dels += cd
            c_ref_total += len(ref_chars)

            # Secondary Whisper normalizer check
            w_norm_fn = en_norm if t_lang == "en" else basic_norm
            rw_w = w_norm_fn(item["ref"]).split()
            hw_w = w_norm_fn(item["hyp"]).split()
            ws, wins, wd, _ = levenshtein_alignment(rw_w, hw_w)
            whisp_edits += (ws + wins + wd)
            whisp_ref_total += len(rw_w)

            utt_wer = (s + ins + d) / max(len(ref_words), 1) * 100.0
            lid_margin = None
            if sub_lid is not None:
                en_z = float(sub_lid[idx_in_sub, 0])
                hi_z = float(sub_lid[idx_in_sub, 7])
                lid_margin = round(en_z - hi_z, 3)

            worst_examples.append({
                "idx": item["idx"],
                "ref": item["ref"],
                "hyp": item["hyp"],
                "ref_norm": ref_norm,
                "hyp_norm": hyp_norm,
                "pred_lang": p_lang,
                "utt_wer": round(utt_wer, 2),
                "edits": int(s + ins + d),
                "ref_len": len(ref_words),
                "lid_en_minus_hi": lid_margin,
                "extra": item.get("extra", {}),
            })

        worst_examples.sort(key=lambda x: (-x["edits"], -x["utt_wer"]))

        lid_stats = {}
        if sub_lid is not None and len(sub_lid) > 0:
            en_z = sub_lid[:, 0]
            hi_z = sub_lid[:, 7]
            margins = en_z - hi_z
            lid_stats = {
                "mean_logit_en": round(float(np.mean(en_z)), 4),
                "std_logit_en": round(float(np.std(en_z)), 4),
                "mean_logit_hi": round(float(np.mean(hi_z)), 4),
                "std_logit_hi": round(float(np.std(hi_z)), 4),
                "mean_margin_en_minus_hi": round(float(np.mean(margins)), 4),
                "std_margin_en_minus_hi": round(float(np.std(margins)), 4),
                "quantiles_margin_en_minus_hi": {
                    q: round(float(np.percentile(margins, p)), 4)
                    for q, p in [("p05", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]
                },
                "frac_en_gt_hi": round(float(np.mean(margins > 0.0)) * 100.0, 2),
            }

        wer = (w_subs + w_ins + w_dels) / max(w_ref_total, 1) * 100.0
        cer = (c_subs + c_ins + c_dels) / max(c_ref_total, 1) * 100.0
        whisp_wer = whisp_edits / max(whisp_ref_total, 1) * 100.0

        return {
            "num_clips": len(sub_out),
            "wer": round(wer, 2),
            "cer": round(cer, 2),
            "whisper_norm_wer": round(whisp_wer, 2),
            "word_edits": int(w_subs + w_ins + w_dels),
            "word_subs": int(w_subs),
            "word_ins": int(w_ins),
            "word_dels": int(w_dels),
            "word_ref_total": int(w_ref_total),
            "char_edits": int(c_subs + c_ins + c_dels),
            "char_ref_total": int(c_ref_total),
            "script_word_counts": {
                "ref_latin_words": int(ref_latin_words),
                "hyp_latin_words": int(hyp_latin_words),
                "latin_overprediction_ratio": round(hyp_latin_words / max(ref_latin_words, 1), 4),
                "ref_devanagari_words": int(ref_dev_words),
                "hyp_devanagari_words": int(hyp_dev_words),
            },
            "lid_distribution": dict(lid_counts),
            "lid_accuracy_pct": round(lid_counts.get(sub_out[0]["target_lang"], 0) / max(len(sub_out), 1) * 100.0, 2) if sub_out else 0.0,
            "lid_logit_stats": lid_stats,
            "failure_taxonomy": {
                "lid_misrouted_clips": failure_buckets["lid_misrouted_clips"],
                "script_mismatch_en_to_dev": failure_buckets["script_mismatch_en_to_dev"],
                "script_mismatch_dev_to_en": failure_buckets["script_mismatch_dev_to_en"],
                "top_hindi_lookalike_errors": dict(failure_buckets["hindi_lookalikes"].most_common(20)),
                "top_matra_nasal_confusions": dict(failure_buckets["matra_nasal_confusions"].most_common(25)),
                "top_word_substitutions": [
                    {"ref": r, "hyp": h, "count": cnt}
                    for (r, h), cnt in sub_pairs.most_common(35)
                ],
                "top_word_deletions": [
                    {"word": w, "count": cnt} for w, cnt in del_words.most_common(20)
                ],
                "top_word_insertions": [
                    {"word": w, "count": cnt} for w, cnt in ins_words.most_common(20)
                ],
            },
            "sample_failures": worst_examples[:20],
        }

    # ========================================================================
    # 4. Run Full GPU CQ-W4A8 STE Evaluation + Step-0 Hidden State Extraction
    # ========================================================================
    eval_plan = [
        ("fleurs_hi", "hi", (0, 200), (200, 418), False),
        ("svarah_en", "en", (0, 300), (300, 600), False),
        ("mucs_hinglish", "hi", (0, 300), (300, 600), True),
        ("lahaja_hi", "hi", (0, 300), (300, 600), False),
        ("librispeech_en", "en", (0, 150), (150, 300), False),
    ]

    full_results: Dict[str, Any] = {}
    step0_archive: Dict[str, np.ndarray] = {}

    for ds_key, forced_lang, dev_slice, test_slice, track_script in eval_plan:
        clips = datasets_map[ds_key]
        t_ds = time.time()
        print(f"\n=== Evaluating {ds_key} ({len(clips)} clips) on GPU [Auto-LID & Forced-{forced_lang}] ===", flush=True)

        auto_out, step0_data, script_comp_auto = decode_dataset_gpu(
            clips,
            forced_lang=None,
            batch_size=24,
            collect_step0=True,
            collect_token_script_stats=track_script,
        )
        forced_out, _, script_comp_forced = decode_dataset_gpu(
            clips,
            forced_lang=forced_lang,
            batch_size=24,
            collect_step0=False,
            collect_token_script_stats=track_script,
        )

        lid_logits = step0_data["lid_logits"] if step0_data else None
        if step0_data is not None:
            step0_archive[f"{ds_key}_h0"] = step0_data["h0"]
            step0_archive[f"{ds_key}_h0q"] = step0_data["h0q"]
            step0_archive[f"{ds_key}_lid_logits"] = step0_data["lid_logits"]

        res_auto_all = score_outputs(auto_out, lid_logits, None)
        res_auto_dev = score_outputs(auto_out, lid_logits, dev_slice)
        res_auto_test = score_outputs(auto_out, lid_logits, test_slice)

        res_forced_all = score_outputs(forced_out, lid_logits, None)
        res_forced_dev = score_outputs(forced_out, lid_logits, dev_slice)
        res_forced_test = score_outputs(forced_out, lid_logits, test_slice)

        full_results[ds_key] = {
            "num_clips_total": len(clips),
            "dev_slice": list(dev_slice),
            "test_slice": list(test_slice),
            "forced_language_tag": forced_lang,
            "elapsed_sec": round(time.time() - t_ds, 2),
            "auto_lid": {
                "all": res_auto_all,
                "dev": res_auto_dev,
                "test": res_auto_test,
                "token_script_competition": script_comp_auto,
            },
            "forced_lid": {
                "all": res_forced_all,
                "dev": res_forced_dev,
                "test": res_forced_test,
                "token_script_competition": script_comp_forced,
            },
        }
        # If Svarah, also compute per-native-language LID & WER breakdown
        if ds_key == "svarah_en":
            l1_breakdown: Dict[str, Dict[str, Any]] = {}
            by_l1_auto: Dict[str, List[Dict[str, Any]]] = {}
            by_l1_forced: Dict[str, List[Dict[str, Any]]] = {}
            for a_item, f_item in zip(auto_out, forced_out):
                l1 = a_item.get("extra", {}).get("primary_language", "") or "Unknown"
                by_l1_auto.setdefault(l1, []).append(a_item)
                by_l1_forced.setdefault(l1, []).append(f_item)
            for l1, a_list in sorted(by_l1_auto.items(), key=lambda kv: -len(kv[1])):
                f_list = by_l1_forced[l1]
                sa = score_outputs(a_list)
                sf = score_outputs(f_list)
                l1_breakdown[l1] = {
                    "clips": len(a_list),
                    "auto_lid_accuracy_pct": sa["lid_accuracy_pct"],
                    "auto_wer": sa["wer"],
                    "forced_en_wer": sf["wer"],
                    "wer_penalty_from_lid": round(sa["wer"] - sf["wer"], 2),
                }
            full_results[ds_key]["per_native_language_breakdown"] = l1_breakdown

        print(
            f"  [{ds_key}] Auto-LID WER={res_auto_all['wer']}% (CER={res_auto_all['cer']}%, LID-Acc={res_auto_all['lid_accuracy_pct']}%) | "
            f"Forced-{forced_lang} WER={res_forced_all['wer']}% (CER={res_forced_all['cer']}%)",
            flush=True,
        )
        with open("/data/eval_cache/phase2_partial_eval.json", "w", encoding="utf-8") as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)
        vol.commit()

    # ========================================================================
    # 5. Silence & Synthetic Hiss Hallucination Evaluation (100 clips)
    # ========================================================================
    print("\n=== Evaluating 100 Silence & Hiss Clips (50 digital silence + 50 faint Gaussian hiss) ===", flush=True)
    rng = np.random.RandomState(42)
    silence_clips = []
    for i in range(100):
        if i < 50:
            s = np.zeros(SAMPLE_RATE * 4, dtype=np.float32)
        else:
            s = (rng.randn(SAMPLE_RATE * 4) * 1e-4).astype(np.float32)
        silence_clips.append({
            "idx": i,
            "text": "",
            "target_lang": "hi",
            "duration_sec": 4.0,
            "samples": s,
            "extra": {"type": "digital_silence" if i < 50 else "faint_hiss"},
        })
    sil_out, sil_step0, _ = decode_dataset_gpu(silence_clips, forced_lang=None, batch_size=25, collect_step0=True)
    invented_words_gpu = sum(len(x["hyp"].split()) for x in sil_out)
    step0_archive["silence_h0"] = sil_step0["h0"]
    step0_archive["silence_h0q"] = sil_step0["h0q"]
    step0_archive["silence_lid_logits"] = sil_step0["lid_logits"]

    # ========================================================================
    # 6. Native C++ `needle` 5-Beam Engine Verification on `.cact`
    # ========================================================================
    print("\n=== Running Native C++ `needle` 5-Beam Engine Verification on `.cact` ===", flush=True)
    cact_verification: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for ds_key, forced_lang, _, _, _ in eval_plan:
            sub_clips = datasets_map[ds_key][:60]
            auto_hyps = []
            forced_hyps = []
            for c in sub_clips:
                wav_path = os.path.join(tmpdir, f"{ds_key}_{c['idx']}.wav")
                write_wav_16k(wav_path, c["samples"])
                # Auto-LID in needle C++
                proc_auto = subprocess.run(
                    [needle_bin, "--model", cact_path, "--audio", wav_path, "--threads", "4"],
                    capture_output=True, text=True, check=True,
                )
                obj_auto = json.loads(proc_auto.stdout.strip().splitlines()[-1])
                auto_hyps.append({
                    "idx": c["idx"],
                    "ref": c["text"],
                    "hyp": obj_auto.get("text", "").strip(),
                    "target_lang": c["target_lang"],
                    "pred_lang": obj_auto.get("language", "hi"),
                    "pred_tag_id": cfg.tag(obj_auto.get("language", "hi")) if obj_auto.get("language", "hi") in lang_names else -1,
                    "duration_sec": c["duration_sec"],
                    "extra": c.get("extra", {}),
                })
                # Forced-LID in needle C++
                proc_forced = subprocess.run(
                    [needle_bin, "--model", cact_path, "--audio", wav_path, "--audio-language", forced_lang, "--threads", "4"],
                    capture_output=True, text=True, check=True,
                )
                obj_forced = json.loads(proc_forced.stdout.strip().splitlines()[-1])
                forced_hyps.append({
                    "idx": c["idx"],
                    "ref": c["text"],
                    "hyp": obj_forced.get("text", "").strip(),
                    "target_lang": c["target_lang"],
                    "pred_lang": obj_forced.get("language", forced_lang),
                    "pred_tag_id": cfg.tag(forced_lang),
                    "duration_sec": c["duration_sec"],
                    "extra": c.get("extra", {}),
                })
            s_auto = score_outputs(auto_hyps)
            s_forced = score_outputs(forced_hyps)
            cact_verification[ds_key] = {
                "num_clips": len(sub_clips),
                "auto_lid_wer": s_auto["wer"],
                "auto_lid_cer": s_auto["cer"],
                "auto_lid_accuracy_pct": s_auto["lid_accuracy_pct"],
                "forced_lid_wer": s_forced["wer"],
                "forced_lid_cer": s_forced["cer"],
            }
            print(
                f"  [needle C++ 5-beam | {ds_key} (60 clips)] Auto WER={s_auto['wer']}% (LID-Acc={s_auto['lid_accuracy_pct']}%) | "
                f"Forced-{forced_lang} WER={s_forced['wer']}%",
                flush=True,
            )

        # Also test 20 silence/hiss clips in C++ needle engine
        cact_sil_words = 0
        for i in [0, 1, 2, 3, 4, 50, 51, 52, 53, 54]:
            wav_path = os.path.join(tmpdir, f"sil_{i}.wav")
            write_wav_16k(wav_path, silence_clips[i]["samples"])
            proc_sil = subprocess.run(
                [needle_bin, "--model", cact_path, "--audio", wav_path, "--threads", "4"],
                capture_output=True, text=True, check=True,
            )
            obj_sil = json.loads(proc_sil.stdout.strip().splitlines()[-1])
            cact_sil_words += len(obj_sil.get("text", "").strip().split())

    # Save step-0 hidden states archive to Modal Volume AND return serialized bytes
    step0_npz_path = "/data/eval_cache/phase2_step0_states.npz"
    np.savez_compressed(step0_npz_path, **step0_archive)
    vol.commit()
    with open(step0_npz_path, "rb") as f:
        step0_npz_bytes = f.read()

    summary = {
        "phase": "Phase 2 Baseline Evaluation & Diagnostic Extraction",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_device": str(jax.devices()[0]),
        "total_elapsed_sec": round(time.time() - t0, 2),
        "checkpoints": {
            "safetensors": "shruti.safetensors (sha256: 5c43105db153a47d02c47131d37d066343f04f1c3ff82315ff5da1ce264fbfa7)",
            "cact": "shruti.cact (sha256: b189e954aaed4a9fd41acbad3ac05e486381e96325b22a4c695b9ea5153d7065)",
        },
        "datasets": full_results,
        "silence_hallucination_test": {
            "num_clips": 100,
            "digital_silence_clips": 50,
            "faint_hiss_clips": 50,
            "gpu_ste_invented_words": int(invented_words_gpu),
            "cact_needle_invented_words_sample10": int(cact_sil_words),
        },
        "cact_needle_5beam_verification": cact_verification,
    }

    with open("/data/eval_cache/phase2_baseline_eval.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    vol.commit()

    return {
        "summary": summary,
        "step0_npz_bytes": step0_npz_bytes,
    }


@app.local_entrypoint()
def main():
    out_dir = ROOT_LOCAL / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_phase2_evaluation.remote()
    summary = res["summary"]
    npz_bytes = res["step0_npz_bytes"]

    json_path = out_dir / "phase2_baseline_eval.json"
    npz_path = out_dir / "phase2_step0_states.npz"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(npz_path, "wb") as f:
        f.write(npz_bytes)

    print(f"\nSaved Phase 2 JSON report -> {json_path}")
    print(f"Saved Step-0 Hidden States NPZ ({len(npz_bytes)/1e6:.2f} MB) -> {npz_path}")
