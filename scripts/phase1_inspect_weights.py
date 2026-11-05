#!/usr/bin/env python3
"""Phase 1: Deep Weight-Space Inspection of Udgam Labs Shruti (55,517,513 params, 116 tensors).

Computes exact weight distributions, singular values, CQ-W4 quantization fidelity,
gate activations, Sinkhorn lane-mixing matrices, HadamardMLP orthogonality drift,
causal tap lag profiles, embedding subspace geometry, language-ID tag geometry,
and Engram hash-slot utilization.
"""
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, "/home/rgukt/Music/SHRUTHI/HINGLISH/src")
from shruti_io import (
    RefTokenizer,
    WhistleConfig,
    cq_quantize_dequantize_np,
    load_safetensors,
    sha256_file,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def sinkhorn_knopp(logits: np.ndarray, steps: int = 20) -> np.ndarray:
    """Exact numpy equivalent of _sinkhorn in whistle.py."""
    log_a = logits - np.max(logits, axis=-1, keepdims=True)
    for _ in range(steps):
        log_a = log_a - np.log(np.sum(np.exp(log_a), axis=-1, keepdims=True) + 1e-12)
        log_a = log_a - np.log(np.sum(np.exp(log_a), axis=-2, keepdims=True) + 1e-12)
    return np.exp(log_a)


def engram_slot_for_ngram(tokens: List[int], orders=(2, 3), slots: int = 18432, hashes: int = 2, pad: int = 0, bos: int = 2) -> List[Tuple[int, int, int]]:
    """Compute the exact (table_idx, order, slot_id) in engrams_0/1 for the final position of a token sequence."""
    arr = np.asarray(tokens, dtype=np.uint32)
    T = len(tokens)
    results = []
    t_idx = 0
    for k in orders:
        shifted = []
        for j in range(k):
            idx = T - 1 - j
            shifted.append(int(arr[idx]) if idx >= 0 else pad)
        for h in range(hashes):
            mix = np.uint32(2166136261 + h * 1000003 + k * 999983)
            for j, tok_id in enumerate(shifted):
                mix = (mix ^ (np.uint32(tok_id) * np.uint32(0x9E3779B9 + j * 0x85EBCA6B))) * np.uint32(0x01000193)
            slot = int(mix % np.uint32(slots - 1)) + 1
            if shifted[0] == pad or shifted[0] == bos:
                slot = 0
            results.append((t_idx, k, slot))
            t_idx += 1
    return results


def effective_rank(mat: np.ndarray) -> Tuple[float, float, float]:
    """Return (sigma_max, sigma_min, effective_rank) of a 2D matrix."""
    s = np.linalg.svd(mat.astype(np.float32), compute_uv=False)
    s_pos = s[s > 1e-12]
    if len(s_pos) == 0:
        return 0.0, 0.0, 0.0
    p = s_pos / np.sum(s_pos)
    ent = -np.sum(p * np.log(p + 1e-15))
    return float(s[0]), float(s[-1]), float(np.exp(ent))


def main():
    st_path = "/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/original/shruti.safetensors"
    cact_path = "/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/original/shruti.cact"

    print("Loading shruti.safetensors and shruti.cact...")
    tensors, meta = load_safetensors(st_path)
    cfg_dict = json.loads(meta["config"])
    run_dict = json.loads(meta["run"])
    cfg = WhistleConfig.from_saved(cfg_dict)
    tok = RefTokenizer.from_cact(cact_path)

    report: Dict[str, Any] = {
        "hashes": {
            "shruti.safetensors": sha256_file(st_path),
            "shruti.cact": sha256_file(cact_path),
        },
        "config": cfg_dict,
        "run_metadata": {
            "grafted_pieces_count": len(run_dict.get("pieces", [])),
            "first_10_grafted": run_dict.get("pieces", [])[:10],
            "last_10_grafted": run_dict.get("pieces", [])[-10:],
        },
        "tensors": {},
    }

    # 1. Per-tensor statistics
    for name in sorted(tensors.keys()):
        arr = tensors[name].astype(np.float32)
        info: Dict[str, Any] = {
            "shape": list(arr.shape),
            "size": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "l2_norm": float(np.linalg.norm(arr)),
            "abs_mean": float(np.mean(np.abs(arr))),
        }
        if arr.ndim == 2 and arr.size <= 5_000_000:
            s_max, s_min, erank = effective_rank(arr)
            info["sigma_max"] = s_max
            info["sigma_min"] = s_min
            info["effective_rank"] = erank
        report["tensors"][name] = info

    # 2. Gate and Hyper-Connection (LaneStack) analysis
    pe_gate_val = float(tensors["pe_gate"])
    enc_attn_gate = tensors["encoder/layers/block/attn_gate"].tolist()
    dec_attn_gate = tensors["stack/layers/block/attn_gate"].tolist()
    dec_cross_gate = tensors["stack/layers/block/cross_gate"].tolist()

    def analyze_mhc(prefix: str) -> Dict[str, Any]:
        L = 8
        lanes = 4
        pre_off = -np.log(lanes - 1.0)
        b_pre = tensors[f"{prefix}/mhc_b_pre"]
        b_post = tensors[f"{prefix}/mhc_b_post"]
        b_res = tensors[f"{prefix}/mhc_b_res"]
        a_pre = tensors[f"{prefix}/mhc_a_pre"]
        a_post = tensors[f"{prefix}/mhc_a_post"]
        a_res = tensors[f"{prefix}/mhc_a_res"]

        h_pre_base = sigmoid(b_pre + pre_off)
        h_post_base = 2.0 * sigmoid(b_post)
        H_res_base = np.array([sinkhorn_knopp(b_res[l]) for l in range(L)])
        diag_dominance = [float(np.mean(np.diag(H_res_base[l]))) for l in range(L)]

        return {
            "a_pre": [round(float(x), 4) for x in a_pre],
            "a_post": [round(float(x), 4) for x in a_post],
            "a_res": [round(float(x), 4) for x in a_res],
            "h_pre_base_per_layer": [[round(float(v), 4) for v in row] for row in h_pre_base],
            "h_post_base_per_layer": [[round(float(v), 4) for v in row] for row in h_post_base],
            "H_res_base_diag_mean": [round(d, 4) for d in diag_dominance],
            "H_res_base_matrices": [[[round(float(v), 4) for v in r] for r in mat] for mat in H_res_base],
        }

    report["gates_and_lanes"] = {
        "pe_gate": {"raw": pe_gate_val, "sigmoid": float(sigmoid(np.array(pe_gate_val)))},
        "encoder_attn_gate": {
            "raw": [round(x, 4) for x in enc_attn_gate],
            "sigmoid": [round(float(sigmoid(np.array(x))), 4) for x in enc_attn_gate],
        },
        "decoder_attn_gate": {
            "raw": [round(x, 4) for x in dec_attn_gate],
            "sigmoid": [round(float(sigmoid(np.array(x))), 4) for x in dec_attn_gate],
        },
        "decoder_cross_gate": {
            "raw": [round(x, 4) for x in dec_cross_gate],
            "sigmoid": [round(float(sigmoid(np.array(x))), 4) for x in dec_cross_gate],
        },
        "encoder_mhc": analyze_mhc("encoder"),
        "decoder_mhc": analyze_mhc("stack"),
    }

    # 3. HadamardMLP analysis (orthogonality drift, diagonal scales, conditioner norms)
    def analyze_hada(prefix: str) -> List[Dict[str, Any]]:
        out = []
        for l in range(8):
            row = {"layer": l}
            for k, dim_n in [("w1a", 16), ("w1b", 32), ("w2a", 16), ("w2b", 32), ("w3a", 16), ("w3b", 32)]:
                W = tensors[f"{prefix}/{k}"][l]
                orth_err = float(np.linalg.norm(W.T @ W - np.eye(dim_n, dtype=np.float32)))
                row[f"{k}_orth_err"] = round(orth_err, 4)
            for d in ("d1", "d2", "d3", "d4", "b2"):
                vec = tensors[f"{prefix}/{d}"][l]
                row[f"{d}_mean"] = round(float(np.mean(vec)), 4)
                row[f"{d}_std"] = round(float(np.std(vec)), 4)
            cv = tensors[f"{prefix}/cond_v"][l]
            cu = tensors[f"{prefix}/cond_u"][l]
            row["cond_v_norm"] = round(float(np.linalg.norm(cv)), 4)
            row["cond_u_norm"] = round(float(np.linalg.norm(cu)), 4)
            out.append(row)
        return out

    report["hadamard_mlp"] = {
        "encoder_mlp_0": analyze_hada("encoder/layers/block/hadamard_mlp_0"),
        "encoder_mlp_1": analyze_hada("encoder/layers/block/hadamard_mlp"),
        "decoder_mlp": analyze_hada("stack/layers/block/hadamard_mlp"),
    }

    # 4. Causal Temporal Conv Taps analysis
    q_taps = tensors["stack/layers/block/self_attn/q_taps"]  # [8, 3, 384]
    k_taps = tensors["stack/layers/block/self_attn/k_taps"]  # [8, 3, 96]
    v_taps = tensors["stack/layers/block/self_attn/v_taps"]  # [8, 3, 128]
    taps_summary = []
    for l in range(8):
        taps_summary.append({
            "layer": l,
            "q_taps_rms_t_minus_2_1_0": [round(float(np.sqrt(np.mean(q_taps[l, t] ** 2))), 4) for t in range(3)],
            "k_taps_rms_t_minus_2_1_0": [round(float(np.sqrt(np.mean(k_taps[l, t] ** 2))), 4) for t in range(3)],
            "v_taps_rms_t_minus_2_1_0": [round(float(np.sqrt(np.mean(v_taps[l, t] ** 2))), 4) for t in range(3)],
            "q_taps_mean_t_minus_2_1_0": [round(float(np.mean(q_taps[l, t])), 4) for t in range(3)],
            "k_taps_mean_t_minus_2_1_0": [round(float(np.mean(k_taps[l, t])), 4) for t in range(3)],
            "v_taps_mean_t_minus_2_1_0": [round(float(np.mean(v_taps[l, t])), 4) for t in range(3)],
        })
    report["causal_taps"] = {
        "decoder_qkv_taps": taps_summary,
        "engrams_0_taps_rms": [round(float(np.sqrt(np.mean(tensors["engrams_0/taps"][t] ** 2))), 4) for t in range(4)],
        "engrams_1_taps_rms": [round(float(np.sqrt(np.mean(tensors["engrams_1/taps"][t] ** 2))), 4) for t in range(4)],
        "engrams_0_taps_mean": [round(float(np.mean(tensors["engrams_0/taps"][t])), 4) for t in range(4)],
        "engrams_1_taps_mean": [round(float(np.mean(tensors["engrams_1/taps"][t])), 4) for t in range(4)],
    }

    # 5. Token Embedding & Language Identification Subspace Analysis
    emb = tensors["embedding/embedding"]  # [8914, 512]
    emb_q = cq_quantize_dequantize_np(emb, bits=4, group=128)
    emb_norms = np.linalg.norm(emb, axis=1)
    emb_unit = emb / np.maximum(emb_norms[:, None], 1e-12)

    special_norms = {tok.pieces[i]: round(float(emb_norms[i]), 4) for i in range(4)}
    byte_norms = emb_norms[4:260]
    latin_norms = emb_norms[260:8192]
    dev_norms = emb_norms[8192:8906]
    lang_norms = emb_norms[8906:8914]

    # Centroids
    latin_centroid = np.mean(emb[260:8192], axis=0)
    latin_centroid_u = latin_centroid / np.linalg.norm(latin_centroid)
    dev_centroid = np.mean(emb[8192:8906], axis=0)
    dev_centroid_u = dev_centroid / np.linalg.norm(dev_centroid)

    # Language tags (8906..8913)
    lang_names = list(cfg.languages)
    lang_ids = list(range(8906, 8914))
    lang_vecs = emb[lang_ids]
    lang_vecs_q = emb_q[lang_ids]
    lang_unit = emb_unit[lang_ids]

    lang_cos_matrix = (lang_unit @ lang_unit.T).tolist()
    lang_dot_matrix = (lang_vecs @ lang_vecs.T).tolist()

    lang_details = {}
    for idx, (lid, lname) in enumerate(zip(lang_ids, lang_names)):
        v = emb[lid]
        vq = emb_q[lid]
        u = emb_unit[lid]
        lang_details[lname] = {
            "token_id": lid,
            "piece": tok.pieces[lid],
            "l2_norm_fp32": round(float(np.linalg.norm(v)), 4),
            "l2_norm_cq4": round(float(np.linalg.norm(vq)), 4),
            "cq4_rel_err": round(float(np.linalg.norm(v - vq) / np.linalg.norm(v)), 4),
            "cos_with_bos": round(float(np.dot(u, emb_unit[2])), 4),
            "cos_with_eos": round(float(np.dot(u, emb_unit[1])), 4),
            "cos_with_latin_centroid": round(float(np.dot(u, latin_centroid_u)), 4),
            "cos_with_devanagari_centroid": round(float(np.dot(u, dev_centroid_u)), 4),
            "mean_dot_with_latin_tokens": round(float(np.mean(emb[260:8192] @ v)), 4),
            "mean_dot_with_devanagari_tokens": round(float(np.mean(emb[8192:8906] @ v)), 4),
        }

    # High-confusion Hindi word pairs in embedding space & Engram tables
    confusion_pairs = [
        ("▁है", "▁हैं"),
        ("▁मैं", "▁में"),
        ("▁की", "▁के"),
        ("▁का", "▁को"),
        ("▁पर", "▁पे"),
        ("▁वह", "▁वो"),
        ("▁और", "▁या"),
        ("▁नहीं", "▁ना"),
        ("ं", "ँ"),
        ("ी", "ि"),
        ("ू", "ु"),
    ]
    pair_diagnostics = []
    for p1, p2 in confusion_pairs:
        id1 = tok.p2id.get(p1)
        id2 = tok.p2id.get(p2)
        if id1 is not None and id2 is not None:
            cos_sim = float(np.dot(emb_unit[id1], emb_unit[id2]))
            l2_dist = float(np.linalg.norm(emb[id1] - emb[id2]))
            slots_1 = engram_slot_for_ngram([8913, id1])
            slots_2 = engram_slot_for_ngram([8913, id2])
            pair_diagnostics.append({
                "pair": f"{p1} vs {p2}",
                "ids": [id1, id2],
                "norms": [round(float(emb_norms[id1]), 4), round(float(emb_norms[id2]), 4)],
                "cosine_similarity": round(cos_sim, 4),
                "l2_distance": round(l2_dist, 4),
                "engram_slots_after_hi": [slots_1, slots_2],
            })

    report["embedding_analysis"] = {
        "partition_norms": {
            "special_0_3": special_norms,
            "bytes_4_259": {"mean": round(float(np.mean(byte_norms)), 4), "std": round(float(np.std(byte_norms)), 4)},
            "latin_260_8191": {
                "mean": round(float(np.mean(latin_norms)), 4),
                "std": round(float(np.std(latin_norms)), 4),
                "min": round(float(np.min(latin_norms)), 4),
                "max": round(float(np.max(latin_norms)), 4),
            },
            "devanagari_8192_8905": {
                "mean": round(float(np.mean(dev_norms)), 4),
                "std": round(float(np.std(dev_norms)), 4),
                "min": round(float(np.min(dev_norms)), 4),
                "max": round(float(np.max(dev_norms)), 4),
            },
            "languages_8906_8913": {
                "mean": round(float(np.mean(lang_norms)), 4),
                "std": round(float(np.std(lang_norms)), 4),
            },
        },
        "centroid_cosine_latin_vs_devanagari": round(float(np.dot(latin_centroid_u, dev_centroid_u)), 4),
        "language_tags": lang_details,
        "language_cosine_matrix": {
            "languages": lang_names,
            "matrix": [[round(float(x), 4) for x in row] for row in lang_cos_matrix],
        },
        "confusion_pairs": pair_diagnostics,
    }

    # 6. Engram Hash Memory Analysis
    engram_analysis = {}
    for site_idx, name in [(3, "engrams_0"), (7, "engrams_1")]:
        table = tensors[f"{name}/embedding"]  # [4, 18432, 128]
        slot_norms = np.linalg.norm(table, axis=-1)  # [4, 18432]
        kp = tensors[f"{name}/key_proj/kernel"]
        vp = tensors[f"{name}/value_proj/kernel"]
        engram_analysis[name] = {
            "decoder_layer_site": site_idx,
            "slot_0_norm_per_table": [round(float(slot_norms[t, 0]), 6) for t in range(4)],
            "nonzero_slots_mean_norm_per_table": [round(float(np.mean(slot_norms[t, 1:])), 4) for t in range(4)],
            "nonzero_slots_std_norm_per_table": [round(float(np.std(slot_norms[t, 1:])), 4) for t in range(4)],
            "nonzero_slots_max_norm_per_table": [round(float(np.max(slot_norms[t, 1:])), 4) for t in range(4)],
            "near_zero_slots_count_lt_0_05": [int(np.sum(slot_norms[t, 1:] < 0.05)) for t in range(4)],
            "key_proj_fro_norm": round(float(np.linalg.norm(kp)), 4),
            "value_proj_fro_norm": round(float(np.linalg.norm(vp)), 4),
        }
    report["engram_analysis"] = engram_analysis

    out_json = "/home/rgukt/Music/SHRUTHI/HINGLISH/reports/phase1_weight_inspection.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved complete Phase 1 weight inspection JSON to {out_json}")

    # Print key findings to stdout
    print("\n=== EMBEDDING PARTITION NORMS ===")
    print(json.dumps(report["embedding_analysis"]["partition_norms"], indent=2))
    print("\n=== LANGUAGE TAG DETAILS ===")
    print(json.dumps(report["embedding_analysis"]["language_tags"], indent=2))
    print("\n=== LANGUAGE COSINE MATRIX (en, de, fr, es, it, nl, pl, hi) ===")
    for lname, row in zip(lang_names, report["embedding_analysis"]["language_cosine_matrix"]["matrix"]):
        print(f"  {lname:>2s}: {row}")
    print("\n=== GATES & POSITIONAL ENCODING ===")
    print("  pe_gate:", report["gates_and_lanes"]["pe_gate"])
    print("  enc_attn_gate (sigmoid):", report["gates_and_lanes"]["encoder_attn_gate"]["sigmoid"])
    print("  dec_attn_gate (sigmoid):", report["gates_and_lanes"]["decoder_attn_gate"]["sigmoid"])
    print("  dec_cross_gate (sigmoid):", report["gates_and_lanes"]["decoder_cross_gate"]["sigmoid"])
    print("\n=== HINDI CONFUSION PAIRS ===")
    for p in pair_diagnostics:
        print(f"  {p['pair']:12s} | IDs={p['ids']} | norms={p['norms']} | cos={p['cosine_similarity']:.4f} | L2={p['l2_distance']:.4f}")
    print("\n=== ENGRAM SUMMARY ===")
    print(json.dumps(report["engram_analysis"], indent=2))


if __name__ == "__main__":
    main()
