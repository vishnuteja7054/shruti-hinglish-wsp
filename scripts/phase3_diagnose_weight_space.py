#!/usr/bin/env python3
"""Phase 3 Weight-Space Diagnosis & Mathematical Hypothesis Formulation for Shruti WSP Research.

Correlates:
  1. Phase 1 exact weight-space geometry (`reports/phase1_weight_inspection.json` + `shruti.safetensors`)
  2. Phase 2 baseline evaluation metrics & failure taxonomy (`reports/phase2_baseline_eval.json`)
  3. Phase 2 empirical step-0 decoder representations h_0, aq(h_0), and LID logits (`reports/phase2_step0_states.npz`)

Performs rigorous mathematical decomposition of:
  - D1 (Step-0 Language ID Failure Mechanism):
      Decomposes step-0 quantized representations h_{0,q}(x) = mu_0 + r(x) into the global audio-independent
      BOS prior mu_0 and the acoustic-conditioned residual r(x). Computes why <h_{0,q}(x), e_en - e_hi>
      misroutes Indian English (Svarah) to <|hi|> or European tags, and derives closed-form Weight-Space
      Programming vectors on the Dev split ONLY (with strictly held-out Test split verification).
  - D2 (Hinglish & Hindi Script-Competition Norm Asymmetry):
      Correlates the 11.4% L2 norm deficit of grafted Devanagari tokens (8192..8905, mean norm 62.31)
      relative to Latin BPE tokens (260..8191, mean norm 70.33) with empirical Dev->Latin substitution
      rates and logit margins on MUCS Hinglish.
  - D3 (Devanagari Look-alike & Matra/Nasalization High-Cosine Subspace):
      Correlates empirical word/character substitutions in FLEURS Hindi, Lahaja, and MUCS Hinglish with
      pairwise token embedding cosines (0.85 - 0.93) and L2 norm imbalances.
  - D4 (Encoder/Decoder Gate & Engram Activation Sensitivity):
      Analyzes gate scaling and engram table norm geometry for potential acoustic/lexical sharpening.

Outputs:
  - `reports/phase3_diagnosis.json`: Complete quantitative diagnostic tables & closed-form WSP parameters
  - `reports/phase3_wsp_vectors.npz`: Exact Dev-calibrated perturbation vectors for Phase 4 experiments
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shruti_io import load_safetensors_flat, RefTokenizer, WHISTLE_SMALL_CFG


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    ckpt_path = ROOT / "checkpoints" / "original" / "shruti.safetensors"
    cact_path = ROOT / "checkpoints" / "original" / "shruti.cact"
    p1_path = ROOT / "reports" / "phase1_weight_inspection.json"
    p2_json_path = ROOT / "reports" / "phase2_baseline_eval.json"
    p2_npz_path = ROOT / "reports" / "phase2_step0_states.npz"

    assert p2_json_path.exists(), f"Missing {p2_json_path}"
    assert p2_npz_path.exists(), f"Missing {p2_npz_path}"

    flat = load_safetensors_flat(str(ckpt_path))
    tok = RefTokenizer.from_cact(str(cact_path))
    with open(p1_path, "r", encoding="utf-8") as f:
        p1 = json.load(f)
    with open(p2_json_path, "r", encoding="utf-8") as f:
        p2 = json.load(f)
    step0 = np.load(str(p2_npz_path), allow_pickle=False)

    E = flat["embedding/embedding"].astype(np.float32)  # [8914, 512]
    lang_names = list(WHISTLE_SMALL_CFG["languages"])   # ['en', 'de', 'fr', 'es', 'it', 'nl', 'pl', 'hi']
    text_vocab = WHISTLE_SMALL_CFG["text"]              # 8906
    E_lang = E[text_vocab:text_vocab + 8].copy()        # [8, 512]
    e_en = E[8906].copy()
    e_hi = E[8913].copy()

    # ========================================================================
    # Diagnosis 1: Step-0 Language ID Subspace & Decision Boundary Analysis
    # ========================================================================
    ds_Order = ["fleurs_hi", "lahaja_hi", "mucs_hinglish", "svarah_en", "librispeech_en"]
    dev_slices = {
        "fleurs_hi": (0, 200),
        "lahaja_hi": (0, 300),
        "mucs_hinglish": (0, 300),
        "svarah_en": (0, 300),
        "librispeech_en": (0, 150),
    }
    test_slices = {
        "fleurs_hi": (200, 418),
        "lahaja_hi": (300, 600),
        "mucs_hinglish": (300, 600),
        "svarah_en": (300, 600),
        "librispeech_en": (150, 300),
    }
    target_langs = {
        "fleurs_hi": "hi",
        "lahaja_hi": "hi",
        "mucs_hinglish": "hi",
        "svarah_en": "en",
        "librispeech_en": "en",
    }

    # Global audio-independent BOS prior mu_0 computed strictly on Dev splits
    all_dev_h0q = np.concatenate(
        [step0[f"{k}_h0q"][dev_slices[k][0]:dev_slices[k][1]] for k in ds_Order],
        axis=0,
    )
    mu_0 = np.mean(all_dev_h0q, axis=0)
    mu_0_norm = float(np.linalg.norm(mu_0))
    mu_0_unit = mu_0 / max(mu_0_norm, 1e-12)

    # Analyze per-dataset step-0 hidden state geometry & decompose logit margin
    w_orig = e_en - e_hi
    prior_bias_b0 = float(np.dot(mu_0, w_orig))

    d1_datasets: Dict[str, Any] = {}
    for k in ds_Order + ["silence"]:
        h0q = step0[f"{k}_h0q"].astype(np.float32)
        logits8 = h0q @ E_lang.T  # [N, 8]
        preds = np.argmax(logits8, axis=-1)
        pred_dist = {lang_names[i]: int(np.sum(preds == i)) for i in range(8) if np.sum(preds == i) > 0}

        r_x = h0q - mu_0[None, :]
        acoustic_signal = r_x @ w_orig
        total_margin = logits8[:, 0] - logits8[:, 7]
        eu_other_max = np.max(logits8[:, 1:7], axis=-1)

        d1_datasets[k] = {
            "num_clips": int(len(h0q)),
            "mean_h0q_l2_norm": round(float(np.mean(np.linalg.norm(h0q, axis=-1))), 4),
            "mean_residual_l2_norm": round(float(np.mean(np.linalg.norm(r_x, axis=-1))), 4),
            "cos_with_mu0_mean": round(float(np.mean([cosine(v, mu_0) for v in h0q])), 4),
            "prior_bias_b0_en_minus_hi": round(prior_bias_b0, 4),
            "acoustic_signal_en_minus_hi_mean": round(float(np.mean(acoustic_signal)), 4),
            "acoustic_signal_en_minus_hi_std": round(float(np.std(acoustic_signal)), 4),
            "total_margin_en_minus_hi_mean": round(float(np.mean(total_margin)), 4),
            "total_margin_en_minus_hi_std": round(float(np.std(total_margin)), 4),
            "eu_other_max_logit_mean": round(float(np.mean(eu_other_max)), 4),
            "pred_lang_distribution": pred_dist,
        }

    # Construct Dev-only Discriminant Directions for Step-0 Language ID
    # Class +1 (English): svarah_en[dev] + librispeech_en[dev]
    # Class -1 (Hindi/Hinglish): fleurs_hi[dev] + lahaja_hi[dev] + mucs_hinglish[dev]
    h0q_en_dev = np.concatenate([
        step0["svarah_en_h0q"][0:300],
        step0["librispeech_en_h0q"][0:150],
    ], axis=0)
    h0q_hi_dev = np.concatenate([
        step0["fleurs_hi_h0q"][0:200],
        step0["lahaja_hi_h0q"][0:300],
        step0["mucs_hinglish_h0q"][0:300],
    ], axis=0)

    # Also compute Svarah-specific vs Hindi-specific dev mean difference
    mu_en_dev = 0.7 * np.mean(step0["svarah_en_h0q"][0:300], axis=0) + 0.3 * np.mean(step0["librispeech_en_h0q"][0:150], axis=0)
    mu_hi_dev = (
        0.4 * np.mean(step0["fleurs_hi_h0q"][0:200], axis=0)
        + 0.4 * np.mean(step0["lahaja_hi_h0q"][0:300], axis=0)
        + 0.2 * np.mean(step0["mucs_hinglish_h0q"][0:300], axis=0)
    )
    delta_mu_dev = mu_en_dev - mu_hi_dev
    u_md = delta_mu_dev / max(float(np.linalg.norm(delta_mu_dev)), 1e-12)

    # Regularized Fisher discriminant on Dev split
    cov_en = np.cov(h0q_en_dev, rowvar=False)
    cov_hi = np.cov(h0q_hi_dev, rowvar=False)
    # Test multiple regularization strengths on Dev split
    best_dev_score = -1e9
    best_cfg = None
    grid_search_results = []

    for reg in [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        cov_w = 0.5 * (cov_en + cov_hi) + reg * np.eye(512, dtype=np.float64)
        v_raw = np.linalg.solve(cov_w, delta_mu_dev.astype(np.float64)).astype(np.float32)
        u_fisher = v_raw / max(float(np.linalg.norm(v_raw)), 1e-12)

        for gamma in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]:
            for beta in [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
                # Perturbation vector delta_w applied symmetrically (+0.5 to en, -0.5 to hi)
                # and European unused tag attenuation (-2.0 * mu_0_unit on de..pl)
                delta_w = gamma * u_fisher + beta * mu_0_unit
                E_lang_mod = E_lang.copy()
                E_lang_mod[0] = e_en + 0.5 * delta_w
                E_lang_mod[7] = e_hi - 0.5 * delta_w
                for eu_idx in range(1, 7):
                    E_lang_mod[eu_idx] = E_lang[eu_idx] - 2.0 * mu_0_unit

                # Evaluate Dev LID accuracy across all 5 datasets
                dev_accs = {}
                for k in ds_Order:
                    lo, hi_idx = dev_slices[k]
                    hq_sub = step0[f"{k}_h0q"][lo:hi_idx]
                    l_mod = hq_sub @ E_lang_mod.T
                    p_mod = np.argmax(l_mod, axis=-1)
                    target_idx = 0 if target_langs[k] == "en" else 7
                    dev_accs[k] = float(np.mean(p_mod == target_idx)) * 100.0

                # Objective on Dev: maximize min accuracy across speech datasets + weighted average
                # Penalty on norm of delta_w to keep input embedding perturbation minimal
                norm_dw = float(np.linalg.norm(delta_w))
                mean_acc = np.mean([
                    dev_accs["fleurs_hi"],
                    dev_accs["lahaja_hi"],
                    dev_accs["mucs_hinglish"],
                    dev_accs["svarah_en"],
                    dev_accs["librispeech_en"],
                ])
                min_acc = min(dev_accs.values())
                score = mean_acc + 0.5 * min_acc - 0.02 * norm_dw
                if score > best_dev_score:
                    best_dev_score = score
                    best_cfg = {
                        "reg": reg,
                        "gamma": gamma,
                        "beta": beta,
                        "u_fisher": u_fisher,
                        "delta_w": delta_w,
                        "norm_delta_w": round(norm_dw, 4),
                        "norm_delta_e_en": round(norm_dw * 0.5, 4),
                        "cos_e_en_orig_new": round(cosine(e_en, e_en + 0.5 * delta_w), 6),
                        "cos_e_hi_orig_new": round(cosine(e_hi, e_hi - 0.5 * delta_w), 6),
                        "dev_lid_acc": {k: round(v, 2) for k, v in dev_accs.items()},
                    }

    # Now evaluate the Dev-selected best_cfg on the strictly held-out Test splits!
    E_lang_best = E_lang.copy()
    dw_best = best_cfg["delta_w"]
    E_lang_best[0] = e_en + 0.5 * dw_best
    E_lang_best[7] = e_hi - 0.5 * dw_best
    for eu_idx in range(1, 7):
        E_lang_best[eu_idx] = E_lang[eu_idx] - 2.0 * mu_0_unit

    test_lid_comparison = {}
    for k in ds_Order:
        lo, hi_idx = test_slices[k]
        hq_test = step0[f"{k}_h0q"][lo:hi_idx]
        hq_all = step0[f"{k}_h0q"]
        target_idx = 0 if target_langs[k] == "en" else 7

        orig_test_preds = np.argmax(hq_test @ E_lang.T, axis=-1)
        new_test_preds = np.argmax(hq_test @ E_lang_best.T, axis=-1)

        orig_all_preds = np.argmax(hq_all @ E_lang.T, axis=-1)
        new_all_preds = np.argmax(hq_all @ E_lang_best.T, axis=-1)

        test_lid_comparison[k] = {
            "target_lang": target_langs[k],
            "orig_dev_lid_acc_pct": round(
                float(np.mean(np.argmax(step0[f"{k}_h0q"][dev_slices[k][0]:dev_slices[k][1]] @ E_lang.T, axis=-1) == target_idx)) * 100.0, 2
            ),
            "new_dev_lid_acc_pct": best_cfg["dev_lid_acc"][k],
            "orig_test_lid_acc_pct": round(float(np.mean(orig_test_preds == target_idx)) * 100.0, 2),
            "new_test_lid_acc_pct": round(float(np.mean(new_test_preds == target_idx)) * 100.0, 2),
            "orig_all_lid_acc_pct": round(float(np.mean(orig_all_preds == target_idx)) * 100.0, 2),
            "new_all_lid_acc_pct": round(float(np.mean(new_all_preds == target_idx)) * 100.0, 2),
        }

    # Also test a pure bias/norm calibration variant (H1b: only mu_0_unit shift, zero covariance)
    best_bias_score = -1e9
    best_bias_beta = 0.0
    for beta in np.linspace(-2.0, 6.0, 161):
        E_lang_b = E_lang.copy()
        E_lang_b[0] = e_en + 0.5 * float(beta) * mu_0_unit
        E_lang_b[7] = e_hi - 0.5 * float(beta) * mu_0_unit
        for eu_idx in range(1, 7):
            E_lang_b[eu_idx] = E_lang[eu_idx] - 2.0 * mu_0_unit
        accs = {}
        for k in ds_Order:
            lo, hi_idx = dev_slices[k]
            p_b = np.argmax(step0[f"{k}_h0q"][lo:hi_idx] @ E_lang_b.T, axis=-1)
            t_idx = 0 if target_langs[k] == "en" else 7
            accs[k] = float(np.mean(p_b == t_idx)) * 100.0
        sc = np.mean(list(accs.values())) + 0.5 * min(accs.values())
        if sc > best_bias_score:
            best_bias_score = sc
            best_bias_beta = round(float(beta), 4)

    # ========================================================================
    # Diagnosis 2: Devanagari vs. Latin Norm Asymmetry on MUCS Hinglish
    # ========================================================================
    norms = np.linalg.norm(E, axis=1)
    latin_norms = norms[260:8192]
    dev_norms = norms[8192:8906]
    norm_ratio_lat_to_dev = float(np.mean(latin_norms) / np.mean(dev_norms))

    mucs_stats = p2["datasets"]["mucs_hinglish"]
    d2_hinglish = {
        "mean_latin_token_norm": round(float(np.mean(latin_norms)), 4),
        "std_latin_token_norm": round(float(np.std(latin_norms)), 4),
        "mean_devanagari_token_norm": round(float(np.mean(dev_norms)), 4),
        "std_devanagari_token_norm": round(float(np.std(dev_norms)), 4),
        "latin_to_devanagari_norm_ratio": round(norm_ratio_lat_to_dev, 5),
        "mucs_auto_script_word_counts": mucs_stats["auto_lid"]["all"]["script_word_counts"],
        "mucs_forced_hi_script_word_counts": mucs_stats["forced_lid"]["all"]["script_word_counts"],
        "mucs_auto_token_competition": mucs_stats["auto_lid"]["token_script_competition"],
        "mucs_forced_hi_token_competition": mucs_stats["forced_lid"]["token_script_competition"],
        "mucs_dev_to_en_word_subs_auto": mucs_stats["auto_lid"]["all"]["failure_taxonomy"]["script_mismatch_dev_to_en"],
        "mucs_en_to_dev_word_subs_auto": mucs_stats["auto_lid"]["all"]["failure_taxonomy"]["script_mismatch_en_to_dev"],
        "mucs_dev_to_en_word_subs_forced": mucs_stats["forced_lid"]["all"]["failure_taxonomy"]["script_mismatch_dev_to_en"],
        "mucs_en_to_dev_word_subs_forced": mucs_stats["forced_lid"]["all"]["failure_taxonomy"]["script_mismatch_en_to_dev"],
    }

    # ========================================================================
    # Diagnosis 3: High-Confusion Devanagari Pairs & Embedding Geometry
    # ========================================================================
    # Aggregate top Devanagari substitutions across fleurs_hi, lahaja_hi, and mucs_hinglish (Dev splits Only for WSP design!)
    dev_hindi_subs = {}
    for k in ["fleurs_hi", "lahaja_hi", "mucs_hinglish"]:
        top_subs = p2["datasets"][k]["forced_lid"]["dev"]["failure_taxonomy"]["top_word_substitutions"]
        for item in top_subs:
            pair = (item["ref"], item["hyp"])
            dev_hindi_subs[pair] = dev_hindi_subs.get(pair, 0) + item["count"]

    piece_to_id = {tok.pieces[i]: i for i in range(8192, 8906)}
    d3_pair_geometry = []
    for (rw, hw), cnt in sorted(dev_hindi_subs.items(), key=lambda kv: -kv[1])[:30]:
        tid_r = piece_to_id.get("▁" + rw) or piece_to_id.get(rw)
        tid_h = piece_to_id.get("▁" + hw) or piece_to_id.get(hw)
        entry: Dict[str, Any] = {
            "ref_word": rw,
            "hyp_word": hw,
            "dev_split_error_count": cnt,
            "ref_tid": tid_r,
            "hyp_tid": tid_h,
        }
        if tid_r is not None and tid_h is not None:
            er = E[tid_r]
            eh = E[tid_h]
            entry["cosine_similarity"] = round(cosine(er, eh), 4)
            entry["ref_l2_norm"] = round(float(np.linalg.norm(er)), 4)
            entry["hyp_l2_norm"] = round(float(np.linalg.norm(eh)), 4)
            entry["hyp_minus_ref_norm"] = round(float(np.linalg.norm(eh) - np.linalg.norm(er)), 4)
        d3_pair_geometry.append(entry)

    # Save WSP perturbation vectors for Phase 4
    wsp_npz_path = ROOT / "reports" / "phase3_wsp_vectors.npz"
    np.savez_compressed(
        str(wsp_npz_path),
        mu_0=mu_0,
        mu_0_unit=mu_0_unit,
        u_md=u_md,
        u_fisher=best_cfg["u_fisher"],
        delta_w_best=best_cfg["delta_w"],
        E_lang_orig=E_lang,
        E_lang_best=E_lang_best,
    )

    diagnosis_report = {
        "phase": "Phase 3 Weight-Space Diagnosis & Mathematical Hypotheses",
        "diagnosis_1_step0_lid": {
            "mathematical_decomposition": (
                "z_en(x) - z_hi(x) = <mu_0, e_en - e_hi> + <r(x), e_en - e_hi>, where mu_0 = E[h_{0,q}] "
                "is the deterministic BOS step-0 prior vector (||mu_0||_2 = "
                f"{mu_0_norm:.4f}) and r(x) = h_{{0,q}}(x) - mu_0 is the acoustic-conditioned residual."
            ),
            "mu_0_l2_norm": round(mu_0_norm, 4),
            "e_en_l2_norm": round(float(np.linalg.norm(e_en)), 4),
            "e_hi_l2_norm": round(float(np.linalg.norm(e_hi)), 4),
            "cos_e_en_e_hi": round(cosine(e_en, e_hi), 4),
            "prior_bias_b0_en_minus_hi": round(prior_bias_b0, 4),
            "per_dataset_step0_geometry": d1_datasets,
            "dev_calibrated_fisher_wsp": {
                k: v for k, v in best_cfg.items() if k not in ("u_fisher", "delta_w")
            },
            "dev_calibrated_pure_bias_beta": best_bias_beta,
            "heldout_test_lid_verification": test_lid_comparison,
        },
        "diagnosis_2_hinglish_script_competition": d2_hinglish,
        "diagnosis_3_devanagari_confusion_geometry": d3_pair_geometry,
        "testable_hypotheses": [
            {
                "id": "H1 (Step-0 Language Tag Subspace Steering & European Tag Suppression)",
                "target_tensors": ["embedding/embedding[8906:8914, :]"],
                "num_params_changed": 4096,
                "pct_of_model": "0.00738%",
                "mathematical_formulation": (
                    "E'[8906] = E[8906] + 0.5 * (gamma * u_Fisher + beta * mu_0_unit), "
                    "E'[8913] = E[8913] - 0.5 * (gamma * u_Fisher + beta * mu_0_unit), "
                    "E'[8907:8913] = E[8907:8913] - 2.0 * mu_0_unit"
                ),
                "expected_impact": (
                    "Eliminates step-0 Hindi/European misrouting on Svarah Indian English and LibriSpeech "
                    "while preserving >99% Hindi Auto-LID accuracy on FLEURS, Lahaja, and MUCS Hinglish, "
                    "with cos(e_orig, e_new) > 0.999 so Forced-LID decoding is unperturbed."
                ),
            },
            {
                "id": "H2 (Devanagari Output Projection Norm Equalization for Hinglish & Hindi)",
                "target_tensors": ["embedding/embedding[8192:8906, :]"],
                "num_params_changed": 365568,
                "pct_of_model": "0.658%",
                "mathematical_formulation": (
                    "E'[8192:8906] = alpha_dev * E[8192:8906] with alpha_dev in [1.03, 1.09] calibrated on Dev split. "
                    "Because Decoder Layer 0 applies scale-invariant _rms_unit(x) to input token embeddings "
                    "whereas the output projection logits = aq(h) @ E.T is linear in ||E[t]||_2, radial scaling "
                    "boosts Devanagari output logits by alpha_dev while keeping input directions invariant."
                ),
                "expected_impact": (
                    "Reduces Latin over-prediction and Devanagari->Latin script mismatches on MUCS Hinglish "
                    "and improves Devanagari word recall on Hindi benchmarks."
                ),
            },
            {
                "id": "H3 (Contrastive Pairwise Decoupling & Norm Balancing of High-Confusion Devanagari Tokens)",
                "target_tensors": ["embedding/embedding[selected_devanagari_ids, :]"],
                "num_params_changed": "5120 - 15360",
                "pct_of_model": "< 0.03%",
                "mathematical_formulation": (
                    "For confused Devanagari token pairs (i, j) with cos(e_i, e_j) > 0.85 and systematic "
                    "substitution bias on Dev splits, apply small orthogonal repulsion "
                    "e_i' = e_i + eta * (e_i - <e_i, hat{e}_j> hat{e}_j) and norm-balance adjustment."
                ),
                "expected_impact": (
                    "Reduces recurring Hindi function-word and nasalization/matra confusions "
                    "(e.g., है/हैं, की/के, मैं/में) without altering acoustic encoder weights."
                ),
            },
            {
                "id": "H4 (Decoder Cross-Attention & Engram Gate Calibration)",
                "target_tensors": ["stack/layers/block/attn_gate", "engrams_0/embedding", "engrams_1/embedding"],
                "num_params_changed": "16 - 2097152",
                "mathematical_formulation": (
                    "Test whether scaling final decoder cross-attention gates or damping noisy bigram/trigram "
                    "engram injections improves accented speech robustness or induces regressions."
                ),
                "expected_impact": (
                    "Provides a direct falsifiable test of decoder internal gate/engram sensitivity vs. "
                    "output-space embedding geometry."
                ),
            },
        ],
    }

    out_json = ROOT / "reports" / "phase3_diagnosis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(diagnosis_report, f, ensure_ascii=False, indent=2)
    print(f"Saved Phase 3 diagnosis JSON -> {out_json}")
    print(f"Saved Phase 3 WSP vectors NPZ -> {wsp_npz_path}")


if __name__ == "__main__":
    main()
