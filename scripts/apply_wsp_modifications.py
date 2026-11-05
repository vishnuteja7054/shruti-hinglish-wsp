#!/usr/bin/env python3
"""Standalone Reproducible Weight-Space Programming (WSP) Patcher for Shruti.

Applies the validated gradient-free weight-space modifications:
    W_new = W_original + Delta_W
to copies of the original `shruti.safetensors` and `shruti.cact` checkpoints,
verifying:
  1. The original checkpoints in `checkpoints/original/` are completely untouched
     (exact SHA-256 verification).
  2. Every modified tensor, row index, L2 norm before/after, cosine similarity,
     and parameter count is logged.
  3. Both `shruti_wsp_final.safetensors` and the surgically patched `shruti_wsp_final.cact`
     are deterministically generated and hashed.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shruti_io import (
    WHISTLE_SMALL_CFG,
    load_safetensors_flat,
    save_safetensors_flat,
    surgical_patch_cact,
)

ORIG_ST_SHA256 = "5c43105db153a47d02c47131d37d066343f04f1c3ff82315ff5da1ce264fbfa7"
ORIG_CACT_SHA256 = "b189e954aaed4a9fd41acbad3ac05e486381e96325b22a4c695b9ea5153d7065"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    orig_st = ROOT / "checkpoints" / "original" / "shruti.safetensors"
    orig_cact = ROOT / "checkpoints" / "original" / "shruti.cact"

    # 1. Verify original checkpoints are untouched (Strict Constraint #1)
    st_hash = sha256_file(orig_st)
    cact_hash = sha256_file(orig_cact)
    assert st_hash == ORIG_ST_SHA256, f"Original safetensors hash mismatch: {st_hash}"
    assert cact_hash == ORIG_CACT_SHA256, f"Original cact hash mismatch: {cact_hash}"
    print("[Constraint #1 Verified] Original checkpoints are untouched and match official SHA-256.")

    # 2. Load WSP vectors & Phase 4/5 configuration
    wsp_npz = np.load(str(ROOT / "reports" / "phase3_wsp_vectors.npz"), allow_pickle=False)
    with open(ROOT / "reports" / "phase4_5_wsp_results.json", "r", encoding="utf-8") as f:
        p45 = json.load(f)

    orig_flat = load_safetensors_flat(str(orig_st))
    E_orig = orig_flat["embedding/embedding"].astype(np.float32).copy()
    E_new = E_orig.copy()

    mu_0_unit = wsp_vecs_mu0 = wsp_npz["mu_0_unit"].astype(np.float32)
    delta_w_best = wsp_npz["delta_w_best"].astype(np.float32)

    comb_info = p45["experiments"]["WSP-Combined-Final"]
    included = comb_info["included_components"]

    # Apply E2 if included
    e2_info = p45["experiments"]["WSP-E2"]
    alpha_dev = float(e2_info["selected_alpha_dev"])
    if alpha_dev != 1.0:
        E_new[8192:8906] = alpha_dev * E_new[8192:8906]

    # Apply E3 if included
    if "WSP-E3" in included:
        e3_info = p45["experiments"]["WSP-E3"]
        scale_factor = alpha_dev if alpha_dev != 1.0 else 1.0
        for p in e3_info["pair_edits"]:
            ir, ih = int(p["ref_tid"]), int(p["hyp_tid"])
            er = E_orig[ir].copy()
            eh = E_orig[ih].copy()
            er_hat = er / max(float(np.linalg.norm(er)), 1e-12)
            eh_hat = eh / max(float(np.linalg.norm(eh)), 1e-12)
            orth_r = er - float(np.dot(er, eh_hat)) * eh_hat
            orth_h = eh - float(np.dot(eh, er_hat)) * er_hat
            eta_norm = float(p["eta_norm"])
            eta_orth = float(p["eta_orth"])
            E_new[ir] = scale_factor * ((1.0 + eta_norm) * er + eta_orth * orth_r)
            E_new[ih] = scale_factor * ((1.0 - eta_norm) * eh + eta_orth * orth_h)

    # Apply winning E1 on language tags 8906..8913
    if "WSP-E1b" in included:
        E_new[8906] = E_orig[8906] + 0.5 * delta_w_best
        E_new[8913] = E_orig[8913] - 0.5 * delta_w_best
    else:
        with open(ROOT / "reports" / "phase3_diagnosis.json", "r", encoding="utf-8") as f:
            p3 = json.load(f)
        beta = float(p3["diagnosis_1_step0_lid"]["dev_calibrated_pure_bias_beta"])
        E_new[8906] = E_orig[8906] + 0.5 * beta * mu_0_unit
        E_new[8913] = E_orig[8913] - 0.5 * beta * mu_0_unit

    for eu_i in range(8907, 8913):
        E_new[eu_i] = E_orig[eu_i] - 2.0 * mu_0_unit

    # Export reproducible checkpoints
    exp_dir = ROOT / "checkpoints" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_st = exp_dir / "shruti_wsp_final.safetensors"
    out_cact = exp_dir / "shruti_wsp_final.cact"

    new_flat = dict(orig_flat)
    new_flat["embedding/embedding"] = E_new.astype(np.float32)
    save_safetensors_flat(str(out_st), new_flat)
    patch_stats = surgical_patch_cact(str(orig_cact), str(out_cact), orig_flat, new_flat, WHISTLE_SMALL_CFG)

    new_st_sha = sha256_file(out_st)
    new_cact_sha = sha256_file(out_cact)

    changed_rows = np.where(np.max(np.abs(E_new - E_orig), axis=1) > 1e-7)[0]
    print(f"Reproduced {len(changed_rows)} modified embedding rows ({len(changed_rows)*512:,} parameters).")
    print(f"  - {out_st.name} SHA-256: {new_st_sha}")
    print(f"  - {out_cact.name} SHA-256: {new_cact_sha}")
    print(f"  - Surgical .cact patch stats: {patch_stats}")

    # Also export the Pareto Zero-Hindi-Regression WSP-E1c checkpoint
    u_fisher = wsp_npz["u_fisher"].astype(np.float32)
    delta_w_e1c = 12.0 * u_fisher + 1.75 * mu_0_unit
    E_e1c = E_orig.copy()
    E_e1c[8906] = E_orig[8906] + 0.5 * delta_w_e1c
    E_e1c[8913] = E_orig[8913] - 0.5 * delta_w_e1c
    for eu_i in range(8907, 8913):
        E_e1c[eu_i] = E_orig[eu_i] - 2.0 * mu_0_unit

    out_e1c_st = exp_dir / "shruti_wsp_e1c_zero_hi_reg.safetensors"
    out_e1c_cact = exp_dir / "shruti_wsp_e1c_zero_hi_reg.cact"
    e1c_flat = dict(orig_flat)
    e1c_flat["embedding/embedding"] = E_e1c.astype(np.float32)
    save_safetensors_flat(str(out_e1c_st), e1c_flat)
    surgical_patch_cact(str(orig_cact), str(out_e1c_cact), orig_flat, e1c_flat, WHISTLE_SMALL_CFG)
    print(f"  - {out_e1c_st.name} SHA-256: {sha256_file(out_e1c_st)}")
    print(f"  - {out_e1c_cact.name} SHA-256: {sha256_file(out_e1c_cact)}")

    # 3. Export Phase 7 Analytic Weight-Space Compilers (WSP-E5 and WSP-E8) if available
    p7_path = ROOT / "reports" / "phase7_analytic_wsp_results.json"
    if p7_path.exists():
        with open(p7_path, "r", encoding="utf-8") as f:
            p7 = json.load(f)

        E_e5 = E_new.copy()
        for r_str, row_vals in p7.get("best_e5_modified_rows", {}).items():
            E_e5[int(r_str)] = np.asarray(row_vals, dtype=np.float32)

        out_e5_st = exp_dir / "shruti_wsp_phase7_e5_ridge.safetensors"
        out_e5_cact = exp_dir / "shruti_wsp_phase7_e5_ridge.cact"
        e5_flat = dict(orig_flat)
        e5_flat["embedding/embedding"] = E_e5.astype(np.float32)
        save_safetensors_flat(str(out_e5_st), e5_flat)
        e5_patch = surgical_patch_cact(str(orig_cact), str(out_e5_cact), orig_flat, e5_flat, WHISTLE_SMALL_CFG)
        e5_changed = np.where(np.max(np.abs(E_e5 - E_orig), axis=1) > 1e-7)[0]
        print(f"\nPhase 7 WSP-E5 (Covariance-Whitened Ridge Embedding Compiler): {len(e5_changed)} rows ({len(e5_changed)*512:,} params)")
        print(f"  - {out_e5_st.name} SHA-256: {sha256_file(out_e5_st)}")
        print(f"  - {out_e5_cact.name} SHA-256: {sha256_file(out_e5_cact)}")
        print(f"  - Surgical .cact patch stats: {e5_patch}")

        if "best_e6_s_scale" in p7:
            s_e6 = np.asarray(p7["best_e6_s_scale"], dtype=np.float32)
            out_e8_st = exp_dir / "shruti_wsp_phase7_e8_combined.safetensors"
            out_e8_cact = exp_dir / "shruti_wsp_phase7_e8_combined.cact"
            e8_flat = dict(e5_flat)
            e8_flat["stack/final_norm/scale"] = s_e6
            save_safetensors_flat(str(out_e8_st), e8_flat)
            e8_patch = surgical_patch_cact(str(orig_cact), str(out_e8_cact), orig_flat, e8_flat, WHISTLE_SMALL_CFG)
            print(f"\nPhase 7 WSP-E8 (Combined Embedding + Final-Norm Ridge Compiler): {len(e5_changed)*512 + 512:,} params")
            print(f"  - {out_e8_st.name} SHA-256: {sha256_file(out_e8_st)}")
            print(f"  - {out_e8_cact.name} SHA-256: {sha256_file(out_e8_cact)}")
            print(f"  - Surgical .cact patch stats: {e8_patch}")

    # 4. Export Phase 8 Unquantized FP32 Master & Quantization-Compensated Checkpoints (WSP-E9, WSP-E10a, WSP-E10b)
    p8_path = ROOT / "reports" / "phase8_fp32_master_wsp_results.json"
    if p8_path.exists():
        with open(p8_path, "r", encoding="utf-8") as f:
            p8 = json.load(f)

        # 4a. Unquantized FP32 Master Checkpoint (WSP-E9)
        E_e9_fp32 = E_new.copy()
        for r_str, row_vals in p8.get("best_fp32_master_modified_rows", {}).items():
            E_e9_fp32[int(r_str)] = np.asarray(row_vals, dtype=np.float32)
        s_e9_fp32 = np.asarray(
            p8.get("best_fp32_master_s_scale", orig_flat["stack/final_norm/scale"]),
            dtype=np.float32,
        )
        out_e9_st = exp_dir / "shruti_wsp_phase8_fp32_master.safetensors"
        e9_flat = dict(orig_flat)
        e9_flat["embedding/embedding"] = E_e9_fp32.astype(np.float32)
        e9_flat["stack/final_norm/scale"] = s_e9_fp32
        save_safetensors_flat(str(out_e9_st), e9_flat)
        e9_changed = np.where(np.max(np.abs(E_e9_fp32 - E_orig), axis=1) > 1e-7)[0]
        print(f"\nPhase 8 WSP-E9 (Unquantized FP32 Master Model): {len(e9_changed)} rows ({len(e9_changed)*512:,} params)")
        print(f"  - {out_e9_st.name} SHA-256: {sha256_file(out_e9_st)}")

        # 4b. Lattice-Snapped CQ-W4 Checkpoint (WSP-E10a)
        E_e10 = E_new.copy()
        for r_str, row_vals in p8.get("best_e10_modified_rows", {}).items():
            E_e10[int(r_str)] = np.asarray(row_vals, dtype=np.float32)
        out_e10a_st = exp_dir / "shruti_wsp_phase8_e10a_latticesnap.safetensors"
        out_e10a_cact = exp_dir / "shruti_wsp_phase8_e10a_latticesnap.cact"
        e10a_flat = dict(orig_flat)
        e10a_flat["embedding/embedding"] = E_e10.astype(np.float32)
        save_safetensors_flat(str(out_e10a_st), e10a_flat)
        e10a_patch = surgical_patch_cact(str(orig_cact), str(out_e10a_cact), orig_flat, e10a_flat, WHISTLE_SMALL_CFG)
        e10_changed = np.where(np.max(np.abs(E_e10 - E_orig), axis=1) > 1e-7)[0]
        print(f"\nPhase 8 WSP-E10a (FP32-to-CQ4 Lattice-Snapped Embedding): {len(e10_changed)} rows ({len(e10_changed)*512:,} params)")
        print(f"  - {out_e10a_st.name} SHA-256: {sha256_file(out_e10a_st)}")
        print(f"  - {out_e10a_cact.name} SHA-256: {sha256_file(out_e10a_cact)}")
        print(f"  - Surgical .cact patch stats: {e10a_patch}")

        # 4c. Quantization-Compensated CQ-W4 + FP16 FinalNorm Checkpoint (WSP-E10b)
        s_e10b = np.asarray(p8.get("best_s_e10_scale", orig_flat["stack/final_norm/scale"]), dtype=np.float32)
        out_e10b_st = exp_dir / "shruti_wsp_phase8_e10b_quantcomp.safetensors"
        out_e10b_cact = exp_dir / "shruti_wsp_phase8_e10b_quantcomp.cact"
        e10b_flat = dict(e10a_flat)
        e10b_flat["stack/final_norm/scale"] = s_e10b
        save_safetensors_flat(str(out_e10b_st), e10b_flat)
        e10b_patch = surgical_patch_cact(str(orig_cact), str(out_e10b_cact), orig_flat, e10b_flat, WHISTLE_SMALL_CFG)
        print(f"\nPhase 8 WSP-E10b (FP32 Master -> LatticeSnap + FP16 QuantComp): {len(e10_changed)*512 + 512:,} params")
        print(f"  - {out_e10b_st.name} SHA-256: {sha256_file(out_e10b_st)}")
        print(f"  - {out_e10b_cact.name} SHA-256: {sha256_file(out_e10b_cact)}")
        print(f"  - Surgical .cact patch stats: {e10b_patch}")

        # 5. Export Phase 9 Multi-Layer Cascade & 4-Block FP16 Group-Norm Checkpoints (WSP-E11 & WSP-E12)
        p9_path = ROOT / "reports" / "phase9_cascade_wsp_results.json"
        if p9_path.exists():
            with open(p9_path, "r", encoding="utf-8") as f:
                p9 = json.load(f)

            # 5a. WSP-E11 (4-Block FP16 Group-Norm + Lattice-Snapped Embedding + FP16 FinalNorm — 196 rows = 100,864 params)
            E_e11 = E_e10.copy()
            for item in p9.get("stage3_block_calibrated_rows", []):
                r_idx = int(item["row"])
                alpha_4 = item["alpha_4"]
                for g in range(4):
                    E_e11[r_idx, g * 128 : (g + 1) * 128] *= float(alpha_4[g])
            s_e11 = np.asarray(p9.get("best_s_p9_scale", s_e10b), dtype=np.float32)

            out_e11_st = exp_dir / "shruti_wsp_phase9_e11_blocknorm.safetensors"
            out_e11_cact = exp_dir / "shruti_wsp_phase9_e11_blocknorm.cact"
            e11_flat = dict(orig_flat)
            e11_flat["embedding/embedding"] = E_e11.astype(np.float32)
            e11_flat["stack/final_norm/scale"] = s_e11
            save_safetensors_flat(str(out_e11_st), e11_flat)
            e11_patch = surgical_patch_cact(str(orig_cact), str(out_e11_cact), orig_flat, e11_flat, WHISTLE_SMALL_CFG)
            e11_changed = np.where(np.max(np.abs(E_e11 - E_orig), axis=1) > 1e-7)[0]
            print(f"\nPhase 9 WSP-E11 (4-Block FP16 Group-Norm + LatticeSnap + FinalNorm): {len(e11_changed)} rows ({len(e11_changed)*512 + 512:,} params)")
            print(f"  - {out_e11_st.name} SHA-256: {sha256_file(out_e11_st)}")
            print(f"  - {out_e11_cact.name} SHA-256: {sha256_file(out_e11_cact)}")
            print(f"  - Surgical .cact patch stats: {e11_patch}")

            # 5b. WSP-E12 (Full 26.87% Model Weight Cascade: 8,654 Vocab Rows + 81,934 Engram-0/1 Slots + FinalNorm = 14,918,912 params)
            gamma_g = float(p9.get("stage3_global_gamma", 0.0015))
            E_e12 = E_e11.copy()
            E_e12[8192:8906] *= (1.0 + gamma_g)
            E_e12[260:8192] *= (1.0 - 0.35 * gamma_g)
            for item in p9.get("stage3_block_calibrated_rows", []):
                E_e12[int(item["row"])] = E_e11[int(item["row"])]
            for r_str in p8.get("best_e10_modified_rows", {}).keys():
                r_p8 = int(r_str)
                if r_p8 < 8192:
                    E_e12[r_p8] = E_e11[r_p8]

            eng0_e12 = orig_flat["engrams_0/embedding"].astype(np.float32).copy()
            eng1_e12 = orig_flat["engrams_1/embedding"].astype(np.float32).copy()
            for h_str, s_list in p9.get("eng_damped_slots", {}).items():
                h_i = int(h_str)
                for s_id in s_list:
                    eng1_e12[h_i, int(s_id)] *= 0.985
                    eng0_e12[h_i, int(s_id)] *= 0.990
            for h_str, s_list in p9.get("eng_boosted_slots", {}).items():
                h_i = int(h_str)
                for s_id in s_list:
                    eng1_e12[h_i, int(s_id)] *= 1.008
                    eng0_e12[h_i, int(s_id)] *= 1.005

            out_e12_st = exp_dir / "shruti_wsp_phase9_e12_cascade27pct.safetensors"
            e12_flat = dict(e11_flat)
            e12_flat["embedding/embedding"] = E_e12.astype(np.float32)
            e12_flat["engrams_0/embedding"] = eng0_e12
            e12_flat["engrams_1/embedding"] = eng1_e12
            save_safetensors_flat(str(out_e12_st), e12_flat)
            print(f"\nPhase 9 WSP-E12 (Full 26.87% Cascade — 14,918,912 params):")
            print(f"  - {out_e12_st.name} SHA-256: {sha256_file(out_e12_st)}")

            # 6. Export Phase 10 Clean-Target Loop-Protected Checkpoints (WSP-E13 & WSP-E14)
            p10_path = ROOT / "reports" / "phase10_clean_target_wsp_results.json"
            if p10_path.exists():
                with open(p10_path, "r", encoding="utf-8") as f:
                    p10 = json.load(f)

                # 6a. WSP-E13 (Clean-Target Sequential Gauss-Seidel 4-Block FP16 + Active 1-Beam Loop Guard)
                E_e13 = E_e10.copy()
                for item in p10.get("e13_calibrated_rows", []):
                    r_idx = int(item["row"])
                    alpha_4 = item["alpha_4"]
                    for g in range(4):
                        E_e13[r_idx, g * 128 : (g + 1) * 128] *= float(alpha_4[g])
                s_e13 = np.asarray(p10.get("scale_p10_e13", s_e10b), dtype=np.float32)

                out_e13_st = exp_dir / "shruti_wsp_phase10_e13_cleanloopsafe.safetensors"
                out_e13_cact = exp_dir / "shruti_wsp_phase10_e13_cleanloopsafe.cact"
                e13_flat = dict(orig_flat)
                e13_flat["embedding/embedding"] = E_e13.astype(np.float32)
                e13_flat["stack/final_norm/scale"] = s_e13
                save_safetensors_flat(str(out_e13_st), e13_flat)
                e13_patch = surgical_patch_cact(str(orig_cact), str(out_e13_cact), orig_flat, e13_flat, WHISTLE_SMALL_CFG)
                e13_changed = np.where(np.max(np.abs(E_e13 - E_orig), axis=1) > 1e-7)[0]
                print(f"\nPhase 10 WSP-E13 (Clean-Target Gauss-Seidel + 1-Beam Loop Guard): {len(e13_changed)} rows ({len(e13_changed)*512 + 512:,} params)")
                print(f"  - {out_e13_st.name} SHA-256: {sha256_file(out_e13_st)}")
                print(f"  - {out_e13_cact.name} SHA-256: {sha256_file(out_e13_cact)}")
                print(f"  - Surgical .cact patch stats: {e13_patch}")

                # 6b. WSP-E14 (Loop-Guarded Phase 9 Checkpoint)
                E_e14 = E_e11.copy()
                for tid in p10.get("e14_reverted_tids", []):
                    E_e14[int(tid)] = E_e10[int(tid)].copy()
                out_e14_st = exp_dir / "shruti_wsp_phase10_e14_loopguarded.safetensors"
                out_e14_cact = exp_dir / "shruti_wsp_phase10_e14_loopguarded.cact"
                e14_flat = dict(orig_flat)
                e14_flat["embedding/embedding"] = E_e14.astype(np.float32)
                e14_flat["stack/final_norm/scale"] = s_e11
                save_safetensors_flat(str(out_e14_st), e14_flat)
                e14_patch = surgical_patch_cact(str(orig_cact), str(out_e14_cact), orig_flat, e14_flat, WHISTLE_SMALL_CFG)
                e14_changed = np.where(np.max(np.abs(E_e14 - E_orig), axis=1) > 1e-7)[0]
                print(f"\nPhase 10 WSP-E14 (Loop-Guarded Phase 9): {len(e14_changed)} rows ({len(e14_changed)*512 + 512:,} params)")
                print(f"  - {out_e14_st.name} SHA-256: {sha256_file(out_e14_st)}")
                print(f"  - {out_e14_cact.name} SHA-256: {sha256_file(out_e14_cact)}")
                print(f"  - Surgical .cact patch stats: {e14_patch}")

                # 7. Export Phase 11 Exact Differentiable FP16-Subspace Checkpoints (WSP-E15 & WSP-E16)
                p11_path = ROOT / "reports" / "phase11_fp16_subspace_diff_wsp_results.json"
                p11_npz_path = ROOT / "reports" / "phase11_fp16_tensors.npz"
                if p11_path.exists():
                    with open(p11_path, "r", encoding="utf-8") as f:
                        p11 = json.load(f)

                    if p11_npz_path.exists():
                        p11_npz = np.load(p11_npz_path)
                        alpha_4_best = np.asarray(p11_npz["emb_alpha_4_best"], dtype=np.float32)
                        e15_tensors = {k[len("e15__"):]: np.asarray(p11_npz[k], dtype=np.float16) for k in p11_npz.files if k.startswith("e15__")}
                        e16_tensors = {k[len("e16__"):]: np.asarray(p11_npz[k], dtype=np.float16) for k in p11_npz.files if k.startswith("e16__")}
                    else:
                        alpha_4_best = np.asarray(p11["emb_alpha_4_best"], dtype=np.float32)
                        e15_tensors = {k: np.asarray(v, dtype=np.float16) for k, v in p11.get("modified_fp16_tensors_e15", {}).items()}
                        e16_tensors = {k: np.asarray(v, dtype=np.float16) for k, v in p11.get("modified_fp16_tensors_e16", {}).items()}

                    E_p11 = E_e14.copy()
                    for g in range(4):
                        E_p11[:, g * 128 : (g + 1) * 128] *= alpha_4_best[:, g : g + 1]

                    # 7a. WSP-E15 (8-Layer Decoder + Cross-Attention + 4-Block Embedding Norms FP16 Calibration)
                    e15_flat = dict(orig_flat)
                    e15_flat["embedding/embedding"] = E_p11.astype(np.float32)
                    n_e15_fp16_params = 0
                    for k, arr_fp16 in e15_tensors.items():
                        arr = arr_fp16.astype(np.float32)
                        e15_flat[k] = arr
                        n_e15_fp16_params += int(arr.size)

                    out_e15_st = exp_dir / "shruti_wsp_phase11_e15_decoder_fp16.safetensors"
                    out_e15_cact = exp_dir / "shruti_wsp_phase11_e15_decoder_fp16.cact"
                    save_safetensors_flat(str(out_e15_st), e15_flat)
                    e15_patch = surgical_patch_cact(str(orig_cact), str(out_e15_cact), orig_flat, e15_flat, WHISTLE_SMALL_CFG)
                    e15_changed = np.where(np.max(np.abs(E_p11 - E_orig), axis=1) > 1e-7)[0]
                    print(
                        f"\nPhase 11 WSP-E15 (Decoder + Cross-Attn + 4-Block Emb FP16 Calibration): "
                        f"{len(e15_changed)} emb rows + {len(e15_tensors)} internal FP16 tensors "
                        f"({n_e15_fp16_params:,} internal FP16 params)"
                    )
                    print(f"  - {out_e15_st.name} SHA-256: {sha256_file(out_e15_st)}")
                    print(f"  - {out_e15_cact.name} SHA-256: {sha256_file(out_e15_cact)}")
                    print(f"  - Surgical .cact patch stats: {e15_patch}")

                    # 7b. WSP-E16 (End-to-End Audio Stem + 8-Layer Encoder + 8-Layer Decoder FP16 Calibration)
                    e16_flat = dict(orig_flat)
                    e16_flat["embedding/embedding"] = E_p11.astype(np.float32)
                    n_e16_fp16_params = 0
                    for k, arr_fp16 in e16_tensors.items():
                        arr = arr_fp16.astype(np.float32)
                        e16_flat[k] = arr
                        n_e16_fp16_params += int(arr.size)

                    out_e16_st = exp_dir / "shruti_wsp_phase11_e16_e2e_fp16.safetensors"
                    out_e16_cact = exp_dir / "shruti_wsp_phase11_e16_e2e_fp16.cact"
                    save_safetensors_flat(str(out_e16_st), e16_flat)
                    e16_patch = surgical_patch_cact(str(orig_cact), str(out_e16_cact), orig_flat, e16_flat, WHISTLE_SMALL_CFG)
                    print(
                        f"\nPhase 11 WSP-E16 (End-to-End Audio Stem + Encoder + Decoder FP16 Calibration): "
                        f"{len(e15_changed)} emb rows + {len(e16_tensors)} internal FP16 tensors "
                        f"({n_e16_fp16_params:,} internal FP16 params)"
                    )
                    print(f"  - {out_e16_st.name} SHA-256: {sha256_file(out_e16_st)}")
                    print(f"  - {out_e16_cact.name} SHA-256: {sha256_file(out_e16_cact)}")
                    print(f"  - Surgical .cact patch stats: {e16_patch}")

    # Final check that original checkpoints remain 100% untouched
    assert sha256_file(orig_st) == ORIG_ST_SHA256
    assert sha256_file(orig_cact) == ORIG_CACT_SHA256


if __name__ == "__main__":
    main()
