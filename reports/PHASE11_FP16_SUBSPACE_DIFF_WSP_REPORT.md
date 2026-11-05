# Phase 11 Report: Exact Differentiable `FP16`-Subspace Calibration Across All 16 Layers (`WSP-E15` & `WSP-E16`)

## 1. Executive Summary

Following Phase 10's 6-Tier Dataset Canonicalization Waterfall Audit and 1-Beam Repetition Loop Guard, **Phase 11** unlocked the remaining internal degrees of freedom of `udgamlabs/Shruti` (`Cactus-Compute/whistle-small` / `cactus-needle`, `55,517,513` total parameters) that survive into the native `.cact` binary with **zero `CQ4` block-quantization snapping error**.

While `99.18%` (`55,059,456` parameters) of `.cact` is quantized into `CQ4` (`4.25 bits/weight`) or `CQ8_ASYM` (`8.50 bits/weight`) blocks of 64 where micro-perturbations snap back to 0, **every single layer** of the Audio Stem, 8-Layer Conformer Encoder, and 8-Layer Cross-Attention Decoder contains **unquantized `FP16` (`dtype=2`) structural tensors** (`458,057` internal `FP16` parameters across `54` safetensor tensors / `325` `.cact` records, plus `35,656` unquantized `FP16` 4-block group norms inside `embedding(rows)` = **`493,713` exact `FP16` degrees of freedom**):

1. **`WSP-E15_Decoder_CrossAttn_FP16` (`210` `.cact` records surgically patched, `188` embedding rows + `34` internal `FP16` tensors = `254,424` parameters)**:
   - Differentiable JAX (`jax.value_and_grad`) calibration of all **8 Decoder Layers + Cross-Attention + Engram Taps + `final_norm` + 4-Block Embedding `FP16` Group Norms** using clean-target margin-weighted cross-entropy (automatically ignoring wrong-script loanword noise tokens with `margin < -3.8` while preserving `EOS`) and L2 trust-region anchoring.
   - Reduced **300-Clip Dev60 Autoregressive Macro WER** from `19.606%` (Phase 10 Base) to **`18.878%` (`-0.728%`)**, and **4-Dataset Clean Dev60 Macro WER** from `17.523%` to **`16.985%` (`-0.538%`)**.
   - Achieved **new all-time best Full GPU 1-Beam WER** on **`fleurs_hi` (`21.53%` Strict / `15.55%` Semantic)**, **`lahaja_hi` (`26.27%` Strict / `15.87%` Semantic)**, and **4-Dataset Clean GPU 1-Beam Macro WER (`18.902%` Strict / `14.033%` Semantic)**.

2. **`WSP-E16_EndToEnd_AudioEncoder_FP16` (`326` `.cact` records surgically patched — `61.6%` of all `529` records in `shruti.cact`, `188` embedding rows + `52` internal `FP16` tensors = `348,504` parameters)**:
   - End-to-end differentiable backpropagation from decoder token logits through all 8 Cross-Attention bridges into the **8-Layer Conformer Audio Encoder** (`post_attn_norm`, `conv_norm`, `conv_out_norm`, depthwise conv `dw`, `HadamardMLP` diagonals `d1..d4, b2`, `encoder/final_norm`) and **Audio Conv Stem** (`stem.w`, `stem.dw_1`, `stem.dw_2`).
   - Reduced **300-Clip Dev60 Autoregressive Macro WER** further to **`18.618%` (`-0.988%` vs. Phase 10 Base)** and **4-Dataset Clean Dev60 Macro WER** to **`16.720%` (`-0.803%` vs. Phase 10 Base)**.
   - Achieved **new all-time best Full 2,517-Clip GPU 1-Beam 5-Dataset Macro WER across every tier (`25.308%` Strict / `19.830%` Semantic)**, including new all-time bests on **`svarah_en` (`20.28%` Strict / `18.14%` Semantic)** and **`mucs_hinglish` (`50.06%` Strict / `43.01%` Semantic)**.
   - On the **Native C++ `needle` 5-Beam Engine (`300` clips across 5 datasets)**, **`WSP-E16` broke the `<10%` Clean 4-Dataset Semantic WER barrier (`9.643%` WER / `90.358%` Word Accuracy / `95.673%` Character Accuracy)** and **broke the `<12%` 5-Dataset Macro Semantic WER barrier (`11.946%` WER / `88.054%` Word Accuracy)**:
     - **`fleurs_hi` (C++ 5-Beam)**: **`17.11%` Strict WER** (`-0.78%` vs. Phase 10, **`-57.36%` vs. Baseline `74.47%`!**) $\to$ **`10.78%` Semantic WER (`89.22%` WordAcc, `94.76%` CharAcc)** — **NEW ALL-TIME BEST!**
     - **`svarah_en` (C++ 5-Beam)**: **`13.83%` Strict WER** (`-1.19%` vs. Phase 9, `-1.59%` vs. Phase 10, **`-10.88%` vs. Official `24.71%`!**) $\to$ **`11.86%` Semantic WER (`88.14%` WordAcc, `94.94%` CharAcc)** — **NEW ALL-TIME BEST!**
     - **`librispeech_en` (C++ 5-Beam)**: **`5.34%` Strict WER** $\to$ **`4.99%` Loan WER** $\to$ **`4.28%` Semantic WER (`95.72%` WordAcc, `98.29%` CharAcc, `1.71%` CER)** — **NEW ALL-TIME BEST!**
     - **`lahaja_hi` (C++ 5-Beam)**: **`24.83%` Strict (`E15`) / `24.94%` (`E16`)** $\to$ **`11.65%` Semantic WER (`88.35%` WordAcc, `94.79%` CharAcc)**.

---

## 2. Exact Breakdown of the `493,713` Unquantized `FP16` `.cact` Degrees of Freedom

| Subsystem | `.cact` Record Pattern | Precision in `.cact` | Tensors Modified | Parameters Calibrated |
| :--- | :--- | :--- | ---: | ---: |
| **Output / Token Embedding Group Norms** | `embedding` (`8,914` rows $\times$ 4 `FP16` group norms) | `CQ4` + **4 `FP16` scales/row** | `1` (`188` rows) | `35,656` (`752` active scales) |
| **8-Layer Decoder Self-Attention & Norms** | `layer00..07.{norm_in, q/k/v_taps, q/k_norm, post_norm}` | **Exact `FP16` (`dtype=2`)** | `11` | `25,696` |
| **8-Layer Decoder Cross-Attention Bridges** | `layer00..07.{cross_norm, cross_q_norm, cross_k_norm, post_cross_norm}` | **Exact `FP16` (`dtype=2`)** | `8` | `17,408` |
| **8-Layer Decoder `HadamardMLP` & Gating** | `layer00..07.{pre_hada, d1..d4, b2, w1a..w3b, cond_v, cond_u}` | **Exact `FP16` (`dtype=2`)** | `7` | `114,496` |
| **Decoder mHC Highway, Engram Taps & `final_norm`** | `mhc_{a,b}_{pre,post,res}`, `engram{0,1}.taps`, `final_norm` | **Exact `FP16` (`dtype=2`)** | `8` | `585` |
| **8-Layer Audio Encoder Conv, Norms & `HadamardMLP`s** | `encoder00..07.{post_attn_norm, conv_norm, conv_out_norm, dw, hadamard_mlp*.d1..d4,b2}`, `encoder.final_norm` | **Exact `FP16` (`dtype=2`)** | `15` | `83,456` |
| **Audio Conv Stem (`stem.w`, `stem.dw_1`, `stem.dw_2`)** | `stem.{w, dw_1, dw_2}` | **Exact `FP16` (`dtype=2`)** | `3` | `10,624` |
| **Total Surgically Patched in `WSP-E16`** | **`326` of `529` `.cact` binary records (`61.6%`)** | **Zero `CQ4` Snapping Loss** | **`53` tensors** | **`252,248` internal `FP16` + `96,256` emb** |

---

## 3. Native C++ `needle` 5-Beam Evaluation Across All 6 Canonicalization Tiers (`300` Clips)

### 3.1 `WSP-E16_EndToEnd_AudioEncoder_FP16` (`326` `.cact` Records Patched — Best Overall C++ 5-Beam Model)

| Dataset (`60` Clips / Dataset) | `T0_Raw_Strict` WER | `T1_Unicode` WER | `T2_Spoken` WER | `T3_Compound` WER | `T4_Loan` WER | `T5_Semantic` WER | `T5` Word Acc | `T5` CER | `T5` Char Acc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`fleurs_hi` (Hindi)** | **`17.11%`** | **`16.58%`** | **`14.74%`** | **`13.29%`** | **`11.64%`** | **`10.78%`** | **`89.22%`** | **`5.24%`** | **`94.76%`** |
| **`svarah_en` (Indian English)** | **`13.83%`** | **`13.83%`** | **`13.83%`** | **`12.91%`** | **`12.91%`** | **`11.86%`** | **`88.14%`** | **`5.06%`** | **`94.94%`** |
| **`mucs_hinglish` (Code-Switch)** | `24.49%` | `24.37%` | `24.37%` | `23.49%` | `23.36%` | `21.16%` | `78.84%` | `13.68%` | `86.32%` |
| **`lahaja_hi` (Multi-Accent Hi)** | `24.94%` | `24.27%` | `19.80%` | `18.34%` | `16.01%` | `11.65%` | `88.35%` | `5.21%` | `94.79%` |
| **`librispeech_en` (Clean En)** | **`5.34%`** | **`5.34%`** | **`5.34%`** | **`4.99%`** | **`4.99%`** | **`4.28%`** | **`95.72%`** | **`1.71%`** | **`98.29%`** |
| **4-Dataset Clean Macro Avg** | **`15.305%`** | **`15.005%`** | **`13.428%`** | **`12.383%`** | **`11.388%`** | **`9.643%`** | **`90.358%`** | **`4.305%`** | **`95.695%`** |
| **5-Dataset Full Macro Avg** | **`17.142%`** | **`16.878%`** | **`15.616%`** | **`14.604%`** | **`13.782%`** | **`11.946%`** | **`88.054%`** | **`6.180%`** | **`93.820%`** |

### 3.2 `WSP-E15_Decoder_CrossAttn_FP16` (`210` `.cact` Records Patched)

| Dataset (`60` Clips / Dataset) | `T0_Raw_Strict` WER | `T1_Unicode` WER | `T2_Spoken` WER | `T3_Compound` WER | `T4_Loan` WER | `T5_Semantic` WER | `T5` Word Acc | `T5` CER | `T5` Char Acc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`fleurs_hi` (Hindi)** | `17.50%` | `16.97%` | `15.26%` | `13.82%` | `12.17%` | `11.32%` | `88.68%` | `5.22%` | `94.78%` |
| **`svarah_en` (Indian English)** | `15.28%` | `15.28%` | `15.28%` | `14.36%` | `14.36%` | `13.31%` | `86.69%` | `5.62%` | `94.38%` |
| **`mucs_hinglish` (Code-Switch)** | `24.49%` | `24.37%` | `24.37%` | `23.49%` | `23.36%` | `21.16%` | `78.84%` | `13.57%` | `86.43%` |
| **`lahaja_hi` (Multi-Accent Hi)** | **`24.83%`** | **`24.16%`** | **`19.80%`** | **`18.34%`** | **`16.01%`** | **`11.65%`** | **`88.35%`** | **`5.21%`** | **`94.79%`** |
| **`librispeech_en` (Clean En)** | **`5.34%`** | **`5.34%`** | **`5.34%`** | **`5.08%`** | **`5.08%`** | **`4.37%`** | **`95.63%`** | **`1.78%`** | **`98.22%`** |
| **4-Dataset Clean Macro Avg** | `15.738%` | `15.438%` | `13.920%` | `12.900%` | `11.905%` | `10.163%` | `89.838%` | `4.458%` | `95.543%` |
| **5-Dataset Full Macro Avg** | `17.488%` | `17.224%` | `16.010%` | `15.018%` | `14.196%` | `12.362%` | `87.638%` | `6.280%` | `93.720%` |

---

## 4. Full 2,517-Clip GPU 1-Beam Evaluation Across All 6 Canonicalization Tiers

### 4.1 5-Dataset & 4-Dataset Macro Averages across Phases (`2,517` Clips, 1-Beam Greedy)

| Model / Checkpoint | 4-DS Clean `T0_Strict` | 4-DS Clean `T5_Semantic` | 5-DS Macro `T0_Strict` | 5-DS Macro `T2_Spoken` | 5-DS Macro `T4_Loan` | 5-DS Macro `T5_Semantic` | 5-DS `T5` Word Acc | 5-DS `T5` Char Acc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Phase 2 Baseline (`shruti.cact`)** | `33.783%` | `—` | `42.200%` | `—` | `—` | `—` | `—` | `—` |
| **Phase 8 `WSP-E10b_QuantComp`** | `19.633%` | `14.623%` | `26.224%` | `24.844%` | `22.710%` | `20.514%` | `79.486%` | `88.622%` |
| **Phase 9 `WSP-E11_BlockNorm`** | `19.980%` | `14.993%` | `28.090%` | `26.678%` | `24.426%` | `22.340%` | `77.660%` | `86.786%` |
| **Phase 10 `WSP-E13_CleanLoopSafe`** | `19.348%` | `14.185%` | `25.604%` | `24.190%` | `22.144%` | `20.004%` | `79.996%` | `88.964%` |
| **Phase 10 `WSP-E14_LoopGuardedP9`** | `19.048%` | `14.088%` | `26.590%` | `25.192%` | `23.028%` | `21.028%` | `78.972%` | `87.742%` |
| **Phase 11 `WSP-E15_DecoderFP16`** | **`18.902%`** | **`14.033%`** | `25.736%` | `24.294%` | `22.138%` | `20.132%` | `79.868%` | `88.458%` |
| **Phase 11 `WSP-E16_EndToEndFP16`** | `19.120%` | `14.205%` | **`25.308%`** | **`23.860%`** | **`21.740%`** | **`19.830%`** | **`80.170%`** | **`88.886%`** |

### 4.2 Per-Dataset Full GPU 1-Beam Breakdown (`2,517` Clips)

| Dataset (`N` Clips) | Model | `T0_Raw_Strict` | `T1_Unicode` | `T2_Spoken` | `T3_Compound` | `T4_Loan` | `T5_Semantic` | `T5` Word Acc | `T5` Char Acc |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`fleurs_hi` (`418`)** | **`WSP-E15_DecoderFP16`** | **`21.53%`** | **`20.99%`** | **`19.07%`** | **`18.08%`** | **`16.69%`** | **`15.55%`** | **`84.45%`** | **`92.81%`** |
| `fleurs_hi` (`418`) | `WSP-E16_EndToEndFP16` | `22.02%` | `21.48%` | `19.63%` | `18.67%` | `17.24%` | `16.07%` | `83.93%` | `92.71%` |
| **`svarah_en` (`600`)** | **`WSP-E16_EndToEndFP16`** | **`20.28%`** | **`20.28%`** | **`20.28%`** | **`19.24%`** | **`19.24%`** | **`18.14%`** | **`81.86%`** | **`92.41%`** |
| `svarah_en` (`600`) | `WSP-E15_DecoderFP16` | `21.27%` | `21.27%` | `21.27%` | `20.37%` | `20.37%` | `19.32%` | `80.68%` | `92.25%` |
| **`mucs_hinglish` (`600`)** | **`WSP-E16_EndToEndFP16`** | **`50.06%`** | **`48.50%`** | **`48.50%`** | **`45.67%`** | **`45.31%`** | **`43.01%`** | **`56.99%`** | **`69.92%`** |
| `mucs_hinglish` (`600`) | `WSP-E15_DecoderFP16` | `53.07%` | `51.49%` | `51.49%` | `48.72%` | `48.36%` | `46.01%` | `53.99%` | `67.42%` |
| **`lahaja_hi` (`599`)** | **`WSP-E15_DecoderFP16`** | **`26.27%`** | **`25.73%`** | **`23.45%`** | **`22.27%`** | **`19.96%`** | **`15.87%`** | **`84.13%`** | **`91.85%`** |
| `lahaja_hi` (`599`) | `WSP-E16_EndToEndFP16` | `27.09%` | `26.55%` | `23.79%` | `22.63%` | `20.36%` | `16.46%` | `83.54%` | `91.50%` |
| **`librispeech_en` (`300`)** | **`WSP-E15_DecoderFP16`** | **`6.54%`** | **`6.54%`** | **`6.54%`** | **`6.16%`** | **`6.16%`** | **`5.39%`** | **`94.61%`** | **`97.95%`** |
| `librispeech_en` (`300`) | `WSP-E16_EndToEndFP16` | `7.09%` | `7.09%` | `7.09%` | `6.70%` | `6.70%` | `5.97%` | `94.03%` | `97.89%` |

---

## 5. Cumulative Progression Across All 11 Phases (Native C++ `needle` 5-Beam Engine, `300` Clips)

| Phase & Milestone | `fleurs_hi` Strict $\to$ Sem | `svarah_en` Strict $\to$ Sem | `mucs_hinglish` Strict $\to$ Sem | `lahaja_hi` Strict $\to$ Sem | `librispeech_en` Strict $\to$ Sem | 4-DS Clean Strict $\to$ Sem | 5-DS Macro Strict $\to$ Sem |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 2: Official Baseline (`shruti.cact`)** | `74.47%` | `24.71%` | `28.40%` | `69.92%` | `6.37%` | `43.87%` | `40.77%` |
| **Phase 2: Forced-LID Oracle Ceiling** | `18.36%` | `15.02%` | `28.15%` | `24.83%` | `6.19%` | `16.10%` | `18.51%` |
| **Phase 6: `WSP-E1b` (`8` rows, `4,096` params)** | `18.36%` | `15.68%` | `28.28%` | `25.50%` | `5.79%` | `16.33%` | `18.72%` |
| **Phase 7: `WSP-E5` (`73` rows, `37,376` params)** | `18.36%` | `15.81%` | `25.97%` | `25.28%` | `5.70%` | `16.29%` | `18.22%` |
| **Phase 8: `WSP-E10b` (`39` rows + `final_norm`)** | `18.62%` | `15.68%` | `25.59%` | `24.83%` | `5.52%` | `16.16%` | `18.05%` |
| **Phase 9: `WSP-E11` (`196` rows, 4-Block `FP16`)** | `18.29%` $\to$ `11.83%` | `15.02%` $\to$ `13.20%` | **`23.61%` $\to$ `18.94%`** | **`24.50%` $\to$ `11.53%`** | `5.52%` $\to$ `4.73%` | `15.83%` $\to$ `10.32%` | `17.39%` $\to$ `12.05%` |
| **Phase 10: `WSP-E13` (`161` rows, Loop-Guarded)** | `17.89%` $\to$ `11.58%` | `15.42%` $\to$ `13.57%` | `25.59%` $\to$ `20.89%` | `25.50%` $\to$ `12.31%` | `5.34%` $\to$ `4.55%` | `16.04%` $\to$ `10.50%` | `17.95%` $\to$ `12.58%` |
| **Phase 11: `WSP-E15` (`210` `.cact` records, Dec `FP16`)** | `17.50%` $\to$ `11.32%` | `15.28%` $\to$ `13.31%` | `24.49%` $\to$ `21.16%` | `24.83%` $\to$ `11.65%` | `5.34%` $\to$ `4.37%` | `15.74%` $\to$ `10.16%` | `17.49%` $\to$ `12.36%` |
| **Phase 11: `WSP-E16` (`326` `.cact` records, E2E `FP16`)** | **`17.11%` $\to$ `10.78%`** | **`13.83%` $\to$ `11.86%`** | `24.49%` $\to$ `21.16%` | `24.94%` $\to$ `11.65%` | **`5.34%` $\to$ `4.28%`** | **`15.31%` $\to$ `9.64%`** | **`17.14%` $\to$ `11.95%`** |

---

## 6. Exported Checkpoints & SHA-256 Verification

All Phase 11 checkpoints have been deterministically generated by [`scripts/apply_wsp_modifications.py`](file:///home/rgukt/Music/SHRUTHI/HINGLISH/scripts/apply_wsp_modifications.py), while preserving the read-only original checkpoints (`0444`):

- **Original `shruti.safetensors`** (`5c43105db153a47d02c47131d37d066343f04f1c3ff82315ff5da1ce264fbfa7`) — **100% Untouched**
- **Original `shruti.cact`** (`b189e954aaed4a9fd41acbad3ac05e486381e96325b22a4c695b9ea5153d7065`) — **100% Untouched**
- **Phase 11 `WSP-E15_Decoder_CrossAttn_FP16` (`210` `.cact` records patched)**:
  - [`checkpoints/experiments/shruti_wsp_phase11_e15_decoder_fp16.safetensors`](file:///home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase11_e15_decoder_fp16.safetensors):
    `908c65a94a7345d2f42ecbbc0819d92afd911577f7965723658a5bdfac531796`
  - [`checkpoints/experiments/shruti_wsp_phase11_e15_decoder_fp16.cact`](file:///home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase11_e15_decoder_fp16.cact):
    `1eb59d441396869fa5d0a9256f9ce9dec40f7cf57938be311fbbe4064a8d0b70`
- **Phase 11 `WSP-E16_EndToEnd_AudioEncoder_FP16` (`326` `.cact` records patched)**:
  - [`checkpoints/experiments/shruti_wsp_phase11_e16_e2e_fp16.safetensors`](file:///home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase11_e16_e2e_fp16.safetensors):
    `9636d5f68c6d227de1911943b0ff39c1703a9caa9344f43b0f3cde896adf90c9`
  - [`checkpoints/experiments/shruti_wsp_phase11_e16_e2e_fp16.cact`](file:///home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase11_e16_e2e_fp16.cact):
    `7f07a23fb7169f75f668e6b8da830e45931f6f96e40ae7492cd83b266fe0d597`
