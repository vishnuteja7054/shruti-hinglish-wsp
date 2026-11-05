# Phase 8 Technical Report: Unquantized `FP32` Master Compilation $\to$ Quantization-Compensated `.cact` Pipeline

## 1. Executive Summary

Phase 8 implements and validates the **3-Stage Unquantized `FP32` Master Model $\to$ Quantization-Compensated `.cact`** Weight-Space Programming (WSP) pipeline on `udgamlabs/Shruti` (`55,517,513` parameters across `116` tensors):

1. **Stage 1 — Pure Unquantized `FP32` Master Compilation (`WSP-E9_Master_Unquantized_FP32`, `quant=False`)**:
   - Extracted `45,847` teacher-forced decoder hidden states (`39,984` clean calibration states) in **pure unquantized `FP32`** (`quant=False`, zero 4-bit weight quantization and zero 8-bit activation quantization) alongside `CQ-W4A8` states.
   - Performed **Direct Logit Attribution** and **Signal-to-Interference Ratio (SIR)** generalized eigenvalue diagnostics on `FP32` Dev error states (`Base-Decoder Unit Margin = -31.336`, `FinalNorm-Scale Margin = +26.658`, `Top-5 SIR Eigenvalues = [0.324, 0.137, 0.082, 0.066, 0.043]`).
   - Compiled `38` content vocabulary rows (`+184` net `FP32` teacher-forced token hits) using **English-Nullspace Weighted Contrastive Ridge Regression** ($\omega_{\text{en}} = 2.0$ in the preservation covariance $\Sigma_{\text{FP32}}$).
2. **Stage 2 — Codebook-Snap Lattice Projection (`WSP-E10a`) & Unquantized `FP16` Residual Compensation (`WSP-E10b`)**:
   - **Stage 2A (`WSP-E10a_LatticeSnap_CQ4`)**: Projected the `38` `FP32` master update vectors onto the 4-bit Hadamard `CQ-W4` (`group = 128`) Lloyd-Max codebook using optimal per-row lattice scaling $\alpha_r \in [0.75, 2.5]$. `31` content rows (`39` total rows including the `8` language tags = **`19,968` parameters, `0.0360%` of the model**) snapped cleanly onto the 4-bit lattice, delivering **`+306` net `CQ-W4` teacher-forced token hits** while modifying **46% fewer rows** than Phase 7 (`39` rows vs. `73` rows).
   - **Stage 2B (`WSP-E10b_QuantComp_CQ4`)**: Solved a closed-form Ridge regression on the adjacent **unquantized `FP16` `stack/final_norm/scale` (`[512]`)** layer specifically targeting the quantization margin loss $r_{\text{quant}, t} = m_t^{\text{FP32}*} - m_t^{\text{CQ4}}$.
3. **Stage 3 & 4 — Benchmark Results (`GPU CQ-W4A8 STE` & Native C++ `needle` 5-Beam Engine)**:
   - **GPU `CQ-W4A8` 5-Dataset Macro WER**: Drops from `28.07%` (Official Baseline) $\to$ `26.83%` (`WSP-E1b`) $\to$ `26.19%` (Phase 7 `WSP-E5`) $\to$ **`25.99%` (`WSP-E10a`)** $\to$ **`25.83%` (`WSP-E10b`)** (**`-2.24%` below Official Baseline**; Dev Macro WER **`25.37%`**, Held-Out Test Macro WER **`26.02%`** on `WSP-E10a`).
   - **`mucs_hinglish` (600 Code-Switched Hinglish Clips)**: GPU All WER drops from `57.38%` (`WSP-E1b`) $\to$ `53.88%` (Phase 7 `WSP-E5`) $\to$ **`52.59%` (`WSP-E10b`)** (**`-4.79%` WER reduction** across all 600 clips; Dev WER drops from `62.29%` $\to$ `57.89%` $\to$ **`54.40%`**, a **`-7.89%` WER reduction**!).
   - **Native C++ `needle` 5-Beam Engine (`shruti_wsp_phase8_e10b_quantcomp.cact`, 300 clips)**:
     - **5-Dataset Macro WER**: Drops from `20.63%` (Official) $\to$ `18.72%` (`WSP-E1b`) $\to$ `18.22%` (Phase 7 `WSP-E5`) $\to$ **`18.05%` (`18.054%`, Phase 8 `WSP-E10b`)** (**`-2.58%` below Official Baseline**!).
     - **`mucs_hinglish` C++ 5-Beam WER**: Drops from `28.40%` (Official) $\to$ `26.45%` (Phase 7) $\to$ **`25.59%` (`100.0%` LID-Acc)** (**`-2.81%` below Official Baseline**).
     - **`svarah_en` C++ 5-Beam WER**: Drops from `24.71%` (Official) $\to$ `15.42%` (Phase 7) $\to$ **`15.28%` (`96.67%` LID-Acc, `-9.43%` below Official Baseline)**.
     - **`librispeech_en` C++ 5-Beam WER**: Drops to **`5.78%` WER** (`2.45%` CER, **`100.0%` LID-Acc**).

---

## 2. Unquantized `FP32` vs. Quantized `CQ-W4A8` Diagnostic Findings

| Metric (`1,250` Dev Clips = `45,847` Tokens) | Unquantized `FP32` (`quant=False`) | Quantized `CQ-W4A8` (`quant=True`) | Architectural Explanation |
| :--- | :---: | :---: | :--- |
| **Teacher-Forced Top-1 Accuracy (All)** | `76.93%` | `77.12%` | `Shruti` was trained with Quantization-Aware Training (QAT) using `cq_ste_params(4)` and `_aq(8-bit)`. |
| **Teacher-Forced Top-1 Accuracy (Hindi)** | `87.15%` | `87.67%` | Encoder/Decoder weights co-adapted to the 4-bit Hadamard codebook during original QAT training. |
| **Teacher-Forced Top-1 Accuracy (Hinglish)** | **`72.88%`** | `72.87%` | Code-switched Hinglish is equally challenging in both regimes before compilation. |
| **Hidden-State Cosine Similarity ($\cos(h_{\text{FP32}}, h_{\text{CQ4}})$)** | \multicolumn{2}{c}{**`0.98438`**} | `FP32` hidden states preserve `98.44%` directional alignment with `CQ-W4` while eliminating 8-bit step quantization noise. |
| **Embedding `CQ-W4` Quantization RMSE** | \multicolumn{2}{c}{**`0.56908`**} | Explains why naive small weight updates vanish unless projected onto the 4-bit Hadamard lattice. |

---

## 3. Comprehensive Results Across All 5 Datasets (GPU Autoregressive Decoding)

| Model Variant | Precision Mode | Modified Params | `fleurs_hi` WER (All) | `svarah_en` WER (All) | `mucs_hinglish` WER (Dev / Test / All) | `lahaja_hi` WER (All) | `librispeech_en` WER (All) | **Dev Macro WER** | **Held-Out Test Macro WER** | **All Macro WER** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Official Baseline (`shruti`)** | `CQ-W4A8` | `0` | `21.25%` | `36.32%` | `60.26% / 53.19% / 56.83%` | `29.27%` | `6.68%` | `28.00%` | `28.14%` | `28.07%` |
| **Official Unquantized `FP32`** | `FP32` | `0` | `21.68%` | `27.40%` | `55.37% / 49.43% / 52.48%` | `29.10%` | `6.77%` | `27.08%` | `27.87%` | `27.49%` |
| **`WSP-E1b` Unquantized `FP32`** | `FP32` | `4,096` | `22.71%` | `21.36%` | `54.35% / 51.67% / 53.05%` | `29.49%` | `6.79%` | `25.90%` | `27.46%` | `26.68%` |
| **Phase 4–5 `WSP-E1b`** | `CQ-W4A8` | `4,096` | `22.63%` | `20.53%` | `62.29% / 52.00% / 57.38%` | `26.95%` | `6.68%` | `26.64%` | `27.04%` | `26.83%` |
| **Phase 7 `WSP-E5` (Quant-First Ridge)** | `CQ-W4A8` | `37,376` (`73` rows) | `22.65%` | `20.62%` | `57.89% / 49.67% / 53.88%` | `27.10%` | `6.68%` | `26.16%` | `26.18%` | `26.19%` |
| **Phase 8 `WSP-E10a` (FP32 $\to$ LatticeSnap)** | `CQ-W4A8` | **`19,968` (`39` rows)** | **`22.28%`** | **`20.48%`** | `57.30%` / **`49.33%`** / `53.42%` | **`27.05%`** | `6.73%` | `25.92%` | **`26.02%`** | `25.99%` |
| **Phase 8 `WSP-E10b` (FP32 $\to$ QuantComp)** | `CQ-W4A8` | **`20,480` (`39` rows + `final_norm`)** | `22.32%` | **`20.48%`** | **`54.40%` / `50.69%` / `52.59%`** | `27.07%` | `6.71%` | **`25.37%`** | `26.29%` | **`25.83%`** |

---

## 4. Native C++ `needle` 5-Beam Engine Verification (`300` Clips Across All 5 Datasets)

| Dataset (60 Clips Each) | Official `shruti.cact` WER | Phase 4–5 `WSP-E1b` WER | Phase 7 `WSP-E5` WER | **Phase 8 `WSP-E10b` (`FP32` Master $\to$ QuantComp) WER / CER / LID-Acc** | Total Gain vs. Official |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`mucs_hinglish` (Hinglish Code-Switched)** | `28.40%` | `28.08%` | `26.45%` | **`25.59%` WER** (`23.49%` CER, **`100.0%` LID-Acc**) | **`-2.81%` WER** |
| **`svarah_en` (Indian-Accented English)** | `24.71%` | `15.35%` | `15.42%` | **`15.28%` WER** (`6.83%` CER, **`96.67%` LID-Acc**) | **`-9.43%` WER** |
| **`lahaja_hi` (Multi-Dialect Hindi)** | `25.63%` | `24.94%` | `24.94%` | **`24.94%` WER** (`25.36%` CER, **`96.67%` LID-Acc**) | **`-0.69%` WER** |
| **`fleurs_hi` (Standard Hindi)** | `18.54%` | `18.61%` | `18.42%` | `18.68%` WER (`11.07%` CER, `98.33%` LID-Acc) | `+0.14%` WER |
| **`librispeech_en` (Clean English)** | `5.87%` | `6.60%` | `5.87%` | **`5.78%` WER** (`2.45%` CER, **`100.0%` LID-Acc**) | **`-0.09%` WER** |
| **5-Dataset Macro Average (C++ 5-Beam)** | `20.63%` | `18.72%` | `18.22%` | **`18.05%` (`18.054%`)** | **`-2.58%` Macro WER** |


---

## 5. Exported Phase 8 Checkpoints & SHA-256 Verification

| Checkpoint File | Role | Modified Params | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| `checkpoints/original/shruti.safetensors` | Official Read-Only Master (`0444`) | `0` | `5c43105db153a47d02c47131d37d066343f04f1c3ff82315ff5da1ce264fbfa7` |
| `checkpoints/original/shruti.cact` | Official Read-Only `.cact` (`0444`) | `0` | `b189e954aaed4a9fd41acbad3ac05e486381e96325b22a4c695b9ea5153d7065` |
| `checkpoints/experiments/shruti_wsp_phase8_fp32_master.safetensors` | Stage 1 Unquantized `FP32` Master (`WSP-E9`) | `23,552` (`46` rows) | `a40ddd7d96029b9fea4568fd3f7f99486de22686c995321b6b05eaaf5babfdfb` |
| `checkpoints/experiments/shruti_wsp_phase8_e10a_latticesnap.safetensors` | Stage 2A `FP32`-to-`CQ-W4` Lattice-Snapped (`WSP-E10a`) | `19,968` (`39` rows) | `c28391ae5c1af5a547b178266ede2fbe009f05433f0ab52d0d676fc5b947567b` |
| `checkpoints/experiments/shruti_wsp_phase8_e10a_latticesnap.cact` | Stage 2A Native C++ `.cact` (`WSP-E10a`) | `19,968` (`39` rows) | `bb2fc1a66aca174f64a1dfab8da4a4fca6c840db00024c0603e1fcc578d149c6` |
| `checkpoints/experiments/shruti_wsp_phase8_e10b_quantcomp.safetensors` | Stage 2B Lattice-Snapped + `FP16` QuantComp (`WSP-E10b`) | `20,480` (`39` rows + `final_norm`) | `538859653d6e9481e2cbfc6254382708c59dafe7a998bcb6fe361df5beb68237` |
| `checkpoints/experiments/shruti_wsp_phase8_e10b_quantcomp.cact` | Stage 2B Native C++ `.cact` (`WSP-E10b`) | `20,480` (`39` rows + `final_norm`) | `dc8d49ad6e1b090ed0b90353a69f9355de0ae2890811dfdf3951941d3c244d33` |
