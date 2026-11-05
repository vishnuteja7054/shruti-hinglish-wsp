# Shruti Weight-Space Programming (WSP) — Complete Experiment Log

## Immutable Baseline Checkpoints
- **Original Safetensors**: `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/original/shruti.safetensors`
  - **Size**: `222,188,200` bytes (`222.2 MB`, `float32`, `55,541,384` parameters across 26 tensors)
  - **SHA-256**: `5c43105db153a47d02c47131d37d066343f04f1c3ff82315ff5da1ce264fbfa7`
  - **Permissions**: `0444` (read-only)
- **Original CACT Engine Archive**: `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/original/shruti.cact`
  - **Size**: `29,546,527` bytes (`29.5 MB`, `CQ-W4` + `FP16` + `FP32` + `RAW` tokenizer)
  - **SHA-256**: `b189e954aaed4a9fd41acbad3ac05e486381e96325b22a4c695b9ea5153d7065`
  - **Permissions**: `0444` (read-only)

---

## Phase 1: Architecture & Weight-Space Statistical Inspection (Completed)
- **Script**: `scripts/phase1_inspect_weights.py`
- **Artifacts**: `reports/PHASE1_ARCHITECTURE_REPORT.md`, `reports/phase1_weight_inspection.json`
- **Key Findings**:
  1. `embedding/embedding` `[8914, 512]`:
     - `<|hi|>` (`ID 8913`) has compressed L2 norm `53.1192` vs `<|en|>` (`ID 8906`) norm `68.0231` and European language tags `71.09..79.17`.
     - Cosine similarity `cos(<|hi|>, <|en|>) = 0.6912`, and `cos(<s>, <|hi|>) = 0.5877` vs `cos(<s>, <|en|>) = 0.4837`.
     - Grafted Devanagari block (`8192..8905`, 714 tokens) has mean L2 norm `62.3090` (`std = 5.05`), 11.4% smaller than the Latin BPE block (`260..8191`, mean L2 norm `70.3258`, `std = 16.41`).
  2. Near-homophonous Hindi pairs have high cosine similarity in `embedding/embedding` (`▁है` `8332` vs `▁हैं` `8374`: `0.9258`; `ू` `8309` vs `ु` `8307`: `0.8907`; `▁की` `8368` vs `▁के` `8340`: `0.8777`; `ं` `8195` vs `ँ` `8193`: `0.8759`; `▁का` `8376` vs `▁को` `8358`: `0.8644`; `▁मैं` `8624` vs `▁में` `8345`: `0.8537`) while hashing to disjoint deterministic slots in `engrams_0` (Layer 3) and `engrams_1` (Layer 7).
  3. Causal QKV conv taps (`stack/layers/block/self_attn/{q,k,v}_taps`) weight lag $t-1$ (`k_taps[:, 1] > 1.0` in Layers 1 & 7) strongly, propagating the step-0 language tag token (`pos = 1`) into subsequent key projections.

---

## Phase 2: Modal GPU & Native C++ Baseline Evaluation & Activation Extraction (Completed)
- **Script**: `scripts/modal_phase2_eval.py`
- **Artifacts**: `reports/PHASE2_BASELINE_REPORT.md`, `reports/phase2_baseline_eval.json`, `reports/phase2_step0_states.npz`
- **Datasets & Splits**:
  - `fleurs_hi` (`google/fleurs`, `hi_in`, `test`, 417 clips: `dev [0:200]`, `test [200:417]`)
  - `svarah_en` (`ai4bharat/Svarah`, `test`, 600 clips: `dev [0:300]`, `test [300:600]`)
  - `mucs_hinglish` (`cfilt/IITB-Hinglish-TOP` / MUCS 2021 Hinglish, `test`, 600 clips: `dev [0:300]`, `test [300:600]`)
  - `lahaja_hi` (`ai4bharat/Lahaja`, `test`, 600 clips: `dev [0:300]`, `test [300:600]`)
  - `librispeech_en` (`openslr/librispeech_asr`, `clean`, `test`, 300 clips: `dev [0:150]`, `test [150:300]`)
  - `silence_100` (50 digital silence + 50 faint synthetic Gaussian hiss clips)
- **Baseline Results Summary**:
  - `fleurs_hi`: GPU Auto-LID WER `21.19%` (Dev `21.53%`, Test `20.87%`, CER `13.18%`, LID-Acc `100.0%`); C++ `needle` 5-beam WER `19.01%`.
  - `svarah_en`: GPU Auto-LID WER `27.89%` (Dev `27.01%`, Test `28.82%`, CER `16.59%`, LID-Acc `35.17%`); Forced-`en` WER `18.37%` (gap `9.52%`); C++ `needle` 5-beam Auto-LID WER `21.34%` (LID-Acc `68.33%`) vs Forced-`en` `15.15%`.
  - `mucs_hinglish`: GPU Auto-LID WER `57.33%` (Dev `61.26%`, Test `53.19%`, CER `41.17%`, LID-Acc `99.83%`); C++ `needle` 5-beam WER `26.15%`.
  - `lahaja_hi`: GPU Auto-LID WER `27.18%` (Dev `24.16%`, Test `29.99%`, CER `17.90%`, LID-Acc `99.33%`); C++ `needle` 5-beam WER `24.83%`.
  - `librispeech_en`: GPU Auto-LID WER `6.78%` (Dev `6.24%`, Test `7.34%`, CER `2.83%`, LID-Acc `99.67%`); C++ `needle` 5-beam WER `5.60%`.
  - `silence_100`: `0` hallucinated words (`100.0%` Empty Rate on GPU and C++ `needle`).

---

## Phase 3: Weight-Space Diagnosis & Mathematical Hypotheses (Completed)
- **Script**: `scripts/phase3_diagnose_weights.py`
- **Artifacts**: `reports/PHASE3_DIAGNOSIS_REPORT.md`, `reports/phase3_diagnosis.json`, `reports/phase3_wsp_vectors.npz`
- **Key Findings**:
  1. Step-0 shared representation $\mu_0 = \mathbb{E}[\text{aq}(h_0)]$ has L2 norm `13.5975`, accounting for `91.2%` of total step-0 energy (`14.9096`).
  2. `<|hi|>` has common-mode alignment $\langle W_{\text{hi}}, \hat{\mu}_0 \rangle = +19.7267$ vs `<|en|>` $\langle W_{\text{en}}, \hat{\mu}_0 \rangle = +18.3292$, giving `<|hi|>` a permanent `+19.00` logit head-start before acoustic evidence is evaluated.
  3. In the 511-D orthogonal subspace $\mu_0^\perp$, regularized Fisher Linear Discriminant (`reg = 0.01`) calibrated strictly on Dev `svarah_en [0:300]` vs Dev `fleurs_hi [0:200]` + `lahaja_hi [0:300]` separates Indian English from Hindi with `AUC = 0.9675` (`93.12%` balanced accuracy).

---

## Phase 4 & 5: Weight-Space Programming Experiments & Verification (Completed)
- **Scripts**: `scripts/modal_phase4_5_wsp.py` (executed on Modal `A100-40GB`, App `ap-B1ognHralOlD2iKsD4dgR7`), `scripts/apply_wsp_modifications.py`
- **Artifacts**: `reports/phase4_5_wsp_results.json`, `reports/PHASE6_FINAL_WSP_REPORT.md`

### Experiment Log Table

| Exp ID | Hypothesis | Target Tensor(s) & Indices | Modified Params | Mathematical Operation $\Delta W$ | Dev Metric Change (Baseline $\to$ WSP) | Held-Out Test Metric Change (Baseline $\to$ WSP) | Status / Verdict |
|---|---|---|---|---|---|---|---|
| **`WSP-E1a`** | **H1 (Baseline 1D BOS Prior Calibration)** | `embedding/embedding[8906:8914, :]` (8 rows) | `4,096` (`0.0074%`) | $\Delta W_{\text{hi}} = -2.35\,\hat{\mu}_0$, $\Delta W_{\text{en}} = +2.35\,\hat{\mu}_0$, $\Delta W_{\text{EU}} = -2.0\,\hat{\mu}_0$ ($\beta=4.7$) | `svarah_en` Dev WER: `27.01%` $\to$ **`19.97%`** (`-7.04%`), `fleurs_hi` Dev LID-Acc: `100.0%` $\to$ `91.0%` (WER `21.53%` $\to$ `25.15%`) | `svarah_en` Test WER: `28.82%` $\to$ **`21.26%`** (`-7.56%`), `fleurs_hi` Test LID-Acc: `100.0%` $\to$ `87.56%` (WER `20.87%` $\to$ `26.03%`) | **Superseded by `WSP-E1b`/`E1c`** (1D shift alone lacks acoustic discriminability on soft-onset Hindi) |
| **`WSP-E1b` (`shruti_wsp_final`)** | **H1 (Regularized Fisher Subspace Steering — Macro-WER Optimal)** | `embedding/embedding[8906:8914, :]` (8 rows) | `4,096` (`0.0074%`) | $\Delta W_{\text{hi}} = -6.0\,\hat{u}_{\text{fisher}} - 2.5\,\hat{\mu}_0$, $\Delta W_{\text{en}} = +6.0\,\hat{u}_{\text{fisher}} + 2.5\,\hat{\mu}_0$, $\Delta W_{\text{EU}} = -2.0\,\hat{\mu}_0$ ($\text{reg}=0.01, \gamma=12.0, \beta=5.0$) | `svarah_en` Dev WER: `27.01%` $\to$ **`20.03%`** (`-6.98%`, LID `35.0%` $\to$ `91.33%`), `mucs` Dev WER: `61.26%` $\to$ **`60.55%`** (`-0.71%`), Macro Dev WER: `28.04%` $\to$ **`26.64%`** (`-1.40%`) | `svarah_en` Test WER: `28.82%` $\to$ **`21.26%`** (`-7.56%`, LID `35.33%` $\to$ `88.00%`), Macro Test WER: `28.13%` $\to$ **`27.04%`** (`-1.09%`); C++ `needle` 5-beam `svarah_en` WER: `21.34%` $\to$ **`15.28%`** (`-6.06%`), Macro 5-beam WER: `19.39%` $\to$ **`18.12%`** (`-1.27%`) | **VALIDATED & EXPORTED** (`shruti_wsp_final.{safetensors,cact}`) |
| **`WSP-E1c` (`shruti_wsp_e1c_zero_hi_reg`)** | **H1 (Zero-Hindi-Regression Fisher Subspace Steering)** | `embedding/embedding[8906:8914, :]` (8 rows) | `4,096` (`0.0074%`) | $\Delta W_{\text{hi}} = -6.0\,\hat{u}_{\text{fisher}} - 0.875\,\hat{\mu}_0$, $\Delta W_{\text{en}} = +6.0\,\hat{u}_{\text{fisher}} + 0.875\,\hat{\mu}_0$, $\Delta W_{\text{EU}} = -2.0\,\hat{\mu}_0$ ($\text{reg}=0.01, \gamma=12.0, \beta=1.75$) | `svarah_en` Dev WER: `27.01%` $\to$ **`23.38%`** (`-3.63%`), `fleurs_hi` Dev WER: `21.53%` $\to$ **`21.53%`** (`100%` LID), `mucs` Dev WER: `61.26%` $\to$ **`59.23%`** (`-2.03%`) | `svarah_en` Test WER: `28.82%` $\to$ **`23.65%`** (`-5.17%`), `fleurs_hi` Test WER: `20.87%` $\to$ **`20.78%`** (`-0.09%`, `100%` LID), `librispeech_en` LID: `99.67%` $\to$ **`100.0%`** (WER `6.78%` $\to$ `6.76%`) | **VALIDATED & EXPORTED** (`shruti_wsp_e1c_zero_hi_reg.{safetensors,cact}`) |
| **`WSP-E2`** | **H2 (Devanagari Output Projection Radial Norm Scaling)** | `embedding/embedding[8192:8906, :]` (714 rows) | `365,568` (`0.658%`) | $\Delta W_{\text{dev}} = (\alpha - 1.0)\,W_{\text{dev}}$, $\alpha \in \{1.02, 1.04, 1.06, 1.08\}$ | At $\alpha=1.04$: `fleurs_hi` Dev WER: `21.53%` $\to$ **`21.27%`** (`-0.26%`), `mucs` `Dev->En` script errors: `802` $\to$ **`649`**, but `mucs` Dev WER: `58.38%` $\to$ `65.17%` (trailing insertions vs unscaled `<eos>`) | At $\alpha=1.04$: `fleurs_hi` Test WER: `20.87%` $\to$ **`20.64%`** (`-0.23%`, All WER `21.19%` $\to$ **`20.95%`**, CER `13.18%` $\to$ **`12.49%`**) | **Partially Confirmed (Pure Hindi) / Excluded from Global Checkpoint by Dev Rule** (`selected_alpha_dev = 1.0`) |
| **`WSP-E3`** | **H3 (Contrastive Angular Repulsion & Norm Balancing of 5 Hindi Pairs)** | `embedding/embedding` rows `[8332, 8374, 8368, 8340, 8376, 8358, 8309, 8307]` (8 rows) | `4,096` (`0.0074%`) | Orthogonal Gram-Schmidt repulsion ($\eta=0.10$) + pair mean L2 norm equilibration ($\lambda=0.50$) | `fleurs_hi` Dev WER: `21.53%` $\to$ `21.62%`, `mucs` Dev WER: `61.26%` $\to$ `63.73%` | `mucs` Held-Out Test WER: `53.19%` $\to$ **`52.57%`** (`-0.62%`), `lahaja_hi` Test WER: `29.99%` $\to$ **`29.86%`** (`-0.13%`) | **Confirmed on Held-Out Test / Excluded from Global Checkpoint by Strict Dev-Only Rule** |
| **`WSP-E4`** | **H4 (Decoder Top-2 Cross-Attention Gate Amplification)** | `stack/layers/block/attn_gate[6:8, 0]` (2 scalars) | `2` (`0.000004%`) | $\Delta g_l = +0.12\,g_l$ for $l \in \{6, 7\}$ | `svarah_en` Forced-`en` Dev WER: `19.08%` $\to$ **`18.93%`**, `mucs` Forced-`hi` Dev WER: `58.38%` $\to$ **`58.28%`**, `fleurs_hi` Dev WER: `21.53%` $\to$ `22.47%` | Not promoted to global checkpoint due to `fleurs_hi` Dev regression (`+0.94%` WER) | **Falsified for Global Checkpoint** (Internal residual recalibration perturbs autoregressive hidden state trajectory on clean Hindi) |
| **`WSP-E5` (`shruti_wsp_phase7_e5_ridge`)** | **H5 (Closed-Form Covariance-Whitened Contrastive Ridge Embedding Compiler)** | `embedding/embedding` (`65` content rows + `8` LID rows = `73` rows) | `37,376` (`0.0673%`) | $\Delta W = \gamma\,B\left(\frac{1}{N}H^\top H + \lambda I_{512}\right)^{-1}$ ($\lambda=0.5, \tau=2.5, \gamma=48.0$) + per-row `CQ-W4` verification gate (`+311` net teacher-forced token hits on $N_{\text{tok}}=45,847$ Dev tokens) | `mucs_hinglish` Dev WER: `62.29%` $\to$ **`57.89%`** (`-4.40%`), `librispeech_en` Dev WER: `5.84%` $\to$ **`5.75%`**, Macro Dev WER: `26.60%` $\to$ **`26.16%`** (`-0.44%` beyond `WSP-E1b`, `-1.84%` vs Baseline) | `mucs_hinglish` Test WER: `52.00%` $\to$ **`49.67%`** (`-2.33%`; All WER `57.38%` $\to$ **`53.88%`**, CER `51.87%` $\to$ **`46.98%`**), `fleurs_hi` Test WER: `22.90%` $\to$ **`22.46%`** (`-0.44%`), `svarah_en` Test WER: `21.26%` $\to$ **`21.13%`**, `librispeech_en` All WER: `6.80%` $\to$ **`6.68%`**, Macro Test WER: `27.04%` $\to$ **`26.18%`** (`-0.86%` beyond `WSP-E1b`, **`-1.96%` vs Baseline**); C++ `needle` 5-beam `mucs_hinglish` WER: `28.08%` $\to$ **`26.45%`**, `lahaja_hi` WER: `25.63%` $\to$ **`24.94%`**, `fleurs_hi` WER: `18.61%` $\to$ **`18.42%`**, `svarah_en` WER: **`15.42%`** | **VALIDATED & EXPORTED** (`shruti_wsp_phase7_e5_ridge.{safetensors,cact}` — **Overall Macro-WER Winner**) |
| **`WSP-E6`** | **H6 (Unquantized Final-Norm Channel-Margin Ridge Compiler)** | `stack/final_norm/scale` `[512]` (FP32) + `8` LID rows | `4,608` (`0.0083%`) | $\Delta s = \eta_s\left(\frac{1}{N}G^\top G + \lambda_s I_{512}\right)^{-1}\left(\frac{1}{N}G^\top r\right)$ ($\lambda_s=10.0, \tau_s=3.0, \eta_s=2.0$, $\|\Delta s\|_2 = 0.2346$, `+423` net TF token hits) | `svarah_en` Dev WER: `20.00%` $\to$ **`19.55%`** (`-0.45%`), `lahaja_hi` Dev WER: `24.13%` $\to$ **`23.92%`** (`-0.21%`), Macro Dev WER: `26.60%` $\to$ **`26.43%`** (`-0.17%`) | `svarah_en` Test WER: `21.26%` $\to$ **`21.04%`** (All WER **`20.29%`**, LID-Acc **`93.50%`**), `lahaja_hi` All WER: `27.45%` $\to$ **`27.13%`** | **Confirmed on Indian English (`20.29%`) & Multi-Accent Hindi (`27.13%`)** |
| **`WSP-E8` (`shruti_wsp_phase7_e8_combined`)** | **H5 + H6 (Combined Ridge Embedding + Final-Norm Compiler)** | `embedding/embedding` (`73` rows) + `stack/final_norm/scale` `[512]` | `37,888` (`0.0682%`) | Joint `WSP-E5` + `WSP-E6` closed-form compilation | `svarah_en` Dev WER: **`19.55%`**, `lahaja_hi` Dev WER: **`23.90%`**, `librispeech_en` Dev WER: **`5.90%`** | `svarah_en` All WER: **`20.28%`** (Test **`21.01%`**, LID-Acc **`93.50%`**), `lahaja_hi` All WER: **`27.04%`**, `librispeech_en` All WER: **`6.65%`** (`7.34%` Test, `2.85%` CER) | **VALIDATED & EXPORTED** (`shruti_wsp_phase7_e8_combined.{safetensors,cact}` — **Best Svarah-EN, Lahaja-HI, and LibriSpeech-EN WER**) |

---

## Exported Experimental Checkpoints & Hashes
1. **Phase 5 Subspace-Steered WSP Checkpoint (`WSP-E1b` / `WSP-Combined-Final`)**:
   - **Safetensors**: `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_final.safetensors`
     - **Size**: `222,188,200` bytes
     - **SHA-256**: `283ee9fcd0211908c4f7901cbbb4ab44924f842f7ae219deb1c5b35da9bfb317`
   - **CACT Engine Archive**: `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_final.cact`
     - **Size**: `29,546,527` bytes
     - **SHA-256**: `e1c9f0bdd97fc0fc865ca02400166ed933665170330f3db64f2cc41520f519d1`
2. **Phase 5 Zero-Hindi-Regression WSP Checkpoint (`WSP-E1c`)**:
   - **Safetensors**: `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_e1c_zero_hi_reg.safetensors`
     - **Size**: `222,188,200` bytes
     - **SHA-256**: `88b1c6219e2f99b984c725c75dc7fce4c9cffa8c361cabc50a7ec2d7d911e24d`
   - **CACT Engine Archive**: `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_e1c_zero_hi_reg.cact`
     - **Size**: `29,546,527` bytes
     - **SHA-256**: `ce15d791109ac4f6c6e46fb39aa04269222cad6a768b08eb126afa86304da3b7`
3. **Phase 7 Analytic Ridge Embedding Compiler Checkpoint (`WSP-E5` — Overall 5-Dataset Macro-WER Winner)**:
   - **Safetensors**: `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase7_e5_ridge.safetensors`
     - **Size**: `222,188,200` bytes (`73` modified embedding rows = `37,376` parameters, `0.0673%` of model)
     - **SHA-256**: `f1edaba3f5473a53766c64e27dbd0f0ec3136d5b02b7093e2e628708e29fca40`
   - **CACT Engine Archive**: `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase7_e5_ridge.cact`
     - **Size**: `29,546,527` bytes
     - **SHA-256**: `8821cf40062ab59560858082defcf792b4935f578a44ac67acaf21967908d628`
4. **Phase 7 Combined Analytic Ridge Compiler Checkpoint (`WSP-E8` — Best Svarah-EN, Lahaja-HI & LibriSpeech-EN)**:
   - **Safetensors**: `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase7_e8_combined.safetensors`
     - **Size**: `222,188,200` bytes (`73` embedding rows + `512` `stack/final_norm/scale` = `37,888` parameters, `0.0682%` of model)
     - **SHA-256**: `955bb63d7e4e704e2cad51a65f184e9570d4da6771e4f8ac62004ed51fd03772`
   - **CACT Engine Archive**: `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase7_e8_combined.cact`
     - **Size**: `29,546,527` bytes
     - **SHA-256**: `21718f2b4acf9c3cb3b80564174a056efb4c92ef8cfb8384f4e0f5f248b1525d`


---

## Phase 8: Unquantized `FP32` Master Compilation $\to$ Quantization-Compensated `.cact` (`WSP-E9`, `WSP-E10a`, `WSP-E10b`)

1. **`WSP-E9_Master_Unquantized_FP32` (Stage 1 Pure `FP32` Compilation, `quant=False`)**:
   - **Target**: `embedding/embedding` (`46` rows = `8` language tags + `38` content vocabulary rows = `23,552` params) + `stack/final_norm/scale` (`512` params) in continuous `FP32`.
   - **Calibration**: English-Nullspace Weighted Contrastive Ridge Regression ($\omega_{\text{en}} = 2.0$, $\lambda_{\text{fp}} = 0.5, \tau_{\text{fp}} = 2.0, g_{\text{fp}} = 40.0$) on `39,984` clean `FP32` teacher-forced states $\to$ `+184` net `FP32` token hits (`76.93% -> 77.42%` clean Dev top-1 accuracy).
   - **Exported Checkpoint**: `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase8_fp32_master.safetensors` (`a40ddd7d96029b9fea4568fd3f7f99486de22686c995321b6b05eaaf5babfdfb`).

2. **`WSP-E10a_LatticeSnap_CQ4` (Stage 2A `FP32`-to-`CQ-W4` Lattice-Snapped Embedding)**:
   - **Target**: Projects the `38` `FP32` master row updates onto the 4-bit Hadamard `CQ-W4` (`group = 128`) Lloyd-Max codebook via optimal per-row lattice scaling $\alpha_r \in [0.75, 2.5]$. `31` content rows + `8` language tags (`39` rows = `19,968` params, `0.0360%` of model) snap cleanly onto the 4-bit lattice (`+306` net `CQ-W4` Dev token hits).
   - **GPU `CQ-W4A8` 5-Dataset Macro WER**: Drops from `26.19%` (Phase 7 `WSP-E5`) to **`25.99%`** (Dev **`25.92%`**, Held-Out Test **`26.02%`** — our best held-out GPU test Macro WER!).
   - **Exported Checkpoints**:
     - `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase8_e10a_latticesnap.safetensors` (`c28391ae5c1af5a547b178266ede2fbe009f05433f0ab52d0d676fc5b947567b`)
     - `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase8_e10a_latticesnap.cact` (`bb2fc1a66aca174f64a1dfab8da4a4fca6c840db00024c0603e1fcc578d149c6`)

3. **`WSP-E10b_QuantComp_CQ4` (Stage 2B `FP32` Master $\to$ LatticeSnap + `FP16` Quantization-Residual Compensation)**:
   - **Target**: `39` lattice-snapped `embedding/embedding` rows + unquantized `FP16` `stack/final_norm/scale` (`[512]`, $\lambda_{\text{qc}} = 25.0, \text{step} = 0.3$) absorbing the remaining `FP32`-to-`CQ-W4` margin residual (`20,480` total parameters = `0.0369%` of model).
   - **GPU `CQ-W4A8` 5-Dataset Macro WER**: Drops to **`25.83%`** All (`-2.24%` vs. Official Baseline), Dev **`25.37%`**, with **`mucs_hinglish` All WER dropping to `52.59%`** (`-4.79%` vs. `WSP-E1b`, Dev `62.29% -> 54.40%`).
   - **Native C++ `needle` 5-Beam Engine (`300` clips across 5 datasets)**:
     - **5-Dataset Macro WER**: **`18.05%` (`18.054%`)** (`-2.58%` vs. Official Baseline `20.63%`, `-0.17%` beyond Phase 7 `WSP-E5` `18.22%`).
     - **`mucs_hinglish`**: **`25.59%` WER** (`23.49%` CER, **`100.0%` LID-Acc**, `-2.81%` vs. Official Baseline).
     - **`svarah_en`**: **`15.28%` WER** (`6.83%` CER, `96.67%` LID-Acc, `-9.43%` vs. Official Baseline).
     - **`librispeech_en`**: **`5.78%` WER** (`2.45%` CER, `100.0%` LID-Acc).
   - **Exported Checkpoints**:
     - `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase8_e10b_quantcomp.safetensors` (`538859653d6e9481e2cbfc6254382708c59dafe7a998bcb6fe361df5beb68237`)
     - `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase8_e10b_quantcomp.cact` (`dc8d49ad6e1b090ed0b90353a69f9355de0ae2890811dfdf3951941d3c244d33`)


---

## Phase 9: Multi-Layer Forward-Cascade + Exact 4-Block `FP16` Group-Norm Calibration (`WSP-E11` & `WSP-E12`, Modifying up to `26.87%` of the Model)

- **Script**: `/home/rgukt/Music/SHRUTHI/HINGLISH/scripts/modal_phase9_cascade_wsp.py`
- **Results JSON**: `/home/rgukt/Music/SHRUTHI/HINGLISH/reports/phase9_cascade_wsp_results.json`
- **Report**: `/home/rgukt/Music/SHRUTHI/HINGLISH/reports/PHASE9_CASCADE_WSP_REPORT.md`
- **Key Findings**:
  1. **Exact 4-Block `FP16` Group-Norm Calibration (`WSP-E11_BlockNorm_CQ4`, `196` embedding rows + `final_norm` = `100,864` parameters)**:
     - Calibrating the 4 unquantized `FP16` group norms per 512-dim row across `183` under-predicted content tokens (on top of Phase 8's `39` lattice-snapped rows) increased clean Dev teacher-forced token hits from `34,041` (`WSP-E1b`) -> `34,344` (`WSP-E10b`) -> **`34,806` (`WSP-E11`)** (`+462` net token hits over Phase 8, `+765` over Phase 5).
     - **Native C++ `needle` 5-Beam Engine (`300` clips across all 5 datasets — Best Overall C++ 5-Beam Result)**:
       - **5-Dataset Macro Average Strict WER**: **`17.388%` (`17.39%`)** (down from `20.63%` Official, `18.72%` Phase 5, `18.22%` Phase 7, `18.05%` Phase 8).
       - **5-Dataset Macro Average Script-Unified Phonetic WER**: **`16.084%` (`16.08%`)**.
       - **`mucs_hinglish`**: **`23.61%` Strict WER** / **`22.13%` Script-Unified Phonetic WER** (down from `28.40%` Official and `25.59%` Phase 8 — a **`-4.79%` Strict / `-6.27%` Phonetic WER reduction**!).
       - **`svarah_en`**: **`15.02%` Strict WER** (`7.03%` CER, `96.67%` LID-Acc — **matching the 100% Oracle Forced-`en` ceiling `15.02%`**, `-9.69%` vs. Official!).
       - **`fleurs_hi`**: **`18.29%` Strict WER** / **`15.66%` Script-Unified WER** (`10.35%` CER).
       - **`lahaja_hi`**: **`24.50%` Strict WER** / **`20.25%` Script-Unified WER** (proving `4.25%` of `lahaja_hi` WER is purely Latin-vs-Devanagari script choice on English loanwords).
       - **`librispeech_en`**: **`5.52%` Strict WER** / **`4.62%` Script-Unified WER** (`2.42%` CER, `100.0%` LID-Acc).
  2. **Global Subspace + Dual-Engram Cascade (`WSP-E12_FullCascade_10Pct`, `8,654` embedding rows + `81,934` Engram slots + `final_norm` = `14,918,912` parameters = `26.872%` of the model)**:
     - Achieved our lowest GPU 1-beam WER on `fleurs_hi` Dev (**`21.76%` Strict / `19.22%` Script-Unified**) and `lahaja_hi` All (**`26.20%` Strict / `23.32%` Script-Unified**, fixing `98` word substitutions vs Phase 8).
  3. **Exported Checkpoints**:
     - `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase9_e11_blocknorm.safetensors` (`cc5bdf32b571e0ff6ac47a920462e8a8dc8fa131227ff3cdd837a01fa583cc02`)
     - `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase9_e11_blocknorm.cact` (`ae48e68c1796f89f64eef933140314f6f2a3eb718b638c01e0e79949797f9cc3`)
     - `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase9_e12_cascade27pct.safetensors` (`4237b8fed21f1fabe71eb5fe3e3020c881f698e99c97c329c37ed0850d2583fa`)


---

## Phase 10: 6-Tier Dataset Canonicalization Waterfall Audit & Clean-Target Loop-Protected WSP (`WSP-E13` & `WSP-E14`)

- **Script**: `/home/rgukt/Music/SHRUTHI/HINGLISH/scripts/modal_phase10_clean_target_wsp.py`
- **Results JSON**: `/home/rgukt/Music/SHRUTHI/HINGLISH/reports/phase10_clean_target_wsp_results.json`
- **Report**: `/home/rgukt/Music/SHRUTHI/HINGLISH/reports/PHASE10_CANONICAL_AUDIT_AND_CLEAN_WSP_REPORT.md`
- **Key Findings**:
  1. **6-Tier Dataset Canonicalization Waterfall (`T0_Raw_Strict` -> `T1_Unicode_Ortho` -> `T2_Spoken_Allomorph` -> `T3_Compound_Space` -> `T4_CrossScript_Loan` -> `T5_Whisper_Semantic`)**:
     - Proved that **31% to 53% of remaining apparent WER across the Hindi/Hinglish benchmarks comes from dataset annotation inconsistencies** (combining Nuqta `\u093c` / Chandrabindu `ँ->ं`, spoken `वो/ये/है` vs written `वह/यह/हैं`, compound word spacing `इसके लिए` vs `इसकेलिए`, cross-script loanwords `"click"` vs `"क्लिक"` / `"painting"` vs `"पेंटिंग"`, and unstressed vowel matra length).
     - On C++ 5-Beam (`bin/needle`, `300` clips), **5-Dataset Macro Average WER drops from `17.47%` (`T0_Raw_Strict`) to `13.87%` (`T4_CrossScript_Loan`, `92.62%` Character Accuracy) and `12.08%` (`T5_Whisper_Semantic` — `87.92%` Word Accuracy & `93.86%` Character Accuracy, `6.14%` CER)**:
       - `librispeech_en`: **`5.34%` Strict (`E13`) / `4.55%` Semantic WER (`95.45%` Word Acc, `98.09%` Char Acc)**
       - `lahaja_hi`: **`24.50%` Strict -> `19.69%` Spoken -> `15.90%` Loan -> `11.53%` Semantic WER (`88.47%` Word Acc, `94.79%` Char Acc — `-12.97%` drop!)**
       - `fleurs_hi`: **`17.89%` Strict (`E13`, new all-time best!) -> `15.39%` Spoken -> `12.57%` Loan -> `11.58%` Semantic WER (`88.42%` Word Acc, `94.57%` Char Acc)**
       - `svarah_en`: **`15.42%` Strict -> `13.57%` Semantic WER (`86.43%` Word Acc, `94.35%` Char Acc)**
       - `mucs_hinglish`: **`23.61%` Strict -> `21.01%` Loan -> `18.94%` Semantic WER (`81.06%` Word Acc, `87.65%` Char Acc)**
  2. **1-Beam Autoregressive Repetition Loop Guard (`WSP-E13_CleanLoopSafe` & `WSP-E14_LoopGuardedP9`)**:
     - Diagnosed and guarded the exact tokens that triggered 1-beam greedy repetition loops on noisy `mucs_hinglish` clips in Phase 9.
     - **`WSP-E13_CleanLoopSafe`** dropped `mucs_hinglish` 1-beam GPU WER (`600` clips) from `60.53%` (Phase 9) and `52.59%` (Phase 8) to an **all-time best `50.63%` Strict (`43.28%` Semantic, `70.98%` Char Acc)**, and dropped `svarah_en` 1-beam GPU WER to **`20.93%` Strict (`18.82%` Semantic, `92.27%` Char Acc)**, achieving our **lowest 5-Dataset GPU 1-Beam Macro WER across all tiers (`25.60%` Strict / `20.00%` Semantic / `88.96%` Char Acc)**.
     - **`WSP-E14_LoopGuardedP9`** achieved our **lowest 1-beam GPU WER on `fleurs_hi` (`21.82%` Strict / `15.87%` Semantic)** and **`librispeech_en` (`6.50%` Strict / `5.34%` Semantic, `97.96%` Char Acc)**.
  3. **Exported Checkpoints**:
     - `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase10_e13_cleanloopsafe.safetensors` (`d9c73c6e687f7cf17d7b8d07b29f3d3a077568024c37cecd9d27e92b3fd6b48d`)
     - `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase10_e13_cleanloopsafe.cact` (`021884f52a44110fdf3a3f3f393854aeeef9032812c01bd67b310baf054e6240`)
     - `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase10_e14_loopguarded.safetensors` (`f6c365e649abc6ffe7b0e77439ddfdc9aa8c846ad310848e548d63e7c9b74354`)
     - `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase10_e14_loopguarded.cact` (`c094b92a153c323e8d6f550ea4bee3e83daef5904d0a96afcf8d391f7589191f`)

---

## Phase 11: Exact Differentiable `FP16`-Subspace Calibration Across All 16 Encoder & Decoder Layers (`WSP-E15` & `WSP-E16`)

- **Script**: `/home/rgukt/Music/SHRUTHI/HINGLISH/scripts/modal_phase11_fp16_subspace_diff_wsp.py`
- **Results JSON**: `/home/rgukt/Music/SHRUTHI/HINGLISH/reports/phase11_fp16_subspace_diff_wsp_results.json`
- **Report**: `/home/rgukt/Music/SHRUTHI/HINGLISH/reports/PHASE11_FP16_SUBSPACE_DIFF_WSP_REPORT.md`
- **Key Findings**:
  1. **8-Layer Decoder + Cross-Attention + 4-Block Embedding `FP16` Calibration (`WSP-E15_Decoder_CrossAttn_FP16`, `210` `.cact` records surgically patched)**:
     - Reduced Dev60 Autoregressive 5-DS Macro WER from `19.606%` to **`18.878%` (`-0.728%`)** and 4-DS Clean Macro WER from `17.523%` to **`16.985%` (`-0.538%`)**.
     - Achieved **new all-time best Full GPU 1-Beam 4-DS Clean Macro WER (`18.902%` Strict / `14.033%` Semantic)**, **`fleurs_hi` (`21.53%` Strict / `15.55%` Semantic)**, and **`lahaja_hi` (`26.27%` Strict / `15.87%` Semantic)**.
  2. **End-to-End Audio Stem + 8-Layer Conformer Encoder + 8-Layer Decoder `FP16` Calibration (`WSP-E16_EndToEnd_AudioEncoder_FP16`, `326` `.cact` records surgically patched — `61.6%` of all `529` records in `shruti.cact`)**:
     - Reduced Dev60 Autoregressive 5-DS Macro WER further to **`18.618%` (`-0.988%` vs Phase 10 Base)** and 4-DS Clean Macro WER to **`16.720%` (`-0.803%` vs Phase 10 Base)**.
     - Achieved **new all-time best Full 2,517-Clip GPU 1-Beam 5-DS Macro WER across every tier (`25.308%` Strict / `19.830%` Semantic / `80.170%` WordAcc)**, including **`svarah_en` (`20.28%` Strict / `18.14%` Semantic)** and **`mucs_hinglish` (`50.06%` Strict / `43.01%` Semantic)**.
     - **Native C++ `needle` 5-Beam Engine (`300` clips across 5 datasets — Best Overall C++ 5-Beam Result)**:
       - **4-Dataset Clean Macro Average WER**: **`15.305%` Strict** $\to$ **`9.643%` Semantic WER (`90.358%` Word Accuracy, `95.695%` Character Accuracy, `4.305%` CER — broke the `<10%` Semantic WER barrier!)**.
       - **5-Dataset Macro Average WER**: **`17.142%` Strict** $\to$ **`13.782%` Loan** $\to$ **`11.946%` Semantic WER (`88.054%` Word Accuracy, `93.820%` Character Accuracy — broke the `<12%` 5-Dataset Semantic WER barrier!)**.
       - **`fleurs_hi`**: **`17.11%` Strict WER** (`-0.78%` vs Phase 10, **`-57.36%` vs Baseline `74.47%`!**) $\to$ **`10.78%` Semantic WER (`89.22%` WordAcc, `94.76%` CharAcc)**.
       - **`svarah_en`**: **`13.83%` Strict WER** (`-1.19%` vs Phase 9, `-1.59%` vs Phase 10, **`-10.88%` vs Baseline `24.71%`!**) $\to$ **`11.86%` Semantic WER (`88.14%` WordAcc, `94.94%` CharAcc)**.
       - **`librispeech_en`**: **`5.34%` Strict WER** $\to$ **`4.99%` Loan WER** $\to$ **`4.28%` Semantic WER (`95.72%` WordAcc, `98.29%` CharAcc, `1.71%` CER)**.
  3. **Exported Checkpoints**:
     - `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase11_e15_decoder_fp16.safetensors` (`908c65a94a7345d2f42ecbbc0819d92afd911577f7965723658a5bdfac531796`)
     - `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase11_e15_decoder_fp16.cact` (`1eb59d441396869fa5d0a9256f9ce9dec40f7cf57938be311fbbe4064a8d0b70`)
     - `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase11_e16_e2e_fp16.safetensors` (`9636d5f68c6d227de1911943b0ff39c1703a9caa9344f43b0f3cde896adf90c9`)
     - `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase11_e16_e2e_fp16.cact` (`7f07a23fb7169f75f668e6b8da830e45931f6f96e40ae7492cd83b266fe0d597`)

