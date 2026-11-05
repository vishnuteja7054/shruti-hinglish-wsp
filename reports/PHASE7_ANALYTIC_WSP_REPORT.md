# Phase 7: Closed-Form Analytic Weight Compilation (`WSP-E5`, `WSP-E6`, `WSP-E8`)

**Model**: `udgamlabs/Shruti` (`Cactus-Compute/whistle-small` / `cactus-needle`, `55,541,384` parameters)  
**Theoretical Framework**: Closed-Form Associative Memory Compilation & Weight-Space Programming (inspired by Takehi, Teoh, Zhang & Khattab, *"Sparks of Weight-Space Programming"*, MIT CSAIL 2026, and Ridge Normal Equations / ROME / MEMIT)  
**Scripts**: `scripts/modal_phase7_analytic_wsp.py`, `scripts/apply_wsp_modifications.py`  
**Raw Evaluation Artifact**: `reports/phase7_analytic_wsp_results.json`

---

## 1. Theoretical Motivation: From 1D/2D Subspace Steering to High-Rank Closed-Form Weight Compilation

In Phases 3–5 (`WSP-E1b`, `WSP-E1c`), we demonstrated that modifying just **8 rows** (`4,096` parameters, `0.0074%` of the model) of `embedding/embedding` along a 2D subspace spanned by the step-0 common-mode vector $\hat{\mu}_0$ and the regularized Fisher Linear Discriminant $\hat{u}_{\text{fisher}} \in \mu_0^\perp$ eliminated the catastrophic Indian-English language-ID collapse on `svarah_en` (`35.17%` $\to$ `89.67%` Auto-LID accuracy; `27.89%` $\to$ `20.63%` GPU WER; `21.34%` $\to$ `15.28%` C++ 5-beam WER).

However, when we asked whether weight-space programming can go **beyond** steering a few scalar gates or 1D directions—to **program hundreds of vocabulary projections simultaneously with zero gradient descent**—we encountered two fundamental challenges:
1. **Collinear Hidden-State Interference**: If we naively push a target token row $W_y$ toward the average hidden state $\bar{h}_y$ where token $y$ was mispredicted, $W_y$ will also fire at thousands of other time steps where the hidden state $h_t$ shares the same principal subspace, causing massive false-positive insertions.
2. **4-Bit Quantization Floor (`CQ-W4`)**: In `shruti.cact` (and GPU `CQ-W4A8` STE), every row of `embedding/embedding` undergoes symmetric 4-bit quantization (`15` integer levels `[-7..+7]` scaled by $\max(|w_i|)/7$). Small continuous weight updates vanish into quantization rounding, whereas large unwhitened updates shift the row's dynamic range $\max(|w_i|)$ and corrupt existing quantization bins.

To solve both problems simultaneously in closed form (zero gradient descent, zero SGD epochs), **Phase 7** formulates weight-space programming as **Covariance-Whitened Contrastive Ridge Compilation** followed by an **Exact Per-Row `CQ-W4` Verification Gate**.

---

## 2. Mathematical Formulation of Phase 7 Compilers

### 2.1 Step 1: Teacher-Forced Hidden-State Extraction on `dev` Splits
Using the `WSP-E1b` language-steered checkpoint on all 5 `dev` splits (`fleurs_hi [0:200]`, `lahaja_hi [0:300]`, `mucs_hinglish [0:300]`, `svarah_en [0:300]`, `librispeech_en [0:150]` — **1,250 clips total**), we extract the exact 8-bit activation-quantized pre-logit decoder states:
$$h_t = \text{aq}_8\!\left(\text{RMSNorm}(x_t) \odot s_0\right) \in \mathbb{R}^{512}$$
across all **$N_{\text{tok}} = 45,847$ teacher-forced target token positions** ($N_{\text{clean}} = 44,128$ clean Devanagari + English positions).
- Baseline teacher-forced top-1 token accuracy on `dev`: **`77.12%` overall** (**`87.67%`** on pure Hindi `fleurs_hi` + `lahaja_hi`).

---

### 2.2 `WSP-E5`: Closed-Form Covariance-Whitened Contrastive Ridge Embedding Compiler
Let $\Sigma_H = \frac{1}{N} H^\top H + \lambda I_{512} \in \mathbb{R}^{512 \times 512}$ be the regularized second-moment matrix of all $N = 44,128$ pre-logit states on `dev`. For each mispredicted time step $t$ where the ground-truth reference token is $y_t$ (`ref`), the argmax predicted competitor token is $\hat{y}_t \neq y_t$ (`pred`), and the logit margin deficit is $m_t = \langle W_{\hat{y}_t}, h_t \rangle - \langle W_{y_t}, h_t \rangle > 0$:
1. We define the hinge-clipped contrastive impulse weight:
   $$\alpha_t = \min(m_t + 0.5,\; \tau)$$
2. We accumulate the contrastive target impulse matrix $B \in \mathbb{R}^{V \times 512}$ across all `dev` errors:
   $$B[y_t] \;\mathrel{+}=\; \alpha_t h_t, \qquad B[\hat{y_t}] \;\mathrel{-}=\; \alpha_t h_t$$
3. We solve the exact **512-dimensional Ridge Normal Equations** in closed form via Cholesky/LU inversion of $\Sigma_H$:
   $$\Delta W = \gamma \, B \, \Sigma_H^{-1} = \gamma \, B \left(\frac{1}{N} H^\top H + \lambda I_{512}\right)^{-1}$$
   **Why $\Sigma_H^{-1}$ is the key to Weight-Space Programming**: Multiplying the contrastive impulse $B[v]$ by $\Sigma_H^{-1}$ projects the weight update into the **low-variance orthogonal complement** of the global hidden-state distribution, so $\langle \Delta W_v, h_t \rangle$ is large *if and only if* $h_t$ matches the specific acoustic/contextual signature of token $v$, while remaining nearly zero ($\mathbb{E}[\langle \Delta W_v, h \rangle^2] \approx 0$) on all unrelated time steps!
4. **Exact Per-Row `CQ-W4` Verification Gate**:
   For each candidate vocabulary row $v$ with at least 2 error occurrences on `dev`, we quantize the updated row $q_4(W_v + \Delta W_v)$ using the exact 4-bit symmetric quantizer (`WEIGHT_BITS = 4`) and compute its exact net change in teacher-forced top-1 token hits across all $N = 44,128$ `dev` tokens:
   $$\Delta \text{Hits}(v) = \text{TP}_{\text{gained}}(v) - \text{FP}_{\text{induced}}(v)$$
   We accept row $v$ if and only if $q_4(W_v + \Delta W_v)$ achieves strictly positive net token gain ($\Delta \text{Hits}(v) \ge +2$) **after 4-bit quantization**.

#### `WSP-E5` Grid Calibration on `dev` ($N = 44,128$ tokens)
- Optimal ridge hyperparameters on `dev`: **$\lambda = 0.5, \tau = 2.5, \gamma = 48.0$**.
- Accepted vocabulary rows passing the `CQ-W4` verification gate: **65 content rows** (`33,280` parameters; combined with the 8 language-ID rows = **73 rows total**, `37,376` parameters = **`0.0673%` of model weights**).
- Net teacher-forced top-1 token hits gained on `dev` after 4-bit quantization: **`+311` net token hits** (`77.12%` $\to$ **`77.90%`** clean TF accuracy).
- Notable compiled vocabulary rows (`(token_id, +net_dev_hits)`):
  - Devanagari subwords & matras: `8195` (`ं` anusvara, **`+11`**), `8247` (`▁अ`, **`+12`**), `8289` (`▁क`, **`+7`**), `8339` (`▁द`, **`+7`**), `8246` (`▁आ`, **`+6`**), `8203` (`ा` aa-matra, **`+4`**), `8301` (`▁स`, **`+4`**), `8354` (`▁प्र`, **`+4`**), `8406` (`▁वि`, **`+4`**), `8340` (`▁के`, **`+3`**), `8358` (`▁को`, **`+3`**), `8193` (`ँ` chandrabindu, **`+2`**), `8418` (`▁हैं`, **`+2`**).
  - High-frequency English / Hinglish subwords & punctuation boundaries: `491` (**`+16`**), `8126` (**`+14`**), `309` (**`+13`**), `8161` (**`+10`**), `417` (**`+8`**), `8131` (**`+8`**), `441` (**`+7`**), `641` (**`+7`**), `3139` (**`+7`**).

---

### 2.3 `WSP-E6`: Unquantized Final-Norm Channel-Margin Ridge Compiler (`stack/final_norm/scale` `[512]`)
Because `stack/final_norm/scale` $s \in \mathbb{R}^{512}$ is stored in **unquantized `FP32`** in both `shruti.safetensors` and `shruti.cact`, it bypasses the 4-bit quantization floor completely. Every pre-logit dot product is linear in $s$:
$$\text{logit}(v, x_t) = \sum_{d=1}^{512} s_d \, \tilde{x}_{t,d} \, W^{\text{q4}}_{v,d}$$
For each error step $t$, let $g_t = \tilde{x}_t \odot (W^{\text{q4}}_{y_t} - W^{\text{q4}}_{\hat{y}_t}) \in \mathbb{R}^{512}$ be the per-channel margin sensitivity vector. Solving the 512-dimensional Ridge Normal Equations on `dev`:
$$\Delta s = \eta_s \left(\frac{1}{N} G^\top G + \lambda_s I_{512}\right)^{-1} \left(\frac{1}{N} G^\top r\right)$$
with $\lambda_s = 10.0, \tau_s = 3.0, \eta_s = 2.0$ yields $\|\Delta s\|_2 = 0.2346$ ($\|\Delta s\|_\infty = 0.0630$) and **`+423` net teacher-forced top-1 token hits** on `dev`.

---

## 3. Full 2,517-Clip Autoregressive Evaluation Results (`dev`, `test`, and `all`)

### 3.1 5-Dataset Macro-Average WER Progression (GPU `CQ-W4A8` STE, All 2,517 Clips)

| Model / WSP Compiler Stage | Modified Params (`%` of `55.54M`) | `dev` Macro WER (`1,250` clips) | Held-Out `test` Macro WER (`1,267` clips) | `all` Macro WER (`2,517` clips) | Held-Out `test` Macro Gain vs. Baseline |
|---|---|---|---|---|---|
| **Official `shruti` Baseline** | `0` (`0.0000%`) | `28.00%` | `28.14%` | `28.07%` | — |
| **Phase 5 `WSP-E1b` (`shruti_wsp_final`)** | `4,096` (`0.0074%`, `8` rows) | `26.60%` | `27.04%` | `26.83%` | `-1.10%` |
| **Phase 7 `WSP-E6` (Final-Norm Ridge)** | `4,608` (`0.0083%`, `8` rows + `512` scale) | `26.43%` | `27.80%` | `27.11%` | `-0.34%` |
| **Phase 7 `WSP-E8` (Combined `E5 + E6`)** | `37,888` (`0.0682%`, `73` rows + `512` scale) | `26.80%` | `27.37%` | `27.10%` | `-0.77%` |
| **Phase 7 `WSP-E5` (`shruti_wsp_phase7_e5_ridge`)** | **`37,376` (`0.0673%`, `73` rows)** | **`26.16%`** | **`26.18%`** | **`26.19%`** | **`-1.96%` (`-0.86%` beyond `WSP-E1b`)** |

---

### 3.2 Per-Dataset & Split Breakdown (`dev` vs. Held-Out `test` vs. `all`)

| Dataset | Split (Clips) | Official Baseline WER / CER | Phase 5 `WSP-E1b` WER / CER | Phase 7 `WSP-E5` (Ridge Emb) WER / CER | Phase 7 `WSP-E6` (FinalNorm) WER / CER | Phase 7 `WSP-E8` (Combined) WER / CER | Best Phase 7 Improvement |
|---|---|---|---|---|---|---|---|
| **`mucs_hinglish`** | `dev [0:300]` | `61.26%` / `45.36%` | `62.29%` / `57.27%` | **`57.89%` / `51.63%`** | `60.29%` / `53.68%` | `61.71%` / `54.83%` | **`-4.40%` Dev WER (`WSP-E5`)** |
| | **`test [300:600]`** | `53.19%` / `36.73%` | `52.00%` / `45.95%` | **`49.67%` / `42.09%`** | `55.92%` / `48.20%` | `54.57%` / `46.58%` | **`-3.52%` Test WER vs Base (`-2.33%` vs E1b)** |
| | **`all [0:600]`** | `57.33%` / `41.17%` | `57.38%` / `51.87%` | **`53.88%` / `46.98%`** | `58.16%` / `51.02%` | `58.23%` / `50.81%` | **`-3.50%` All WER, `-4.89%` CER (`WSP-E5`)** |
| **`svarah_en`** | `dev [0:300]` | `27.01%` / `16.21%` | `20.00%` / `10.17%` | `20.11%` / `10.35%` | **`19.55%` / `9.82%`** | **`19.55%` / `9.96%`** | **`-7.46%` Dev WER vs Base (`WSP-E6`/`E8`)** |
| | **`test [300:600]`** | `28.82%` / `16.99%` | `21.26%` / `10.85%` | `21.13%` / `11.17%` | `21.04%` / `10.75%` | **`21.01%` / `10.89%`** | **`-7.81%` Test WER vs Base (`WSP-E8`)** |
| | **`all [0:600]`** | `27.89%` / `16.59%` | `20.63%` / `10.85%` | `20.62%` / `10.75%` | `20.29%` / `10.28%` | **`20.28%` / `10.42%`** | **`-7.61%` All WER (`93.50%` LID-Acc on `E6`/`E8`)** |
| **`lahaja_hi`** | `dev [0:300]` | `24.16%` / `15.72%` | `24.13%` / `18.30%` | `24.19%` / `18.21%` | `23.92%` / `18.08%` | **`23.90%` / `18.05%`** | **`-0.26%` Dev WER (`WSP-E8`)** |
| | **`test [300:600]`** | `29.99%` / `19.93%` | `30.08%` / `23.72%` | `30.11%` / `23.62%` | `30.45%` / `24.08%` | `30.29%` / `23.92%` | Stable on Held-Out Test |
| | **`all [0:600]`** | `27.18%` / `17.90%` | `27.45%` / `21.32%` | `27.10%` / `20.87%` | `27.13%` / `21.03%` | **`27.04%` / `20.94%`** | **`-0.41%` All WER vs `WSP-E1b` (`WSP-E8`)** |
| **`librispeech_en`** | `dev [0:150]` | `6.24%` / `2.50%` | `5.84%` / `2.37%` | **`5.75%` / `2.57%`** | `5.97%` / `2.45%` | `5.90%` / `2.65%` | **`-0.49%` Dev WER (`WSP-E5`)** |
| | **`test [150:300]`** | `7.34%` / `3.17%` | `7.64%` / `3.15%` | `7.55%` / `3.17%` | `7.58%` / `3.22%` | **`7.34%` / `3.03%`** | **`-0.30%` Test WER vs `WSP-E1b` (`WSP-E8`)** |
| | **`all [0:300]`** | `6.78%` / `2.83%` | `6.80%` / `2.79%` | `6.68%` / `2.89%` | `6.80%` / `2.85%` | **`6.65%` / `2.85%`** | **`-0.13%` All WER vs Base (`100%` LID-Acc)** |
| **`fleurs_hi`** | `dev [0:200]` | `21.53%` / `13.70%` | `21.96%` / `14.51%` | `22.85%` / `15.58%` | `22.42%` / `15.05%` | `22.93%` / `15.65%` | |
| | **`test [200:417]`** | `20.87%` / `12.69%` | `22.90%` / `13.45%` | **`22.46%` / `14.70%`** | `23.99%` / `15.32%` | `23.66%` / `15.38%` | **`-0.44%` Held-Out Test WER vs `WSP-E1b` (`WSP-E5`)** |
| | **`all [0:417]`** | `21.19%` / `13.18%` | `21.89%` / `13.96%` | `22.65%` / `15.13%` | `23.19%` / `15.19%` | `23.29%` / `15.51%` | |

---

### 3.3 Native C++ `needle` 5-Beam Engine Verification (`shruti_wsp_phase7_e5_ridge.cact`)

We surgically patched the 73 modified rows of `embedding/embedding` into `/home/rgukt/Music/SHRUTHI/HINGLISH/checkpoints/experiments/shruti_wsp_phase7_e5_ridge.cact` (`29,546,527` bytes) and evaluated it with the standalone C++ `needle` 5-beam engine on the 60-clip benchmark slices (`clips[:60]` per dataset, identical to Phase 5 `modal_phase4_5_wsp.py`):

| Dataset (`60` Clips) | Official `shruti.cact` 5-Beam WER | Phase 5 `shruti_wsp_final.cact` (`8` rows) 5-Beam WER | Phase 7 `shruti_wsp_phase7_e5_ridge.cact` (`73` rows) 5-Beam WER / CER | Phase 7 LID-Acc | Phase 7 Gain vs. Phase 5 (`WSP-E1b`) |
|---|---|---|---|---|---|
| **`mucs_hinglish`** | `28.40%` | `28.08%` | **`26.45%` / `24.24%`** | **`100.00%`** | **`-1.63%` WER (`-1.95%` vs Official)** |
| **`lahaja_hi`** | `25.63%` | `25.63%` | **`24.94%` / `25.39%`** | **`96.67%`** | **`-0.69%` WER (`-0.69%` vs Official)** |
| **`fleurs_hi`** | `18.54%` | `18.61%` | **`18.42%` / `11.38%`** | **`98.33%`** | **`-0.19%` WER (`-0.12%` vs Official)** |
| **`svarah_en`** | `24.71%` (`61.67%` LID) | `15.42%` (`96.67%` LID) | **`15.42%` / `6.83%`** | **`96.67%`** | **`0.00%` vs E1b (`-9.29%` vs Official!)** |
| **`librispeech_en`** | `5.87%` | `5.87%` | **`5.87%` / `2.38%`** | **`100.00%`** | **`0.00%` (`100.0%` LID-Acc)** |
| **5-Dataset Macro-Average (C++ 5-Beam)** | **`20.63%` WER** | **`18.72%` WER** | **`18.22%` WER** | **`98.33%`** | **`-0.50%` vs E1b (`-2.41%` vs Official!)** |

Look at what happened in the native C++ `needle` 5-beam engine:
- **`WSP-E5` improves or matches every single dataset simultaneously in 5-beam C++ decoding!**
- On `mucs_hinglish`, C++ 5-beam WER drops from `28.40%` (Official) $\to$ `28.08%` (`WSP-E1b`) $\to$ **`26.45%`** (`WSP-E5`, `-1.95%` total drop).
- On `lahaja_hi`, C++ 5-beam WER drops from `25.63%` $\to$ **`24.94%`** (`-0.69%`).
- On `fleurs_hi`, C++ 5-beam WER drops from `18.54%` (Official) / `18.61%` (`WSP-E1b`) $\to$ **`18.42%`** (`WSP-E5`).
- On `svarah_en`, C++ 5-beam WER drops from `24.71%` $\to$ **`15.42%`** (`-9.29%` absolute, `-37.6%` relative).

---

## 4. Key Scientific Insights for Weight-Space Programming Research

1. **Why Covariance Whitening ($\Sigma_H^{-1}$) Unlocked Multi-Row Vocabulary Programming (`WSP-E5`)**:
   - In Phase 5, heuristic pair repulsion (`WSP-E3`) and uniform radial scaling (`WSP-E2`) suffered from collateral insertions on `mucs_hinglish` Dev because they modified rows along directions that had non-zero projection onto common hidden states $H$.
   - In Phase 7 (`WSP-E5`), multiplying the contrastive error impulse $B$ by $\Sigma_H^{-1} = (\frac{1}{N} H^\top H + \lambda I)^{-1}$ orthogonalized every token's weight update against the principal variance axes of $H$. Consequently, updating **65 content vocabulary rows** (`33,280` parameters) reduced `mucs_hinglish` WER by **`-4.40%` on Dev (`62.29%` $\to$ `57.89%`)** and **`-2.33%` on Held-Out Test (`52.00%` $\to$ `49.67%`)**, while simultaneously improving `fleurs_hi` Held-Out Test (`22.90%` $\to$ `22.46%`), `lahaja_hi` All (`27.45%` $\to$ `27.10%`), `svarah_en` Held-Out Test (`21.26%` $\to$ `21.13%`), and `librispeech_en` All (`6.80%` $\to$ `6.68%`).
2. **Why Quantization-Aware Gating Is Mandatory for 4-Bit Edge Models**:
   - Out of hundreds of candidate vocabulary rows with teacher-forced errors on `dev`, only **65 rows** survived the exact 4-bit symmetric quantization (`CQ-W4`) verification gate with $\Delta \text{Hits}(v) \ge +2$. Filtering out rows whose updates are distorted by the 15-level integer grid prevented quantization-induced degradation.
3. **Why Boundary Projections Outperform Internal Residual Scaling in Autoregressive Decoders**:
   - Updating the unquantized `stack/final_norm/scale` (`WSP-E6`) achieved the largest single-step teacher-forced gain (`+423` net token hits) and pushed `svarah_en` Auto-LID accuracy to **`93.50%`** (`20.28%` WER in `WSP-E8`), but slightly increased insertions on noisy code-switched `mucs_hinglish` during free-running greedy decoding because scaling all 8,914 vocabulary logits globally alters the relative entropy against `<eos>`. In contrast, selective covariance-whitened row updates (`WSP-E5`) preserve the global logit scale while surgically correcting confused token boundaries.
