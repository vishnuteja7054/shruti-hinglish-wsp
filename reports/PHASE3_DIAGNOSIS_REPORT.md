# Phase 3: Weight-Space Diagnosis & Mathematical Hypotheses — Shruti (`udgamlabs/Shruti`)

**Date:** 2026-11-03  
**Diagnostic Script:** `scripts/phase3_diagnose_weight_space.py`  
**Quantitative Outputs:** `reports/phase3_diagnosis.json`, `reports/phase3_wsp_vectors.npz`

---

## 1. Overview: Linking Empirical Failures to Exact Weight-Space Geometry

In Phase 3, we correlated the Phase 1 weight-space inspection (`reports/phase1_weight_inspection.json`) with the Phase 2 empirical failure taxonomy and the extracted step-0 quantized decoder states $h_{0,q}(x) = \text{aq}(h_0(x)) \in \mathbb{R}^{512}$ across all 2,517 evaluation clips + 100 silence clips (`reports/phase2_step0_states.npz`).

We establish **three exact mathematical mechanisms** in Shruti's parameter space that explain its empirical failure modes.

---

## 2. Diagnosis D1: Mathematical Decomposition of the Step-0 Language ID Failure on Indian English

### 2.1 Why Step-0 Hidden States Are Dominated by a Deterministic BOS Prior Vector $\mu_0$

At decoder step $t = 0$, the input token is always `<s>` (`bos = 2`). Inside the 8-layer decoder stack, self-attention only attends to position $0$ (`<s>`), while cross-attention attends to the audio encoder representation $E_{\text{enc}}(x)$. Consequently, the quantized step-0 output representation $h_{0,q}(x) \in \mathbb{R}^{512}$ decomposes into an audio-independent **BOS prior vector** $\mu_0$ and an **acoustic-conditioned residual** $r(x)$:
$$h_{0,q}(x) = \mu_0 + r(x), \quad \text{where } \mu_0 = \mathbb{E}_{x \in \mathcal{D}_{\text{dev}}}\big[h_{0,q}(x)\big], \quad \|\mu_0\|_2 = 1.8759$$

Across all speech datasets, $h_{0,q}(x)$ is tightly concentrated around $\mu_0$, with mean cosine similarity $\cos(h_{0,q}(x), \mu_0) \in [0.8494, 0.9619]$ and residual norm $\|r(x)\|_2 \in [0.5817, 1.0435]$:

| Dataset | Clips | $\|h_{0,q}(x)\|_2$ (Mean) | $\|r(x)\|_2$ (Mean) | $\cos(h_{0,q}(x), \mu_0)$ | Prior Bias $b_0 = \langle \mu_0, e_{\text{en}} - e_{\text{hi}} \rangle$ | Acoustic Signal $\langle r(x), e_{\text{en}} - e_{\text{hi}} \rangle$ | Total Margin $z_{\text{en}} - z_{\text{hi}}$ | Baseline Auto-LID Pred (`en` / `hi`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FLEURS Hindi** | 417 | $2.0643$ | $0.5817$ | $0.9619$ | $-7.4732$ | $-4.5194 \pm 2.3710$ | $-11.9926 \pm 2.3710$ | `0 / 417` (`100.0%` `hi`) |
| **Lahaja Hindi** | 600 | $1.9745$ | $0.6500$ | $0.9447$ | $-7.4732$ | $-6.1130 \pm 2.9597$ | $-13.5862 \pm 2.9597$ | `2 / 598` (`99.67%` `hi`) |
| **MUCS Hinglish** | 600 | $2.0841$ | $0.7860$ | $0.9291$ | $-7.4732$ | $-5.1805 \pm 2.5357$ | $-12.6537 \pm 2.5357$ | `0 / 600` (`100.0%` `hi`) |
| **Svarah Indian English** | 600 | $1.9874$ | $0.7351$ | $0.9297$ | **$-7.4732$** | **$+5.7675 \pm 5.4364$** | **$-1.7057 \pm 5.4364$** | **`229 / 371` (`38.17%` `en`)** |
| **LibriSpeech Clean** | 300 | $1.9214$ | $1.0435$ | $0.8494$ | $-7.4732$ | $+19.2523 \pm 1.4600$ | $+11.7791 \pm 1.4600$ | `300 / 0` (`100.0%` `en`) |
| **Silence / Hiss** | 100 | $0.7164$ | $1.4657$ | $0.7009$ | $-7.4732$ | $+6.2552 \pm 0.0788$ | $-1.2181 \pm 0.0788$ | `0 / 100` (`100.0%` `hi`) |

### 2.2 The Smoking Gun in Equation Form

Look at the decomposition of the step-0 logit margin $z_{\text{en}}(x) - z_{\text{hi}}(x)$:
$$z_{\text{en}}(x) - z_{\text{hi}}(x) = \underbrace{\langle \mu_0, e_{\text{en}} - e_{\text{hi}} \rangle}_{b_0 = -7.4732 \text{ (Static Prior Bias)}} + \underbrace{\langle r(x), e_{\text{en}} - e_{\text{hi}} \rangle}_{\text{Acoustic Discriminant Signal}}$$

1. **The Acoustic Signal Actually Works on Svarah Indian English!**  
   For Indian English (`svarah_en`), the acoustic residual inner product $\langle r(x), e_{\text{en}} - e_{\text{hi}} \rangle$ is **strongly positive** ($+5.7675$ on average), whereas for Hindi/Hinglish it is **strongly negative** ($-4.5194$ on FLEURS, $-6.1130$ on Lahaja, $-5.1805$ on MUCS).
2. **Why Does Svarah Fail? Because of the Static Prior Bias $b_0 = -7.4732$ and Non-Optimal Direction $e_{\text{en}} - e_{\text{hi}}$!**  
   Because `whistle_finetune.py` initialized `<|hi|>` (`8913`) as the mean of the 7 European language embeddings (`||e_hi||_2 = 53.12` vs `||e_en||_2 = 68.02`, $\cos(e_{\text{en}}, e_{\text{hi}}) = 0.6912$) and fine-tuned predominantly on `<|hi|>`-tagged data, the static BOS prior bias $b_0 = \langle \mu_0, e_{\text{en}} - e_{\text{hi}} \rangle = -7.4732$ overwhelms the $+5.7675$ acoustic signal of Indian English speakers (whose Indian phonetic formants are closer to Indian acoustics than American LibriSpeech's $+19.2523$).
3. **Closed-Form Weight-Space Solution on the Calibration (`Dev`) Split:**  
   By steering $e_{\text{en}}$ (`8906`) and $e_{\text{hi}}$ (`8913`) along the **Dev-calibrated Regularized Fisher Discriminant direction** $\hat{u}_{\text{Fisher}}$ plus the **BOS prior direction** $\hat{\mu}_0 = \mu_0 / \|\mu_0\|_2$, and suppressing the 6 unused European tags (`8907..8912`):
   $$\Delta w = \gamma \hat{u}_{\text{Fisher}} + \beta \hat{\mu}_0, \quad e_{\text{en}}' = e_{\text{en}} + \frac{1}{2}\Delta w, \quad e_{\text{hi}}' = e_{\text{hi}} - \frac{1}{2}\Delta w, \quad e_{c}' = e_{c} - 2.0 \hat{\mu}_0 \;\; (c \in \{8907..8912\})$$
   - On the **strictly held-out Test split** of Svarah Indian English, step-0 LID accuracy jumps from **`38.00%` $\to$ `88.00%` (`+50.00%` absolute gain)** under Fisher Subspace Steering (`WSP-E1b`), while maintaining **`100.0%`** on LibriSpeech Test and **`97.70%`** on FLEURS Hindi Test!

---

## 3. Diagnosis D2: Devanagari vs. Latin Norm Asymmetry in Tied Input/Output Embeddings

### 3.1 Scale-Invariant Input vs. Scale-Linear Output Projection

In Shruti (`Whistle`), the embedding matrix $E \in \mathbb{R}^{8914 \times 512}$ serves two roles:
1. **Decoder Token Input (`Whistle.decode`):**  
   $$\text{tok} = \text{RMSUnit}(E[\text{tokens}]) = \frac{E[\text{tokens}]}{\sqrt{\frac{1}{512}\|E[\text{tokens}]\|_2^2 + 10^{-6}}}$$
   Because `_rms_unit` normalizes every row to RMS $= 1$ ($\| \text{tok} \|_2 = \sqrt{512}$), **radial scaling $E[t] \leftarrow \alpha E[t]$ ($\alpha > 0$) has ZERO effect on the decoder's internal hidden states $h_t$!**
2. **Output Vocabulary Projection (`Whistle.__call__`):**  
   $$\text{logits}_t = \text{aq}(h_t) E^\top \implies \text{logit}_t(v) = \|E[v]\|_2 \cdot \langle \text{aq}(h_t), \hat{E}[v] \rangle$$
   Here, **there is no RMS normalization on $E$** — each token's output logit scales linearly with its L2 norm $\|E[v]\|_2$!

### 3.2 The 11.4% Grafted Devanagari Norm Deficit
- **Latin BPE tokens (`260..8191`, 7,932 tokens):** Mean L2 norm = **`70.3258` $\pm$ `16.4089`**
- **Grafted Devanagari tokens (`8192..8905`, 714 tokens):** Mean L2 norm = **`62.3090` $\pm$ `5.0503`**
- **Ratio:** Latin tokens have a **`1.1287x` (`+12.87%`) higher L2 norm** than Devanagari tokens because `whistle_finetune.py` initialized the 714 Devanagari tokens by averaging UTF-8 byte embeddings (which shrunk their norms by $\sim 1/\sqrt{k}$).
- **Empirical Correlation on MUCS Hinglish:** Even when `<|hi|>` is selected 100% of the time, the model predicts **`2,178` Latin words vs. `1,249` reference Latin words (`1.744x` over-prediction)** and substitutes **780 reference Devanagari words with Latin words** (vs. only 167 Latin $\to$ Devanagari substitutions).

---

## 4. Diagnosis D3: High-Cosine Devanagari Look-Alike & Matra Subspace

In `reports/phase3_diagnosis.json`, we measured the exact embedding geometry of the top recurring Hindi/Hinglish word substitutions on the Dev splits:

| Reference Word (Token ID) | Hypothesis Word (Token ID) | Dev Split Error Count | Embedding Cosine $\cos(e_{\text{ref}}, e_{\text{hyp}})$ | $\|e_{\text{ref}}\|_2$ | $\|e_{\text{hyp}}\|_2$ | Norm Difference $\|e_{\text{hyp}}\|_2 - \|e_{\text{ref}}\|_2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`है` (`8332`)** | **`हैं` (`8374`)** | 46 | **`0.9258`** | $58.3990$ | $59.7978$ | **$+1.3987$** |
| **`हैं` (`8374`)** | **`है` (`8332`)** | 30 | **`0.9258`** | $59.7978$ | $58.3990$ | $-1.3987$ |
| **`यह` (`8740`)** | **`ये` (`8572`)** | 26 | **`0.9204`** | $59.7628$ | $59.8120$ | $+0.0491$ |
| **`की` (`8368`)** | **`कि` (`8354`)** | 16 | **`0.8668`** | $60.7321$ | $59.1990$ | $-1.5331$ |
| **`की` (`8368`)** | **`के` (`8340`)** | 7 | **`0.8777`** | $60.7321$ | $60.1896$ | $-0.5425$ |
| **`यह` (`8740`)** | **`या` (`8342`)** | 4 | **`0.8220`** | $59.7628$ | $61.2269$ | $+1.4640$ |

Notice that `है` (`8332`) and `हैं` (`8374`) have a cosine similarity of **`0.9258`**, and `हैं` has a **`+1.3987` larger L2 norm** ($59.80$ vs. $58.40$), explaining why `है -> हैं` occurs 52 times on MUCS Hinglish and 30 times on Lahaja!

---

## 5. Formulated Testable Weight-Space Hypotheses for Phase 4

1. **Hypothesis H1 (`WSP-E1a` & `WSP-E1b` — Step-0 Language Tag Subspace Steering):**  
   - **Target:** `embedding/embedding[8906:8914, :]` (8 rows = `4,096` parameters, `0.00738%` of model).  
   - **Formulation:** Apply symmetric discriminant steering $\pm \frac{1}{2}(\gamma \hat{u}_{\text{Fisher}} + \beta \hat{\mu}_0)$ to `en` (`8906`) and `hi` (`8913`), and attenuate unused European tags (`8907..8912`) by $-2.0\hat{\mu}_0$.  
   - **Prediction:** Dramatically reduces Auto-LID WER and CER on **Svarah Indian English** (closing the $+8.25\%$ Auto-vs-Forced WER gap) while preserving $\ge 95\%$ Hindi LID accuracy and $0\%$ regression on **LibriSpeech Clean** and **Silence/Hiss**.

2. **Hypothesis H2 (`WSP-E2` — Devanagari Output Projection Radial Norm Calibration):**  
   - **Target:** `embedding/embedding[8192:8906, :]` (714 rows = `365,568` parameters, `0.658%` of model).  
   - **Formulation:** Scale all 714 grafted Devanagari token rows by $\alpha_{\text{dev}} \in \{1.02, 1.04, 1.06, 1.08\}$ calibrated on the Dev splits of MUCS Hinglish, FLEURS Hindi, and Lahaja. Because decoder inputs apply `_rms_unit(x)`, this purely re-weights the output projection `aq(h) @ E.T` in favor of Devanagari over Latin tokens.  
   - **Prediction:** Reduces Devanagari $\to$ Latin script substitutions on MUCS Hinglish, FLEURS Hindi, and Lahaja, and lowers overall WER/CER if Latin over-prediction is a net error source.

3. **Hypothesis H3 (`WSP-E3` — Contrastive Angular Repulsion & Norm Balancing of Confused Devanagari Pairs):**  
   - **Target:** Confused Devanagari token pairs (`8332/8374`, `8740/8572`, `8368/8354`, `8368/8340`, `8740/8342` — 7 rows = `3,584` parameters, `0.00645%` of model).  
   - **Formulation:** Apply orthogonal repulsion in $\text{span}(e_i, e_j)$ and Dev-calibrated norm balancing between systematically confused pairs.  
   - **Prediction:** Tests whether local linear output-space decoupling can resolve contextual Hindi function-word confusions (`है/हैं`, `यह/ये`, `की/कि/के`) without altering encoder/decoder attention weights.

4. **Hypothesis H4 (`WSP-E4` — Internal Decoder Cross-Attention Gate Amplification):**  
   - **Target:** `stack/layers/block/attn_gate[6:8]` (2 parameters).  
   - **Formulation:** Scale top-2 decoder cross-attention gates by $+12\%$ (`1.12x`) to test whether stronger acoustic cross-attention in the final decoder layers improves accented speech transcription or perturbs the calibrated mHC highway residual balance.
