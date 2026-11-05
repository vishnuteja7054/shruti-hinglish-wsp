# Phase 2: Baseline Evaluation & Failure Taxonomy Report — Shruti (`udgamlabs/Shruti`)

**Date:** 2026-11-03  
**Checkpoints Evaluated:**
- `checkpoints/original/shruti.safetensors` (`222,188,200` bytes, SHA-256: `5c43105db153a47d02c47131d37d066343f04f1c3ff82315ff5da1ce264fbfa7`)
- `checkpoints/original/shruti.cact` (`29,546,527` bytes, SHA-256: `b189e954aaed4a9fd41acbad3ac05e486381e96325b22a4c695b9ea5153d7065`)

**Execution Environment:** Modal GPU (`CudaDevice(id=0)`), exact Cactus Quants CQ-W4A8 Straight-Through Estimator (`cq_ste_params` + `_aq` INT8 dynamic activation quantization) and native C++ `needle` 5-beam inference engine (`bin/needle`).

---

## 1. Executive Summary of Baseline Performance

We evaluated the unmodified **Shruti** checkpoint across **2,517 speech clips** spanning five benchmarks plus **100 silence/hiss clips**, under both **Auto-LID** (model selects the step-0 language token $\arg\max_{c \in \{8906..8913\}} \text{aq}(h_0) \cdot E_c^\top$) and **Forced-LID** (step-0 language tag fixed to the ground-truth dataset language: `<|hi|>` `8913` for Hindi/Hinglish, `<|en|>` `8906` for English).

To strictly prevent test-set leakage during Phase 4 Weight-Space Programming, every dataset was partitioned into a **Calibration (`Dev`) split** and a **Held-Out (`Test`) split** prior to analysis.

### 1.1 Full-Dataset GPU CQ-W4A8 Greedy Decoding Results (`shruti.safetensors`)

| Dataset | Total Clips (Dev / Test) | Mode | All WER (%) | All CER (%) | Dev WER (%) | Held-Out Test WER (%) | Whisper-Norm WER (%) | Step-0 LID Accuracy (%) | Word Edits ($S / I / D / N_{\text{ref}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FLEURS Hindi** (`google/fleurs`, `hi_in`) | 417 (200 / 217) | Auto-LID | **21.14%** | **13.18%** | 21.53% | 20.76% | 16.00% | **100.00%** | $1792 / 253 / 123 / 10256$ |
| **FLEURS Hindi** (`google/fleurs`, `hi_in`) | 417 (200 / 217) | Forced-`hi` | **21.14%** | **13.18%** | 21.53% | 20.76% | 16.00% | 100.00% | $1792 / 253 / 123 / 10256$ |
| **Svarah Indian English** (`svarah`) | 600 (300 / 300) | Auto-LID | **27.89%** | **16.93%** | 26.97% | **28.82%** | 26.84% | **35.17%** | $1246 / 373 / 179 / 6446$ |
| **Svarah Indian English** (`svarah`) | 600 (300 / 300) | Forced-`en` | **19.64%** | **8.90%** | 19.08% | **20.20%** | 19.64% | 100.00% | $858 / 263 / 145 / 6446$ |
| **MUCS Hinglish** (`MUCS-Hinglish`) | 600 (300 / 300) | Auto-LID | **55.85%** | **49.42%** | 58.38% | 53.19% | 54.39% | **100.00%** | $2304 / 1379 / 774 / 7980$ |
| **MUCS Hinglish** (`MUCS-Hinglish`) | 600 (300 / 300) | Forced-`hi` | **55.85%** | **49.42%** | 58.38% | 53.19% | 54.39% | 100.00% | $2304 / 1379 / 774 / 7980$ |
| **Lahaja Accented Hindi** (`lahaja`) | 600 (300 / 300) | Auto-LID | **26.53%** | **26.07%** | 23.52% | 29.65% | 27.66% | **99.83%** | $2300 / 183 / 89 / 9693$ |
| **Lahaja Accented Hindi** (`lahaja`) | 600 (300 / 300) | Forced-`hi` | **26.54%** | **26.07%** | 23.54% | 29.65% | 27.65% | 100.00% | $2300 / 184 / 89 / 9693$ |
| **LibriSpeech Clean** (`librispeech_asr`) | 300 (150 / 150) | Auto-LID | **6.74%** | **2.90%** | 5.72% | 7.70% | 6.74% | **100.00%** | $360 / 53 / 23 / 6466$ |
| **LibriSpeech Clean** (`librispeech_asr`) | 300 (150 / 150) | Forced-`en` | **6.74%** | **2.90%** | 5.72% | 7.70% | 6.74% | 100.00% | $360 / 53 / 23 / 6466$ |

> **Note on Normalization:** Standard `WhisperBasicTextNormalizer` strips Unicode `Mn` (Nonspacing Mark) characters, which deletes Devanagari vowel matras (`ा, ि, ी, ु, ू, े, ै, ो, ौ, ं, ँ, ्`) and artificially lowers Hindi WER (e.g., FLEURS Hindi drops from `21.14%` true Devanagari WER to `16.00%` Whisper-normalized WER). All primary WER and CER figures in our research use `normalize_indic_hinglish`, which preserves all Devanagari combining marks (`Mn`/`Mc`) while lowercasing Latin characters and stripping punctuation.

### 1.2 Native C++ `needle` 5-Beam Engine Verification (`shruti.cact`, 60-Clip Stratified Subsets)

| Dataset (60 Clips) | Auto-LID WER (%) | Auto-LID CER (%) | Auto-LID Accuracy (%) | Forced-LID WER (%) | Forced-LID CER (%) | Auto vs. Forced WER Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FLEURS Hindi** | **18.82%** | 11.54% | 100.0% | **18.82%** | 11.54% | 0.00% |
| **Svarah Indian English** | **21.34%** | 11.83% | **45.0%** | **15.02%** | 6.11% | **+6.32% WER penalty** |
| **MUCS Hinglish** | **26.21%** | 23.61% | 100.0% | **26.21%** | 23.61% | 0.00% |
| **Lahaja Accented Hindi** | **24.61%** | 25.17% | 100.0% | **24.61%** | 25.17% | 0.00% |
| **LibriSpeech Clean** | **5.52%** | 2.38% | 100.0% | **5.52%** | 2.38% | 0.00% |

### 1.3 Silence & Hiss Hallucination Regression Test (100 Clips)

| Test Condition | Clips | Invented Words (GPU CQ-W4A8) | Invented Words (Native C++ `needle`) |
| :--- | :---: | :---: | :---: |
| **Digital Silence** (`0.0` amplitude, 4.0s) | 50 | **0 words** | **0 words** |
| **Faint Gaussian Hiss** ($\sigma = 10^{-4}$, 4.0s) | 50 | **0 words** | **0 words** |
| **Total** | **100** | **0 words (0.0% hallucination)** | **0 words (0.0% hallucination)** |

---

## 2. Empirical Failure Taxonomy & Recurring Error Patterns

### Failure Pattern 1: Catastrophic Step-0 Language ID Misrouting on Indian English (`svarah_en`)
- **Quantitative Impact:** On **Svarah Indian English** (600 clips), Auto-LID predicts `<|hi|>` (`8913`) on **389 / 600 clips (`64.83%` error rate; only `35.17%` LID accuracy)** in GPU decoding, and on **55.0%** of clips in the native C++ `needle` 5-beam engine.
- **Downstream Consequence:** When an Indian English speaker's clip is routed to `<|hi|>` at step 0, Shruti's decoder still transcribes many content words in Latin (because it was trained on Hinglish under `<|hi|>`), **but transliterates short English function words and phonetically ambiguous words into Devanagari**:
  - `"or" -> "और"` (16 times)
  - `"in" -> "इन"` (8 times)
  - `"is" -> "इस"` (6 times)
  - `"agree" -> "अगरी"` (6 times)
  - Total English $\to$ Devanagari word substitutions on Svarah Auto-LID: **400 words** (vs. **0 words** under Forced-`en`).
- **WER / CER Penalty:** Step-0 LID misrouting alone inflates Svarah WER from **`19.64%` (Forced-`en`) to `27.89%` (Auto-LID)** — an **`+8.25%` absolute WER degradation** (`+8.03%` CER degradation) on GPU, and **`+6.32%` WER degradation** (`21.34%` vs `15.02%`) on the C++ 5-beam engine.
- **Step-0 Logit Margin Distribution ($z_{\text{en}} - z_{\text{hi}}$):**
  - `fleurs_hi`: mean $-12.20 \pm 2.39$ ($p_{95} = -8.08$, $0\%$ positive)
  - `lahaja_hi`: mean $-13.59 \pm 2.96$ ($0.33\%$ positive)
  - `mucs_hinglish`: mean $-12.65 \pm 2.54$ ($0\%$ positive)
  - `svarah_en`: mean **$-1.71 \pm 5.44$** ($p_{25} = -5.67, p_{50} = -2.23, p_{75} = +2.24$) — **centered just below zero!**
  - `librispeech_en`: mean $+11.78 \pm 1.46$ ($100\%$ positive)

### Failure Pattern 2: Devanagari $\to$ Latin Script Substitution on Hinglish (`mucs_hinglish`) and Accented Hindi (`lahaja_hi`)
- **Quantitative Impact:**
  - On **MUCS Hinglish** (600 clips), reference transcripts contain `1,249` Latin words and `6,695` Devanagari words, but the baseline model predicts **`2,178` Latin words** (`1.744x` Latin over-prediction ratio) and `6,381` Devanagari words.
  - Across MUCS Hinglish, **780 reference Devanagari words are output in Latin script** (`"क्लिक" -> "click"` 46x, `"फाइल" -> "file"` 19x, `"टेक्स्ट" -> "text"` 18x, `"बॉक्स" -> "box"` 13x, `"ट्युटोरियल" -> "tutorial"` 13x, `"मैसेज" -> "message"` 13x), whereas only 167 reference Latin words are output in Devanagari.
  - On **Lahaja Accented Hindi** (600 clips), **1,214 reference Devanagari words** (primarily English loanwords/acronyms written in Devanagari in the reference, such as `"पेंटिंग" -> "painting"` 17x, `"पी" -> "p"` 16x, `"आई" -> "i"` 13x) are output in Latin script.
  - On **FLEURS Hindi** (417 clips), **243 reference Devanagari words** are output in Latin script (`13.48x` Latin word over-prediction ratio: `283` predicted vs `21` reference Latin words).

### Failure Pattern 3: High-Cosine Devanagari Function-Word, Nasalization (Anusvara/Chandrabindu), and Matra Confusions
- Across all three Hindi/Hinglish benchmarks, the single largest class of within-Devanagari substitutions consists of **single-token grammatical function words and nasalized/matra minimal pairs**:
  - **`है` (`8332`) $\leftrightarrow$ `हैं` (`8374`):**
    - FLEURS Hindi: `हैं -> है` (36x), `है -> हैं` (9x)
    - MUCS Hinglish: `है -> हैं` (52x), `हैं -> है` (5x)
    - Lahaja Hindi: `है -> हैं` (30x), `हैं -> है` (26x)
  - **`यह` (`8740`) $\leftrightarrow$ `ये` (`8572`) and `वह` $\leftrightarrow$ `वो` (`8688`):**
    - Lahaja Hindi: `वह -> वो` (87x), `यह -> ये` (60x)
    - FLEURS Hindi: `ये -> यह` (10x)
  - **`की` (`8368`) $\leftrightarrow$ `कि` (`8354`) $\leftrightarrow$ `के` (`8340`):**
    - Lahaja Hindi: `की -> कि` (25x), `की -> के` (7x), `के -> की` (3x)
    - FLEURS Hindi: `के -> की` (7x), `की -> के` (5x), `कि -> की` (3x), `की -> कि` (3x)
  - **Orthographic Nuqta / Chandrabindu / Vowel-Length Pairs:**
    - `यहां -> यहाँ` (10x), `जहां -> जहाँ` (8x), `अफ़्रीका -> अफ्रीका` (7x), `तेज़ -> तेज` (5x), `सामान -> समान` (4x).
