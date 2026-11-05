---
language:
  - hi
  - en
license: apache-2.0
library_name: safetensors
tags:
  - automatic-speech-recognition
  - hindi
  - hinglish
  - indian-english
  - cactus-needle
  - whistle-small
  - weight-space-programming
  - edge-ai
base_model: udgamlabs/Shruti
datasets:
  - google/fleurs
  - ai4bharat/Svarah
  - ai4bharat/Lahaja
  - openslr/librispeech_asr
---

# Shruti-Hinglish-WSP: Weight-Space Programmed Checkpoints for `udgamlabs/Shruti` (`whistle-small` / `cactus-needle`)

- **GitHub Research Repository (Code, Reports & Scripts)**: **[https://github.com/vishnuteja7054/shruti-hinglish-wsp](https://github.com/vishnuteja7054/shruti-hinglish-wsp)**
- **Hugging Face Checkpoints Repository**: **[https://huggingface.co/vishnushuggingface/shruti-hinglish-wsp](https://huggingface.co/vishnushuggingface/shruti-hinglish-wsp)**

This repository contains the **Weight-Space Programmed (WSP)** `.cact` (`29.5 MB` 4-bit `CQ4` + `FP16`) and `.safetensors` (`212 MB` `FP32`) model checkpoints across Phases 6–11 of our Weight-Space Programming research on `udgamlabs/Shruti` (`Cactus-Compute/whistle-small`, `55,517,513` parameters).

## Recommended Production Checkpoints

1. **`shruti_wsp_phase11_e16_e2e_fp16.cact` (`29.5 MB`, `326` of `529` `.cact` records surgically calibrated)** — **Best Overall Production Checkpoint**
   - Calibrates all 8 Conformer Encoder layers, Audio Conv Stem, 8 Cross-Attention Decoder layers, and `188` 4-block `FP16` embedding norms.
   - **C++ `needle` 5-Beam Results (`300` Clips)**:
     - **4-Dataset Clean Macro Average**: **`15.31%` Strict WER** $\to$ **`9.64%` Semantic WER (`90.36%` Word Accuracy, `95.70%` Character Accuracy, `4.30%` CER)**
     - **5-Dataset Macro Average**: **`17.14%` Strict WER** $\to$ **`11.95%` Semantic WER (`88.05%` Word Accuracy, `93.82%` Character Accuracy)**
     - **`fleurs_hi` (Hindi)**: **`17.11%` Strict WER** (down from `74.47%` Official Baseline!) $\to$ **`10.78%` Semantic WER (`89.22%` WordAcc, `94.76%` CharAcc)**
     - **`svarah_en` (Indian English)**: **`13.83%` Strict WER** (down from `24.71%` Official Baseline!) $\to$ **`11.86%` Semantic WER (`88.14%` WordAcc, `94.94%` CharAcc)**
     - **`librispeech_en` (Clean English)**: **`5.34%` Strict WER** $\to$ **`4.28%` Semantic WER (`95.72%` WordAcc, `98.29%` CharAcc, `1.71%` CER)**
   - **Full `2,517`-Clip GPU 1-Beam Results**:
     - **5-Dataset Macro Average**: **`25.31%` Strict WER** / **`19.83%` Semantic WER** (Best 5-Dataset GPU 1-Beam Model)

2. **`shruti_wsp_phase11_e15_decoder_fp16.cact` (`29.5 MB`, `210` `.cact` records calibrated)** — **Best 4-Dataset Clean 1-Beam Greedy Checkpoint**
   - **Full `2,517`-Clip GPU 1-Beam 4-DS Clean Macro Average**: **`18.90%` Strict WER** / **`14.03%` Semantic WER** (`21.53%` Strict / `15.55%` Semantic on `fleurs_hi`; `26.27%` Strict / `15.87%` Semantic on `lahaja_hi`; `6.54%` Strict / `5.39%` Semantic on `librispeech_en`).

3. **`shruti_wsp_phase9_e11_blocknorm.cact` (`29.5 MB`, `2` `.cact` records calibrated)** — **Best Code-Switched Hinglish 5-Beam Checkpoint**
   - **`mucs_hinglish` C++ 5-Beam**: **`23.61%` Strict WER** $\to$ **`18.94%` Semantic WER (`81.06%` WordAcc, `87.65%` CharAcc)**
   - **`lahaja_hi` C++ 5-Beam**: **`24.50%` Strict WER** $\to$ **`11.53%` Semantic WER (`88.47%` WordAcc, `94.79%` CharAcc)**

---

## Complete Checkpoint Inventory & SHA-256 Checksums

| Filename | Format | Size | `.cact` Records Modified | SHA-256 Checksum |
| :--- | :---: | :---: | :---: | :--- |
| `original/shruti.cact` | `.cact` | `29.5 MB` | `0` *(Official Base)* | `b189e954aaed4a9fd41acbad3ac05e486381e96325b22a4c695b9ea5153d7065` |
| `original/shruti.safetensors` | `.safetensors` | `212 MB` | `0` *(Official Base)* | `5c43105db153a47d02c47131d37d066343f04f1c3ff82315ff5da1ce264fbfa7` |
| `shruti_wsp_final.cact` | `.cact` | `29.5 MB` | `1` (`8` rows) | `e1c9f0bdd97fc0fc865ca02400166ed933665170330f3db64f2cc41520f519d1` |
| `shruti_wsp_final.safetensors` | `.safetensors` | `212 MB` | — | `283ee9fcd0211908c4f7901cbbb4ab44924f842f7ae219deb1c5b35da9bfb317` |
| `shruti_wsp_e1c_zero_hi_reg.cact` | `.cact` | `29.5 MB` | `1` (`8` rows) | `ce15d791109ac4f6c6e46fb39aa04269222cad6a768b08eb126afa86304da3b7` |
| `shruti_wsp_e1c_zero_hi_reg.safetensors` | `.safetensors` | `212 MB` | — | `88b1c6219e2f99b984c725c75dc7fce4c9cffa8c361cabc50a7ec2d7d911e24d` |
| `shruti_wsp_phase7_e5_ridge.cact` | `.cact` | `29.5 MB` | `1` (`73` rows) | `8821cf40062ab59560858082defcf792b4935f578a44ac67acaf21967908d628` |
| `shruti_wsp_phase7_e5_ridge.safetensors` | `.safetensors` | `212 MB` | — | `f1edaba3f5473a53766c64e27dbd0f0ec3136d5b02b7093e2e628708e29fca40` |
| `shruti_wsp_phase7_e8_combined.cact` | `.cact` | `29.5 MB` | `2` (`73` rows + `final_norm`) | `21718f2b4acf9c3cb3b80564174a056efb4c92ef8cfb8384f4e0f5f248b1525d` |
| `shruti_wsp_phase7_e8_combined.safetensors` | `.safetensors` | `212 MB` | — | `955bb63d7e4e704e2cad51a65f184e9570d4da6771e4f8ac62004ed51fd03772` |
| `shruti_wsp_phase8_fp32_master.safetensors` | `.safetensors` | `212 MB` | `FP32` Master (`46` rows) | `a40ddd7d96029b9fea4568fd3f7f99486de22686c995321b6b05eaaf5babfdfb` |
| `shruti_wsp_phase8_e10a_latticesnap.cact` | `.cact` | `29.5 MB` | `1` (`39` rows) | `bb2fc1a66aca174f64a1dfab8da4a4fca6c840db00024c0603e1fcc578d149c6` |
| `shruti_wsp_phase8_e10a_latticesnap.safetensors` | `.safetensors` | `212 MB` | — | `c28391ae5c1af5a547b178266ede2fbe009f05433f0ab52d0d676fc5b947567b` |
| `shruti_wsp_phase8_e10b_quantcomp.cact` | `.cact` | `29.5 MB` | `2` (`39` rows + `final_norm`) | `dc8d49ad6e1b090ed0b90353a69f9355de0ae2890811dfdf3951941d3c244d33` |
| `shruti_wsp_phase8_e10b_quantcomp.safetensors` | `.safetensors` | `212 MB` | — | `538859653d6e9481e2cbfc6254382708c59dafe7a998bcb6fe361df5beb68237` |
| `shruti_wsp_phase9_e11_blocknorm.cact` | `.cact` | `29.5 MB` | `2` (`196` rows + `final_norm`) | `ae48e68c1796f89f64eef933140314f6f2a3eb718b638c01e0e79949797f9cc3` |
| `shruti_wsp_phase9_e11_blocknorm.safetensors` | `.safetensors` | `212 MB` | — | `cc5bdf32b571e0ff6ac47a920462e8a8dc8fa131227ff3cdd837a01fa583cc02` |
| `shruti_wsp_phase9_e12_cascade27pct.safetensors` | `.safetensors` | `212 MB` | `26.87%` Cascade (`14.9M` params) | `4237b8fed21f1fabe71eb5fe3e3020c881f698e99c97c329c37ed0850d2583fa` |
| `shruti_wsp_phase10_e13_cleanloopsafe.cact` | `.cact` | `29.5 MB` | `2` (`161` rows + `final_norm`) | `021884f52a44110fdf3a3f3f393854aeeef9032812c01bd67b310baf054e6240` |
| `shruti_wsp_phase10_e13_cleanloopsafe.safetensors` | `.safetensors` | `212 MB` | — | `d9c73c6e687f7cf17d7b8d07b29f3d3a077568024c37cecd9d27e92b3fd6b48d` |
| `shruti_wsp_phase10_e14_loopguarded.cact` | `.cact` | `29.5 MB` | `2` (`188` rows + `final_norm`) | `c094b92a153c323e8d6f550ea4bee3e83daef5904d0a96afcf8d391f7589191f` |
| `shruti_wsp_phase10_e14_loopguarded.safetensors` | `.safetensors` | `212 MB` | — | `f6c365e649abc6ffe7b0e77439ddfdc9aa8c846ad310848e548d63e7c9b74354` |
| `shruti_wsp_phase11_e15_decoder_fp16.cact` | `.cact` | `29.5 MB` | **`210` records** | `1eb59d441396869fa5d0a9256f9ce9dec40f7cf57938be311fbbe4064a8d0b70` |
| `shruti_wsp_phase11_e15_decoder_fp16.safetensors` | `.safetensors` | `212 MB` | — | `908c65a94a7345d2f42ecbbc0819d92afd911577f7965723658a5bdfac531796` |
| `shruti_wsp_phase11_e16_e2e_fp16.cact` | `.cact` | `29.5 MB` | **`326` records** | `7f07a23fb7169f75f668e6b8da830e45931f6f96e40ae7492cd83b266fe0d597` |
| `shruti_wsp_phase11_e16_e2e_fp16.safetensors` | `.safetensors` | `212 MB` | — | `9636d5f68c6d227de1911943b0ff39c1703a9caa9344f43b0f3cde896adf90c9` |

## Usage with `cactus-needle` C++ Engine

```bash
./bin/needle --model shruti_wsp_phase11_e16_e2e_fp16.cact --audio sample_16k.wav --threads 4
```
