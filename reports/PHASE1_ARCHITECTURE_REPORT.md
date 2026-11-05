# Phase 1 Technical Report: Shruti Architecture & Weight-Space Investigation

## 1. Checkpoint Verification & Workspace Setup

All original model artifacts and the native inference runtime have been downloaded, verified against official SHA-256 checksums, and write-protected (`chmod 444`) inside `/home/rgukt/Music/SHRUTHI/HINGLISH`:

| File | Size | SHA-256 Checksum | Status |
| :--- | :---: | :--- | :--- |
| `checkpoints/original/shruti.safetensors` | 222.2 MB | `5c43105db153a47d02c47131d37d066343f04f1c3ff82315ff5da1ce264fbfa7` | Verified, Read-Only (`0444`) |
| `checkpoints/original/shruti.cact` | 29.5 MB | `b189e954aaed4a9fd41acbad3ac05e486381e96325b22a4c695b9ea5153d7065` | Verified, Read-Only (`0444`) |
| `bin/needle` (Linux x86_64 C++ engine) | 1.5 MB | `388c2b91e1f36bfa29b7a3204310166483f6df7a388110eebf9885f6fcfa89fb` | Verified, Executable |

---

## 2. Model Configuration & Parameter Inventory

From `__metadata__["config"]` in `shruti.safetensors` and the `.cact` binary header:

* **Total Tensors**: `116` (`F32` in `.safetensors`; mixed `CQ-W4`, `FP16`, `FP32`, `RAW` in `.cact`)
* **Total Parameters**: `55,517,513`
* **Core Dimensions**: `dim = 512`, `enc_layers = 8`, `dec_layers = 8`, `heads = 8`, `kv_heads = 2`, `qk_dim = 48`, `v_dim = 64`
* **Multi-Lane Hyper-Connections (`LaneStack`)**: `enc_lanes = 4`, `lanes = 4`
* **Engram Hash Memory**: `sites = (3, 7)`, `orders = (2, 3)`, `hashes = 2`, `slots = 18432`, `sub_dim = 128`, `conv_taps = 4`, `dilation = 3`
* **Audio Front-End**: `rate = 16000`, `win = 400` (25 ms), `hop = 160` (10 ms), `fft = 512`, `mels = 80`, `stem = 128`, `stem_stages = 3` ($8\times$ downsampling $\to$ 80 ms frame rate)
* **Vocabulary**: `text = 8906` (`8192` base Whistle BPE + `714` grafted Devanagari pieces), `languages = ["en", "de", "fr", "es", "it", "nl", "pl", "hi"]` (`8906..8913`), total `vocab = 8914`.

---

## 3. Empirical Weight-Space Findings (from `shruti.safetensors`)

### 3.1 Token Embedding & Output Projection (`embedding/embedding` `[8914, 512]`)

`embedding/embedding` serves both as the input token table (scaled by $\sqrt{512} \approx 22.627$) and the untied-bias output projection ($\text{logits} = \text{aq}(\text{final\_norm}(x)) E^\top$).

#### Partition L2 Norm Statistics
| Vocabulary Slice | Token IDs | Count | Mean L2 Norm | Std | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Special (`<pad>`, `</s>`, `<s>`, `<unk>`) | `0..3` | 4 | `186.71, 57.59, 72.18, 187.19` | — | `57.59` | `187.19` |
| UTF-8 Byte Fallback (`<0x00>..<0xFF>`) | `4..259` | 256 | `139.9937` | `61.3222` | — | — |
| Base Latin/European BPE | `260..8191` | 7,932 | **`70.3258`** | `16.4089` | `49.7110` | `189.1844` |
| Grafted Devanagari Pieces | `8192..8905` | 714 | **`62.3090`** | `5.0503` | `49.0002` | `72.3756` |
| Language Tags (`<|en|>..<|hi|>`) | `8906..8913` | 8 | `71.2349` | `7.5512` | `53.1192` | `79.1657` |

> **Key Weight-Space Discovery #1 (Latin vs. Devanagari Norm Imbalance)**:
> In `whistle_finetune.py` (`initial_rows`), the 714 grafted Devanagari tokens (`8192..8905`) were initialized as the average of their UTF-8 byte-fallback embeddings (`0xE0 0xA4/0xA5 ...`). Averaging multiple byte vectors compressed their initial L2 norms to **`62.31`** ($\pm 5.05$), which is **11.4% smaller** than the mean norm of the Latin BPE tokens (**`70.33`** $\pm 16.41$, reaching up to `189.18`). Because the output projection `x @ E.T` has no bias term and does not normalize rows of $E$, Latin tokens enjoy a structural inner-product advantage during code-switched Hinglish decoding—directly explaining why Shruti over-emits Latin words on MUCS Hinglish (`10,713` predicted Latin words vs. `9,122` reference Latin words).

### 3.2 Language Tag Geometry (`8906..8913`) & Step-0 LID Bias

| Lang | Token ID | L2 Norm (FP32) | L2 Norm (CQ4) | $\cos(\cdot, \text{<bos>})$ | $\cos(\cdot, \text{Latin Centroid})$ | $\cos(\cdot, \text{Devanagari Centroid})$ | $\cos(\cdot, \text{<\|en\|>})$ | $\cos(\cdot, \text{<\|hi\|>})$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `en` | `8906` | `68.0231` | `67.9466` | `0.4837` | `0.3840` | `0.3720` | `1.0000` | **`0.6912`** |
| `de` | `8907` | `73.1974` | `73.7997` | `0.4781` | `0.4658` | `0.4493` | `0.3838` | `0.6615` |
| `fr` | `8908` | `73.8788` | `72.6982` | `0.4466` | `0.4565` | `0.4228` | `0.3978` | `0.6459` |
| `es` | `8909` | `71.0910` | `71.3438` | `0.4763` | `0.4577` | `0.4362` | `0.3977` | `0.6618` |
| `it` | `8910` | `77.1173` | `76.3708` | `0.4698` | `0.4725` | `0.4628` | `0.3491` | `0.6556` |
| `nl` | `8911` | `74.2869` | `73.8912` | `0.4829` | `0.4852` | `0.4658` | `0.4047` | `0.6691` |
| `pl` | `8912` | `79.1657` | `78.3514` | `0.4852` | `0.4974` | `0.4879` | `0.3385` | `0.6612` |
| `hi` | `8913` | **`53.1192`** | **`52.9621`** | **`0.5877`** | `0.4921` | **`0.5383`** | **`0.6912`** | `1.0000` |

> **Key Weight-Space Discovery #2 (Origin of the 64.25% Svarah Indian-English LID Failure)**:
> In `whistle_finetune.py` (`tag = embedding[cfg.text:].mean(0)`), `<|hi|>` (`8913`) was initialized as the **exact arithmetic mean of the 7 European language embeddings** (`en, de, fr, es, it, nl, pl`). Consequently:
> 1. Pairwise cosines among the original 7 European tags are low (`0.3385 .. 0.5273`), whereas `<|hi|>` sits at the center of the cone with cosine **`0.6912`** to `<|en|>` and `0.6459 .. 0.6691` to `de..pl`.
> 2. `<|hi|>` has the highest cosine similarity with the step-0 input token `<bos>` (`0.5877` vs. `0.4837` for `<|en|>`), giving `<|hi|>` an intrinsic step-0 residual bias before cross-attention even differentiates the utterance.
> 3. Because $E_{\text{hi}}$ was trained on a mixture that included Indian-accented speech (and shares `0.6912` cosine alignment with $E_{\text{en}}$), Indian-English audio on Svarah projects more strongly onto $E_{\text{hi}}$ than $E_{\text{en}}$ on 257 / 400 clips.

### 3.3 High-Confusion Hindi Token Pairs & Deterministic Engram Slots

| Pair | Token IDs | L2 Norms | Cosine Similarity | L2 Distance | `engrams_0/1` Slots after `<\|hi\|>` (`t0, t1, t2, t3`) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `▁है` vs `▁हैं` | `8332` vs `8374` | `58.40` vs `59.80` | **`0.9258`** | `22.80` | `(16165, 976, 17901, 14685)` vs `(11258, 13447, 5152, 8598)` |
| `ू` vs `ु` | `8309` vs `8307` | `69.90` vs `69.26` | **`0.8907`** | `32.54` | `(12832, 3417, 17075, 4214)` vs `(3161, 17118, 9220, 14515)` |
| `▁की` vs `▁के` | `8368` vs `8340` | `60.73` vs `60.19` | **`0.8777`** | `29.91` | `(1089, 17496, 14358, 5207)` vs `(6012, 11381, 5673, 748)` |
| `ं` vs `ँ` | `8195` vs `8193` | `68.59` vs `64.86` | **`0.8759`** | `33.44` | `(13759, 646, 1785, 6392)` vs `(4088, 14347, 12362, 16705)` |
| `▁का` vs `▁को` | `8376` vs `8358` | `59.50` vs `62.02` | **`0.8644`** | `31.74` | `(14570, 5683, 10847, 1502)` vs `(7863, 7094, 15288, 7039)` |
| `▁मैं` vs `▁में` | `8624` vs `8345` | `61.82` vs `60.81` | **`0.8537`** | `33.18` | `(243, 8106, 6168, 17759)` vs `(15037, 2902, 17912, 13209)` |

### 3.4 Gates, Positional Encoding, and Multi-Lane Hyper-Connections (`LaneStack`)
* **Sinusoidal Positional Gate (`pe_gate`)**: Raw value `-6.2170`, $\sigma(\text{pe\_gate}) = 0.001991$. The encoder-to-decoder cross-attention memory `mem` relies almost entirely on the encoder's RoPE + depthwise conv temporal structure, adding only a `0.2%` amplitude sinusoidal positional anchor.
* **Residual Attention Gates (`attn_gate`, `cross_gate`)**:
  * Decoder self-attention `attn_gate` and cross-attention `cross_gate` are saturated at $\sigma(\cdot) = 1.0000$ across all 8 layers.
  * Encoder `attn_gate` is $\sigma(\cdot) = 1.0000$ on layers `0, 1, 2, 5, 6, 7`, dipping slightly at **Layer 3** ($\sigma = 0.9189$) and **Layer 4** ($\sigma = 0.9995$).
* **Multi-Lane Hyper-Connections (`mHC`)**:
  * The input-dependent scaling coefficients `a_pre`, `a_post`, `a_res` are small (`0.002 .. 0.28`), meaning the 4-lane routing is governed primarily by the static bias parameters `mhc_b_pre`, `mhc_b_post`, `mhc_b_res` (which are stored in **FP16** in `.cact`!).
  * In the Encoder, Sinkhorn diagonal dominance `mean(diag(H_res))` drops to `0.3879` at Layer 3 (strong cross-lane mixing).
  * In the Decoder, Sinkhorn diagonal dominance drops at **Layer 5** (`0.4105`) and **Layer 7** (`0.3554`), right where high-level lexical and Engram features are integrated across the 4 lanes.

### 3.5 Causal QKV Taps & Engram Dilated Taps
* In `MultiHeadAttention` (`q_taps`, `k_taps`, `v_taps`, index `0` = current step $t$, `1` = $t-1$, `2` = $t-2$):
  * **Decoder Layer 0** `v_taps` weights all three steps nearly equally (`RMS = [0.3035, 0.3510, 0.2812]`).
  * **Decoder Layer 3** (first Engram site) `v_taps` puts its **largest weight on lag $t-2$** (`RMS = [0.9850, 0.8794, 1.3148]`).
  * **Decoder Layers 1 & 7** `k_taps` weight lag $t-1$ (`1.0506` and `1.0139`) more strongly than current step $t$ (`0.8494` and `0.9076`), meaning the language token at `pos = 1` (`<|hi|>` vs. `<|en|>`) strongly modulates the keys of `pos = 2`!
* In `engrams_0` and `engrams_1` (`taps` `[4, 512]` with dilation `3`, so lags $0, 3, 6, 9$):
  * Tap `0` (current $n$-gram) dominates (`RMS = 1.0373` in `engrams_0`, `2.4307` in `engrams_1`), while dilated context taps `1..3` contribute `5%–11%` amplitude modulation.
