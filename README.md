# MiniTrans

Transformer-based Vietnamese-to-English machine translation, trained from scratch on the PhoMT dataset.

## Model

The model follows the standard Transformer-Base architecture (Vaswani et al., 2017) with Pre-Layer Normalization, trained jointly on Vietnamese and English using a shared BPE tokenizer.

| Parameter | Value |
|---|---|
| Architecture | Transformer-Base, Pre-LN |
| Encoder / Decoder layers | 6 + 6 |
| d_model | 512 |
| Attention heads | 8 |
| d_ff | 2048 |
| Parameters | 56.5M |
| BPE vocabulary | 24,000 (shared vi + en) |
| Training pairs | 600,000 (PhoMT) |
| Validation pairs | 5,000 |
| Epochs | 30 |
| Best valid NLL | 2.0882 (epoch 26) |
| Best valid PPL | ~8.07 |

## Training Techniques

### Architecture

**Pre-LN (norm_first=True).**
LayerNorm is placed before each sub-layer rather than after. This stabilizes gradients during early training and removes the risk of loss explosion during warmup, at the cost of losing PyTorch's nested-tensor fast path which only supports Post-LN.

**Weight tying.**
The source embedding, target embedding, and output projection matrix share the same weights. This is possible because both languages use the same tokenizer, and reduces the parameter count by roughly 24.6M (~25%), while also enforcing a consistent representation between input and output vocabulary.

**Sinusoidal positional encoding.**
Fixed (non-learned) position vectors added to embeddings. Dropout p=0.1 is applied after the sum of embedding and positional encoding, following the original paper.

**Embedding scaling.**
Embeddings are multiplied by sqrt(d_model) before adding positional encoding, balancing their relative magnitudes.

**Initialization.**
Transformer weight matrices use Xavier uniform initialization. Embeddings use normal initialization scaled by d_model^-0.5, with the padding token vector zeroed out explicitly.

---

### Tokenizer

**BPE with Metaspace pre-tokenizer.**
The tokenizer is trained jointly on the Vietnamese and English training corpus, resulting in a shared vocabulary of 24,000 subword units. The Metaspace pre-tokenizer marks word boundaries with the `▁` character before BPE splitting, so the decoder can reconstruct whitespace exactly (verified by round-trip tests on both languages).

**NFKC normalization.**
Applied before tokenization to canonicalize Unicode and collapse duplicate whitespace, preventing token mismatches due to different character representations.

**BPE dropout (p=0.1).**
A separate tokenizer instance with dropout=0.1 is used during training. On each forward pass, a random subset of merge rules is skipped, producing different segmentations for the same sentence. This acts as a form of subword regularization (Provilkov et al., 2020) and improves generalization. The inference tokenizer is kept deterministic.

---

### Optimizer and Training Dynamics

**AdamW with decoupled weight decay.**
betas=(0.9, 0.98), eps=1e-8. Weight decay of 0.01 is applied only to parameter matrices with 2+ dimensions; biases, LayerNorm parameters, and embeddings use weight decay=0.0 to avoid shrinking values that should not be penalized.

**Noam learning rate schedule.**
`lr = peak_lr * min(step / warmup, sqrt(warmup / step))`, with warmup=4000 steps and peak_lr=5e-4. This linearly ramps up the learning rate during warmup to avoid large updates when the model is unstable, then decays as 1/sqrt(step).

**Label smoothing (0.1).**
Softens the one-hot target distribution to discourage overconfident predictions and improve generalization.

**Gradient clipping (max norm = 1.0).**
Clips the global gradient norm before each optimizer step to prevent gradient explosions.

**Gradient accumulation (ACCUM_STEPS=2).**
Gradients are accumulated over 2 micro-batches before each optimizer update, effectively doubling the batch size without additional VRAM.

**Mixed precision (AMP).**
`torch.amp.autocast` combined with `GradScaler` computes the forward and backward passes in FP16, reducing memory usage and improving throughput. The scaler handles loss scaling to prevent FP16 underflow.

**NaN/Inf guard.**
The loss is checked with `torch.isfinite()` before each backward pass; batches with non-finite loss are skipped. The gradient norm is also checked before the optimizer step, and non-finite gradients are skipped (in addition to GradScaler's own overflow handling).

---

### Data Pipeline

**Quality filtering.**
Sentence pairs are removed if either side is empty, longer than 400 characters, or has a Vietnamese/English length ratio outside [0.5, 3.0].

**Pre-encoding.**
All training sentences are tokenized and stored as integer ID lists before training begins. This avoids repeated tokenization across 30 epochs.

**Token-based batching (MAX_TOKENS=4096).**
Batches are formed by grouping sentences until their total token count reaches the limit, rather than using a fixed sentence count. This keeps GPU utilization stable across variable-length inputs.

**Bucket sorting.**
Within each shuffled chunk, sentences are sorted by length before batching, reducing padding tokens per batch.

---

### Checkpointing

- `best.pt` is saved whenever validation NLL improves.
- `last.pt` is saved every 5 epochs as a recovery checkpoint.
- Both are automatically uploaded to a Kaggle Dataset after saving, ensuring checkpoints persist beyond the 12-hour Kaggle session limit.

---

### Inference

**Batched beam search.**
Beam search is implemented as a fully vectorized operation over the entire `batch x beam` tensor, avoiding Python-level loops over sentences or beams.

**GNMT length penalty.**
`score / ((5 + length + 1) / 6)^alpha`, with alpha=0.6. This corrects beam search's natural bias toward shorter sequences, since log-probabilities accumulate negatively with length.

---

## Repository Structure

```
MiniTrans/
├── app.py                                          # Flask server and inference code
├── index.html                                      # Web UI
├── phomt_bpe.json                                  # Tokenizer (HuggingFace tokenizers format)
├── merges.txt                                      # BPE merge rules
├── vocab.json                                      # BPE vocabulary
├── transformer-phomt-600k.ipynb                    # Training notebook (Kaggle)
├── Tong hop ky thuat - Transformer PhoMT vi-en.html  # Vietnamese technical summary
└── README.md
```

The model weights (`best.pt`, ~216 MB) are not stored in this repository. Download them from Kaggle and place the file in the project root before running.

**Download:** https://www.kaggle.com/datasets/maihongsn/transformer-phomt-600k-ckpt

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/sonmhust/MiniTrans.git
cd MiniTrans
```

**2. Install dependencies**

```bash
# With CUDA (recommended, requires CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install tokenizers flask

# CPU only
pip install torch tokenizers flask
```

**3. Download model weights**

Download `best.pt` from https://www.kaggle.com/datasets/maihongsn/transformer-phomt-600k-ckpt and place it in the project root.

```
MiniTrans/
├── best.pt   <-- place here
├── app.py
└── ...
```

**4. Run the server**

```bash
python app.py
```

Then open `http://localhost:5000` in a browser.

---

## API

**POST /translate**

```json
// request
{ "text": "Hom nay troi dep.", "beam": 4 }

// response
{ "translation": "Today the weather is nice.", "sentences": 1, "device": "cuda", "beam": 4 }
```

**GET /status**

```json
{ "model_loaded": true, "device": "cuda", "cuda": true, "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU" }
```

---

## Dataset

[PhoMT](https://github.com/ura-hcmut/PhoMT) — a Vietnamese-English parallel corpus of approximately 3 million sentence pairs, released by URA-HCMUT.