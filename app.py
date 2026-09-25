"""
Flask backend for PhoMT Vietnamese-to-English Transformer translation.
"""
import os, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR   = Path(__file__).parent
CKPT_PATH  = BASE_DIR / "best.pt"
TOK_PATH   = BASE_DIR / "phomt_bpe.json"

PAD, UNK, BOS, EOS = 0, 1, 2, 3
VOCAB_SIZE = 24000
MAX_LEN    = 128


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, n_heads=8, n_layers=6,
                 d_ff=2048, dropout=0.1, max_len=256, pad_id=0, tie_weights=True):
        super().__init__()
        self.pad_id  = pad_id
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads,
            num_encoder_layers=n_layers, num_decoder_layers=n_layers,
            dim_feedforward=d_ff, dropout=dropout,
            activation="relu", batch_first=True, norm_first=True,
        )
        self.out_proj = nn.Linear(d_model, tgt_vocab)
        self.scale    = math.sqrt(d_model)
        if tie_weights and src_vocab == tgt_vocab:
            self.src_emb.weight  = self.tgt_emb.weight
            self.out_proj.weight = self.tgt_emb.weight

    def make_src_key_padding_mask(self, src):
        return src == self.pad_id

    def make_tgt_mask(self, tgt):
        return nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)

    def encode(self, src, src_key_padding_mask):
        x = self.pos_enc(self.src_emb(src) * self.scale)
        return self.transformer.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, enc_out, tgt_mask, src_key_padding_mask, tgt_key_padding_mask=None):
        x = self.pos_enc(self.tgt_emb(tgt) * self.scale)
        return self.transformer.decoder(
            x, enc_out, tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

    def forward(self, src, tgt):
        src_pm = self.make_src_key_padding_mask(src)
        tgt_pm = self.make_src_key_padding_mask(tgt)
        tgt_m  = self.make_tgt_mask(tgt)
        enc = self.encode(src, src_pm)
        dec = self.decode(tgt, enc, tgt_m, src_pm, tgt_pm)
        return self.out_proj(dec)


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    tokenizer = Tokenizer.from_file(str(TOK_PATH))
    print(f"[INFO] Tokenizer loaded. Vocab: {tokenizer.get_vocab_size()}")

    model = Transformer(VOCAB_SIZE, VOCAB_SIZE, pad_id=PAD)
    ck    = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in ck["model"].items()}
    model.load_state_dict(state)
    model.to(device).eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    epoch  = ck.get("epoch", "?")
    vnll   = ck.get("valid_nll", float("nan"))
    print(f"[INFO] Loaded epoch={epoch}, valid_nll={vnll:.4f}, params={params:.1f}M")
    del ck, state
    return model, tokenizer, device


@torch.no_grad()
def beam_search(model, src_tensor, device, beam=4, max_len=MAX_LEN, alpha=0.6):
    src = src_tensor.to(device)
    B   = src.size(0)

    mask = model.make_src_key_padding_mask(src)
    enc  = model.encode(src, mask)
    enc  = enc.repeat_interleave(beam, 0)
    mask = mask.repeat_interleave(beam, 0)

    seqs     = torch.full((B * beam, 1), BOS, dtype=torch.long, device=device)
    scores   = torch.zeros(B, beam, device=device)
    scores[:, 1:] = -1e9
    scores   = scores.view(-1)
    finished = torch.zeros(B * beam, dtype=torch.bool, device=device)
    base     = (torch.arange(B, device=device) * beam).unsqueeze(1)

    for _ in range(max_len):
        tgt_m  = model.make_tgt_mask(seqs)
        dec    = model.decode(seqs, enc, tgt_m, mask)[:, -1]
        logp   = F.log_softmax(model.out_proj(dec).float(), dim=-1)
        V      = logp.size(-1)

        logp[finished] = -1e9
        logp[finished, PAD] = 0.0

        cand       = (scores.unsqueeze(1) + logp).view(B, beam * V)
        top_s, top_i = cand.topk(beam, dim=1)
        beam_i     = top_i // V
        tok        = (top_i % V).view(-1)
        sel        = (base + beam_i).view(-1)

        seqs     = torch.cat([seqs[sel], tok.unsqueeze(1)], dim=1)
        scores   = top_s.view(-1)
        finished = finished[sel] | (tok == EOS)
        if finished.all():
            break

    seqs   = seqs.view(B, beam, -1)[:, :, 1:]
    scores = scores.view(B, beam)
    best   = []
    for b in range(B):
        cands = []
        for k in range(beam):
            ids = seqs[b, k].tolist()
            if EOS in ids:
                ids = ids[:ids.index(EOS)]
            ids = [i for i in ids if i != PAD]
            lp  = ((5 + len(ids) + 1) / 6) ** alpha
            cands.append((scores[b, k].item() / lp, ids))
        best.append(max(cands, key=lambda x: x[0])[1])
    return best


def translate_batch(texts, model, tokenizer, device, beam=4, batch_size=8):
    order   = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    results = [None] * len(texts)
    for s in range(0, len(order), batch_size):
        idxs  = order[s : s + batch_size]
        batch = [[BOS] + tokenizer.encode(texts[i]).ids[: MAX_LEN - 2] + [EOS] for i in idxs]
        L     = max(len(b) for b in batch)
        src   = torch.full((len(batch), L), PAD, dtype=torch.long)
        for j, b in enumerate(batch):
            src[j, : len(b)] = torch.tensor(b)
        out_ids = beam_search(model, src, device, beam=beam)
        for i, ids in zip(idxs, out_ids):
            results[i] = tokenizer.decode(ids)
    return results


# ── Flask ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
_model = _tokenizer = _device = None


def _ensure_loaded():
    global _model, _tokenizer, _device
    if _model is None:
        _model, _tokenizer, _device = load_model()


@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/translate", methods=["POST"])
def translate_api():
    _ensure_loaded()
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    beam = max(1, min(int(data.get("beam", 4)), 8))

    if not text:
        return jsonify({"error": "Van ban dau vao khong duoc de trong."}), 400

    sentences = [s.strip() for s in text.split("\n") if s.strip()]
    if not sentences:
        return jsonify({"error": "Khong co cau hop le."}), 400

    try:
        results = translate_batch(sentences, _model, _tokenizer, _device, beam=beam)
        return jsonify({
            "translation": "\n".join(results),
            "sentences"  : len(sentences),
            "device"     : str(_device),
            "beam"       : beam,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def status_api():
    cuda = torch.cuda.is_available()
    return jsonify({
        "model_loaded": _model is not None,
        "device"      : str(_device) if _device else None,
        "cuda"        : cuda,
        "gpu"         : torch.cuda.get_device_name(0) if cuda else None,
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  PhoMT Vi->En Translator")
    print("  Open browser: http://localhost:5000")
    print("=" * 60)
    _ensure_loaded()
    app.run(host="0.0.0.0", port=5000, debug=False)