#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Myanmar TTS NAR  —  FastSpeech2-style Non-Autoregressive Transformer
====================================================================
Filename: myanmar_tts_transformer_nar.py

Sibling of myanmar_tts_transformer.py (AR TransformerTTS + guided attention).
This file keeps the same mel frontend / vocoder / train loop scaffolding,
but replaces the AR decoder with:

  Text → CharEmbed+SinPE → TransformerEncoder(N)
              ↓
        DurationPredictor  (trained on forced durations)
              ↓
        LengthRegulator    (expand each token by d_i frames)
              ↓
        TransformerDecoder(N)   ← non-causal FFT blocks (NAR)
              ↓
        Linear → mel_before → Postnet → mel_after

Why NAR here (vs tutorial AR v5)
--------------------------------
AR + stop-token + soft cross-attention needs guided attention on small
data.  NAR instead uses EXPLICIT forced durations so length/alignment
are hard-coded by the length regulator — no PreNet, no teacher forcing,
no stop token, no guided attention.

Forced durations (no MFA required)
----------------------------------
prep computes proportional (uniform) char→mel durations so
  sum(d_i) == mel_frames
for every utterance.  That is a simple forced-alignment baseline.
You can later swap in MFA / CTC alignments by writing a `durations`
list into each sample JSON (same length as text_ids).

Training loss = masked mel L1/SC  +  dur_loss_weight * duration MSE
Inference    = predict durations → length-regulate → one-shot mel

NOTE: use a SEPARATE --output_dir from the AR tutorial (different
      checkpoints / sample schema with durations).  Re-run --stage prep.
"""
from __future__ import annotations

import argparse, json, logging, math, os, random, sys, time, unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ══════════════════════════════════════════════════════════════════════════════
# §1  Logging & dependency check
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(log_file=None, verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s  [%(levelname)-8s]  %(message)s"
    h = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        h.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt,
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=h)
    return logging.getLogger("tts")


def check_dependencies(logger):
    ok = True
    for pkg, hint in [("torch","pip install torch"),
                      ("soundfile","pip install soundfile"),
                      ("librosa","pip install librosa"),
                      ("matplotlib","pip install matplotlib"),
                      ("scipy","pip install scipy")]:
        try:    __import__(pkg); logger.info(f"  ✓ {pkg}")
        except ImportError: logger.error(f"  ✗ {pkg} → {hint}"); ok = False
    for pkg, hint in [("gi","apt install python3-gi python3-gi-cairo gir1.2-pango-1.0")]:
        try:    __import__(pkg); logger.info(f"  ✓ {pkg}")
        except ImportError: logger.info(f"  – {pkg}  (optional) → {hint}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# §2  Myanmar text
# ══════════════════════════════════════════════════════════════════════════════

_MYA_LO, _MYA_HI = 0x1000, 0x109F

def is_myanmar_char(c): return _MYA_LO <= ord(c) <= _MYA_HI

def normalize_text(text):
    text = unicodedata.normalize("NFC", text.strip())
    return "".join(c for c in text
                   if is_myanmar_char(c) or c.isascii() or c == " ")


class MyanmarCharVocab:
    PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"

    def __init__(self):
        self.char2idx = {self.PAD:0, self.UNK:1, self.BOS:2, self.EOS:3}
        self.idx2char = {v:k for k,v in self.char2idx.items()}

    def build(self, texts):
        for t in texts:
            for c in normalize_text(t):
                if c not in self.char2idx:
                    i = len(self.char2idx)
                    self.char2idx[c] = i; self.idx2char[i] = c

    def encode(self, text):
        n = normalize_text(text)
        return ([self.char2idx[self.BOS]]
                + [self.char2idx.get(c, self.char2idx[self.UNK]) for c in n]
                + [self.char2idx[self.EOS]])

    def decode(self, ids):
        sp = {self.PAD, self.UNK, self.BOS, self.EOS}
        return "".join(self.idx2char.get(i, self.UNK)
                       for i in ids if self.idx2char.get(i,self.UNK) not in sp)

    @property
    def size(self): return len(self.char2idx)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char2idx": self.char2idx}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f: d = json.load(f)
        v = cls()
        v.char2idx = {k: int(i) for k,i in d["char2idx"].items()}
        v.idx2char = {int(i): k for k,i in d["char2idx"].items()}
        return v


# ══════════════════════════════════════════════════════════════════════════════
# §3  Mel utilities  — fixed [-80,0]→[-1,+1] normalisation
# ══════════════════════════════════════════════════════════════════════════════

_MEL_MIN_DB, _MEL_MAX_DB, _MEL_RANGE = -80.0, 0.0, 80.0

def mel_normalize(mel_db):
    return (2.0*(np.clip(mel_db,_MEL_MIN_DB,_MEL_MAX_DB)-_MEL_MIN_DB)
            /_MEL_RANGE - 1.0).astype(np.float32)

def mel_denormalize(mel_norm):
    return ((np.clip(mel_norm,-1.0,1.0)+1.0)/2.0
            *_MEL_RANGE+_MEL_MIN_DB).astype(np.float32)


@dataclass
class MelConfig:
    sample_rate: int   = 22050
    n_fft:       int   = 1024
    hop_length:  int   = 256
    win_length:  int   = 1024
    n_mels:      int   = 80
    fmin:        float = 0.0
    fmax:        float = 8000.0

    def to_dict(self): return {k:v for k,v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k:v for k,v in d.items()
                      if k in cls.__dataclass_fields__})


def load_audio(path, sr):
    import soundfile as sf
    a, s = sf.read(path, dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    if s != sr:
        import librosa; a = librosa.resample(a, orig_sr=s, target_sr=sr)
    return a


def audio_to_mel(audio, cfg):
    import librosa
    m = librosa.feature.melspectrogram(
        y=audio, sr=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, win_length=cfg.win_length,
        n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax)
    return mel_normalize(librosa.power_to_db(m, ref=np.max).T)  # (T,n_mels)


def mel_to_audio_gl(mel_norm, cfg, n_iter=200):
    """Normalized mel → audio via improved Griffin-Lim (200 iter, power=1.5)."""
    import librosa
    from scipy.signal import butter, sosfiltfilt
    p = librosa.db_to_power(mel_denormalize(mel_norm).T.astype(np.float64))
    a = librosa.feature.inverse.mel_to_audio(
        p, sr=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, win_length=cfg.win_length,
        n_iter=n_iter, power=1.5, center=True)
    nyq = cfg.sample_rate/2.0
    sos = butter(4, [max(80.,cfg.fmin+20)/nyq,
                     min(7800.,cfg.fmax-200)/nyq],
                 btype="band", output="sos")
    a = sosfiltfilt(sos, a)
    pk = np.max(np.abs(a))
    if pk > 1e-8: a = a/pk*0.9
    return a.astype(np.float32)


def save_wav(audio, path, sr):
    import soundfile as sf; sf.write(path, audio.astype(np.float32), sr)


# ══════════════════════════════════════════════════════════════════════════════
# §4  Vocoder wrapper
# ══════════════════════════════════════════════════════════════════════════════

class Vocoder:
    def __init__(self, cfg, logger, use_neural=True):
        self.cfg = cfg; self.logger = logger; self._wg = None
        if use_neural:
            try:
                wg = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub",
                    "nvidia_waveglow", model_math="fp32",
                    pretrained=True, verbose=False)
                wg.eval()
                for m in wg.modules():
                    if hasattr(m,"weight_g"):
                        try: nn.utils.remove_weight_norm(m)
                        except: pass
                self._wg = wg; logger.info("WaveGlow loaded ✓")
            except Exception as e:
                logger.warning(f"WaveGlow unavailable ({type(e).__name__}), "
                               "using Griffin-Lim.")

    def synthesize(self, mel_norm):
        if self._wg is not None: return self._waveglow(mel_norm)
        return mel_to_audio_gl(mel_norm, self.cfg)

    def _waveglow(self, mel_norm):
        t = torch.from_numpy(mel_denormalize(mel_norm).T).float().unsqueeze(0)
        d = next(self._wg.parameters()).device
        with torch.no_grad(): a = self._wg.infer(t.to(d), sigma=0.9)
        a = a.squeeze().cpu().numpy()
        pk = np.max(np.abs(a)); return (a/pk*0.9).astype(np.float32) if pk>1e-8 else a


# ══════════════════════════════════════════════════════════════════════════════
# §5  Shared encoder building blocks
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class ConvFFN(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size=9, dropout=0.1):
        super().__init__()
        pad = (kernel_size-1)//2
        self.c1   = nn.Conv1d(d_model, d_ff,    kernel_size, padding=pad)
        self.c2   = nn.Conv1d(d_ff,    d_model, kernel_size, padding=pad)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        r = x; x = x.transpose(1,2)
        x = self.drop(F.relu(self.c1(x)))
        return self.norm(r + self.drop(self.c2(x).transpose(1,2)))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, ffn_kernel=9):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                            dropout=dropout, batch_first=True)
        self.ffn   = ConvFFN(d_model, d_ff, ffn_kernel, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.drop(a))
        return self.ffn(x)


# ══════════════════════════════════════════════════════════════════════════════
# §6  TransformerTTSNAR — FastSpeech2-style non-autoregressive model
# ══════════════════════════════════════════════════════════════════════════════

def proportional_durations(n_tokens: int, n_frames: int) -> List[int]:
    """
    Forced duration baseline (no MFA): distribute mel frames across tokens
    so sum(durs) == n_frames.  Each token gets at least 0 frames; if
    n_frames >= n_tokens every token gets ≥1 via the frame→token map.
    """
    if n_tokens <= 0:
        return []
    if n_frames <= 0:
        return [1] * n_tokens
    durs = [0] * n_tokens
    for i in range(n_frames):
        durs[min(n_tokens - 1, (i * n_tokens) // n_frames)] += 1
    # rare edge case: ensure non-empty prediction path
    if sum(durs) == 0:
        durs[0] = 1
    return durs


class Postnet(nn.Module):
    """5-layer Conv1d postnet (explicit layers)."""
    def __init__(self, n_mels, d_hidden=512, dropout=0.1):
        super().__init__()
        self.c0 = nn.Conv1d(n_mels,   d_hidden, 5, padding=2)
        self.b0 = nn.BatchNorm1d(d_hidden)
        self.c1 = nn.Conv1d(d_hidden, d_hidden, 5, padding=2)
        self.b1 = nn.BatchNorm1d(d_hidden)
        self.c2 = nn.Conv1d(d_hidden, d_hidden, 5, padding=2)
        self.b2 = nn.BatchNorm1d(d_hidden)
        self.c3 = nn.Conv1d(d_hidden, d_hidden, 5, padding=2)
        self.b3 = nn.BatchNorm1d(d_hidden)
        self.c4 = nn.Conv1d(d_hidden, n_mels,   5, padding=2)
        self.b4 = nn.BatchNorm1d(n_mels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):              # x: (B,T,n_mels)
        x = x.transpose(1,2)
        x = self.drop(torch.tanh(self.b0(self.c0(x))))
        x = self.drop(torch.tanh(self.b1(self.c1(x))))
        x = self.drop(torch.tanh(self.b2(self.c2(x))))
        x = self.drop(torch.tanh(self.b3(self.c3(x))))
        x = self.drop(self.b4(self.c4(x)))
        return x.transpose(1,2)        # (B,T,n_mels)


class DurationPredictor(nn.Module):
    """
    FastSpeech-style duration predictor.
    Predicts log(duration) per encoder token from encoder hidden states.
    """
    def __init__(self, d_model, n_layers=2, kernel_size=3, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size, padding=pad),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.proj  = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        # x: (B, T, d)
        h = x.transpose(1, 2)          # (B, d, T)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h)
            h = norm(h.transpose(1, 2)).transpose(1, 2)
        log_dur = self.proj(h.transpose(1, 2)).squeeze(-1)   # (B, T)
        if mask is not None:
            log_dur = log_dur.masked_fill(mask, 0.0)
        return log_dur


class LengthRegulator(nn.Module):
    """Expand encoder states by integer durations (hard alignment)."""

    def forward(self, x, durations, mel_max_len=None):
        """
        x         : (B, T_text, d)
        durations : (B, T_text) int — padded tokens should have duration 0
        returns   : (B, T_mel, d), mel_lengths (B,)
        """
        B, T, D = x.shape
        outs, lens = [], []
        for b in range(B):
            expanded = []
            for t in range(T):
                d = int(durations[b, t].item())
                if d > 0:
                    expanded.append(x[b, t].unsqueeze(0).expand(d, -1))
            if expanded:
                e = torch.cat(expanded, dim=0)
            else:
                e = x.new_zeros(1, D)
            outs.append(e)
            lens.append(e.size(0))

        max_len = mel_max_len if mel_max_len is not None else max(lens)
        y = x.new_zeros(B, max_len, D)
        for b, e in enumerate(outs):
            L = min(e.size(0), max_len)
            y[b, :L] = e[:L]
        return y, torch.tensor(lens, device=x.device, dtype=torch.long)


class TransformerTTSNAR(nn.Module):
    """
    Non-autoregressive Transformer TTS (FastSpeech2-style).

    Training:
      model(text_ids, mel_targets, mel_lengths, durations, dur_loss_weight)
      → (mel_before, mel_after, log_dur_pred, loss)

    Inference:
      mel_norm = model.infer(text_ids, speed_factor=1.0)
    """

    def __init__(self, vocab_size, n_mels,
                 d_model=256, n_heads=4,
                 n_enc_layers=4, n_dec_layers=4,
                 d_ff=1024, ffn_kernel=9, dropout=0.1):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model

        # ── Encoder ──────────────────────────────────────────────────────────
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.enc_pe  = SinusoidalPE(d_model, dropout=dropout)
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout, ffn_kernel)
            for _ in range(n_enc_layers)])

        # ── Duration + length regulator ──────────────────────────────────────
        self.duration_predictor = DurationPredictor(d_model, dropout=dropout)
        self.length_regulator   = LengthRegulator()

        # ── NAR decoder (same FFT block as encoder; no causal / cross-attn) ──
        self.dec_pe  = SinusoidalPE(d_model, dropout=dropout)
        self.decoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout, ffn_kernel)
            for _ in range(n_dec_layers)])

        # ── Output heads ─────────────────────────────────────────────────────
        self.mel_proj = nn.Linear(d_model, n_mels)
        self.postnet  = Postnet(n_mels, min(d_model * 2, 512), dropout)

    def _encode(self, text_ids):
        pad_mask = (text_ids == 0)                # (B, T_text) True = pad
        x = self.enc_pe(self.embed(text_ids))
        for blk in self.encoder:
            x = blk(x, key_padding_mask=pad_mask)
        return x, pad_mask

    def forward(self, text_ids, mel_targets, mel_lengths, durations,
                dur_loss_weight=1.0):
        """
        Teacher durations (forced) expand the encoder; duration predictor
        is trained to match log(durations).  Mel decoder is fully parallel.

        durations : (B, T_text) int64 — 0 on pad positions
        """
        B, T_mel, _ = mel_targets.shape
        enc_out, enc_pad = self._encode(text_ids)

        log_dur_pred = self.duration_predictor(enc_out, mask=enc_pad)
        # duration loss on non-pad tokens only
        dur_tgt = durations.float().clamp(min=0)
        token_mask = (~enc_pad).float()
        log_tgt = torch.log(dur_tgt + 1.0)
        dur_loss = (
            ((log_dur_pred - log_tgt) ** 2) * token_mask
        ).sum() / token_mask.sum().clamp(min=1.0)

        # expand with GT durations so mel length matches targets
        dec_in, _ = self.length_regulator(
            enc_out, durations, mel_max_len=T_mel)

        idx = torch.arange(T_mel, device=enc_out.device).unsqueeze(0)
        mel_pad = idx >= mel_lengths.to(enc_out.device).unsqueeze(1)

        dec_h = self.dec_pe(dec_in)
        for blk in self.decoder:
            dec_h = blk(dec_h, key_padding_mask=mel_pad)

        mel_before = self.mel_proj(dec_h)
        mel_after  = mel_before + self.postnet(mel_before)

        mel_loss = masked_mel_loss(mel_before, mel_after, mel_targets, mel_lengths)
        total = mel_loss + dur_loss_weight * dur_loss
        return mel_before, mel_after, log_dur_pred, total

    @torch.no_grad()
    def infer(self, text_ids, speed_factor=1.0, min_duration=1):
        """
        One-shot NAR inference: predict durations → expand → mel.
        speed_factor < 1 → slower (longer durations); > 1 → faster.
        Returns: (T, n_mels) float32 numpy, normalized mel [-1,+1]
        """
        enc_out, enc_pad = self._encode(text_ids)
        log_dur = self.duration_predictor(enc_out, mask=enc_pad)
        dur = torch.clamp(
            (torch.exp(log_dur) - 1.0) / max(speed_factor, 1e-3),
            min=0.0,
        )
        dur = torch.round(dur).long()
        dur = torch.clamp(dur, min=0)
        # force at least min_duration on real tokens
        real = ~enc_pad
        dur = torch.where(real & (dur < min_duration),
                          torch.full_like(dur, min_duration), dur)
        dur = torch.where(enc_pad, torch.zeros_like(dur), dur)

        dec_in, mel_lens = self.length_regulator(enc_out, dur)
        T_mel = dec_in.size(1)
        idx = torch.arange(T_mel, device=dec_in.device).unsqueeze(0)
        mel_pad = idx >= mel_lens.unsqueeze(1)

        dec_h = self.dec_pe(dec_in)
        for blk in self.decoder:
            dec_h = blk(dec_h, key_padding_mask=mel_pad)

        mel_before = self.mel_proj(dec_h)
        mel_after  = mel_before + self.postnet(mel_before)
        L = int(mel_lens[0].item())
        return mel_after[0, :L].detach().cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# §7  Loss
# ══════════════════════════════════════════════════════════════════════════════

def make_length_mask(lengths, max_len):
    """(B, max_len, 1) float mask: 1.0 for real frames, 0.0 for padding."""
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return (idx < lengths.unsqueeze(1)).unsqueeze(-1).float()


def masked_mel_loss(mel_before, mel_after, targets, mel_lengths):
    """
    Masked L1  +  Masked Spectral Convergence.

    CRITICAL: divides by n_valid_ELEMENTS (frames × mel_bins), not just frames.
    Previous bug: mask.sum() counted frames only → loss 80× too large →
    effective lr = 1e-3/80 = 1.25e-5 → model barely trained.
    """
    tgt  = targets.clamp(-1.0, 1.0)
    B, T, M = mel_before.shape

    if mel_lengths is not None:
        mask  = make_length_mask(mel_lengths, T).to(mel_before.device)
        n_el  = (mask.sum() * M).clamp(min=1.0)   # frames × bins

        l1_b  = ((mel_before - tgt).abs() * mask).sum() / n_el
        l1_a  = ((mel_after  - tgt).abs() * mask).sum() / n_el

        pm = mel_after * mask; tm = tgt * mask
        sc = ((tm-pm).norm(p="fro",dim=(-2,-1)) /
              tm.norm(p="fro",dim=(-2,-1)).clamp(min=1e-8)).mean()
    else:
        l1_b = F.l1_loss(mel_before, tgt)
        l1_a = F.l1_loss(mel_after,  tgt)
        diff = (tgt - mel_after).norm(p="fro", dim=(-2,-1))
        sc   = (diff / tgt.norm(p="fro",dim=(-2,-1)).clamp(min=1e-8)).mean()

    return l1_b + l1_a + sc


# ══════════════════════════════════════════════════════════════════════════════
# §8  Dataset & DataLoader  (4-tuple: text_ids, mel, mel_length, durations)
# ══════════════════════════════════════════════════════════════════════════════

class TTSDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s  = self.samples[idx]
        ti = torch.tensor(s["text_ids"], dtype=torch.long)
        m  = torch.from_numpy(np.load(s["mel"]).astype(np.float32))
        ml = torch.tensor(m.shape[0], dtype=torch.long)
        if "durations" in s and len(s["durations"]) == len(s["text_ids"]):
            dur = torch.tensor(s["durations"], dtype=torch.long)
        else:
            # fallback: recompute proportional durations on the fly
            dur = torch.tensor(
                proportional_durations(len(s["text_ids"]), int(m.shape[0])),
                dtype=torch.long)
        return ti, m, ml, dur


def _collate(batch):
    texts, mels, mlens, durs = zip(*batch)
    B = len(texts)
    max_t = max(t.size(0) for t in texts)
    max_m = max(m.size(0) for m in mels)
    n_mels = mels[0].size(1)

    tp = torch.zeros(B, max_t, dtype=torch.long)
    dp = torch.zeros(B, max_t, dtype=torch.long)
    mp = torch.full((B, max_m, n_mels), -2.0)   # -2.0 sentinel (outside [-1,+1])
    for i, (t, m, d) in enumerate(zip(texts, mels, durs)):
        tp[i, :t.size(0)] = t
        mp[i, :m.size(0)] = m
        dp[i, :d.size(0)] = d
    return tp, mp, torch.stack(list(mlens)), dp


def make_dataloader(samples, batch_size, shuffle=True, num_workers=2):
    ds = TTSDataset(samples)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=_collate,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=(shuffle and len(ds) >= batch_size*2))


# ══════════════════════════════════════════════════════════════════════════════
# §9  LR scheduler  (linear warmup + cosine decay)
# ══════════════════════════════════════════════════════════════════════════════

def warmup_cosine(optimizer, warmup_epochs, total_epochs, min_lr=0.05):
    def _lr(ep):
        if ep < warmup_epochs: return (ep+1)/max(warmup_epochs,1)
        p = (ep-warmup_epochs)/max(total_epochs-warmup_epochs,1)
        return min_lr + (1-min_lr)*0.5*(1+math.cos(math.pi*p))
    return optim.lr_scheduler.LambdaLR(optimizer, _lr)


# ══════════════════════════════════════════════════════════════════════════════
# §10  Mel plots
# ══════════════════════════════════════════════════════════════════════════════

def _pango_render(text, font_path, size=14):
    try:
        import gi; gi.require_version("Pango","1.0"); gi.require_version("PangoCairo","1.0")
        from gi.repository import Pango, PangoCairo; import cairo
        t = cairo.ImageSurface(cairo.FORMAT_ARGB32,1,1); tc=cairo.Context(t)
        lay=PangoCairo.create_layout(tc); fd=Pango.FontDescription.from_string(f"Myanmar3 {size}")
        lay.set_font_description(fd); lay.set_text(text,-1)
        w,h=lay.get_pixel_size(); p=6; w+=2*p; h+=2*p
        s=cairo.ImageSurface(cairo.FORMAT_ARGB32,w,h); c=cairo.Context(s)
        c.set_source_rgb(0,0,0); l=PangoCairo.create_layout(c)
        l.set_font_description(fd); l.set_text(text,-1)
        c.move_to(p,p); PangoCairo.show_layout(c,l)
        r=np.frombuffer(s.get_data(),dtype=np.uint8).reshape(h,w,4)
        return r[...,[2,1,0,3]].copy()
    except: return None


def plot_mel(mel, title, save_path, logger, vmin=-1.0, vmax=1.0):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
        mf="/usr/share/fonts/truetype/mm3-multi-os.ttf"
        if os.path.exists(mf):
            font_manager.fontManager.addfont(mf)
            fp=font_manager.FontProperties(fname=mf)
            plt.rcParams["font.family"]="sans-serif"
            plt.rcParams["font.sans-serif"]=[fp.get_name(),"DejaVu Sans"]
        fig,ax=plt.subplots(figsize=(12,4))
        im=ax.imshow(mel.T,aspect="auto",origin="lower",cmap="viridis",
                     vmin=vmin,vmax=vmax,interpolation="none")
        plt.colorbar(im,ax=ax,label="Normalised mel [-1,+1]")
        ax.set_xlabel("Frame"); ax.set_ylabel("Mel bin")
        rgba=_pango_render(title,mf)
        if rgba is not None:
            ab=AnnotationBbox(OffsetImage(rgba,zoom=1),(0.5,1.12),
                              xycoords="axes fraction",
                              box_alignment=(0.5,0.5),frameon=False)
            ax.add_artist(ab)
        else:
            ax.set_title(title.encode("ascii","replace").decode()[:80],fontsize=9)
        plt.tight_layout(); plt.savefig(save_path,dpi=150,bbox_inches="tight")
        plt.close(fig); logger.info(f"Mel plot → {save_path}")
    except Exception as e: logger.warning(f"Mel plot failed: {e}")


def plot_comparison(pred, gt, save_path, logger):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,figsize=(12,6))
        for ax,m,t in zip(axes,[gt,pred],["GT mel","TTS predicted mel"]):
            im=ax.imshow(m.T,aspect="auto",origin="lower",cmap="viridis",
                         vmin=-1,vmax=1,interpolation="none")
            ax.set_title(t); ax.set_xlabel("Frame"); ax.set_ylabel("Mel bin")
            plt.colorbar(im,ax=ax)
        plt.tight_layout(); plt.savefig(save_path,dpi=150); plt.close(fig)
        logger.info(f"Comparison plot → {save_path}")
    except Exception as e: logger.warning(f"Comparison plot failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# §11  Data preparation
# ══════════════════════════════════════════════════════════════════════════════

def _load_split(output_dir, split):
    p = Path(output_dir)/f"{split}.json"
    if not p.exists(): return []
    with open(p, encoding="utf-8") as f: return json.load(f)


def prepare_data(data_dir, tsv_file, output_dir, train_ratio, dev_ratio,
                 seed, mel_cfg, max_samples, logger):
    random.seed(seed); np.random.seed(seed)
    dp = Path(data_dir); tp = dp/tsv_file
    if not tp.exists(): logger.error(f"TSV not found: {tp}"); sys.exit(1)
    wd = dp/"wavs"

    def _wav(fid):
        for p in [wd/f"{fid}.wav", dp/f"{fid}.wav"]:
            if p.exists(): return p

    raw = []
    with open(tp, encoding="utf-8") as f:
        for line in f:
            ps = line.strip().split(maxsplit=1)
            if len(ps)!=2: continue
            fid,text = ps; w=_wav(fid)
            if w: raw.append({"id":fid,"wav":str(w),"text":text})

    if max_samples>0 and len(raw)>max_samples:
        random.shuffle(raw); raw=raw[:max_samples]
    logger.info(f"Found {len(raw)} audio files.")

    vocab = MyanmarCharVocab(); vocab.build([s["text"] for s in raw])
    logger.info(f"Vocabulary: {vocab.size} characters")

    out = Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"mels").mkdir(exist_ok=True)
    vocab.save(str(out/"vocab.json"))
    with open(out/"mel_config.json","w") as f: json.dump(mel_cfg.to_dict(),f)
    with open(out/"normalization.json","w") as f:
        json.dump({"type":"fixed_linear","mel_min_db":_MEL_MIN_DB,
                   "mel_max_db":_MEL_MAX_DB,"norm_min":-1.0,"norm_max":1.0,
                   "formula":"norm=2*(db-(-80))/80-1"},f)

    logger.info("Computing mel spectrograms ...")
    valid=[]; skip=0
    for i,s in enumerate(raw):
        try:
            a=load_audio(s["wav"],mel_cfg.sample_rate)
            d=len(a)/mel_cfg.sample_rate
            if d<0.3 or d>15.0: skip+=1; continue
            mn=audio_to_mel(a,mel_cfg)
            mp=out/"mels"/f"{s['id']}.npy"; np.save(mp,mn)
            ti=vocab.encode(s["text"])
            durs = proportional_durations(len(ti), int(mn.shape[0]))
            assert sum(durs) == int(mn.shape[0]), (
                f"duration sum {sum(durs)} != mel frames {mn.shape[0]}")
            valid.append({"id":s["id"],"wav":s["wav"],"text":s["text"],
                          "text_ids":ti,"mel":str(mp),
                          "durations":durs,
                          "mel_frames":int(mn.shape[0]),"text_len":len(ti)})
        except Exception as e: logger.debug(f"Skip {s['id']}: {e}"); skip+=1
        if (i+1)%500==0: logger.info(f"  {i+1}/{len(raw)} ...")

    logger.info(f"Valid: {len(valid)} | Skipped: {skip}")

    ml=[s["mel_frames"] for s in valid]; tl=[s["text_len"] for s in valid]
    fpc=[m/t for m,t in zip(ml,tl)]
    mfpc=float(np.mean(fpc))
    smp=[np.load(s["mel"]) for s in valid[:50]]
    logger.info(f"Mel norm check: min={np.min([m.min() for m in smp]):.3f} "
                f"max={np.max([m.max() for m in smp]):.3f} "
                f"mean={np.mean([m.mean() for m in smp]):.3f}")
    logger.info(f"Mel frames : mean={np.mean(ml):.0f} ± {np.std(ml):.0f}")
    logger.info(f"Frames/char: mean={mfpc:.1f} ± {np.std(fpc):.1f}")

    with open(out/"dataset_stats.json","w") as f:
        json.dump({"mean_frames_per_char":mfpc,
                   "mean_mel_frames":float(np.mean(ml)),
                   "n_utterances":len(valid)},f)

    random.shuffle(valid)
    n=len(valid); nt=int(n*train_ratio); nd=int(n*dev_ratio)
    splits={"train":valid[:nt],"dev":valid[nt:nt+nd],"test":valid[nt+nd:]}
    for name,data in splits.items():
        with open(out/f"{name}.json","w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False)
        logger.info(f"  {name}: {len(data)}")
    logger.info("Data preparation complete ✓")


# ══════════════════════════════════════════════════════════════════════════════
# §12  Training
# ══════════════════════════════════════════════════════════════════════════════

def _load_artifacts(output_dir):
    out=Path(output_dir)
    v=MyanmarCharVocab.load(str(out/"vocab.json"))
    with open(out/"mel_config.json") as f: mc=MelConfig.from_dict(json.load(f))
    return v, mc


def _load_mean_fpc(output_dir, logger=None, default=7.3):
    """
    Load mean frames-per-char from dataset_stats.json (written by prepare_data).
    Used for logging expected length; NAR length comes from predicted durations.
    """
    p = Path(output_dir)/"dataset_stats.json"
    if p.exists():
        try:
            with open(p) as f:
                return float(json.load(f)["mean_frames_per_char"])
        except Exception:
            pass
    if logger:
        logger.warning(f"dataset_stats.json not found; using fallback "
                       f"mean_frames_per_char={default}")
    return default


def _build_model(args, vocab_size, n_mels, device):
    return TransformerTTSNAR(
        vocab_size=vocab_size, n_mels=n_mels,
        d_model=args.d_model, n_heads=args.n_heads,
        n_enc_layers=args.n_layers, n_dec_layers=args.n_layers,
        d_ff=args.ffn_dim, dropout=args.dropout
    ).to(device)


def _save_ckpt(path, model, opt, sched, epoch, dev_loss, best, args):
    torch.save({"epoch":epoch,"model_state":model.state_dict(),
                "opt_state":opt.state_dict(),"sched_state":sched.state_dict(),
                "dev_loss":dev_loss,"best_dev_loss":best,
                "model_config":{
                    "arch":"TransformerTTSNAR",
                    "vocab_size":model.embed.num_embeddings,
                    "n_mels":model.n_mels,
                    "d_model":args.d_model,"n_heads":args.n_heads,
                    "n_layers":args.n_layers,"ffn_dim":args.ffn_dim,
                    "dropout":args.dropout}}, path)


def run_train(args, logger):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    vocab, mel_cfg = _load_artifacts(args.output_dir)
    train_s = _load_split(args.output_dir,"train")
    dev_s   = _load_split(args.output_dir,"dev")
    if not train_s: logger.error("No training data. Run --stage prep."); sys.exit(1)

    if "durations" not in train_s[0]:
        logger.warning("Sample JSON has no 'durations' — recomputing "
                       "proportional durations on the fly. Prefer re-running "
                       "--stage prep with this NAR script.")

    logger.info(f"Vocab:{vocab.size}  Train:{len(train_s)}  Dev:{len(dev_s)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model  = _build_model(args, vocab.size, mel_cfg.n_mels, device)
    np_    = sum(p.numel() for p in model.parameters())
    logger.info(f"TransformerTTSNAR parameters: {np_:,}")

    opt   = optim.AdamW(model.parameters(), lr=args.lr,
                        betas=(0.9,0.98), eps=1e-9,
                        weight_decay=args.weight_decay)
    sched = warmup_cosine(opt, args.warmup_epochs, args.epochs)

    start=1; best=float("inf")
    if args.resume:
        cp=Path(args.resume)
        if not cp.exists(): logger.error(f"Checkpoint not found: {cp}"); sys.exit(1)
        ck=torch.load(cp,map_location=device,weights_only=False)
        model.load_state_dict(ck["model_state"])
        if "opt_state"   in ck: opt.load_state_dict(ck["opt_state"])
        if "sched_state" in ck: sched.load_state_dict(ck["sched_state"])
        start=ck.get("epoch",0)+1; best=ck.get("best_dev_loss",float("inf"))
        logger.info(f"Resumed from epoch {start-1} (best dev={best:.4f})")

    tr_dl = make_dataloader(train_s, args.batch_size,
                            shuffle=True, num_workers=args.num_workers)
    dv_dl = make_dataloader(dev_s,   args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    logger.info(f"Training {args.epochs} epochs "
                f"(warmup={args.warmup_epochs}, lr={args.lr:.1e})")
    logger.info("Architecture: TransformerTTSNAR (FastSpeech2-style)")
    logger.info("  encoder -> duration predictor -> length regulator -> NAR decoder")
    logger.info(f"  duration loss weight = {args.dur_loss_weight}")
    logger.info("Forced durations: proportional char->mel (or sample['durations'])")

    history = []
    for epoch in range(start, args.epochs+1):

        # ── train ─────────────────────────────────────────────────────────────
        model.train(); sl=0.0; nb=0
        for text_ids, mel_tgt, mel_lens, durs in tr_dl:
            text_ids = text_ids.to(device)
            mel_tgt  = mel_tgt.to(device)
            mel_lens = mel_lens.to(device)
            durs     = durs.to(device)

            _,_,_, loss = model(text_ids, mel_tgt, mel_lens, durs,
                                dur_loss_weight=args.dur_loss_weight)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sl += loss.item(); nb += 1

            if args.verbose and nb % args.log_steps == 0:
                logger.debug(f"  Ep{epoch:3d} s{nb:4d} | "
                             f"loss={loss.item():.4f} "
                             f"lr={opt.param_groups[0]['lr']:.2e}")

        avg_tr = sl/max(nb,1); sched.step()

        # ── validate ──────────────────────────────────────────────────────────
        model.eval(); sd=0.0
        with torch.no_grad():
            for text_ids, mel_tgt, mel_lens, durs in dv_dl:
                _,_,_, loss = model(text_ids.to(device), mel_tgt.to(device),
                                    mel_lens.to(device), durs.to(device),
                                    dur_loss_weight=args.dur_loss_weight)
                sd += loss.item()
        dev_loss = sd/max(len(dv_dl),1)
        lr = opt.param_groups[0]["lr"]

        logger.info(f"Epoch {epoch:4d}/{args.epochs} | "
                    f"train={avg_tr:.4f}  dev={dev_loss:.4f} | lr={lr:.2e}")
        history.append({"epoch":epoch,"train":avg_tr,"dev":dev_loss,"lr":lr})

        if dev_loss < best:
            best = dev_loss
            _save_ckpt(out/"best_model.pt", model, opt, sched,
                       epoch, dev_loss, best, args)
            logger.info(f"  ✓ Best model saved (dev={dev_loss:.4f})")

        if epoch % args.save_every == 0:
            _save_ckpt(out/f"checkpoint_ep{epoch:04d}.pt",
                       model, opt, sched, epoch, dev_loss, best, args)

    with open(out/"train_log.json","w") as f: json.dump(history,f,indent=2)
    logger.info(f"Training done. Best dev: {best:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# §13  Inference helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_model(output_dir, logger):
    out  = Path(output_dir)
    ckpt = torch.load(out/"best_model.pt", map_location="cpu",
                      weights_only=False)
    cfg  = ckpt["model_config"]
    arch = cfg.get("arch", "TransformerTTSNAR")
    if arch != "TransformerTTSNAR":
        logger.warning(f"Checkpoint arch={arch}; loading as TransformerTTSNAR.")
    model = TransformerTTSNAR(
        vocab_size=cfg["vocab_size"], n_mels=cfg["n_mels"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_enc_layers=cfg["n_layers"], n_dec_layers=cfg["n_layers"],
        d_ff=cfg["ffn_dim"], dropout=cfg.get("dropout",0.1))
    model.load_state_dict(ckpt["model_state"]); model.eval()
    logger.info(f"Loaded TransformerTTSNAR: epoch={ckpt.get('epoch','?')} "
                f"dev={ckpt.get('dev_loss',float('nan')):.4f}")
    return model


def synthesize_text(text, model, vocab, device, speed_factor=1.0,
                    logger=None, mean_fpc=7.3):
    """
    Text → normalized mel via one-shot NAR inference.
    Length comes from the duration predictor (scaled by speed_factor).
    """
    ids = torch.tensor([vocab.encode(text)], dtype=torch.long, device=device)
    n_tokens = ids.size(1)
    expected = max(10, int(round(n_tokens * mean_fpc / max(speed_factor, 1e-3))))

    if logger:
        logger.info(f"  length   : {n_tokens} tokens × {mean_fpc:.1f} fpc "
                    f"≈ {expected} frames expected  (speed={speed_factor})")

    with torch.no_grad():
        mel_norm = model.infer(ids, speed_factor=speed_factor)

    mel_db = mel_denormalize(mel_norm)
    if logger:
        logger.info(f"  mel_norm : min={mel_norm.min():.3f}  "
                    f"max={mel_norm.max():.3f}  mean={mel_norm.mean():.3f}")
        logger.info(f"  mel_db   : min={mel_db.min():.1f}  "
                    f"max={mel_db.max():.1f}  mean={mel_db.mean():.1f} dB")
        logger.info(f"  shape    : {mel_norm.shape}  "
                    f"({mel_norm.shape[0]*256/22050:.2f}s at default hop/sr)")
        if mel_db.max() < -30.0:
            logger.warning(f"  ⚠ mel_db max={mel_db.max():.1f} dB. "
                           "Model may need more epochs (try 200+).")
        else:
            logger.info(f"  ✓ mel_db max={mel_db.max():.1f} dB — healthy")
    return mel_norm


def _find_gt_id(sid, output_dir, logger):
    for sp in ("train","dev","test"):
        for s in _load_split(output_dir, sp):
            if s["id"]==sid:
                logger.info(f"Found {sid} in {sp}")
                return np.load(s["mel"]), s["text"]
    return None, None


def _find_gt_text(text, output_dir, logger):
    for sp in ("train","dev","test"):
        for s in _load_split(output_dir, sp):
            if s["text"].strip()==text.strip():
                logger.info(f"Found text match in {sp}")
                return np.load(s["mel"])
    return None


# ══════════════════════════════════════════════════════════════════════════════
# §14  Synthesis stage
# ══════════════════════════════════════════════════════════════════════════════

def run_synth(args, logger):
    out = Path(args.output_dir)
    vocab, mel_cfg = _load_artifacts(args.output_dir)
    mean_fpc = _load_mean_fpc(args.output_dir, logger)
    vocoder = Vocoder(mel_cfg, logger, use_neural=not args.no_neural_vocoder)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.list_samples:
        logger.info("Sample IDs (first 20):")
        for sp in ("train","dev","test"):
            items = _load_split(args.output_dir, sp)
            if items:
                for s in items[:20]:
                    logger.info(f"  {s['id']:32s}  {s['text'][:50]}")
                break
        return

    if args.gt_sample_id:
        gm, gt = _find_gt_id(args.gt_sample_id, args.output_dir, logger)
        if gm is None: logger.error(f"Sample '{args.gt_sample_id}' not found."); return
        logger.info(f"GT text: {gt}")
        db = mel_denormalize(gm)
        logger.info(f"GT mel: min={db.min():.1f} max={db.max():.1f} "
                    f"mean={db.mean():.1f} dB  shape={gm.shape}")
        audio = vocoder.synthesize(gm)
        wpath = args.synth_out or f"gt_{args.gt_sample_id}.wav"
        save_wav(audio, wpath, mel_cfg.sample_rate)
        plot_mel(gm, f"GT: {gt[:50]}", str(Path(wpath).with_suffix(".png")), logger)
        logger.info(f"GT audio → {wpath}  ({len(audio)/mel_cfg.sample_rate:.2f}s)")
        if not args.gt_compare: return

    if args.use_gt_mel and args.synth_text:
        gm = _find_gt_text(args.synth_text, args.output_dir, logger)
        if gm is not None:
            audio = vocoder.synthesize(gm)
            wpath = args.synth_out or "gt_synth.wav"
            save_wav(audio, wpath, mel_cfg.sample_rate)
            plot_mel(gm, f"GT: {args.synth_text[:50]}",
                     str(Path(wpath).with_suffix(".png")), logger)
            logger.info(f"GT-matched audio → {wpath}"); return
        logger.warning("No text match for --use_gt_mel; running model inference.")

    model = load_model(args.output_dir, logger); model.to(device)
    text  = args.synth_text or "မင်္ဂလာပါ ကျောင်းသားများ"
    logger.info(f"Synthesizing: {text!r}")

    t0 = time.time()
    mel_pred = synthesize_text(text, model, vocab, device,
                               speed_factor=args.speed_factor, logger=logger,
                               mean_fpc=mean_fpc)
    logger.info(f"Acoustic model: {(time.time()-t0)*1000:.0f} ms")

    wpath = args.synth_out or str(out/"synthesized.wav")
    plot_mel(mel_pred, f"TTS NAR: {text[:50]}",
             str(Path(wpath).with_suffix(".png")), logger)

    t1 = time.time()
    audio = vocoder.synthesize(mel_pred)
    save_wav(audio, wpath, mel_cfg.sample_rate)
    dur = len(audio)/mel_cfg.sample_rate
    logger.info(f"Vocoder: {(time.time()-t1)*1000:.0f} ms  |  "
                f"Audio → {wpath}  ({dur:.2f}s)")

    if args.gt_compare:
        gm = _find_gt_text(text, args.output_dir, logger)
        if gm is not None:
            ga = vocoder.synthesize(gm)
            gw = str(Path(wpath).stem)+"_GT.wav"
            save_wav(ga, gw, mel_cfg.sample_rate)
            logger.info(f"GT comparison → {gw}")
            plot_comparison(mel_pred, gm,
                            str(Path(wpath).stem)+"_compare.png", logger)
        else:
            logger.info("No exact GT match found.")


# ══════════════════════════════════════════════════════════════════════════════
# §15  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_eval(args, logger):
    out = Path(args.output_dir)
    vocab, mel_cfg = _load_artifacts(args.output_dir)
    mean_fpc = _load_mean_fpc(args.output_dir, logger)
    model = load_model(args.output_dir, logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_s = _load_split(args.output_dir, "test")
    n_ev   = min(args.eval_n, len(test_s))
    if n_ev==0: logger.warning("No test samples."); return

    logger.info(f"Evaluating {n_ev} utterances ...")
    ed = out/"eval_output"; ed.mkdir(exist_ok=True)
    vocoder = Vocoder(mel_cfg, logger, use_neural=not args.no_neural_vocoder)
    results=[]; maes=[]; rtfs=[]; ts=[]

    for i, s in enumerate(test_s[:n_ev]):
        t0=time.time()
        try: mel_pred = synthesize_text(s["text"], model, vocab, device,
                                        mean_fpc=mean_fpc)
        except Exception as e:
            logger.warning(f"  [{i+1}/{n_ev}] {s['id']} failed: {e}"); continue
        tt=time.time()-t0; ts.append(tt)

        mae=None
        try:
            gm=np.load(s["mel"]); T=min(mel_pred.shape[0],gm.shape[0])
            mae=float(np.abs(mel_pred[:T]-gm[:T]).mean()); maes.append(mae)
        except: pass

        plot_mel(mel_pred, f"Synth: {s['text'][:40]}",
                 str(ed/f"{s['id']}_mel.png"), logger)

        rtf=None
        try:
            a=vocoder.synthesize(mel_pred); d=len(a)/mel_cfg.sample_rate
            rtf=tt/d if d>0 else None
            save_wav(a, str(ed/f"{s['id']}.wav"), mel_cfg.sample_rate)
            if rtf: rtfs.append(rtf)
        except Exception as e: logger.debug(f"Vocoder: {e}")

        ms=f"{mae:.4f}" if mae else "N/A"; rs=f"{rtf:.3f}" if rtf else "N/A"
        logger.info(f"  [{i+1:3d}/{n_ev}] {s['id']}  mae={ms}  "
                    f"synth={tt*1000:.0f}ms  RTF={rs}")
        results.append({"id":s["id"],"text":s["text"],"mel_mae":mae,
                        "synth_ms":round(tt*1000,1),"rtf":rtf})

    am = float(np.mean(maes)) if maes else float("nan")
    ar = float(np.mean(rtfs)) if rtfs else float("nan")
    at = float(np.mean(ts))*1000
    logger.info(""); logger.info("="*60)
    logger.info("  EVALUATION — Myanmar SLR80 TTS NAR (FastSpeech2-style)")
    logger.info("="*60)
    logger.info(f"  Utterances : {n_ev}")
    logger.info(f"  Avg Mel MAE: {am:.4f}")
    logger.info(f"  Avg Synth  : {at:.1f} ms")
    logger.info(f"  Avg RTF    : {ar:.3f}")
    logger.info("  UTMOS: https://github.com/sarulab-speech/UTMOS22")
    logger.info("="*60)

    with open(ed/"eval_results.json","w",encoding="utf-8") as f:
        json.dump({"summary":{"n_eval":n_ev,"mel_mae":am,"synth_ms":at,"rtf":ar},
                   "utterances":results},f,ensure_ascii=False,indent=2)
    logger.info(f"Results → {ed/'eval_results.json'}")


# ══════════════════════════════════════════════════════════════════════════════
# §16  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="Myanmar TTS NAR — TransformerTTSNAR (FastSpeech2-style)",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--stage", choices=["prep","train","synth","eval","all"],
                   default="all")
    p.add_argument("--check_deps", action="store_true")
    p.add_argument("--data_dir",    default="./data/slr80")
    p.add_argument("--tsv_file",    default="line_index_female.tsv")
    p.add_argument("--output_dir",  default="./tts_nar")
    p.add_argument("--train_ratio", type=float, default=0.80)
    p.add_argument("--dev_ratio",   type=float, default=0.10)
    p.add_argument("--max_samples", type=int,   default=0)
    p.add_argument("--sample_rate", type=int,   default=22050)
    p.add_argument("--n_fft",       type=int,   default=1024)
    p.add_argument("--hop_length",  type=int,   default=256)
    p.add_argument("--win_length",  type=int,   default=1024)
    p.add_argument("--n_mels",      type=int,   default=80)
    p.add_argument("--fmin",        type=float, default=0.0)
    p.add_argument("--fmax",        type=float, default=8000.0)
    p.add_argument("--d_model",     type=int,   default=256)
    p.add_argument("--n_heads",     type=int,   default=4)
    p.add_argument("--n_layers",    type=int,   default=4,
                   help="Encoder AND decoder layers")
    p.add_argument("--ffn_dim",     type=int,   default=1024)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--epochs",        type=int,   default=300)
    p.add_argument("--warmup_epochs", type=int,   default=20)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--weight_decay",  type=float, default=1e-6)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--save_every",    type=int,   default=10)
    p.add_argument("--log_steps",     type=int,   default=20)
    # Backward-compat args (accepted, ignored)
    p.add_argument("--dur_loss_weight", type=float, default=1.0,
                   help="Weight for duration MSE loss (log-domain)")
    p.add_argument("--synth_text",   type=str,
                   default="မင်္ဂလာပါ ကျောင်းသားများ")
    p.add_argument("--synth_out",    type=str,   default=None)
    p.add_argument("--speed_factor", type=float, default=1.0,
                   help="0.8=slower  1.0=normal  1.2=faster")
    p.add_argument("--eval_n",     type=int, default=50)
    p.add_argument("--log_file",   default=None)
    p.add_argument("--verbose",    action="store_true")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--resume",     type=str, default=None)
    p.add_argument("--list_samples",      action="store_true")
    p.add_argument("--gt_sample_id",      type=str, default=None)
    p.add_argument("--use_gt_mel",        action="store_true")
    p.add_argument("--gt_compare",        action="store_true")
    p.add_argument("--no_neural_vocoder", action="store_true")
    return p


def main():
    parser = build_parser(); args = parser.parse_args()
    logger = setup_logging(args.log_file, args.verbose)

    if args.check_deps:
        sys.exit(0 if check_dependencies(logger) else 1)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    mel_cfg = MelConfig(sample_rate=args.sample_rate, n_fft=args.n_fft,
                        hop_length=args.hop_length, win_length=args.win_length,
                        n_mels=args.n_mels, fmin=args.fmin, fmax=args.fmax)

    if args.stage in ("prep","all"):
        prepare_data(args.data_dir, args.tsv_file, args.output_dir,
                     args.train_ratio, args.dev_ratio, args.seed,
                     mel_cfg, args.max_samples, logger)
    if args.stage in ("train","all"): run_train(args, logger)
    if args.stage in ("synth","all"): run_synth(args, logger)
    if args.stage in ("eval", "all"): run_eval(args, logger)


if __name__ == "__main__":
    main()
