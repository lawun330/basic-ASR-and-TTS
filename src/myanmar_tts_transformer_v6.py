#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Myanmar TTS Demo v6  —  TransformerTTS + Guided Attention + HiFi-GAN
====================================================================
Filename: myanmar_tts_transformer_v6.py

Built on v5 (AR TransformerTTS + guided attention + length-aware stop floor).

v6 additions (on top of v5)
---------------------------
1. Vocoder choice via --vocoder {hifigan,waveglow,griffin_lim}
   (HiFi-GAN default; WaveGlow optional; Griffin-Lim fallback /
   --no_neural_vocoder).
2. Scheduled sampling in late training: mix GT / predicted PreNet
   inputs to reduce exposure bias at inference.
3. Softer stop threshold default (0.65) while keeping min_frames floor.
4. Attention-map logging/plotting on synth — visual check for diagonal
   alignment; tune --ga_weight / --ga_g / --stop_thresh if off-diagonal.
5. CLI knobs for GA weight, GA band width g, stop threshold, scheduled
   sampling schedule.

Architecture (same acoustic core as v5):
  Text → CharEmbed+SinPE → TransformerEncoder(N)
                                   ↓  cross-attention (guided)
  [mel t-1] → PreNet → +SinPE → TransformerDecoder(N)
                                       ↓
                               Linear → mel + stop; Postnet
  Training: teacher forcing (+ scheduled sampling) + GA + stop loss
  Inference: AR loop, stop_thresh≈0.65, min_frames from mean_fpc
  Vocoder: HiFi-GAN (default) | WaveGlow | Griffin-Lim

NOTE: retrain from scratch vs v5 if you change GA weights / SS schedule
      meaningfully; prep mels can be reused from v5.

UPDATED FOR AIEF CLASS: Ye Kyaw Thu, LU Lab., Myanmar
DATE: 22 Sep 2026
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
# §4  Vocoder wrapper  —  HiFi-GAN / WaveGlow / Griffin-Lim
# ══════════════════════════════════════════════════════════════════════════════

class Vocoder:
    """
    Mel → wav.  Select with --vocoder {hifigan,waveglow,griffin_lim}.
    --no_neural_vocoder is an alias for griffin_lim.
    Neural backends load from NVIDIA torch.hub; on failure → Griffin-Lim.
    """
    def __init__(self, cfg, logger, backend="hifigan", waveglow_sigma=0.6):
        self.cfg = cfg
        self.logger = logger
        self.backend = backend
        self.waveglow_sigma = float(waveglow_sigma)
        self._model = None
        self._denoiser = None
        if backend == "griffin_lim":
            logger.info("Vocoder: Griffin-Lim")
            return
        if backend == "hifigan":
            self._load_hifigan()
        elif backend == "waveglow":
            self._load_waveglow()
        else:
            raise ValueError(f"Unknown vocoder backend: {backend!r}")

    def _hub_load(self, entry):
        kwargs = dict(pretrained=True, verbose=False)
        try:
            return torch.hub.load(
                "NVIDIA/DeepLearningExamples:torchhub", entry, trust_repo=True,
                **kwargs)
        except TypeError:
            # older torch without trust_repo=
            return torch.hub.load(
                "NVIDIA/DeepLearningExamples:torchhub", entry, **kwargs)

    def _load_hifigan(self):
        try:
            loaded = self._hub_load("nvidia_hifigan")
            if isinstance(loaded, (tuple, list)) and len(loaded) >= 1:
                self._model = loaded[0]
                self._denoiser = loaded[2] if len(loaded) > 2 else None
            else:
                self._model = loaded
            self._model.eval()
            for m in self._model.modules():
                if hasattr(m, "weight_g"):
                    try:
                        nn.utils.remove_weight_norm(m)
                    except Exception:
                        pass
            self.logger.info("HiFi-GAN loaded ✓  (NVIDIA torch.hub)")
        except Exception as e:
            self.logger.warning(
                f"HiFi-GAN unavailable ({type(e).__name__}: {e}); "
                "using Griffin-Lim."
            )
            self.backend = "griffin_lim"
            self._model = None

    def _load_waveglow(self):
        try:
            wg = self._hub_load("nvidia_waveglow")
            # fuse weight-norm for faster infer (NVIDIA API)
            if hasattr(wg, "remove_weightnorm"):
                wg = wg.remove_weightnorm(wg)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = wg.to(device).eval()
            self.logger.info(
                f"WaveGlow loaded ✓  (NVIDIA torch.hub, sigma={self.waveglow_sigma})"
            )
        except Exception as e:
            self.logger.warning(
                f"WaveGlow unavailable ({type(e).__name__}: {e}); "
                "using Griffin-Lim."
            )
            self.backend = "griffin_lim"
            self._model = None

    def synthesize(self, mel_norm):
        if self.backend == "hifigan" and self._model is not None:
            try:
                return self._hifigan_infer(mel_norm)
            except Exception as e:
                self.logger.warning(
                    f"HiFi-GAN infer failed ({type(e).__name__}); "
                    "falling back to Griffin-Lim."
                )
        elif self.backend == "waveglow" and self._model is not None:
            try:
                return self._waveglow_infer(mel_norm)
            except Exception as e:
                self.logger.warning(
                    f"WaveGlow infer failed ({type(e).__name__}); "
                    "falling back to Griffin-Lim."
                )
        return mel_to_audio_gl(mel_norm, self.cfg)

    def _mel_batch(self, mel_norm):
        # mel_norm [-1,+1] → dB → (1, n_mels, T) for NVIDIA vocoders
        mel_db = mel_denormalize(mel_norm).T.astype(np.float32)
        return torch.from_numpy(mel_db).unsqueeze(0)

    def _peak_norm(self, audio_t):
        a = audio_t.squeeze().detach().cpu().numpy().astype(np.float32)
        if a.ndim > 1:
            a = a.reshape(-1)
        pk = float(np.max(np.abs(a))) if a.size else 0.0
        return (a / pk * 0.9).astype(np.float32) if pk > 1e-8 else a

    def _hifigan_infer(self, mel_norm):
        t = self._mel_batch(mel_norm)
        d = next(self._model.parameters()).device
        t = t.to(d)
        with torch.no_grad():
            audio = self._model(t)
            if self._denoiser is not None:
                try:
                    audio = self._denoiser(audio.float(), strength=0.01)
                except Exception:
                    pass
        return self._peak_norm(audio)

    def _waveglow_infer(self, mel_norm):
        t = self._mel_batch(mel_norm)
        d = next(self._model.parameters()).device
        t = t.to(d)
        with torch.no_grad():
            audio = self._model.infer(t, sigma=self.waveglow_sigma)
        return self._peak_norm(audio)


def resolve_vocoder_backend(args):
    """Map CLI flags → backend name.  --no_neural_vocoder wins over --vocoder."""
    if getattr(args, "no_neural_vocoder", False):
        return "griffin_lim"
    return getattr(args, "vocoder", "hifigan") or "hifigan"


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
# §6  TransformerTTS — autoregressive decoder
# ══════════════════════════════════════════════════════════════════════════════

class PreNet(nn.Module):
    """
    Two-layer MLP pre-net for mel decoder input.

    CRITICAL: dropout kept ACTIVE at inference (training=True).
    Tacotron2 paper finding: always-on dropout prevents the decoder
    from memorising training mel sequences and forces it to rely on
    the encoder text context.  Without this, inference collapses.
    """
    def __init__(self, n_mels, d_model, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(n_mels, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.p   = dropout

    def forward(self, x):
        # training=True → dropout always active (also at eval/inference)
        x = F.dropout(F.relu(self.fc1(x)), p=self.p, training=True)
        x = F.dropout(F.relu(self.fc2(x)), p=self.p, training=True)
        return x


class TransformerDecoderBlock(nn.Module):
    """
    Transformer decoder layer:
      1. Causal (masked) self-attention on mel frames
      2. Cross-attention to encoder output (learns text-mel alignment)
      3. Conv FFN
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, ffn_kernel=9):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads,
                                                 dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                 dropout=dropout, batch_first=True)
        self.ffn   = ConvFFN(d_model, d_ff, ffn_kernel, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, enc_out,
                self_mask=None, enc_key_padding_mask=None,
                need_weights=False):
        # 1. Causal self-attention
        sa, _ = self.self_attn(x, x, x, attn_mask=self_mask, need_weights=False)
        x = self.norm1(x + self.drop(sa))
        # 2. Cross-attention to encoder
        #    need_weights=True (training only) returns (B, T_mel, T_text)
        #    attention map, averaged over heads — used for guided-attention loss.
        ca, attn_w = self.cross_attn(
            x, enc_out, enc_out,
            key_padding_mask=enc_key_padding_mask,
            need_weights=need_weights, average_attn_weights=True)
        x = self.norm2(x + self.drop(ca))
        # 3. Conv FFN
        return self.ffn(x), attn_w


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


class TransformerTTS(nn.Module):
    """
    Autoregressive Transformer TTS (v6).

    Training  (teacher forcing + optional scheduled sampling):
      model(text_ids, mel_targets, mel_lengths, ss_prob=..., ga_weight=..., ...)
      → (mel_before, mel_after, stop_logit, total_loss, attn_maps)

    Inference (autoregressive, frame-by-frame):
      mel_norm, attn = model.infer(text_ids, stop_thresh=0.65, ...)
    """

    def __init__(self, vocab_size, n_mels,
                 d_model=256, n_heads=4,
                 n_enc_layers=4, n_dec_layers=4,
                 d_ff=1024, ffn_kernel=9, dropout=0.1):
        super().__init__()
        self.n_mels = n_mels

        # ── Encoder ──────────────────────────────────────────────────────────
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.enc_pe  = SinusoidalPE(d_model, dropout=dropout)
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout, ffn_kernel)
            for _ in range(n_enc_layers)])

        # ── Decoder ──────────────────────────────────────────────────────────
        self.prenet  = PreNet(n_mels, d_model, dropout=0.5)
        self.dec_pe  = SinusoidalPE(d_model, dropout=dropout)
        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout, ffn_kernel)
            for _ in range(n_dec_layers)])

        # ── Output heads ─────────────────────────────────────────────────────
        self.mel_proj  = nn.Linear(d_model, n_mels)
        self.stop_proj = nn.Linear(d_model, 1)
        self.postnet   = Postnet(n_mels, min(d_model*2, 512), dropout)

    def _encode(self, text_ids):
        pad_mask = (text_ids == 0)                # (B, T_text)
        x = self.enc_pe(self.embed(text_ids))
        for blk in self.encoder:
            x = blk(x, key_padding_mask=pad_mask)
        return x, pad_mask                        # enc_out, enc_pad_mask

    def _causal_mask(self, T, device):
        """Upper-triangular bool mask: position i cannot attend to j > i."""
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def _decode(self, dec_in, enc_out, enc_pad, need_weights=True):
        """Run PreNet + decoder stack; return mel heads + attention maps."""
        B, T_mel, _ = dec_in.shape
        causal = self._causal_mask(T_mel, dec_in.device)
        dec_h  = self.dec_pe(self.prenet(dec_in))
        attn_maps = []
        for layer in self.decoder:
            dec_h, attn_w = layer(
                dec_h, enc_out,
                self_mask=causal,
                enc_key_padding_mask=enc_pad,
                need_weights=need_weights,
            )
            if need_weights and attn_w is not None:
                attn_maps.append(attn_w)
        mel_before = self.mel_proj(dec_h)
        mel_after  = mel_before + self.postnet(mel_before)
        stop_logit = self.stop_proj(dec_h).squeeze(-1)
        return mel_before, mel_after, stop_logit, attn_maps

    # ── forward (training — teacher forcing + scheduled sampling) ─────────────

    def forward(self, text_ids, mel_targets, mel_lengths,
                ss_prob=0.0, ga_weight=1.0, ga_g=0.2, stop_loss_weight=0.1):
        """
        Teacher-forced training forward with optional scheduled sampling.

        ss_prob : float in [0,1] — per-frame probability of feeding the
                  previous *predicted* mel (detached) instead of GT into
                  the PreNet.  0.0 = pure teacher forcing (v5 behaviour).
        """
        B, T_mel, _ = mel_targets.shape
        dev = text_ids.device

        enc_out, enc_pad = self._encode(text_ids)
        text_lengths = (~enc_pad).sum(dim=1)

        zeros = torch.zeros(B, 1, self.n_mels, device=dev)
        gt_in = torch.cat([zeros,
                           mel_targets[:, :-1].clamp(-1.0, 1.0)], dim=1)

        # Scheduled sampling (v6): first TF pass → mix GT/pred → second pass
        if self.training and ss_prob > 0.0:
            with torch.no_grad():
                mel_tf, _, _, _ = self._decode(
                    gt_in, enc_out, enc_pad, need_weights=False)
            pred_in = torch.cat([
                zeros, mel_tf[:, :-1].detach().clamp(-1.0, 1.0)], dim=1)
            # Bernoulli mask over time (broadcast over mel bins)
            bern = (torch.rand(B, T_mel, 1, device=dev) < ss_prob).float()
            # do not sample on padding frames
            frame_idx = torch.arange(T_mel, device=dev).view(1, T_mel, 1)
            valid = (frame_idx < mel_lengths.view(B, 1, 1)).float()
            bern = bern * valid
            dec_in = bern * pred_in + (1.0 - bern) * gt_in
        else:
            dec_in = gt_in

        mel_before, mel_after, stop_logit, attn_maps = self._decode(
            dec_in, enc_out, enc_pad, need_weights=True)

        mel_loss = masked_mel_loss(mel_before, mel_after, mel_targets, mel_lengths)

        ga_loss = torch.stack([
            guided_attention_loss(a, text_lengths, mel_lengths, g=ga_g)
            for a in attn_maps
        ]).mean()

        stop_tgt = torch.zeros(B, T_mel, device=dev)
        for b in range(B):
            last = min(int(mel_lengths[b].item()) - 1, T_mel - 1)
            stop_tgt[b, last] = 1.0

        frame_idx = torch.arange(T_mel, device=dev).unsqueeze(0)
        stop_mask = (frame_idx < mel_lengths.to(dev).unsqueeze(1)).float()
        pos_w = stop_mask.sum(dim=1).clamp(min=2.0)
        stop_loss = torch.stack([
            F.binary_cross_entropy_with_logits(
                stop_logit[b], stop_tgt[b],
                pos_weight=pos_w[b:b+1],
                reduction="none"
            ).mul(stop_mask[b]).sum() / stop_mask[b].sum().clamp(min=1)
            for b in range(B)
        ]).mean()

        total = mel_loss + stop_loss_weight * stop_loss + ga_weight * ga_loss
        return mel_before, mel_after, stop_logit, total, attn_maps

    # ── infer (autoregressive) ────────────────────────────────────────────────

    def infer(self, text_ids, max_frames=2000, stop_thresh=0.65, min_frames=10,
              return_attn=True):
        """
        Autoregressive inference.  v6 default stop_thresh=0.65 (softer than
        v5's 0.5).  Optionally returns mean cross-attention map (T_mel, T_text)
        from the last decoder layer for diagonal diagnostics.
        """
        enc_out, enc_pad = self._encode(text_ids)
        dev = text_ids.device

        frames: List[torch.Tensor] = []
        attn_steps: List[torch.Tensor] = []
        dec_in = torch.zeros(1, 1, self.n_mels, device=dev)

        for t in range(max_frames):
            T = dec_in.size(1)
            dec_h = self.dec_pe(self.prenet(dec_in))
            causal = self._causal_mask(T, dev)
            last_attn = None
            for layer in self.decoder:
                dec_h, attn_w = layer(
                    dec_h, enc_out,
                    self_mask=causal,
                    enc_key_padding_mask=enc_pad,
                    need_weights=return_attn,
                )
                if return_attn and attn_w is not None:
                    last_attn = attn_w  # (1, T, T_text)

            last_h = dec_h[:, -1:]
            mel_frame = self.mel_proj(last_h)
            stop_prob = torch.sigmoid(self.stop_proj(last_h))

            frames.append(mel_frame[0, 0])
            if return_attn and last_attn is not None:
                attn_steps.append(last_attn[0, -1].detach().cpu())  # (T_text,)

            dec_in = torch.cat([dec_in, mel_frame], dim=1)

            if stop_prob.item() > stop_thresh and t >= min_frames:
                break

        if not frames:
            empty = np.zeros((1, self.n_mels), dtype=np.float32)
            return (empty, None) if return_attn else empty

        mel_pred = torch.stack(frames).unsqueeze(0)
        mel_after = mel_pred + self.postnet(mel_pred)
        mel_np = mel_after[0].detach().cpu().numpy()

        attn_np = None
        if return_attn and attn_steps:
            # (T_mel, T_text)
            attn_np = torch.stack(attn_steps, dim=0).numpy()

        return (mel_np, attn_np) if return_attn else mel_np


# ══════════════════════════════════════════════════════════════════════════════
# §7  Loss
# ══════════════════════════════════════════════════════════════════════════════

def make_length_mask(lengths, max_len):
    """(B, max_len, 1) float mask: 1.0 for real frames, 0.0 for padding."""
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return (idx < lengths.unsqueeze(1)).unsqueeze(-1).float()


def guided_attention_loss(attn_w, text_lengths, mel_lengths, g=0.2):
    """
    Guided Attention Loss (DC-TTS, Tachibana et al. 2017).

    ═══════════════════════════════════════════════════════════════
    WHY THIS IS NEEDED  (root cause of the "11-frame / wind" failure)
    ═══════════════════════════════════════════════════════════════
    Vanilla cross-attention is free to attend ANYWHERE in the text at
    each decoder step.  With only ~2000 training utterances it never
    discovers on its own that attention should sweep monotonically
    left-to-right as time progresses.  Instead it found a shortcut:
    mostly ignore the text, predict "stop" almost immediately.
    Evidence: a 9-char and a 62-char input both produced exactly the
    same ~11 frames — proof the decoder wasn't actually reading the
    encoder output.

    This loss penalises attention mass that falls far from the
    diagonal line  (text_position/text_len == mel_position/mel_len).
    It gives the model an explicit, strong gradient toward the only
    sensible alignment for speech: monotonic, left-to-right, one
    pass over the text.  This is the standard fix in TTS literature
    for exactly this failure mode on small datasets.

    attn_w       : (B, T_mel, T_text)  — averaged-over-heads attention
                   weights from one decoder layer (rows sum to 1)
    text_lengths : (B,)  real (non-padded) text token counts
    mel_lengths  : (B,)  real (non-padded) mel frame counts
    g            : float — band width; smaller = stricter diagonal
    """
    B, T_mel, T_text = attn_w.shape
    device = attn_w.device
    losses = []
    for b in range(B):
        Tm = min(int(mel_lengths[b].item()),  T_mel)
        Tt = min(int(text_lengths[b].item()), T_text)
        if Tm < 1 or Tt < 1:
            continue
        n = torch.arange(Tm, device=device, dtype=torch.float32).unsqueeze(1) / max(Tm-1, 1)
        t = torch.arange(Tt, device=device, dtype=torch.float32).unsqueeze(0) / max(Tt-1, 1)
        W = 1.0 - torch.exp(-((n - t) ** 2) / (2.0 * g * g))   # (Tm, Tt), 0 on diagonal
        losses.append((attn_w[b, :Tm, :Tt] * W).mean())
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


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
# §8  Dataset & DataLoader  (3-tuple: text_ids, mel_norm, mel_length)
# ══════════════════════════════════════════════════════════════════════════════

class TTSDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s  = self.samples[idx]
        ti = torch.tensor(s["text_ids"], dtype=torch.long)
        m  = torch.from_numpy(np.load(s["mel"]).astype(np.float32))
        ml = torch.tensor(m.shape[0], dtype=torch.long)
        return ti, m, ml


def _collate(batch):
    texts, mels, mlens = zip(*batch)
    B = len(texts)
    max_t = max(t.size(0) for t in texts)
    max_m = max(m.size(0) for m in mels)
    n_mels = mels[0].size(1)

    tp = torch.zeros(B, max_t, dtype=torch.long)
    mp = torch.full((B, max_m, n_mels), -2.0)   # -2.0 sentinel (outside [-1,+1])
    for i,(t,m) in enumerate(zip(texts,mels)):
        tp[i,:t.size(0)] = t
        mp[i,:m.size(0)] = m
    return tp, mp, torch.stack(list(mlens))


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
        from matplotlib import font_manager
        # use the real face name from the TTF (Myanmar3 / Padauk / Noto …)
        face = "sans-serif"
        if font_path and os.path.exists(font_path):
            face = font_manager.FontProperties(fname=font_path).get_name()
        t = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1); tc = cairo.Context(t)
        lay = PangoCairo.create_layout(tc)
        fd = Pango.FontDescription.from_string(f"{face} {size}")
        lay.set_font_description(fd); lay.set_text(text, -1)
        w, h = lay.get_pixel_size(); p = 6; w += 2 * p; h += 2 * p
        s = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h); c = cairo.Context(s)
        c.set_source_rgb(0, 0, 0); l = PangoCairo.create_layout(c)
        l.set_font_description(fd); l.set_text(text, -1)
        c.move_to(p, p); PangoCairo.show_layout(c, l)
        r = np.frombuffer(s.get_data(), dtype=np.uint8).reshape(h, w, 4)
        return r[..., [2, 1, 0, 3]].copy()
    except Exception:
        return None


def _myanmar_font_path():
    """Prefer mm3 / Padauk (Latin+Myanmar), then Noto Myanmar."""
    candidates = [
        "/usr/share/fonts/truetype/mm3-multi-os.ttf",
        "/usr/share/fonts/truetype/padauk/Padauk-Regular.ttf",
        "/usr/share/fonts/truetype/padauk/PadaukBook-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMyanmar-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSerifMyanmar-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _split_latin_prefix(title: str):
    """Split leading ASCII from the rest (e.g. 'GT: ' | 'မင်္ဂလာပါ…')."""
    i = 0
    while i < len(title) and ord(title[i]) < 128:
        i += 1
    return title[:i], title[i:]


def _title_fontproperties(fontsize=11):
    """Padauk/mm3 (Latin+Myanmar) preferred; never default to DejaVu for Myanmar text."""
    from matplotlib import font_manager
    mf = _myanmar_font_path()
    if not mf:
        return None, None
    font_manager.fontManager.addfont(mf)
    return mf, font_manager.FontProperties(fname=mf, size=fontsize)


def _set_mixed_title(ax, title, font_path, fontsize=11):
    """
    Noto Myanmar has no Latin glyphs — split ASCII prefix vs Myanmar body.
    Both halves use TTFs from disk (Padauk for ASCII if available), never DejaVu.
    """
    from matplotlib import font_manager
    title = title[:80]
    latin, myanmar = _split_latin_prefix(title)
    # prefer Padauk for Latin so we never touch DejaVu
    latin_path = font_path
    for p in (
        "/usr/share/fonts/truetype/padauk/Padauk-Regular.ttf",
        "/usr/share/fonts/truetype/padauk/PadaukBook-Regular.ttf",
        font_path,
    ):
        if p and os.path.exists(p):
            latin_path = p
            break
    fp_latin = font_manager.FontProperties(fname=latin_path, size=fontsize)
    fp_mya = font_manager.FontProperties(fname=font_path, size=fontsize)

    if myanmar and latin:
        ax.text(0.5, 1.05, latin, transform=ax.transAxes,
                ha="right", va="bottom", fontproperties=fp_latin,
                clip_on=False)
        ax.text(0.5, 1.05, myanmar, transform=ax.transAxes,
                ha="left", va="bottom", fontproperties=fp_mya,
                clip_on=False)
    elif myanmar:
        ax.set_title(myanmar, fontproperties=fp_mya)
    else:
        ax.set_title(latin or title, fontproperties=fp_latin)


def plot_mel(mel, title, save_path, logger, vmin=-1.0, vmax=1.0):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
        mf, title_fp = _title_fontproperties(11)
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(mel.T, aspect="auto", origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax, interpolation="none")
        plt.colorbar(im, ax=ax, label="Normalised mel [-1,+1]")
        ax.set_xlabel("Frame"); ax.set_ylabel("Mel bin")

        # Pango only works if `gi` is importable (system Python; often missing in conda)
        rgba = _pango_render(title, mf) if mf else None
        if rgba is not None:
            ab = AnnotationBbox(OffsetImage(rgba, zoom=1), (0.5, 1.12),
                                xycoords="axes fraction",
                                box_alignment=(0.5, 0.5), frameon=False)
            ax.add_artist(ab)
        elif mf is not None:
            # Padauk/mm3 cover Latin+Myanmar → one FontProperties is enough.
            # Noto Sans Myanmar is Myanmar-only → split Latin / Myanmar fonts.
            if "Noto" in mf and "Myanmar" in mf:
                _set_mixed_title(ax, title, mf)
            else:
                ax.set_title(title[:80], fontproperties=title_fp)
        else:
            ax.set_title(title[:80], fontsize=9)
            logger.warning(
                "No Myanmar TTF found; install fonts-sil-padauk / fonts-noto-core "
                "or mm3 for correct titles."
            )
        plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig); logger.info(f"Mel plot → {save_path}")
    except Exception as e:
        logger.warning(f"Mel plot failed: {e}")


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



def plot_attention(attn, save_path, logger, title="Cross-attention (last layer)"):
    """
    Plot (T_mel, T_text) attention.  A healthy v5/v6 model shows a clear
    diagonal band (monotonic left-to-right).  If mass is diffuse or stuck
    on one column, raise --ga_weight or tighten --ga_g and retrain.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if attn is None or getattr(attn, "size", 0) == 0:
            logger.warning("No attention map to plot.")
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(attn, aspect="auto", origin="lower", cmap="magma",
                       interpolation="nearest")
        ax.set_xlabel("Text position")
        ax.set_ylabel("Mel frame")
        mf, title_fp = _title_fontproperties(11)
        if title_fp is not None:
            if mf and "Noto" in mf and "Myanmar" in mf:
                _set_mixed_title(ax, title, mf)
            else:
                ax.set_title(title[:80], fontproperties=title_fp)
        else:
            # ascii-only fallback — strip Myanmar to avoid DejaVu glyph warnings
            safe = "".join(c if ord(c) < 128 else "?" for c in title[:80])
            ax.set_title(safe)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # rough diagonal guide
        Tm, Tt = attn.shape
        if Tm > 1 and Tt > 1:
            ax.plot([0, Tt - 1], [0, Tm - 1], "c--", linewidth=1, alpha=0.6,
                    label="ideal diagonal")
            ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        # simple diagonal score: mean mass near diagonal band
        score = _attention_diagonal_score(attn)
        logger.info(f"Attention plot → {save_path}  (diag_score={score:.3f}; "
                    f">0.3 usually OK)")
        if score < 0.25:
            logger.warning(
                "Attention looks off-diagonal.  Consider higher --ga_weight "
                "or smaller --ga_g, then retrain."
            )
    except Exception as e:
        logger.warning(f"Attention plot failed: {e}")


def _attention_diagonal_score(attn, band=0.15):
    """Fraction of attention mass within a relative band of the diagonal."""
    Tm, Tt = attn.shape
    if Tm < 2 or Tt < 2:
        return 0.0
    total = float(attn.sum()) + 1e-8
    near = 0.0
    for i in range(Tm):
        ideal = i / max(Tm - 1, 1) * (Tt - 1)
        lo = max(0, int(ideal - band * Tt))
        hi = min(Tt, int(ideal + band * Tt) + 1)
        near += float(attn[i, lo:hi].sum())
    return near / total



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
            valid.append({"id":s["id"],"wav":s["wav"],"text":s["text"],
                          "text_ids":ti,"mel":str(mp),
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
    Used to compute a length-aware min_frames floor at inference, so output
    duration tracks input text length instead of relying solely on a
    stop-token that can fire prematurely (see infer() docstring).
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
    return TransformerTTS(
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
                    "arch":"TransformerTTSv6",
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

    logger.info(f"Vocab:{vocab.size}  Train:{len(train_s)}  Dev:{len(dev_s)}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {dev}")

    model  = _build_model(args, vocab.size, mel_cfg.n_mels, dev)
    np_    = sum(p.numel() for p in model.parameters())
    logger.info(f"TransformerTTS parameters: {np_:,}")

    opt   = optim.AdamW(model.parameters(), lr=args.lr,
                        betas=(0.9,0.98), eps=1e-9,
                        weight_decay=args.weight_decay)
    sched = warmup_cosine(opt, args.warmup_epochs, args.epochs)

    start=1; best=float("inf")
    if args.resume:
        cp=Path(args.resume)
        if not cp.exists(): logger.error(f"Checkpoint not found: {cp}"); sys.exit(1)
        ck=torch.load(cp,map_location=dev,weights_only=False)
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
    logger.info("Architecture: TransformerTTS v6 + Guided Attention + HiFi-GAN")
    logger.info(f"  ga_weight={args.ga_weight}  ga_g={args.ga_g}  "
                f"stop_loss_weight={args.stop_loss_weight}")
    logger.info(f"  scheduled sampling: start_epoch={args.ss_start_epoch}  "
                f"max_prob={args.ss_max_prob}")
    logger.info("Expected: dig should fall toward ~0.2–0.4 like v5 if GA is healthy.")

    history = []
    for epoch in range(start, args.epochs+1):

        # ── train ─────────────────────────────────────────────────────────────
        # scheduled sampling ramp (0 until ss_start_epoch, then → ss_max_prob)
        if epoch < args.ss_start_epoch:
            ss_prob = 0.0
        else:
            span = max(1, args.epochs - args.ss_start_epoch)
            ss_prob = args.ss_max_prob * min(
                1.0, (epoch - args.ss_start_epoch + 1) / span)

        model.train(); sl=0.0; nb=0
        for text_ids, mel_tgt, mel_lens in tr_dl:
            text_ids = text_ids.to(dev)
            mel_tgt  = mel_tgt.to(dev)
            mel_lens = mel_lens.to(dev)

            *_, loss, _attn = model(
                text_ids, mel_tgt, mel_lens,
                ss_prob=ss_prob,
                ga_weight=args.ga_weight,
                ga_g=args.ga_g,
                stop_loss_weight=args.stop_loss_weight,
            )
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
            for text_ids, mel_tgt, mel_lens in dv_dl:
                *_, loss, _attn = model(
                    text_ids.to(dev), mel_tgt.to(dev), mel_lens.to(dev),
                    ss_prob=0.0,
                    ga_weight=args.ga_weight,
                    ga_g=args.ga_g,
                    stop_loss_weight=args.stop_loss_weight,
                )
                sd += loss.item()
        dev_loss = sd/max(len(dv_dl),1)
        lr = opt.param_groups[0]["lr"]

        logger.info(f"Epoch {epoch:4d}/{args.epochs} | "
                    f"train={avg_tr:.4f}  dev={dev_loss:.4f} | lr={lr:.2e}  ss={ss_prob:.2f}")
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
    arch = cfg.get("arch", "TransformerTTSv6")
    if arch not in ("TransformerTTSv6", "TransformerTTS"):
        logger.warning(f"Checkpoint arch={arch}; loading as TransformerTTSv6.")
    model = TransformerTTS(
        vocab_size=cfg["vocab_size"], n_mels=cfg["n_mels"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_enc_layers=cfg["n_layers"], n_dec_layers=cfg["n_layers"],
        d_ff=cfg["ffn_dim"], dropout=cfg.get("dropout",0.1))
    model.load_state_dict(ckpt["model_state"]); model.eval()
    logger.info(f"Loaded TransformerTTS: epoch={ckpt.get('epoch','?')} "
                f"dev={ckpt.get('dev_loss',float('nan')):.4f}")
    return model


def synthesize_text(text, model, vocab, device, speed_factor=1.0,
                    logger=None, mean_fpc=7.3, stop_thresh=0.65,
                    return_attn=True, attn_path=None):
    """
    Text → normalized mel via autoregressive inference (v6).

    stop_thresh default 0.65 (softer than v5 0.5).  min_frames floor kept.
    If return_attn, also returns / optionally saves attention heatmap.
    """
    ids = torch.tensor([vocab.encode(text)], dtype=torch.long, device=device)
    n_tokens   = ids.size(1)
    expected   = max(10, int(round(n_tokens * mean_fpc)))
    min_frames = max(10, int(round(expected * 0.5)))
    max_frames = int(round(expected * 2.5)) + 20

    if logger:
        logger.info(f"  length   : {n_tokens} tokens × {mean_fpc:.1f} fpc "
                    f"≈ {expected} frames  (floor={min_frames}, cap={max_frames})")
        logger.info(f"  stop_thresh={stop_thresh}")

    with torch.no_grad():
        out = model.infer(
            ids, max_frames=max_frames, min_frames=min_frames,
            stop_thresh=stop_thresh, return_attn=return_attn,
        )
    if return_attn:
        mel_norm, attn = out
    else:
        mel_norm, attn = out, None

    if abs(speed_factor-1.0) > 0.01:
        from scipy.ndimage import zoom
        t_new    = max(10, round(mel_norm.shape[0]/speed_factor))
        mel_norm = zoom(mel_norm, (t_new/mel_norm.shape[0], 1), order=1)
        if attn is not None and attn.shape[0] > 1:
            attn = zoom(attn, (t_new/attn.shape[0], 1), order=1)

    if attn_path and attn is not None and logger is not None:
        plot_attention(attn, attn_path, logger,
                       title=f"Attention: {text[:40]}")

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
    vocoder = Vocoder(
        mel_cfg, logger,
        backend=resolve_vocoder_backend(args),
        waveglow_sigma=args.waveglow_sigma,
    )
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
        wpath = Path(args.synth_out) if args.synth_out else Path(f"gt_{args.gt_sample_id}.wav")
        wpath.parent.mkdir(parents=True, exist_ok=True)
        save_wav(audio, str(wpath), mel_cfg.sample_rate)
        plot_mel(gm, f"GT: {gt[:50]}", str(wpath.with_suffix(".png")), logger)
        logger.info(f"GT audio → {wpath}  ({len(audio)/mel_cfg.sample_rate:.2f}s)")
        if not args.gt_compare: return

    if args.use_gt_mel and args.synth_text:
        gm = _find_gt_text(args.synth_text, args.output_dir, logger)
        if gm is not None:
            audio = vocoder.synthesize(gm)
            wpath = Path(args.synth_out) if args.synth_out else Path("gt_synth.wav")
            wpath.parent.mkdir(parents=True, exist_ok=True)
            save_wav(audio, str(wpath), mel_cfg.sample_rate)
            plot_mel(gm, f"GT: {args.synth_text[:50]}",
                     str(wpath.with_suffix(".png")), logger)
            logger.info(f"GT-matched audio → {wpath}"); return
        logger.warning("No text match for --use_gt_mel; running model inference.")

    model = load_model(args.output_dir, logger); model.to(device)
    text  = args.synth_text or "မင်္ဂလာပါ ကျောင်းသားများ"
    logger.info(f"Synthesizing: {text!r}")

    t0 = time.time()
    wpath = Path(args.synth_out) if args.synth_out else (out / "synthesized.wav")
    wpath.parent.mkdir(parents=True, exist_ok=True)
    attn_path = str(wpath.with_name(wpath.stem + "_attn.png"))
    mel_pred = synthesize_text(
        text, model, vocab, device,
        speed_factor=args.speed_factor, logger=logger,
        mean_fpc=mean_fpc, stop_thresh=args.stop_thresh,
        return_attn=True, attn_path=attn_path,
    )
    logger.info(f"Acoustic model: {(time.time()-t0)*1000:.0f} ms")

    plot_mel(mel_pred, f"TTS v6: {text[:50]}",
             str(wpath.with_suffix(".png")), logger)

    t1 = time.time()
    audio = vocoder.synthesize(mel_pred)
    save_wav(audio, str(wpath), mel_cfg.sample_rate)
    dur = len(audio)/mel_cfg.sample_rate
    logger.info(f"Vocoder: {(time.time()-t1)*1000:.0f} ms  |  "
                f"Audio → {wpath}  ({dur:.2f}s)")

    if args.gt_compare:
        gm = _find_gt_text(text, args.output_dir, logger)
        if gm is not None:
            ga = vocoder.synthesize(gm)
            gw = wpath.with_name(wpath.stem + "_GT.wav")
            save_wav(ga, str(gw), mel_cfg.sample_rate)
            logger.info(f"GT comparison → {gw}")
            plot_comparison(mel_pred, gm,
                            str(wpath.with_name(wpath.stem + "_compare.png")),
                            logger)
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
    vocoder = Vocoder(
        mel_cfg, logger,
        backend=resolve_vocoder_backend(args),
        waveglow_sigma=args.waveglow_sigma,
    )
    results=[]; maes=[]; rtfs=[]; ts=[]

    for i, s in enumerate(test_s[:n_ev]):
        t0=time.time()
        try: mel_pred = synthesize_text(
                s["text"], model, vocab, device,
                mean_fpc=mean_fpc, stop_thresh=args.stop_thresh,
                return_attn=False,
            )
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
    logger.info("  EVALUATION — Myanmar SLR80 TTS Demo v6")
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
        description="Myanmar TTS Demo v6 — TransformerTTS + HiFi-GAN + scheduled sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--stage", choices=["prep","train","synth","eval","all"],
                   default="all")
    p.add_argument("--check_deps", action="store_true")
    p.add_argument("--data_dir",    default="./data/slr80")
    p.add_argument("--tsv_file",    default="line_index_female.tsv")
    p.add_argument("--output_dir",  default="./tts_v6")
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
    # v6: guided attention / stop tuning
    p.add_argument("--ga_weight", type=float, default=1.0,
                   help="Weight for guided attention loss (raise if attn off-diagonal)")
    p.add_argument("--ga_g", type=float, default=0.2,
                   help="Guided-attention band width g (smaller = stricter diagonal)")
    p.add_argument("--stop_loss_weight", type=float, default=0.1,
                   help="Weight for stop-token BCE during training")
    p.add_argument("--stop_thresh", type=float, default=0.65,
                   help="Inference stop threshold (v6 default 0.65; v5 used 0.5)")
    # v6: scheduled sampling
    p.add_argument("--ss_start_epoch", type=int, default=50,
                   help="Epoch to begin scheduled sampling (0=from start)")
    p.add_argument("--ss_max_prob", type=float, default=0.5,
                   help="Max per-frame prob of using predicted mel in PreNet")
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
    p.add_argument("--vocoder", choices=["hifigan", "waveglow", "griffin_lim"],
                   default="hifigan",
                   help="Mel→wav backend (default: hifigan)")
    p.add_argument("--waveglow_sigma", type=float, default=0.6,
                   help="WaveGlow infer noise scale (typical 0.6)")
    p.add_argument("--no_neural_vocoder", action="store_true",
                   help="Alias for --vocoder griffin_lim")
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
