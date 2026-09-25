#!/usr/bin/env python3
"""
get_real_durations.py
======================
Generates REAL per-character durations for myanmar_tts_transformer_nar.py,
replacing the `proportional_durations()` baseline, via CTC forced alignment.

No pronunciation dictionary needed (unlike MFA) — this uses a CTC acoustic
model directly, so the "phone" inventory is just your existing character
vocabulary (MyanmarCharVocab / normalize_text()).

Model: facebook/mms-1b-all, Burmese adapter ("mya"). Character-level output,
which matches your normalize_text() tokenization closely enough that no
dictionary or G2P step is required.

WHAT THIS SCRIPT DOES
----------------------
1. Loads train.json / dev.json exactly as written by your existing
   `--stage prep` (paths, text, text_ids, mel .npy paths).
2. For each sample:
     a. Loads audio at 16kHz (MMS model's expected rate — independent of
        your TTS mel_config sample_rate).
     b. Runs the CTC model -> per-frame log-probs.
     c. Force-aligns those log-probs against the normalized transcript
        characters using torchaudio.functional.forced_align.
     d. Converts the resulting per-character frame spans (in seconds) into
        mel-frame counts using YOUR mel_config.json (sample_rate/hop_length),
        not the CTC model's frame rate.
     e. Distributes rounding remainder so sum(durations) == exact mel length
        (this is a hard requirement of the NAR dataset loader).
     f. Pads BOS/EOS duration = 0, matching len(text_ids).
3. Writes train_real_durations.json / dev_real_durations.json next to the
   originals. Review a few, then swap them in for train.json / dev.json
   (back up the originals first) and re-run --stage train. No need to
   re-run --stage prep — mel .npy files are untouched.

INSTALL
-------
pip install transformers torchaudio soundfile
# WAV I/O uses soundfile (not torchaudio.load / TorchCodec).

USAGE
-----
python get_real_durations.py \
    --output_dir ./tts_output/nar_run_1 \
    --lang mya

Notes:
- facebook/mms-1b-all is a ~1B-param multilingual model. GPU strongly
  recommended; CPU will work but slowly for 2.5k utterances.
- If a handful of utterances fail alignment (mismatched punctuation, empty
  audio, etc.) the script logs and SKIPS them, falling back to proportional
  durations for just that sample rather than crashing the whole run — check
  the printed skip count and inspect those ids manually if it's more than a
  few percent of the corpus.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Text normalization — MUST match myanmar_tts_transformer_nar.normalize_text()
# exactly, since durations are indexed against text_ids produced by that
# function. Duplicated here (not imported) so this script has no dependency
# on your training code.
# ---------------------------------------------------------------------------
import unicodedata

_MYA_LO, _MYA_HI = 0x1000, 0x109F


def is_myanmar_char(c):
    return _MYA_LO <= ord(c) <= _MYA_HI


def normalize_text(text):
    text = unicodedata.normalize("NFC", text.strip())
    return "".join(c for c in text if is_myanmar_char(c) or c.isascii() or c == " ")


def proportional_durations(n_tokens, n_frames):
    """Same fallback as the NAR script, used only when alignment fails."""
    if n_tokens <= 0:
        return []
    if n_frames <= 0:
        return [1] * n_tokens
    durs = [0] * n_tokens
    for i in range(n_frames):
        durs[min(n_tokens - 1, (i * n_tokens) // n_frames)] += 1
    if sum(durs) == 0:
        durs[0] = 1
    return durs


def largest_remainder_round(float_counts, target_sum):
    """
    Round a list of non-negative floats to ints so they sum EXACTLY to
    target_sum, using largest-remainder apportionment. This keeps the
    rounding error from silently drifting the total off the true mel length
    (the NAR dataset loader asserts sum(durations) == n_frames).
    """
    floor_vals = [int(np.floor(v)) for v in float_counts]
    remainder_budget = target_sum - sum(floor_vals)
    if remainder_budget <= 0:
        # Extremely rare (rounding pushed us over) — trim from the largest.
        vals = floor_vals[:]
        idx_sorted = sorted(range(len(vals)), key=lambda i: -vals[i])
        i = 0
        while sum(vals) > target_sum and i < len(idx_sorted):
            if vals[idx_sorted[i]] > 0:
                vals[idx_sorted[i]] -= 1
            i = (i + 1) % max(len(idx_sorted), 1)
        return vals
    fracs = [v - int(np.floor(v)) for v in float_counts]
    order = sorted(range(len(fracs)), key=lambda i: -fracs[i])
    vals = floor_vals[:]
    for i in order[:remainder_budget]:
        vals[i] += 1
    return vals


def load_ctc_model(lang, device):
    from transformers import Wav2Vec2ForCTC, AutoProcessor

    model_id = "facebook/mms-1b-all"
    print(f"Loading {model_id} (adapter='{lang}') — first run downloads weights...")
    processor = AutoProcessor.from_pretrained(model_id, target_lang=lang)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_id, target_lang=lang, ignore_mismatched_sizes=True
    )
    model.load_adapter(lang)
    # keep tokenizer vocab in sync with the language CTC head
    if hasattr(processor.tokenizer, "set_target_lang"):
        processor.tokenizer.set_target_lang(lang)
    model.to(device).eval()
    ctc_dim = int(model.lm_head.out_features)
    print(
        f"  CTC dim={ctc_dim}  tokenizer_vocab={len(processor.tokenizer.get_vocab())}  "
        f"pad/blank={processor.tokenizer.pad_token_id}"
    )
    return model, processor


def build_char_targets(processor, chars, ctc_dim):
    """
    Map our normalized character sequence to the CTC model's vocabulary ids.
    MMS tokenizers use "|" as the word-delimiter for space and are otherwise
    character-level, so this is close to 1:1. Characters not in the model's
    vocab (rare — stray symbols) are dropped from the *alignment target only*;
    we still allocate them a proportional share of whatever their neighbours
    get, at the very end, so text_ids/durations stay the same length.

    Important: facebook/mms-1b-all Burmese adapter has CTC dim 140 (ids 0..139)
    but tokenizer still lists "|" with id 140.  torchaudio.forced_align requires
    every target id < CTC dim, so we must skip out-of-range ids (spaces mapped
    to "|").  Skipped positions get duration 0 and borrow via largest-remainder.
    """
    vocab = processor.tokenizer.get_vocab()

    ids, kept_positions = [], []
    n_oor = 0
    for i, c in enumerate(chars):
        tok = "|" if c == " " else c
        if tok not in vocab:
            continue
        tid = int(vocab[tok])
        if tid < 0 or tid >= ctc_dim:
            n_oor += 1
            continue
        ids.append(tid)
        kept_positions.append(i)
    return ids, kept_positions, n_oor


def align_one(model, processor, device, wav_path, chars, mel_sr, mel_hop):
    # soundfile avoids torchaudio's TorchCodec backend (often broken .so /
    # ffmpeg mismatch in conda). torchaudio still used for resample + align.
    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)  # (T, C)
    audio = torch.from_numpy(wav.T.copy())  # (C, T)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    target_sr = processor.feature_extractor.sampling_rate  # 16000 for MMS
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    inputs = processor(audio.squeeze(0).numpy(), sampling_rate=target_sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits  # (1, T_ctc, V)
    log_probs = torch.log_softmax(logits, dim=-1).cpu()
    ctc_dim = int(log_probs.shape[-1])

    target_ids, kept_positions, _n_oor = build_char_targets(processor, chars, ctc_dim)
    if len(target_ids) == 0:
        return None

    blank_id = int(processor.tokenizer.pad_token_id)
    if blank_id is None or blank_id >= ctc_dim:
        blank_id = int(getattr(model.config, "pad_token_id", 0) or 0)
    targets = torch.tensor(target_ids, dtype=torch.long).unsqueeze(0)

    # torchaudio>=2.1 API
    aligned_tokens, scores = torchaudio.functional.forced_align(
        log_probs, targets, blank=blank_id
    )
    # aligned_tokens: (1, T)
    aligned_1d = aligned_tokens[0]

    # CTC frame duration in seconds (wav2vec2-family: 320x downsample @16kHz)
    ctc_frame_sec = audio.shape[-1] / target_sr / log_probs.shape[1]

    # Recover per-target-token durations via run-length over the aligned path.
    # Prefer torchaudio's merge_tokens/TokenSpan helper when available.
    try:
        from torchaudio.functional import merge_tokens
        spans = merge_tokens(aligned_1d, scores[0], blank=blank_id)
        # spans: list of TokenSpan(token, start, end, score) in target order,
        # already skipping blanks — one span per target_ids entry.
        assert len(spans) == len(target_ids), \
            f"span/target length mismatch ({len(spans)} vs {len(target_ids)})"
        durations_sec = [(s.end - s.start) * ctc_frame_sec for s in spans]
    except Exception:
        # Fallback for older torchaudio / API quirks without merge_tokens.
        durations_sec = _manual_run_length(
            aligned_1d.tolist(), blank_id, len(target_ids), ctc_frame_sec
        )
        if durations_sec is None:
            return None

    mel_fps = mel_sr / mel_hop
    kept_mel_frames = [max(0, d * mel_fps) for d in durations_sec]

    return kept_positions, kept_mel_frames


def _manual_run_length(aligned_tokens, blank_id, n_targets, ctc_frame_sec):
    """Fallback if torchaudio.functional.merge_tokens isn't available."""
    spans = []
    prev = None
    run_start = 0
    for i, tok in enumerate(aligned_tokens + [None]):
        if tok != prev:
            if prev is not None and prev != blank_id:
                spans.append((run_start, i))
            run_start = i
            prev = tok
    if len(spans) != n_targets:
        # Can't reliably recover per-token spans — signal failure.
        return None
    return [(e - s) * ctc_frame_sec for s, e in spans]


def process_split(samples, model, processor, device, mel_sr, mel_hop, char_of):
    out, n_ok, n_fallback = [], 0, 0
    for s in samples:
        chars = list(normalize_text(s["text"]))
        mel = np.load(s["mel"])
        n_frames = int(mel.shape[0])
        n_tokens = len(s["text_ids"])  # includes BOS/EOS

        result = None
        try:
            result = align_one(model, processor, device, s["wav"], chars, mel_sr, mel_hop)
        except Exception as e:
            print(f"  [align error] {s['id']}: {e}")

        if result is None:
            durations = proportional_durations(n_tokens, n_frames)
            n_fallback += 1
        else:
            kept_positions, kept_mel_frames = result
            # Scatter aligned char durations back into full char-length array,
            # giving any dropped characters 0 (they'll borrow from rounding).
            full = [0.0] * len(chars)
            for pos, val in zip(kept_positions, kept_mel_frames):
                full[pos] = val
            total_float = sum(full) if sum(full) > 0 else 1.0
            scale = n_frames / total_float
            full = [v * scale for v in full]
            char_durs = largest_remainder_round(full, n_frames)
            # BOS + chars + EOS, matching text_ids layout (BOS/EOS get 0)
            durations = [0] + char_durs + [0]
            if len(durations) != n_tokens:
                durations = proportional_durations(n_tokens, n_frames)
                n_fallback += 1
            else:
                n_ok += 1

        s2 = dict(s)
        s2["durations"] = durations
        out.append(s2)

    print(f"  aligned OK: {n_ok}   fell back to proportional: {n_fallback}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True, help="Same output_dir used in --stage prep")
    ap.add_argument("--lang", default="mya", help="MMS language code (Burmese = mya)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    mel_cfg = json.loads((out / "mel_config.json").read_text())
    mel_sr, mel_hop = mel_cfg["sample_rate"], mel_cfg["hop_length"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_ctc_model(args.lang, device)

    for split in ["train", "dev"]:
        src = out / f"{split}.json"
        if not src.exists():
            print(f"skip: {src} not found")
            continue
        samples = json.loads(src.read_text())
        print(f"Aligning {split}.json ({len(samples)} utterances)...")
        aligned = process_split(samples, model, processor, device, mel_sr, mel_hop, None)
        dst = out / f"{split}_real_durations.json"
        dst.write_text(json.dumps(aligned, ensure_ascii=False))
        print(f"  wrote {dst}")

    print("\nDone. Inspect a few *_real_durations.json entries, then:")
    print("  cp train.json train_proportional_backup.json")
    print("  cp dev.json dev_proportional_backup.json")
    print("  cp train_real_durations.json train.json")
    print("  cp dev_real_durations.json dev.json")
    print("  python src/myanmar_tts_transformer_nar.py --stage train --output_dir ...")


if __name__ == "__main__":
    main()