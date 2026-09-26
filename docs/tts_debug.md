# Myanmar TTS Experiments Brief (SLR80 / AIEF)

Context for other AIs: custom Transformer TTS on OpenSLR80 Burmese (~2.5k utts).
Core acoustic recipe lives in `src/myanmar_tts_transformer*.py`. Notebooks:
`tts_transformer_slr80_v1.0.ipynb`, `tts_transformer_slr80_v1.1.ipynb`.
Outputs under `tts_output/`.

**Critical diagnostic:** teacher-forcing **dev loss ≈ 0.2–0.3 does NOT mean good synth**.
Judge with attention heatmaps (`*_attn.png`), mel plots, and audio — not loss alone.

---

## Version lineage (why each existed)

| Ver | Architecture | Main failure | Why next version |
|-----|--------------|--------------|------------------|
| **v1–v3** | NAR: Encoder → Gaussian upsampling → Decoder → mel | **Wind / white-noise audio** | Soft upsampling + L1 → conditional **mean** mel (featureless). Dev loss plateaued (~1.06) while audio stayed wind. |
| **v4** | **AR** TransformerTTS (PreNet + causal decoder + stop) | Train OK (dev ~0.25); infer → **~11 silent frames** for any text length | Exposure bias + unconstrained cross-attn: at infer, attn locks, stop fires early. Length ignored text. |
| **v5** | v4 + **Guided Attention (GA)** + length-aware `min_frames`/`max_frames` from mean frames/char | Baseline AR tutorial that can work on small data | GA pushes attn toward **monotonic diagonal**. Must retrain from scratch vs v4. File: `myanmar_tts_transformer.py`. |
| **NAR sibling** | FastSpeech2-style + DurationPredictor + LengthRegulator | With **proportional** (uniform) durations only, dig/quality weak (~constant duration) | Needs **real** durations (MFA/CTC), not proportional. File: `myanmar_tts_transformer_nar.py`. |
| **v6** | v5 acoustic core + extras | Same AR+GA core; quality depends on GA/SS training | Vocoder choice, scheduled sampling, softer stop, attn plots, CLI GA/SS knobs. File: `myanmar_tts_transformer_v6.py`. |

### v6 additions (on top of v5)
1. `--vocoder {hifigan,waveglow,griffin_lim}` (default HiFi-GAN; `--no_neural_vocoder` = Griffin-Lim)
2. Scheduled sampling (`--ss_start_epoch`, `--ss_max_prob`) to reduce exposure bias
3. Softer stop default (`--stop_thresh 0.65`) + length floor kept
4. Synth attention maps + `diag_score` (aim ≳ 0.3 with mass near cyan diagonal)
5. CLI: `--ga_weight`, `--ga_g`, `--stop_loss_weight`, etc.

Prep mels (`output_dir/mels/*.npy`) are reusable across v5/v6 if mel config unchanged.
**Changing GA/SS meaningfully → retrain from scratch** (new `output_dir`), don’t expect late `--resume` to un-collapse attn.

---

## How to read attention plots

- **Healthy:** thin bright band along dashed **ideal diagonal** (text 0→T, mel 0→Tm).
- **Collapse / wind:** vertical stripe (stuck on one token), blocky stacks, path stops mid-text, or random blobs at wrong corners.
- **Blank / silent mel:** often attn collapsed; vocoder cannot fix empty acoustic mel.
- Log line: `diag_score=…; >0.3 usually OK` — below that, treat as failed alignment.

---

## Recent v6 experiment folder notes (`tts_output/`)

| Exp | Intent | Attn / audio observation | Lesson |
|-----|--------|--------------------------|--------|
| **v6_exp_1.0** | Longer train, milder GA/SS (e.g. ga≈1, ss later) | Single vertical stripe (stuck ~one text pos); blank/wind | Low loss ≠ alignment |
| **v6_exp_1.1** | Stronger GA (`ga_weight=3`, `ga_g=0.15`), more SS | Slightly less pure sink, still scrambled / off-diagonal (`diag_score≈0.21`) | Better GA helps a bit; not enough |
| **v6_exp_1.2** | Fresh scout (~50 eps), stronger GA | Clearer path early, then **loop** mid-text; covers ~half sentence only | Structured but unfinished alignment |
| **v6_exp_1.3** | Another GA/SS scout; short + long synth | Short (`ရုပ်ရှင်…`): blocky vertical, skips tokens. Long (`bur_5362…`): stall ~text 10, path to ~30/65 | Short text still fails → not “sentence too long” only |

**Path pitfall:** folder rename `v6_exp_1_1` ↔ `v6_exp_1.1` breaks `train.json` mel paths (`FileNotFoundError`). Prefer underscore names; don’t rename after prep without rewriting JSON mel paths.

---

## Suggested train / eval protocol

1. Prep once → stable `--output_dir` (no dots in name).
2. Scout **50 epochs** with strong GA, e.g. `--ga_weight 5–8 --ga_g 0.08–0.1`, SS start ~20% of epochs.
3. Synth fixed probe texts; inspect `*_attn.png` + mel + wav.
4. **Continue to 200 only if** attn near diagonal end-to-end and mel not flat at −1 / not “mel_db max≈−48 dB”.
5. If still vertical/stall after 50: **new run**, higher GA — do not resume mid-collapse.

Example scout:
```bash
python src/myanmar_tts_transformer_v6.py --stage train \
  --output_dir ./tts_output/v6_exp_1_4 \
  --epochs 50 \
  --ga_weight 8.0 --ga_g 0.08 \
  --ss_start_epoch 10 --ss_max_prob 0.8
```

Synth check:
```bash
python src/myanmar_tts_transformer_v6.py --stage synth \
  --output_dir ./tts_output/v6_exp_1_4 \
  --gt_sample_id bur_5362_4875790905 \
  --synth_out ./tts_output/v6_exp_1_4/gt_output/probe.wav \
  --gt_compare
```

---

## Open questions for other AIs

1. Why does GA (even weight 3–5) still yield mid-text stall on ~2.5k Burmese chars with this TransformerTTS?
2. Better next step: much higher GA, location-sensitive attention, ForwardAttention, or force MFA durations + NAR?
3. Is 50-epoch scout too early to judge, or is collapsed shape already decisive?
4. Mel domain mismatch for HiFi-GAN/WaveGlow vs librosa dB mels — secondary after blank mel?

---

## Key files

- `src/myanmar_tts_transformer.py` — v5 AR + GA (tutorial baseline)
- `src/myanmar_tts_transformer_v6.py` — v6 (vocoders + SS + attn diagnostics)
- `src/myanmar_tts_transformer_nar.py` — NAR / proportional durations
- Dataset: OpenSLR80; typical prep stats ~mean frames/char ≈ 7.3