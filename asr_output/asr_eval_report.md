# ASR Evaluation Report — Myanmar SLR80

## Results

| Metric | Value |
|--------|-------|
| **WER** | 58.30% |
| **CER (Myanmar)** | 13.69% |
| Test utterances decoded | 200 |
| Total words | 1,746 |
| Total characters | 13,537 |
| Substitutions | 939 |
| Deletions | 457 |
| Insertions | 457 |
| Failed decodes | 0 |

## Model
- Base checkpoint: `facebook/wav2vec2-base`
- Tokeniser: Custom Myanmar character vocabulary (built from training set)
- Fine-tuned on: SLR80 Myanmar Female Speech Corpus (CC BY-SA 4.0)

## Metric notes
- **CER** is the primary metric for Myanmar. Space usage is inconsistent
  in Myanmar text, so WER is less reliable.
- CER = (Substitutions + Deletions + Insertions) / Total reference characters
- "|" in hypotheses represents the word-boundary token (maps back to space).
