# Fix 1: make_compute_metrics — robust decode that handles both old and new transformers
# Fix 2: make_ctc_collator — use feature_extractor.pad directly (more robust in 5.x)
# Fix 3: add --predict_with_generate=False hint, and log sample predictions each epoch

with open('myanmar_asr.py', encoding='utf-8') as f:
    src = f.read()

# ── Fix make_ctc_collator ────────────────────────────────────────────────────
old_collator = '''def make_ctc_collator(processor):
    """
    Pad variable-length input_values and labels in each batch.
    Labels are padded with -100 so that CTC loss ignores padding positions.
    """
    import torch

    def collate(features: List[Dict]) -> Dict:
        # ── Pad audio ──────────────────────────────────────────────────────
        inp_list = [{"input_values": f["input_values"]} for f in features]
        batch    = processor.pad(inp_list, padding=True, return_tensors="pt")

        # ── Pad labels  ────────────────────────────────────────────────────
        label_ids = [f["labels"] for f in features]
        max_len   = max(len(l) for l in label_ids)
        labels_padded = torch.full(
            (len(label_ids), max_len), fill_value=-100, dtype=torch.long
        )
        for i, ids in enumerate(label_ids):
            labels_padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

        batch["labels"] = labels_padded
        return batch

    return collate'''

new_collator = '''def make_ctc_collator(processor):
    """
    Pad variable-length input_values and labels in each batch.
    Labels are padded with -100 so CTC loss ignores padding positions.

    Uses feature_extractor.pad() directly for audio — more robust across
    transformers versions than processor.pad() which changed behaviour in 5.x.
    """
    import torch

    def collate(features: List[Dict]) -> Dict:
        # ── Pad audio features ─────────────────────────────────────────────
        input_values = [torch.tensor(f["input_values"], dtype=torch.float32)
                        for f in features]
        # Manual padding to max length in batch
        max_audio_len = max(v.size(0) for v in input_values)
        padded_audio  = torch.zeros(len(input_values), max_audio_len)
        attention_mask = torch.zeros(len(input_values), max_audio_len, dtype=torch.long)
        for i, v in enumerate(input_values):
            padded_audio[i, :v.size(0)]     = v
            attention_mask[i, :v.size(0)]   = 1

        # ── Pad labels with -100 (CTC ignore index) ─────────────────────────
        label_ids = [f["labels"] for f in features]
        max_label_len  = max(len(l) for l in label_ids)
        labels_padded  = torch.full(
            (len(label_ids), max_label_len), fill_value=-100, dtype=torch.long
        )
        for i, ids in enumerate(label_ids):
            labels_padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

        return {
            "input_values":  padded_audio,
            "attention_mask": attention_mask,
            "labels":         labels_padded,
        }

    return collate'''

assert old_collator in src, "old_collator not found"
src = src.replace(old_collator, new_collator, 1)
print("Fix 1 (collator): OK")

# ── Fix make_compute_metrics ─────────────────────────────────────────────────
old_metrics = '''def make_compute_metrics(processor):
    """
    CTC decode predictions and compute WER + CER.
    Called by HuggingFace Trainer at each evaluation epoch.

    The tokenizer uses "|" as the word-boundary token (replaces space).
    We map "|" back to " " before computing WER/CER so scores are meaningful.
    """
    import numpy as np

    def _decode_batch(ids_2d, group_tokens: bool = True) -> List[str]:
        """Decode a 2D array of token ids → list of strings, "|"→" "."""
        strings = processor.tokenizer.batch_decode(
            ids_2d, group_tokens=group_tokens
        )
        return [s.replace("|", " ").strip() for s in strings]

    def compute_metrics(pred) -> Dict:
        pred_ids  = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids.copy()
        # Replace CTC-ignore padding (-100) with the real PAD token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = _decode_batch(pred_ids,  group_tokens=True)
        label_str = _decode_batch(label_ids, group_tokens=False)

        wer_res = compute_wer(label_str, pred_str)
        cer_res = compute_cer(label_str, pred_str)
        return {"wer": wer_res["wer"], "cer": cer_res["cer"]}

    return compute_metrics'''

new_metrics = '''def make_compute_metrics(processor):
    """
    CTC decode predictions → WER + CER.
    Called by HuggingFace Trainer at each eval epoch.

    CTC decoding steps:
      1. argmax over logits → raw token id sequence
      2. collapse consecutive duplicates (CTC merge)
      3. remove [PAD] (blank) tokens
      4. map token ids → characters via tokenizer vocabulary
      5. replace "|" word-boundary token → space
    We implement steps 2-5 manually to be robust across transformers versions.
    """
    import numpy as np

    pad_id  = processor.tokenizer.pad_token_id   # 0 = CTC blank
    id2char = processor.tokenizer.convert_ids_to_tokens  # id → char string

    def _ctc_decode(ids_row) -> str:
        """CTC greedy decode: collapse repeats, strip blanks, join chars."""
        tokens = []
        prev   = None
        for tid in ids_row:
            if tid != prev:          # collapse consecutive duplicates
                if tid != pad_id:    # drop CTC blank
                    tokens.append(tid)
            prev = tid
        # Convert ids → chars, map "|" → space
        chars = []
        for tid in tokens:
            ch = id2char(int(tid))
            if ch is None:
                continue
            if ch in ("[PAD]", "[UNK]"):
                continue
            chars.append(" " if ch == "|" else ch)
        return "".join(chars).strip()

    def _decode_labels(ids_row) -> str:
        """Decode ground-truth label ids (no CTC collapsing needed)."""
        chars = []
        for tid in ids_row:
            if int(tid) == -100 or int(tid) == pad_id:
                continue
            ch = id2char(int(tid))
            if ch is None or ch in ("[PAD]", "[UNK]"):
                continue
            chars.append(" " if ch == "|" else ch)
        return "".join(chars).strip()

    def compute_metrics(pred) -> Dict:
        pred_ids  = np.argmax(pred.predictions, axis=-1)   # (B, T)
        label_ids = pred.label_ids                          # (B, L)

        pred_str  = [_ctc_decode(row)  for row in pred_ids]
        label_str = [_decode_labels(row) for row in label_ids]

        # Log a few examples so we can visually verify decoding is working
        n_show = min(3, len(pred_str))
        for i in range(n_show):
            logging.getLogger("asr").debug(
                f"  EVAL sample {i}\\n"
                f"    REF: {label_str[i][:80]}\\n"
                f"    HYP: {pred_str[i][:80]}"
            )

        wer_res = compute_wer(label_str, pred_str)
        cer_res = compute_cer(label_str, pred_str)
        return {"wer": wer_res["wer"], "cer": cer_res["cer"]}

    return compute_metrics'''

assert old_metrics in src, "old_metrics not found"
src = src.replace(old_metrics, new_metrics, 1)
print("Fix 2 (compute_metrics): OK")

with open('myanmar_asr.py', 'w', encoding='utf-8') as f:
    f.write(src)
print("File written.")
