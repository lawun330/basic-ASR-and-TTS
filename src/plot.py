#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot.py — TTS / NAR training diagnostics (matplotlib)

Notebooks already cover: listen (Audio), vocoder wavs, GT vs pred mel
side-by-side (`*_compare.png`).  This script adds what those cells skip:

  1. train / dev loss (+ lr) curves from train_log.json or a text log
  2. duration QC: real vs proportional, sum==mel_frames, spread stats
  3. mel error |pred−gt| for an in-set sample (or two explicit .npy paths)

Examples
--------
  # loss curves (JSON written by myanmar_tts_transformer_nar.py / v6)
  python src/plot.py loss --output_dir ./tts_output/nar_version

  # if train_log.json is empty after --resume, parse a saved train transcript:
  python src/plot.py loss --from_log ./tts_output/nar_version/train_stdout.txt \\
      --save ./tts_output/nar_version/plots/loss.png

  # duration quality on train.json (after real-duration swap)
  python src/plot.py durations --output_dir ./tts_output/nar_version --split train

  # in-set mel error (GT from prep mels/; pred from --save_mel synth)
  python src/plot.py mel_error --output_dir ./tts_output/nar_version \\
      --sample_id bur_5362_4875790905

  # custom panel titles (optional 4th = figure title)
  python src/plot.py mel_error --output_dir ./tts_output/nar_version \\
      --sample_id bur_5362_4875790905 \\
      --titles "GT mel" "Predicted mel" "|pred − gt|" "Mel error"

  # or pass explicit paths
  python src/plot.py mel_error \\
      --gt_mel  ./tts_output/nar_version/mels/bur_….npy \\
      --pred_mel ./tts_output/nar_version/gt_output/bur_…_synth.npy
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


# ─── shared helpers ───────────────────────────────────────────────────────────

def _ensure_parent(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _savefig(path, fig):
    import matplotlib.pyplot as plt
    path = _ensure_parent(path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {path}")


def _load_history_json(output_dir: Path):
    p = output_dir / "train_log.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if not data:
        return None
    return data


_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s*/\s*(\d+)\s*\|\s*"
    r"train=([0-9.]+)\s+dev=([0-9.]+)"
    r"(?:\s*\|\s*lr=([0-9.eE+-]+))?"
    r"(?:\s+ss=([0-9.]+))?"
)


def _parse_history_log(log_path: Path):
    """Parse 'Epoch N/M | train=…  dev=… | lr=…' lines from a text log."""
    text = Path(log_path).read_text(errors="replace")
    rows = []
    for m in _EPOCH_RE.finditer(text):
        rows.append({
            "epoch": int(m.group(1)),
            "train": float(m.group(3)),
            "dev": float(m.group(4)),
            "lr": float(m.group(5)) if m.group(5) else None,
            "ss": float(m.group(6)) if m.group(6) else None,
        })
    # keep last occurrence per epoch (resume / re-runs)
    by_ep = {}
    for r in rows:
        by_ep[r["epoch"]] = r
    return [by_ep[k] for k in sorted(by_ep)]


def _proportional_durations(n_tokens: int, n_frames: int):
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


# ─── 1) loss curves ───────────────────────────────────────────────────────────

def cmd_loss(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = None
    if args.from_log:
        hist = _parse_history_log(Path(args.from_log))
        if not hist:
            print(f"no Epoch lines found in {args.from_log}", file=sys.stderr)
            sys.exit(1)
        print(f"parsed {len(hist)} epochs from {args.from_log}")
    else:
        out = Path(args.output_dir)
        hist = _load_history_json(out)
        if hist is None:
            print(
                f"empty/missing {out/'train_log.json'}.\n"
                "  Tip: after --resume the NAR/v6 script overwrites history with "
                "only the new run. Re-train briefly, or paste stdout into a file "
                "and use --from_log path/to/stdout.txt",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"loaded {len(hist)} epochs from {out/'train_log.json'}")

    epochs = [h["epoch"] for h in hist]
    train = [h["train"] for h in hist]
    dev = [h["dev"] for h in hist]
    lrs = [h.get("lr") for h in hist]
    has_lr = any(lr is not None for lr in lrs)

    if has_lr:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        axes = np.array([axes])

    ax = axes[0]
    ax.plot(epochs, train, label="train", color="C0")
    ax.plot(epochs, dev, label="dev", color="C1")
    best_i = int(np.argmin(dev))
    ax.scatter([epochs[best_i]], [dev[best_i]], color="C1", zorder=5,
               label=f"best dev={dev[best_i]:.4f} @ ep{epochs[best_i]}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if has_lr:
        axes[1].plot(epochs, lrs, color="C2", label="lr")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("learning rate")
        axes[1].set_title("Learning rate")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")

    fig.suptitle(str("Training Results"), fontsize=10)
    fig.tight_layout()

    save = args.save
    if save is None and args.output_dir:
        save = str(Path(args.output_dir) / "plots" / "loss_curves.png")
    if save is None:
        save = "loss_curves.png"
    _savefig(save, fig)


# ─── 2) duration QC ───────────────────────────────────────────────────────────

def cmd_durations(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.output_dir)
    split = args.split
    path = out / f"{split}.json"
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        sys.exit(1)
    samples = json.loads(path.read_text())
    if not samples or "durations" not in samples[0]:
        print(f"{path} has no 'durations' field", file=sys.stderr)
        sys.exit(1)

    # optional proportional backup for comparison
    prop_path = out / f"{split}_proportional_backup.json"
    prop_by_id = {}
    if prop_path.exists():
        for s in json.loads(prop_path.read_text()):
            if "durations" in s:
                prop_by_id[s["id"]] = s["durations"]

    n = min(args.n, len(samples))
    stds, maxs, match_ok, near_uniform = [], [], 0, 0
    scatter_real, scatter_prop = [], []
    example = None

    for s in samples[:n]:
        d = s["durations"]
        mel = np.load(s["mel"])
        n_frames = int(mel.shape[0])
        if sum(d) == n_frames and len(d) == len(s["text_ids"]):
            match_ok += 1
        mid = d[1:-1] if len(d) > 2 else d  # drop BOS/EOS zeros
        if not mid:
            continue
        stds.append(float(np.std(mid)))
        maxs.append(int(max(mid)))
        # "near uniform" heuristic: ≤3 distinct positive values
        pos = [x for x in mid if x > 0]
        if len(set(pos)) <= 3:
            near_uniform += 1
        if example is None:
            example = s
        if s["id"] in prop_by_id:
            pr = prop_by_id[s["id"]]
            L = min(len(d), len(pr))
            scatter_real.extend(d[1:L - 1] if L > 2 else d[:L])
            scatter_prop.extend(pr[1:L - 1] if L > 2 else pr[:L])

    print(f"split={split}  inspected={n}/{len(samples)}")
    print(f"  sum(durations)==mel_frames : {match_ok}/{n}")
    print(f"  mid-duration std  mean={np.mean(stds):.2f}  "
          f"median={np.median(stds):.2f}")
    print(f"  near-uniform (≤3 distinct >0 values): {near_uniform}/{n}  "
          f"(high ⇒ looks proportional)")
    if example is not None:
        mid = example["durations"][1:-1]
        print(f"  example id={example['id']}")
        print(f"    top counts: {Counter(mid).most_common(8)}")
        print(f"    std={np.std(mid):.2f}  min={min(mid)}  max={max(mid)}")

    fig, axes = plt.subplots(1, 2 if scatter_real else 1, figsize=(11, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axes[0].hist(stds, bins=40, color="C0", edgecolor="white")
    axes[0].set_xlabel("std of mid-token durations")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"{split}: duration spread (higher ⇒ less uniform)")
    axes[0].grid(True, alpha=0.3)

    if scatter_real:
        # subsample for plotting
        idx = np.random.RandomState(0).choice(
            len(scatter_real), size=min(5000, len(scatter_real)), replace=False)
        xr = np.asarray(scatter_prop)[idx]
        yr = np.asarray(scatter_real)[idx]
        axes[1].scatter(xr, yr, s=4, alpha=0.25, c="C1")
        lim = max(xr.max(), yr.max()) + 1
        axes[1].plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
        axes[1].set_xlabel("proportional duration")
        axes[1].set_ylabel("train.json duration (real if swapped)")
        axes[1].set_title("real vs proportional (should leave y=x)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        print(f"  (no {prop_path.name} — skip real-vs-prop scatter)")

    save = args.save or str(out / "plots" / f"durations_{split}.png")
    _savefig(save, fig)


# ─── 3) mel absolute error (in-set sample or two .npy paths) ───────────────────

def _resolve_mel_from_split(output_dir: Path, sample_id: str):
    """Return (mel_path, text) from train/dev/test JSON, or (None, None)."""
    for sp in ("train", "dev", "test"):
        jp = output_dir / f"{sp}.json"
        if not jp.is_file():
            continue
        with open(jp, encoding="utf-8") as f:
            items = json.load(f)
        for s in items:
            if s.get("id") == sample_id:
                mp = Path(s["mel"])
                if not mp.is_file():
                    mp = output_dir / "mels" / f"{sample_id}.npy"
                return mp, s.get("text", "")
    return None, None


def _find_pred_mel(output_dir: Path, sample_id: str, explicit: str | None):
    """Locate predicted mel .npy from --pred_mel or common synth save locations."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            print(f"pred mel not found: {p}", file=sys.stderr)
            sys.exit(1)
        return p
    candidates = [
        output_dir / "gt_output" / f"{sample_id}_synth.npy",
        output_dir / "gt_output" / f"{sample_id}.npy",
        output_dir / "gt_output" / f"gt_{sample_id}_synthesized.npy",
        output_dir / "gt_output" / f"gt_{sample_id}_synth.npy",
        output_dir / f"{sample_id}_synth.npy",
        output_dir / f"{sample_id}.npy",
    ]
    for c in candidates:
        if c.is_file():
            return c
    print(
        "pred mel not found. Run synth with --gt_sample_id --gt_compare "
        "--save_mel (writes {output_dir}/gt_output/{id}_synth.npy), "
        "or pass --pred_mel explicitly.\n"
        "looked for:\n  " + "\n  ".join(str(c) for c in candidates),
        file=sys.stderr,
    )
    sys.exit(1)


def cmd_mel_error(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_id = args.sample_id
    out = Path(args.output_dir) if args.output_dir else None

    if sample_id and out is None:
        print("mel_error --sample_id requires --output_dir", file=sys.stderr)
        sys.exit(1)

    if sample_id:
        gt_path, gt_text = _resolve_mel_from_split(out, sample_id)
        if gt_path is None or not gt_path.is_file():
            # fallback: prep layout mels/{id}.npy
            gt_path = out / "mels" / f"{sample_id}.npy"
            gt_text = sample_id
        if not gt_path.is_file():
            print(f"GT mel not found for {sample_id!r} under {out}",
                  file=sys.stderr)
            sys.exit(1)
        pred_path = _find_pred_mel(out, sample_id, args.pred_mel)
        title_suffix = f"  [{sample_id}]"
        if gt_text and gt_text != sample_id:
            title_suffix += f"\n{gt_text[:80]}"
        default_save = str(out / "plots" / f"mel_err_{sample_id}.png")
    else:
        if not args.gt_mel or not args.pred_mel:
            print(
                "mel_error: provide --output_dir --sample_id, "
                "or both --gt_mel and --pred_mel",
                file=sys.stderr,
            )
            sys.exit(1)
        gt_path = Path(args.gt_mel)
        pred_path = Path(args.pred_mel)
        if not gt_path.is_file():
            print(f"GT mel not found: {gt_path}", file=sys.stderr); sys.exit(1)
        if not pred_path.is_file():
            print(f"pred mel not found: {pred_path}", file=sys.stderr); sys.exit(1)
        title_suffix = ""
        default_save = "mel_error.png"

    gt = np.load(gt_path).astype(np.float32)
    pred = np.load(pred_path).astype(np.float32)
    if gt.ndim != 2 or pred.ndim != 2:
        print("expected (T, n_mels) arrays", file=sys.stderr)
        sys.exit(1)
    T = min(gt.shape[0], pred.shape[0])
    M = min(gt.shape[1], pred.shape[1])
    gt, pred = gt[:T, :M], pred[:T, :M]
    err = np.abs(pred - gt)
    mae = float(err.mean())

    # defaults; --titles overrides panel titles in order, optional 4th = figure title
    t0 = f"GT mel:  {gt_path.name}"
    t1 = f"Pred mel:  {pred_path.name}"
    t2 = (f"Absolute Mel Difference: MAE={mae:.4f}   "
          f"(T={T}, aligned to min length)")
    t_sup = f"In-set mel error{title_suffix}"
    custom = list(args.titles) if args.titles else []
    if len(custom) >= 1:
        t0 = custom[0]
    if len(custom) >= 2:
        t1 = custom[1]
    if len(custom) >= 3:
        t2 = custom[2]
    if len(custom) >= 4:
        t_sup = custom[3]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    kw = dict(aspect="auto", origin="lower", interpolation="none")
    im0 = axes[0].imshow(gt.T, cmap="viridis", vmin=-1, vmax=1, **kw)
    axes[0].set_ylabel("mel bin")
    axes[0].set_title(t0)
    plt.colorbar(im0, ax=axes[0], fraction=0.02)
    im1 = axes[1].imshow(pred.T, cmap="viridis", vmin=-1, vmax=1, **kw)
    axes[1].set_ylabel("mel bin")
    axes[1].set_title(t1)
    plt.colorbar(im1, ax=axes[1], fraction=0.02)
    im2 = axes[2].imshow(err.T, cmap="magma", vmin=0, vmax=max(0.5, float(err.max())), **kw)
    axes[2].set_ylabel("mel bin"); axes[2].set_xlabel("frame")
    axes[2].set_title(t2)
    plt.colorbar(im2, ax=axes[2], fraction=0.02)
    fig.suptitle(t_sup)

    save = args.save or default_save
    _savefig(save, fig)
    print(f"GT   = {gt_path}")
    print(f"pred = {pred_path}")
    print(f"MAE  = {mae:.4f}  (T={T})")
    print(f"plot → {save}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="TTS NAR / AR diagnostic plots (loss, durations, mel error)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("loss", help="train/dev loss curves")
    pl.add_argument("--output_dir", type=str, default=None,
                    help="dir containing train_log.json")
    pl.add_argument("--from_log", type=str, default=None,
                    help="parse Epoch lines from a text log / notebook export")
    pl.add_argument("--save", type=str, default=None)
    pl.set_defaults(func=cmd_loss)

    pd = sub.add_parser("durations", help="duration QC histograms / vs proportional")
    pd.add_argument("--output_dir", type=str, required=True)
    pd.add_argument("--split", choices=["train", "dev", "test"], default="train")
    pd.add_argument("--n", type=int, default=500, help="max samples to scan")
    pd.add_argument("--save", type=str, default=None)
    pd.set_defaults(func=cmd_durations)

    pe = sub.add_parser(
        "mel_error",
        help="|pred−gt| for an in-set --sample_id (or two .npy paths)")
    pe.add_argument("--output_dir", type=str, default=None,
                    help="run dir with mels/ + train.json (required with --sample_id)")
    pe.add_argument("--sample_id", type=str, default=None,
                    help="in-set utterance id; GT = mels/{id}.npy")
    pe.add_argument("--gt_mel", type=str, default=None,
                    help="explicit GT .npy (alt to --sample_id)")
    pe.add_argument("--pred_mel", type=str, default=None,
                    help="explicit pred .npy; else look under gt_output/")
    pe.add_argument(
        "--titles", nargs="+", default=None, metavar="TITLE",
        help="custom titles: panel0 panel1 panel2 [figure]. "
             "omit to keep defaults. example: --titles 'GT' 'Pred' '|diff|' 'MAE'")
    pe.add_argument("--save", type=str, default=None)
    pe.set_defaults(func=cmd_mel_error)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "loss" and not args.output_dir and not args.from_log:
        parser.error("loss: provide --output_dir and/or --from_log")
    args.func(args)


if __name__ == "__main__":
    main()
