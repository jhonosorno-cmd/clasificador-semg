"""Leave-one-subject-out (LOSO) runner for NinaPro DB3 (7 classes).

Per (model, held-out subject): pretrain once on the other subjects, then evaluate
three adaptation conditions on the held-out subject's rep 6: ``zero`` (no
fine-tuning), ``ft1`` (fine-tune on rep 1) and ``ft2`` (fine-tune on reps 1-2).
Normalization statistics only use repetitions allowed by the protocol.

Usage:
    python experimento_loso/run_loso.py --models v10 --subjects 1 --conditions zero ft1
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # config paths (data/raw, checkpoints) are relative to the repo root

from experimento_loso.loso_core import ALL_SUBJECTS, CONDITIONS, FoldPlan, fold_plan  # noqa: E402
from experimento_loso.loso_dataset import LOSODataset, LOSOExtendedDataset  # noqa: E402
from src.config import BATCH_SIZE, SEED  # noqa: E402
from src.train_loro_cv import (  # noqa: E402
    build_model,
    calculate_class_weights,
    evaluate,
    train_epoch,
)

CKPT_DIR = HERE / "checkpoints"
RES_DIR = HERE / "resultados"
PROBS_DIR = RES_DIR / "probs"
LR = 5e-4
FT_BACKBONE_LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 7


def set_seed(seed: int, device: torch.device) -> None:
    """Seeds python, numpy and torch before each run.

    CUDA is seeded only when training runs on it: touching an unusable GPU
    (see ``resolve_device``) raises even if ``torch.cuda.is_available()`` is True.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


class CappedLoader:
    """Wraps a DataLoader limiting batches per iteration (smoke tests only)."""

    def __init__(self, loader: DataLoader, max_batches: int | None) -> None:
        self.loader = loader
        self.dataset = loader.dataset
        self.max_batches = max_batches

    def __len__(self) -> int:
        n = len(self.loader)
        return n if self.max_batches is None else max(1, min(n, self.max_batches))

    def __iter__(self) -> Iterator:
        if self.max_batches is None:
            return iter(self.loader)
        return itertools.islice(self.loader, self.max_batches)


def make_loader(ds: LOSODataset, shuffle: bool, max_batches: int | None) -> CappedLoader:
    """Builds a loader (drop_last only for training sets larger than one batch)."""
    drop_last = shuffle and len(ds) > BATCH_SIZE
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                        num_workers=0, drop_last=drop_last)
    return CappedLoader(loader, max_batches)


def make_dataset(
    model_name: str, subjects: list[int], reps: tuple[int, ...],
    stats_reps: dict[int, tuple[int, ...]], mode: str, augment: bool, target: int,
) -> LOSODataset:
    """Creates the LOSO dataset appropriate for the model (v17 uses 132D features)."""
    cls = LOSOExtendedDataset if model_name == "v17" else LOSODataset
    return cls(subjects=subjects, reps_to_use=list(reps), stats_reps=stats_reps,
               mode=mode, augment=augment, target_subject=target)


def class_dist(ds: LOSODataset) -> list[int]:
    """Per-class window counts."""
    return np.bincount(ds.labels, minlength=NUM_CLASSES).tolist()


@torch.no_grad()
def predict_probs(
    model: nn.Module, loader: CappedLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities and labels over a loader, in dataset order."""
    model.eval()
    probs, labels = [], []
    for signals, features, y in loader:
        logits = model(signals.to(device), features.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.numpy())
    return (np.concatenate(probs).astype(np.float32),
            np.concatenate(labels).astype(np.int64))


def fit(
    model: nn.Module, train_loader: CappedLoader, val_loader: CappedLoader,
    optimizer: torch.optim.Optimizer, device: torch.device, use_augment: bool,
    max_epochs: int, patience: int, tag: str,
) -> dict:
    """Trains with early stopping on val accuracy and restores the best weights.

    Returns:
        Dict with ``best_val``, ``best_epoch``, ``epochs_run`` and ``state_dict``.
    """
    weights = calculate_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)
    best_val, best_epoch, bad = 0.0, -1, 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        _, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_augment)
        _, val_acc, val_bal, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        epochs_run = epoch
        improved = val_acc > best_val
        mark = " *" if improved else ""
        print(f"  [{tag}] ep {epoch:3d}/{max_epochs} train {train_acc*100:5.2f}% "
              f"val {val_acc*100:5.2f}% bal {val_bal*100:5.2f}% "
              f"lr {optimizer.param_groups[0]['lr']:.1e}{mark}", flush=True)
        if improved:
            best_val, best_epoch, bad = val_acc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                print(f"  [{tag}] early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return {"best_val": float(best_val), "best_epoch": best_epoch,
            "epochs_run": epochs_run, "state_dict": best_state}


def pretrain(
    model_name: str, subject: int, plan: FoldPlan, args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict, Path]:
    """Trains (or loads) the condition-independent pretrained model for a fold."""
    ckpt_path = CKPT_DIR / f"{model_name}_S{subject}_pretrain_seed{args.seed}.pth"
    if ckpt_path.exists() and not args.force:
        model = build_model(model_name).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  [pretrain] loaded {ckpt_path.name}")
        return model, ckpt["meta"], ckpt_path

    set_seed(args.seed, device)
    model = build_model(model_name).to(device)
    stats = {s: plan.train_reps for s in plan.train_subjects}
    train_ds = make_dataset(model_name, list(plan.train_subjects), plan.train_reps, stats,
                            "train", model_name == "v15", subject)
    val_ds = make_dataset(model_name, list(plan.train_subjects), plan.val_reps, stats,
                          "train", False, subject)
    train_loader = make_loader(train_ds, True, args.max_batches)
    val_loader = make_loader(val_ds, False, args.max_batches)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    res = fit(model, train_loader, val_loader, optimizer, device, model_name == "v15",
              args.epochs, args.patience, "pretrain")
    meta = {
        "best_val": res["best_val"], "best_epoch": res["best_epoch"],
        "epochs_run": res["epochs_run"],
        "train_windows": len(train_ds), "val_windows": len(val_ds),
        "train_class_distribution": class_dist(train_ds),
    }
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": res["state_dict"], "meta": meta,
                "model_name": model_name, "held_out_subject": subject, "seed": args.seed},
               ckpt_path)
    return model, meta, ckpt_path


def finetune(
    model: nn.Module, model_name: str, subject: int, plan: FoldPlan,
    args: argparse.Namespace, device: torch.device,
) -> tuple[dict, dict, Path]:
    """Fine-tunes ``model`` in place on the held-out subject's calibration reps."""
    set_seed(args.seed, device)
    stats = {subject: plan.calib_reps}
    ft_ds = make_dataset(model_name, [subject], plan.finetune_reps, stats, "train",
                         model_name == "v15", subject)
    val_ds = make_dataset(model_name, [subject], plan.finetune_val_reps, stats, "train",
                          False, subject)
    # Both CyborgHybrid and CyborgHybridAttention expose the final head as `.classifier`.
    clf_params = list(model.classifier.parameters())
    clf_ids = {id(p) for p in clf_params}
    rest = [p for p in model.parameters() if id(p) not in clf_ids]
    optimizer = torch.optim.AdamW(
        [{"params": clf_params, "lr": LR}, {"params": rest, "lr": FT_BACKBONE_LR}],
        weight_decay=WEIGHT_DECAY)
    res = fit(model, make_loader(ft_ds, True, args.max_batches),
              make_loader(val_ds, False, args.max_batches), optimizer, device,
              model_name == "v15", args.ft_epochs, args.ft_patience, plan.condition)
    ckpt_path = CKPT_DIR / f"{model_name}_S{subject}_{plan.condition}_seed{args.seed}.pth"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": res["state_dict"], "model_name": model_name,
                "held_out_subject": subject, "condition": plan.condition}, ckpt_path)
    meta = {"best_val": res["best_val"], "best_epoch": res["best_epoch"],
            "epochs_run": res["epochs_run"], "train_windows": len(ft_ds),
            "val_windows": len(val_ds), "train_class_distribution": class_dist(ft_ds)}
    return meta, {"finetune_windows": len(ft_ds)}, ckpt_path


def run_condition(
    model_name: str, subject: int, plan: FoldPlan, pre_model: nn.Module,
    pre_meta: dict, pre_ckpt: Path, args: argparse.Namespace, device: torch.device,
    started: float,
) -> dict:
    """Evaluates one condition on rep 6 and writes its JSON and probs files."""
    model = copy.deepcopy(pre_model)
    if plan.finetune_reps:
        meta, extra, ckpt_path = finetune(model, model_name, subject, plan, args, device)
    else:
        meta, extra, ckpt_path = pre_meta, {}, pre_ckpt

    test_ds = make_dataset(model_name, [subject], plan.test_reps,
                           {subject: plan.calib_reps}, "test", False, subject)
    probs, labels = predict_probs(model, make_loader(test_ds, False, args.max_batches), device)
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    record = {
        "model": model_name, "subject": subject, "seed": args.seed,
        "condition": plan.condition, "protocol": "LOSO",
        "train_subjects": list(plan.train_subjects),
        "calib_reps": list(plan.calib_reps), "finetune_reps": list(plan.finetune_reps),
        "norm": "calibration_reps",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_min": round((time.time() - started) / 60, 2),
        "device": str(device),
        "epochs_run": meta["epochs_run"], "best_epoch": meta["best_epoch"],
        "best_val_accuracy": meta["best_val"],
        "pretrain_epochs_run": pre_meta["epochs_run"],
        "pretrain_best_epoch": pre_meta["best_epoch"],
        "pretrain_best_val_accuracy": pre_meta["best_val"],
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "per_class_recall": [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None
                             for i in range(NUM_CLASSES)],
        "confusion_matrix": cm.tolist(),
        "window_counts": {"train_windows": meta["train_windows"],
                          "val_windows": meta["val_windows"],
                          "test_windows": len(test_ds),
                          "pretrain_train_windows": pre_meta["train_windows"], **extra},
        "train_class_distribution": meta["train_class_distribution"],
        "pretrain_class_distribution": pre_meta["train_class_distribution"],
        "test_class_distribution": class_dist(test_ds),
        "checkpoint": str(ckpt_path.relative_to(ROOT)).replace("\\", "/"),
        "torch_version": torch.__version__, "python_version": platform.python_version(),
    }
    tag = f"{model_name}_S{subject}_{plan.condition}_seed{args.seed}"
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PROBS_DIR / f"{tag}.npz", probs=probs, labels=labels)
    (RES_DIR / f"{tag}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [{plan.condition}] acc {record['accuracy']*100:.2f}% "
          f"bal {record['balanced_accuracy']*100:.2f}% -> {tag}.json", flush=True)
    return record


def run_fold(model_name: str, subject: int, args: argparse.Namespace,
             device: torch.device) -> list[tuple[str, float, float]]:
    """Runs all pending conditions of one (model, held-out subject) fold."""
    RES_DIR.mkdir(parents=True, exist_ok=True)
    pending = [c for c in args.conditions
               if args.force
               or not (RES_DIR / f"{model_name}_S{subject}_{c}_seed{args.seed}.json").exists()]
    if not pending:
        print(f"[SKIP] {model_name} S{subject}: all conditions done")
        return []
    started = time.time()
    print("=" * 68)
    print(f"LOSO | {model_name.upper()} | held-out S{subject} | {pending}")
    print("=" * 68, flush=True)
    base_plan = fold_plan(subject, pending[0])
    pre_model, pre_meta, pre_ckpt = pretrain(model_name, subject, base_plan, args, device)
    out = []
    for cond in pending:
        r = run_condition(model_name, subject, fold_plan(subject, cond), pre_model,
                          pre_meta, pre_ckpt, args, device, started)
        out.append((cond, r["accuracy"] * 100, r["balanced_accuracy"] * 100))
    print(f"Fold time: {(time.time() - started) / 60:.2f} min", flush=True)
    return out


_CUDA_PROBE = (
    "import sys, torch; "
    "ok = torch.cuda.is_available() and (torch.zeros(1, device='cuda') + 1).item() == 1; "
    "sys.exit(0 if ok else 1)"
)


def cuda_usable() -> bool:
    """Checks in a separate process whether CUDA can actually run a kernel.

    ``torch.cuda.is_available()`` can be True on GPUs the installed build cannot
    run (old architectures, devices held by another process). Recent PyTorch
    optimizers query the accelerator even when training on CPU, so a broken GPU
    must be hidden before CUDA is ever touched; probing in a child process keeps
    this process clean.

    Returns:
        True if a CUDA kernel ran successfully.
    """
    try:
        return subprocess.run([sys.executable, "-c", _CUDA_PROBE], timeout=180,
                              capture_output=True).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def resolve_device(choice: str) -> torch.device:
    """Returns the training device; GPU visibility was settled in ``main``.

    Args:
        choice: 'auto', 'cuda' or 'cpu'.

    Returns:
        The device to use.

    Raises:
        RuntimeError: If 'cuda' was requested and is not available.
    """
    if choice != "cpu" and torch.cuda.is_available():
        print(f"[INFO] Device: cuda ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    if choice == "cuda":
        raise RuntimeError("CUDA requested but not available")
    print("[INFO] Device: cpu")
    return torch.device("cpu")


def relaunch_without_gpu() -> int:
    """Re-runs this script with every GPU hidden and returns its exit code."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")  # "" counts as unset on Windows
    return subprocess.call([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                           env=env)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="LOSO runner for NinaPro DB3")
    parser.add_argument("--models", nargs="+", choices=["v10", "v15", "v17", "v18"],
                        default=["v10", "v15", "v17", "v18"])
    parser.add_argument("--subjects", type=int, nargs="+", default=list(ALL_SUBJECTS))
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS),
                        default=list(CONDITIONS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--ft-epochs", type=int, default=60)
    parser.add_argument("--ft-patience", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Cap batches per epoch (train and eval); smoke tests only")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="auto = CUDA if usable, else CPU")
    args = parser.parse_args()

    gpu_hidden = os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1")
    if not gpu_hidden and torch.cuda.is_available() and args.device != "cuda":
        if args.device == "cpu":
            return relaunch_without_gpu()
        if not cuda_usable():
            print("[WARN] A GPU is present but CUDA cannot run on it; "
                  "relaunching on CPU with the GPU hidden")
            return relaunch_without_gpu()
    device = resolve_device(args.device)
    missing = [s for s in ALL_SUBJECTS
               if not (ROOT / "data" / "raw" / f"S{s}_E1_A1.mat").exists()]
    if missing:
        print(f"[WARN] Missing .mat files for subjects {missing}; LOSO needs all 11")

    summary: list[tuple[str, int, str, str, str, str]] = []
    for model_name in args.models:
        for subject in args.subjects:
            if subject not in ALL_SUBJECTS or missing:
                reason = ("unknown subject" if subject not in ALL_SUBJECTS
                          else f"missing .mat {missing}")
                print(f"[SKIP] {model_name} S{subject}: {reason}")
                summary.append((model_name, subject, "-", "---", "---", reason))
                continue
            try:
                for cond, acc, bal in run_fold(model_name, subject, args, device):
                    summary.append((model_name, subject, cond, f"{acc:.2f}%", f"{bal:.2f}%", ""))
            except Exception as exc:  # one failing fold must not stop the batch
                print(f"[FAIL] {model_name} S{subject}: {type(exc).__name__}: {exc}")
                summary.append((model_name, subject, "-", "---", "---",
                                f"{type(exc).__name__}: {exc}"))

    print("\n" + "=" * 68 + "\nSUMMARY\n" + "=" * 68)
    print(f"{'model':<6}{'subj':<6}{'cond':<6}{'acc':>9}{'bal':>9}  note")
    for m, s, c, a, b, note in summary:
        print(f"{m:<6}S{s:<5}{c:<6}{a:>9}{b:>9}  {note}")
    failed = [row for row in summary if row[5]]
    if failed:
        print(f"\n[ERROR] {len(failed)} run(s) failed or were skipped; see notes above.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
