"""Entrena un modelo del trabajo de grado sobre un sujeto arbitrario de NinaPro DB3.

Reutiliza el modelo, los pesos de clase, el bucle de entrenamiento y la evaluacion de
`src/train_loro_cv.py`, de modo que el procedimiento sea identico al de la validacion
cruzada ya reportada en el documento. Lo unico que cambia es el sujeto objetivo.

No escribe nada fuera de esta carpeta. En particular, NO toca `checkpoints/loro/`.

Uso:
    python experimento_multisujeto/run_subject.py --model v10 --subjects 1 2 3 4
    python experimento_multisujeto/run_subject.py --model v10 --subjects 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.config import BATCH_SIZE, DEVICE, EPOCHS, LEARNING_RATE, SEED  # noqa: E402
from src.data_loader import NinaProDB3Dataset  # noqa: E402
from src.train_loro_cv import (  # noqa: E402
    build_model,
    calculate_class_weights,
    evaluate,
    train_epoch,
)
from src.train_v17_features import NinaProExtendedFeatures  # noqa: E402

# Particion estandar del documento: train reps 1-4, val rep 5, test rep 6.
TRAIN_REPS = [1, 2, 3, 4]
VAL_REPS = [5]
TEST_REPS = [6]
PATIENCE_ES = 15

CKPT_DIR = HERE / "checkpoints"
RES_DIR = HERE / "resultados"


def set_seed(seed: int) -> None:
    """Fija las tres fuentes de aleatoriedad antes de cada corrida.

    Se llama una vez por combinacion de sujeto y modelo, no al importar, para que el
    resultado de una corrida no dependa de las que se ejecutaron antes en el mismo
    proceso.

    Args:
        seed: Semilla a fijar.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders_for_subject(
    model_name: str, subject: int, batch_size: int = BATCH_SIZE
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Crea los DataLoaders del sujeto indicado con la particion estandar.

    Replica `src.train_loro_cv.build_loaders` cambiando el sujeto, que alli esta fijo
    en 1.

    Args:
        model_name: 'v10', 'v15', 'v17' o 'v18'.
        subject: Id del sujeto de DB3.
        batch_size: Tamano de lote.

    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_cls = NinaProExtendedFeatures if model_name == "v17" else NinaProDB3Dataset
    use_augment = model_name == "v15"

    train_ds = dataset_cls(
        subjects=[subject], reps_to_use=TRAIN_REPS, mode="train",
        augment=use_augment, target_subject=subject,
    )
    val_ds = dataset_cls(
        subjects=[subject], reps_to_use=VAL_REPS, mode="val", target_subject=subject,
    )
    test_ds = dataset_cls(
        subjects=[subject], reps_to_use=TEST_REPS, mode="test", target_subject=subject,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def run_one(model_name: str, subject: int, seed: int) -> dict:
    """Entrena y evalua un modelo sobre un sujeto, y devuelve el registro crudo.

    Args:
        model_name: 'v10', 'v15', 'v17' o 'v18'.
        subject: Id del sujeto de DB3.
        seed: Semilla, fijada antes de construir datos y modelo.

    Returns:
        Diccionario con metricas, conteos y metadatos de la corrida.
    """
    print("=" * 68)
    print(f"MULTI-SUJETO  |  Modelo: {model_name.upper()}  |  Sujeto: S{subject}  |  Semilla: {seed}")
    print("=" * 68)

    set_seed(seed)
    device = torch.device(DEVICE)
    started = time.time()

    print(f"\n[1/4] Cargando datos de S{subject}...")
    train_loader, val_loader, test_loader = build_loaders_for_subject(model_name, subject)
    counts = {
        "train_windows": len(train_loader.dataset),
        "val_windows": len(val_loader.dataset),
        "test_windows": len(test_loader.dataset),
    }
    train_dist = np.bincount(train_loader.dataset.labels, minlength=7).tolist()
    test_dist = np.bincount(test_loader.dataset.labels, minlength=7).tolist()
    print(f"  ventanas -> train {counts['train_windows']} | val {counts['val_windows']} "
          f"| test {counts['test_windows']}")
    print(f"  distribucion test por clase: {test_dist}")

    print("\n[2/4] Creando modelo...")
    model = build_model(model_name).to(device)

    class_weights = calculate_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    use_augment = model_name == "v15"
    best_val_acc, best_epoch, patience_counter = 0.0, -1, 0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{model_name}_S{subject}_seed{seed}.pth"

    print("\n[3/4] Entrenando...")
    print(f"Epochs: {EPOCHS} | LR: {LEARNING_RATE} | Patience: {PATIENCE_ES}")
    print("-" * 68)

    history = []
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, use_augment
        )
        val_loss, val_acc, val_bal_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch + 1, "train_loss": float(train_loss), "train_acc": float(train_acc),
            "val_loss": float(val_loss), "val_acc": float(val_acc),
            "val_balanced_acc": float(val_bal_acc), "lr": float(lr),
        })
        print(f"Ep {epoch+1:3d}/{EPOCHS} | Train {train_acc*100:.2f}% | "
              f"Val {val_acc*100:.2f}% (BalAcc {val_bal_acc*100:.2f}%) | LR {lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, patience_counter = val_acc, epoch + 1, 0
            torch.save(
                {
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc, "val_balanced_accuracy": val_bal_acc,
                    "subject": subject, "model_name": model_name, "seed": seed,
                    "train_reps": TRAIN_REPS, "val_reps": VAL_REPS, "test_reps": TEST_REPS,
                },
                ckpt_path,
            )
            print(f"  [SAVED] Val Acc: {val_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE_ES:
                print(f"\nEarly stopping (epoch {epoch+1})")
                break

    print("\n[4/4] Evaluando en test...")
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_acc, test_bal_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    cm = confusion_matrix(labels, preds, labels=list(range(7)))
    per_class_recall = [
        float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i in range(7)
    ]

    record = {
        "model": model_name,
        "subject": subject,
        "seed": seed,
        "protocol": "particion estandar: train reps 1-4, val rep 5, test rep 6; argmax sin umbral",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_min": round((time.time() - started) / 60, 2),
        "device": str(DEVICE),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_acc),
        "accuracy": float(test_acc),
        "balanced_accuracy": float(test_bal_acc),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "window_counts": counts,
        "train_class_distribution": train_dist,
        "test_class_distribution": test_dist,
        "checkpoint": str(ckpt_path.relative_to(ROOT)).replace("\\", "/"),
    }

    RES_DIR.mkdir(parents=True, exist_ok=True)
    out = RES_DIR / f"{model_name}_S{subject}_seed{seed}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"RESULTADO  |  {model_name.upper()}  |  S{subject}")
    print("=" * 68)
    print(f"Accuracy:          {test_acc*100:.2f}%")
    print(f"Balanced Accuracy: {test_bal_acc*100:.2f}%")
    print(f"Epocas: {len(history)} (mejor: {best_epoch}) | Tiempo: {record['elapsed_min']} min")
    print(f"Guardado en: {out.relative_to(ROOT)}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrenamiento sujeto-especifico multi-sujeto")
    parser.add_argument("--model", choices=["v10", "v15", "v17", "v18"], required=True)
    parser.add_argument("--subjects", type=int, nargs="+", required=True,
                        help="Ids de sujeto de DB3, por ejemplo: 1 2 3 4")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    resumen = []
    for subject in args.subjects:
        mat = ROOT / "data" / "raw" / f"S{subject}_E1_A1.mat"
        if not mat.exists():
            print(f"[OMITIDO] S{subject}: no existe {mat}")
            resumen.append((subject, None, None, "archivo ausente"))
            continue
        try:
            r = run_one(args.model, subject, args.seed)
            resumen.append((subject, r["accuracy"] * 100, r["balanced_accuracy"] * 100, ""))
        except Exception as exc:  # el protocolo exige documentar el fallo, no ocultarlo
            print(f"[FALLO] S{subject}: {type(exc).__name__}: {exc}")
            resumen.append((subject, None, None, f"{type(exc).__name__}: {exc}"))

    print("\n" + "=" * 68)
    print(f"RESUMEN DEL LOTE  |  {args.model.upper()}  |  semilla {args.seed}")
    print("=" * 68)
    print(f"{'Sujeto':<10}{'Acc':>10}{'BalAcc':>10}   nota")
    for s, acc, bal, nota in resumen:
        a = f"{acc:.2f}%" if acc is not None else "---"
        b = f"{bal:.2f}%" if bal is not None else "---"
        print(f"S{s:<9}{a:>10}{b:>10}   {nota}")
    ok = [a for _, a, _, _ in resumen if a is not None]
    if len(ok) > 1:
        print(f"\nmedia {np.mean(ok):.2f}%  ·  desviacion {np.std(ok, ddof=1):.2f} pp  ·  n={len(ok)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
