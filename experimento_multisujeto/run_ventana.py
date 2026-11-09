"""Entrena con longitudes de ventana distintas a los 250 ms del trabajo de grado.

Responde dos preguntas que la medición de separabilidad dejó abiertas:

  1. ¿El aumento de separabilidad sobrevive a la pérdida de ventanas de entrenamiento?
     → v10 con 500 ms, contra la corrida de referencia con 250 ms.
  2. ¿Los quince marcos eran de verdad la limitación del Transformer?
     → v23 con 1000 ms (62 marcos), contra v23 con 250 ms (15 marcos) bajo el mismo protocolo.

La constante de ventana se sustituye en memoria sobre el módulo ya importado; NO se
modifica `src/config.py` ni `src/data_loader.py`. Se verifica en cada corrida que la
sustitución tuvo efecto comprobando la forma real de las ventanas.

El encoding posicional de `CyborgTransformer` está fijado en max_len=32 dentro de la clase,
así que con más de 32 marcos falla. Se reemplaza el buffer por uno mayor sobre la instancia
ya construida. Es una tabla sinusoidal fija, sin parámetros aprendidos, de modo que la
arquitectura no cambia: solo se amplía el rango de posiciones que puede indexar.

Uso:
    python experimento_multisujeto/run_ventana.py --model v10 --ventana 250
    python experimento_multisujeto/run_ventana.py --model v23 --ventana 125
    python experimento_multisujeto/run_ventana.py --model v23 --ventana 500
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

import src.data_loader as dl  # noqa: E402
from src.config import BATCH_SIZE, DEVICE, EPOCHS, LEARNING_RATE, SEED  # noqa: E402
from src.models_cyborg import CyborgHybrid  # noqa: E402
from src.models_v23 import CyborgTransformer, PositionalEncoding  # noqa: E402
from src.train_loro_cv import calculate_class_weights, evaluate, train_epoch  # noqa: E402

TRAIN_REPS, VAL_REPS, TEST_REPS = [1, 2, 3, 4], [5], [6]
PATIENCE_ES = 15
FRECUENCIA = 500
CKPT_DIR = HERE / "checkpoints"
RES_DIR = HERE / "resultados"


def set_seed(seed: int) -> None:
    """Fija las tres fuentes de aleatoriedad antes de cada corrida."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(subject: int, ventana: int) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Crea los DataLoaders con la ventana indicada, manteniendo 80 % de traslape.

    Raises:
        RuntimeError: si la sustitución de la constante no tuvo efecto.
    """
    dl.WINDOW_SIZE, dl.STRIDE = ventana, max(1, ventana // 5)
    conjuntos = {}
    for nombre, reps, modo in (("train", TRAIN_REPS, "train"),
                               ("val", VAL_REPS, "val"),
                               ("test", TEST_REPS, "test")):
        conjuntos[nombre] = dl.NinaProDB3Dataset(
            subjects=[subject], reps_to_use=reps, mode=modo,
            augment=False, target_subject=subject)

    real = np.asarray(conjuntos["train"].samples).shape[2]
    if real != ventana:
        raise RuntimeError(f"se pidió ventana {ventana} y se generó {real}")

    loaders = (
        DataLoader(conjuntos["train"], batch_size=BATCH_SIZE, shuffle=True,
                   num_workers=0, drop_last=True),
        DataLoader(conjuntos["val"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        DataLoader(conjuntos["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    )
    conteos = {k: len(v) for k, v in conjuntos.items()}
    return (*loaders, conteos)


def build_model(nombre: str, marcos: int) -> nn.Module:
    """Instancia el modelo, ampliando el encoding posicional del Transformer si hace falta.

    Args:
        nombre: 'v10' o 'v23'.
        marcos: marcos que saldrán de la CNN, es decir ventana // 8.
    """
    if nombre == "v10":
        return CyborgHybrid(manual_features_dim=48)
    if nombre == "v23":
        modelo = CyborgTransformer(manual_features_dim=48)
        actual = modelo.pos_encoding.pe.shape[1]
        if marcos > actual:
            print(f"  [AJUSTE] encoding posicional de {actual} a {marcos + 8} posiciones "
                  f"(tabla sinusoidal fija, sin parámetros aprendidos)")
            modelo.pos_encoding = PositionalEncoding(
                d_model=modelo.d_model, max_len=marcos + 8, dropout=0.1)
        return modelo
    raise ValueError(f"modelo desconocido: {nombre}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrenamiento con ventana variable")
    ap.add_argument("--model", choices=["v10", "v23"], required=True)
    ap.add_argument("--ventana", type=int, required=True, help="muestras a 500 Hz")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    ms = round(a.ventana / FRECUENCIA * 1000)
    marcos = a.ventana // 8
    etiqueta = f"{a.model}_S{a.subject}_w{a.ventana}_seed{a.seed}"

    print("=" * 72)
    print(f"VENTANA VARIABLE | {a.model.upper()} | S{a.subject} | {a.ventana} muestras "
          f"({ms} ms, ~{marcos} marcos) | semilla {a.seed}")
    print("=" * 72)

    set_seed(a.seed)
    device = torch.device(DEVICE)
    t0 = time.time()

    print("\n[1/4] Cargando datos...")
    train_loader, val_loader, test_loader, conteos = build_loaders(a.subject, a.ventana)
    print(f"  ventanas -> train {conteos['train']} | val {conteos['val']} | test {conteos['test']}")

    print("\n[2/4] Creando modelo...")
    model = build_model(a.model, marcos).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parámetros entrenables: {n_par:,}".replace(",", " "))

    criterion = nn.CrossEntropyLoss(weight=calculate_class_weights(train_loader.dataset).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=5)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"{etiqueta}.pth"
    mejor_val, mejor_epoca, sin_mejora, historia = 0.0, -1, 0, []

    print("\n[3/4] Entrenando...")
    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, False)
        va_loss, va_acc, va_bal, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_acc)
        historia.append({"epoch": epoch + 1, "train_acc": float(tr_acc),
                         "val_acc": float(va_acc), "val_bal": float(va_bal)})
        print(f"Ep {epoch+1:3d}/{EPOCHS} | Train {tr_acc*100:.2f}% | "
              f"Val {va_acc*100:.2f}% (BalAcc {va_bal*100:.2f}%) | "
              f"LR {optimizer.param_groups[0]['lr']:.6f}")
        if va_acc > mejor_val:
            mejor_val, mejor_epoca, sin_mejora = va_acc, epoch + 1, 0
            torch.save({"model_state_dict": model.state_dict(), "val_accuracy": va_acc,
                        "ventana": a.ventana, "model_name": a.model, "seed": a.seed}, ckpt)
            print(f"  [SAVED] Val Acc: {va_acc*100:.2f}%")
        else:
            sin_mejora += 1
            if sin_mejora >= PATIENCE_ES:
                print(f"\nEarly stopping (epoch {epoch+1})")
                break

    print("\n[4/4] Evaluando en test...")
    model.load_state_dict(torch.load(ckpt, weights_only=False)["model_state_dict"])
    _, acc, bal, preds, labels = evaluate(model, test_loader, criterion, device)
    cm = confusion_matrix(labels, preds, labels=list(range(7)))

    registro = {
        "model": a.model, "subject": a.subject, "seed": a.seed,
        "ventana_muestras": a.ventana, "ventana_ms": ms, "marcos_tras_cnn": marcos,
        "stride_muestras": a.ventana // 5,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "parametros": int(n_par),
        "epochs_run": len(historia), "best_epoch": mejor_epoca,
        "best_val_accuracy": float(mejor_val),
        "accuracy": float(acc), "balanced_accuracy": float(bal),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "per_class_recall": [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None
                             for i in range(7)],
        "confusion_matrix": cm.tolist(),
        "window_counts": conteos,
        "test_class_distribution": np.bincount(labels, minlength=7).tolist(),
    }
    RES_DIR.mkdir(parents=True, exist_ok=True)
    salida = RES_DIR / f"ventana_{etiqueta}.json"
    salida.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"RESULTADO | {a.model.upper()} | {ms} ms | {marcos} marcos")
    print("=" * 72)
    print(f"Accuracy:          {acc*100:.2f}%")
    print(f"Balanced Accuracy: {bal*100:.2f}%")
    print(f"Ventanas de entrenamiento: {conteos['train']} | Épocas: {len(historia)} "
          f"| Tiempo: {registro['elapsed_min']} min")
    print(f"Guardado en: {salida.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
