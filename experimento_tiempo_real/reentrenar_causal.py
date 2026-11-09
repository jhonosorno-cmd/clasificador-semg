"""Reentrena los cuatro componentes del Ensemble 4M con preprocesamiento causal.

La evaluación previa (`eval_causal.py`) midió -1,83 pp al aplicar un pipeline causal sobre
modelos entrenados con el pipeline fuera de línea. Ese número mezcla dos efectos: el costo
real de perder la fase cero, y el desajuste entre cómo se entrenó y cómo se evaluó.

Este script elimina el segundo: entrena de cero con el mismo pipeline causal con el que
después se infiere, de modo que la diferencia restante sea atribuible solo a la causalidad.

Cada componente conserva exactamente lo que lo distingue en el trabajo original:
    v10  CyborgHybrid 48D            + Mixup (alpha=0,4)
    v15  CyborgHybrid 48D            + aumentación de señal (ruido, escalado, canal caído)
    v17  CyborgHybrid 132D           + aumentación de señal
    v18  CyborgHybridAttention 48D   + aumentación de señal

NO modifica el repositorio: sustituye `_preprocess_emg` en memoria y escribe sus
checkpoints y resultados en esta carpeta.

Uso:
    python experimento_tiempo_real/reentrenar_causal.py
    python experimento_tiempo_real/reentrenar_causal.py --solo v10
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
from scipy.signal import butter, lfilter
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import src.data_loader as dl  # noqa: E402
from src.config import BATCH_SIZE, DEVICE, EPOCHS, LEARNING_RATE, SEED, TARGET_FREQUENCY  # noqa: E402
from src.models_cyborg import CyborgHybrid  # noqa: E402
from src.models_v18 import CyborgHybridAttention  # noqa: E402
from src.train_cyborg import train_epoch as train_epoch_mixup  # noqa: E402
from src.train_loro_cv import calculate_class_weights, evaluate  # noqa: E402
from src.train_loro_cv import train_epoch as train_epoch_aug  # noqa: E402
from src.train_v17_features import NinaProExtendedFeatures  # noqa: E402

SUJETO = 1
TRAIN_REPS, VAL_REPS, TEST_REPS = [1, 2, 3, 4], [5], [6]
PATIENCE_ES = 15
THETA_DOC = 0.36
CKPT = HERE / "checkpoints"
_ORIGINAL = dl.NinaProDB3Dataset._preprocess_emg

# alias -> (clase de modelo, dim de features, dataset extendido, usa Mixup)
COMPONENTES = {
    "v10": (CyborgHybrid, 48, False, True),
    "v15": (CyborgHybrid, 48, False, False),
    "v17": (CyborgHybrid, 132, True, False),
    "v18": (CyborgHybridAttention, 48, False, False),
}

_STATS: dict = {}


def fijar_semilla(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def preprocesar_causal(self, emg):
    """Filtro causal y z-score con estadísticos fijados en la calibración.

    La primera invocación corresponde al conjunto de entrenamiento y fija los
    estadísticos; validación y prueba reutilizan esos mismos valores, como haría
    un sistema en vivo calibrado antes de usarse.
    """
    nyq = TARGET_FREQUENCY / 2.0
    b, a = butter(4, [20.0 / nyq, (nyq - 1.0) / nyq], btype="band")
    f = np.zeros_like(emg)
    for ch in range(emg.shape[1]):
        f[:, ch] = lfilter(b, a, emg[:, ch])
    if "mu" not in _STATS:
        mu, sd = f.mean(axis=0, keepdims=True), f.std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1e-8
        _STATS["mu"], _STATS["sd"] = mu, sd
        print(f"  [CALIBRACIÓN] estadísticos fijados desde las reps {TRAIN_REPS}")
    return ((f - _STATS["mu"]) / _STATS["sd"]).astype(np.float32)


def construir_loaders(extendido: bool) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    Clase = NinaProExtendedFeatures if extendido else dl.NinaProDB3Dataset
    conjuntos = {}
    for nombre, reps, modo in (("train", TRAIN_REPS, "train"),
                               ("val", VAL_REPS, "val"),
                               ("test", TEST_REPS, "test")):
        conjuntos[nombre] = Clase(subjects=[SUJETO], reps_to_use=reps, mode=modo,
                                  augment=False, target_subject=SUJETO)
    return (
        DataLoader(conjuntos["train"], batch_size=BATCH_SIZE, shuffle=True,
                   num_workers=0, drop_last=True),
        DataLoader(conjuntos["val"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        DataLoader(conjuntos["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        {k: len(v) for k, v in conjuntos.items()},
    )


def entrenar(alias: str) -> dict:
    Clase, dim, extendido, usa_mixup = COMPONENTES[alias]
    print("\n" + "=" * 74)
    print(f"{alias.upper()} · causal · features {dim}D · "
          f"{'Mixup' if usa_mixup else 'aumentación de señal'}")
    print("=" * 74)

    fijar_semilla()
    device = torch.device(DEVICE)
    t0 = time.time()

    train_loader, val_loader, test_loader, conteos = construir_loaders(extendido)
    print(f"  ventanas -> train {conteos['train']} | val {conteos['val']} | test {conteos['test']}")

    model = Clase(manual_features_dim=dim).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=calculate_class_weights(train_loader.dataset).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=5)
    CKPT.mkdir(parents=True, exist_ok=True)
    ruta = CKPT / f"{alias}_causal.pth"
    mejor, mejor_ep, sin_mejora, historia = 0.0, -1, 0, []

    for ep in range(EPOCHS):
        if usa_mixup:
            tr_loss, tr_acc = train_epoch_mixup(model, train_loader, criterion,
                                                optimizer, device, use_mixup=True)
        else:
            tr_loss, tr_acc = train_epoch_aug(model, train_loader, criterion,
                                              optimizer, device, use_augment=True)
        va_loss, va_acc, va_bal, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_acc)
        historia.append({"epoch": ep + 1, "train_acc": float(tr_acc),
                         "val_acc": float(va_acc), "val_bal": float(va_bal)})
        print(f"Ep {ep+1:3d}/{EPOCHS} | Train {tr_acc*100:.2f}% | "
              f"Val {va_acc*100:.2f}% (Bal {va_bal*100:.2f}%)")
        if va_acc > mejor:
            mejor, mejor_ep, sin_mejora = va_acc, ep + 1, 0
            torch.save({"model_state_dict": model.state_dict(), "val_accuracy": va_acc,
                        "alias": alias, "pipeline": "causal", "seed": SEED}, ruta)
            print(f"  [SAVED] {va_acc*100:.2f}%")
        else:
            sin_mejora += 1
            if sin_mejora >= PATIENCE_ES:
                print(f"\nEarly stopping (época {ep+1})")
                break

    model.load_state_dict(torch.load(ruta, weights_only=False)["model_state_dict"])
    _, acc, bal, _, _ = evaluate(model, test_loader, criterion, device)
    minutos = round((time.time() - t0) / 60, 2)
    print(f"  -> test {acc*100:.2f}% · balanceada {bal*100:.2f}% · {minutos} min")
    return {"alias": alias, "epocas": len(historia), "mejor_epoca": mejor_ep,
            "mejor_val": float(mejor), "test_accuracy": float(acc * 100),
            "test_balanced": float(bal * 100), "minutos": minutos,
            "ventanas": conteos, "historia": historia}


def evaluar_ensemble() -> dict:
    """Promedia las softmax de los cuatro modelos causales sobre prueba y validación."""
    modelos = []
    for alias, (Clase, dim, extendido, _) in COMPONENTES.items():
        m = Clase(manual_features_dim=dim)
        m.load_state_dict(torch.load(CKPT / f"{alias}_causal.pth",
                                     weights_only=False)["model_state_dict"])
        m.eval()
        modelos.append((alias, m, dim, extendido))

    salida = {}
    for split, reps in (("val", VAL_REPS), ("test", TEST_REPS)):
        ds48 = dl.NinaProDB3Dataset(subjects=[SUJETO], reps_to_use=reps, mode="test",
                                    augment=False, target_subject=SUJETO)
        ds132 = NinaProExtendedFeatures(subjects=[SUJETO], reps_to_use=reps, mode="test",
                                        augment=False, target_subject=SUJETO)
        señales = torch.from_numpy(np.asarray(ds48.samples, dtype=np.float32))
        y = np.asarray(ds48.labels)
        f48 = torch.from_numpy(np.stack(
            [ds48._extract_manual_features(s[0, :, :12]) for s in ds48.samples])).float()
        f132 = torch.from_numpy(np.stack(
            [ds132._extract_manual_features(s[0, :, :12]) for s in ds132.samples])).float()

        acum = None
        with torch.no_grad():
            for _, m, dim, _ in modelos:
                p = torch.softmax(m(señales, f132 if dim == 132 else f48), dim=1)
                acum = p if acum is None else acum + p
        salida[split] = {"probs": (acum / len(modelos)).numpy(), "labels": y}

    # theta del documento, y theta re-barrido sobre la validación causal
    vp, vy = salida["val"]["probs"], salida["val"]["labels"]
    rejilla = np.arange(0.10, 1.00, 0.02)
    accs = [((np.where(vp[:, 0] > t, 0, vp[:, 1:].argmax(1) + 1)) == vy).mean() for t in rejilla]
    theta_nuevo = float(rejilla[int(np.argmax(accs))])

    tp, ty = salida["test"]["probs"], salida["test"]["labels"]
    res = {"theta_documento_valor": THETA_DOC, "theta_recalibrado_valor": round(theta_nuevo, 2)}
    for nombre, pred in (("argmax", tp.argmax(1)),
                         ("theta_documento", np.where(tp[:, 0] > THETA_DOC, 0, tp[:, 1:].argmax(1) + 1)),
                         ("theta_recalibrado", np.where(tp[:, 0] > theta_nuevo, 0, tp[:, 1:].argmax(1) + 1))):
        res[nombre] = {
            "accuracy": float((pred == ty).mean() * 100),
            "balanced_accuracy": float(balanced_accuracy_score(ty, pred) * 100),
            "f1_macro": float(f1_score(ty, pred, average="macro", zero_division=0)),
            "recall_por_clase": [float((pred[ty == c] == c).mean() * 100) for c in range(7)],
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Reentrena el ensemble con pipeline causal")
    ap.add_argument("--solo", choices=list(COMPONENTES), help="entrena un solo componente")
    a = ap.parse_args()

    dl.NinaProDB3Dataset._preprocess_emg = preprocesar_causal
    t0 = time.time()
    aliases = [a.solo] if a.solo else list(COMPONENTES)
    individuales = {alias: entrenar(alias) for alias in aliases}

    registro = {
        "descripcion": "Reentrenamiento de los 4 componentes con filtro causal (lfilter) y "
                       "z-score de calibracion, para separar el costo de la causalidad del "
                       "desajuste entre entrenamiento e inferencia.",
        "sujeto": SUJETO, "semilla": SEED,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "individuales": individuales,
    }
    if not a.solo:
        print("\n" + "=" * 74 + "\nENSEMBLE 4M CAUSAL\n" + "=" * 74)
        registro["ensemble"] = evaluar_ensemble()
        e = registro["ensemble"]
        print(f"  argmax              {e['argmax']['accuracy']:.2f}%  "
              f"bal {e['argmax']['balanced_accuracy']:.2f}%")
        print(f"  θ=0,36 (documento)  {e['theta_documento']['accuracy']:.2f}%  "
              f"bal {e['theta_documento']['balanced_accuracy']:.2f}%")
        print(f"  θ={e['theta_recalibrado_valor']:.2f} (recalibrado)  "
              f"{e['theta_recalibrado']['accuracy']:.2f}%  "
              f"bal {e['theta_recalibrado']['balanced_accuracy']:.2f}%")

    registro["minutos_totales"] = round((time.time() - t0) / 60, 2)
    dl.NinaProDB3Dataset._preprocess_emg = _ORIGINAL
    salida = HERE / ("resultados_reentreno.json" if not a.solo
                     else f"resultados_reentreno_{a.solo}.json")
    salida.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {salida.relative_to(ROOT)} · {registro['minutos_totales']} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
