"""
Script temporal: genera la matriz de confusión del Ensemble 4M (v10+v15+v17+v18)
y la guarda en results/ensemble_confusion.json para uso en LaTeX.

Ejecutar desde la raíz del proyecto:
    python gen_confusion_matrix.py
"""
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.models_cyborg import CyborgHybrid
from src.models_v18 import CyborgHybridAttention
from src.data_loader import create_data_loaders
from src.train_v17_features import create_data_loaders_v17

CHECKPOINTS = ROOT / "checkpoints"
DEVICE = "cpu"

GESTOS = ["Reposo", "Gesto 1", "Gesto 2", "Gesto 3", "Gesto 4", "Gesto 5", "Gesto 6"]


def load_model(ModelClass, path, features_dim=None):
    if features_dim is not None:
        model = ModelClass(manual_features_dim=features_dim)
    else:
        model = ModelClass()
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def main():
    print("Cargando modelos...")
    m10 = load_model(CyborgHybrid,          CHECKPOINTS / "best_model_cyborg.pth",          48)
    m15 = load_model(CyborgHybrid,          CHECKPOINTS / "best_model_v15_augmentation.pth", 48)
    m17 = load_model(CyborgHybrid,          CHECKPOINTS / "best_model_v17_features.pth",    132)
    m18 = load_model(CyborgHybridAttention, CHECKPOINTS / "best_model_v18_attention.pth",   None)

    print("Cargando datos de prueba (S1, rep 6)...")
    _, _, loader_48  = create_data_loaders(batch_size=512)
    _, _, loader_132 = create_data_loaders_v17(batch_size=512)

    signals_list, feat48_list, feat132_list, labels_list = [], [], [], []
    for (sig, f48, lbl), (_, f132, _) in zip(loader_48, loader_132):
        signals_list.append(sig.numpy())
        feat48_list.append(f48.numpy())
        feat132_list.append(f132.numpy())
        labels_list.append(lbl.numpy())

    signals  = np.concatenate(signals_list)
    feat48   = np.concatenate(feat48_list)
    feat132  = np.concatenate(feat132_list)
    y_true   = np.concatenate(labels_list)

    print(f"Total muestras test: {len(y_true)}")
    print(f"Distribución clases: {np.bincount(y_true)}")

    print("Calculando predicciones del ensemble...")
    def infer(model, sig, feat):
        with torch.no_grad():
            logits = model(torch.tensor(sig), torch.tensor(feat))
            return torch.softmax(logits, dim=1).numpy()

    p10 = infer(m10, signals, feat48)
    p15 = infer(m15, signals, feat48)
    p17 = infer(m17, signals, feat132)
    p18 = infer(m18, signals, feat48)

    probs = (p10 + p15 + p17 + p18) / 4
    y_pred = np.argmax(probs, axis=1)

    acc = np.mean(y_true == y_pred) * 100
    print(f"\nAccuracy Ensemble 4M: {acc:.2f}%")

    # Matriz de confusión (counts)
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de confusión (counts):")
    print(cm)

    # Normalizada por fila (%)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    print("\nMatriz de confusión normalizada por fila (%):")
    np.set_printoptions(precision=1, suppress=True)
    print(cm_pct)

    # Accuracy por clase
    print("\nAccuracy por clase:")
    for i in range(7):
        total = cm[i].sum()
        correct = cm[i, i]
        print(f"  Clase {i} ({GESTOS[i]:12s}): {correct:4d}/{total:4d} = {correct/total*100:.2f}%")

    print("\nReporte completo:")
    print(classification_report(y_true, y_pred, target_names=GESTOS))

    # Guardar resultados
    results = {
        "accuracy": float(acc),
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_pct": cm_pct.tolist(),
        "class_names": GESTOS,
        "per_class": {
            str(i): {
                "correct": int(cm[i, i]),
                "total": int(cm[i].sum()),
                "accuracy_pct": float(cm[i, i] / cm[i].sum() * 100)
            }
            for i in range(7)
        }
    }

    out_path = ROOT / "results" / "ensemble_confusion.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
