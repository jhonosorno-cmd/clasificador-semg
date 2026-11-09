"""
Evaluación del Sistema en Cascada: Modelo A (detector binario) → Modelo B (clasificador 6 gestos)
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.config import *
from src.models import HybridCNNLSTM
from src.data_loader import create_data_loaders


def evaluate_cascade(model_binary, model_gestures, loader, device):
    """
    Evalúa el sistema en cascada:
    1. Modelo A predice si es reposo (0) o gesto (1)
    2. Si es gesto, Modelo B clasifica cuál gesto (0-5 → 1-6)
    """
    model_binary.eval()
    model_gestures.eval()
    
    all_preds = []
    all_labels = []
    
    # Estadísticas de cascada
    total_samples = 0
    routed_to_binary_rest = 0
    routed_to_gesture_classifier = 0
    
    with torch.no_grad():
        for signals, labels in tqdm(loader, desc="Evaluating Cascade"):
            signals = signals.to(device)
            batch_size = signals.size(0)
            total_samples += batch_size
            
            # Paso 1: Modelo A (detector binario)
            binary_outputs = model_binary(signals)
            _, binary_preds = torch.max(binary_outputs, 1)  # 0=reposo, 1=gesto
            
            # Paso 2: Para cada muestra, decidir la predicción final
            final_preds = torch.zeros(batch_size, dtype=torch.long)
            
            for i in range(batch_size):
                if binary_preds[i] == 0:
                    # Modelo A dice "reposo" → predicción final = 0
                    final_preds[i] = 0
                    routed_to_binary_rest += 1
                else:
                    # Modelo A dice "gesto" → pasar a Modelo B
                    routed_to_gesture_classifier += 1
                    signal_i = signals[i:i+1]  # (1, 1, 125, 13)
                    gesture_output = model_gestures(signal_i)
                    _, gesture_pred = torch.max(gesture_output, 1)
                    # Remapear: Modelo B predice 0-5, necesitamos 1-6
                    final_preds[i] = gesture_pred[0] + 1
            
            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calcular métricas
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(7)))
    
    # Estadísticas de routing
    print(f"\n{'='*60}")
    print(f"ESTADÍSTICAS DE CASCADA")
    print(f"{'='*60}")
    print(f"Total muestras: {total_samples}")
    print(f"Ruteadas a 'Reposo' (Modelo A): {routed_to_binary_rest} ({routed_to_binary_rest/total_samples*100:.1f}%)")
    print(f"Ruteadas a 'Clasificador Gestos' (Modelo B): {routed_to_gesture_classifier} ({routed_to_gesture_classifier/total_samples*100:.1f}%)")
    
    return accuracy, balanced_acc, cm, all_preds, all_labels


def main():
    print("=" * 60)
    print("EVALUACION SISTEMA EN CASCADA (A -> B)")
    print("=" * 60)
    
    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")
    
    # Cargar datos
    print("\n[1/4] Cargando datos...")
    _, _, test_loader = create_data_loaders(batch_size=BATCH_SIZE)
    print(f"Test batches: {len(test_loader)}")
    
    # Cargar Modelo A (detector binario)
    print("\n[2/4] Cargando Modelo A (detector binario)...")
    model_binary = HybridCNNLSTM(num_classes=2).to(device)
    checkpoint_binary = torch.load(
        CHECKPOINTS_PATH / "best_model_binary.pth",
        weights_only=False
    )
    model_binary.load_state_dict(checkpoint_binary['model_state_dict'])
    print(f"Modelo A cargado (Val Acc: {checkpoint_binary['val_accuracy']*100:.2f}%)")
    
    # Cargar Modelo B (clasificador 6 gestos)
    print("\n[3/4] Cargando Modelo B (clasificador 6 gestos)...")
    model_gestures = HybridCNNLSTM(num_classes=6).to(device)
    checkpoint_gestures = torch.load(
        CHECKPOINTS_PATH / "best_model_gestures.pth",
        weights_only=False
    )
    model_gestures.load_state_dict(checkpoint_gestures['model_state_dict'])
    print(f"Modelo B cargado (Val Acc: {checkpoint_gestures['val_accuracy']*100:.2f}%)")
    
    # Evaluar sistema en cascada
    print("\n[4/4] Evaluando sistema en cascada...")
    test_acc, test_bal_acc, cm, test_preds, test_labels = evaluate_cascade(
        model_binary, model_gestures, test_loader, device
    )
    
    print(f"\n{'='*60}")
    print(f"RESULTADOS FINALES - SISTEMA EN CASCADA")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_bal_acc*100:.2f}%")
    
    # Confusion matrix
    print(f"\nConfusion Matrix (7 clases):")
    print("       ", end="")
    for i in range(7):
        print(f"Pred{i} ", end="")
    print()
    for i in range(7):
        print(f"Real{i}: ", end="")
        for j in range(7):
            print(f"{cm[i,j]:5d} ", end="")
        print()
    
    # Métricas por clase
    print(f"\nAccuracy por clase:")
    for i in range(7):
        class_name = "Reposo" if i == 0 else f"Gesto {i}"
        class_acc = cm[i,i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  Clase {i} ({class_name}): {class_acc*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")
    
    # Comparación con modelo monolítico
    print(f"\n{'='*60}")
    print(f"COMPARACIÓN")
    print(f"{'='*60}")
    print(f"Sistema Cascada:     {test_acc*100:.2f}%")
    print(f"Modelo Monolítico v8: 69.38%")
    print(f"Objetivo:            79.72%")
    
    if test_acc > 0.7972:
        print(f"\n¡OBJETIVO ALCANZADO! (+{(test_acc-0.7972)*100:.2f}%)")
    else:
        print(f"\nGap al objetivo: {(0.7972-test_acc)*100:.2f}%")


if __name__ == "__main__":
    main()
