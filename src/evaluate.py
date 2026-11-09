"""Script de evaluación para modelo entrenado de clasificación EMG"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import pandas as pd

from src.data_loader import create_data_loaders
from src.models import HybridCNNLSTM
from src.config import DEVICE, SELECTED_GESTURES

def load_best_model(checkpoint_path, model):
    """
    Carga el mejor modelo desde checkpoint.
    
    Args:
        checkpoint_path: Path al checkpoint
        model: Modelo a cargar
    
    Returns:
        model con pesos cargados
    """
    print(f"Cargando modelo desde {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Modelo cargado - Epoch: {checkpoint['epoch']}")
    print(f"Val Accuracy en entrenamiento: {checkpoint['val_accuracy']:.4f}")
    
    return model

def collect_predictions(model, test_loader, device):
    """
    Colecta todas las predicciones y etiquetas del test set.
    
    Returns:
        tuple: (all_predictions, all_labels)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    print("Colectando predicciones del test set...")
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(signals)
            _, predicted = torch.max(outputs, 1)
            
            # Guardar predicciones y etiquetas
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Genera y guarda matriz de confusión.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        class_names: Nombres de las clases
        save_path: Path para guardar la imagen
    """
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalizar para mostrar porcentajes
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Crear figura
    plt.figure(figsize=(12, 10))
    
    # Matriz de confusión normalizada
    plt.subplot(2, 2, 1)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión Normalizada')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    
    # Matriz de confusión absoluta
    plt.subplot(2, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión Absoluta')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    
    # Precisión por clase
    plt.subplot(2, 2, 3)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    bars = plt.bar(class_names, class_accuracy)
    plt.title('Accuracy por Clase')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Añadir valores sobre las barras
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Distribución de predicciones
    plt.subplot(2, 2, 4)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    
    x_pos = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x_pos - width/2, counts_true, width, label='Verdadero', alpha=0.7)
    plt.bar(x_pos + width/2, counts_pred, width, label='Predicho', alpha=0.7)
    plt.title('Distribución de Clases')
    plt.xlabel('Clase')
    plt.ylabel('Número de Muestras')
    plt.xticks(x_pos, class_names)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matriz de confusión guardada en: {save_path}")

def print_detailed_report(y_true, y_pred, class_names):
    """
    Imprime reporte detallado de clasificación.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        class_names: Nombres de las clases
    """
    print("\n" + "="*60)
    print("REPORTE DE CLASIFICACIÓN DETALLADO")
    print("="*60)
    
    # Reporte general
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                  digits=4, output_dict=True)
    
    # Métricas globales
    accuracy = (y_true == y_pred).mean()
    print(f"\nAccuracy General: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}")
    print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Weighted Avg Recall: {report['weighted avg']['recall']:.4f}")
    print(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    # Tabla detallada por clase
    print(f"\n{'Clase':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Soporte':<8}")
    print("-" * 56)
    
    for class_name in class_names:
        if class_name in report:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            support = report[class_name]['support']
            print(f"{class_name:<8} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<8}")
    
    # Análisis de errores
    print(f"\nANÁLISIS DE ERRORES")
    print("-" * 30)
    
    cm = confusion_matrix(y_true, y_pred)
    
    for i, class_name in enumerate(class_names):
        # Errores por clase (falsos negativos)
        fn = cm[i].sum() - cm[i, i]  # False Negatives
        fp = cm[:, i].sum() - cm[i, i]  # False Positives
        tp = cm[i, i]  # True Positives
        
        if fn + fp > 0:
            error_rate = (fn + fp) / (tp + fn + fp)
            print(f"{class_name}: {fn + fp} errores ({error_rate:.3f}) "
                  f"[FN: {fn}, FP: {fp}, TP: {tp}]")
    
    # Comparación con baseline (aleatorio)
    random_baseline = 1.0 / len(class_names)
    print(f"\nBaseline (aleatorio): {random_baseline:.4f} ({random_baseline*100:.2f}%)")
    improvement = (accuracy - random_baseline) / random_baseline * 100
    print(f"Mejora vs baseline: {improvement:.1f}%")

def save_predictions_csv(y_true, y_pred, save_path):
    """
    Guarda predicciones en archivo CSV para análisis adicional.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        save_path: Path para guardar CSV
    """
    df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'correct': y_true == y_pred
    })
    
    df.to_csv(save_path, index=False)
    print(f"Predicciones guardadas en: {save_path}")

def evaluate_model():
    """Función principal de evaluación"""
    print("=== EVALUACIÓN DEL MODELO ENTRENADO ===")
    print(f"Device: {DEVICE}")
    print("=" * 50)
    
    # Cargar data loaders
    print("Cargando datasets...")
    _, _, test_loader = create_data_loaders(batch_size=64)  # Batch más grande para evaluación
    
    # Crear modelo
    print("Creando modelo...")
    model = HybridCNNLSTM().to(DEVICE)
    
    # Cargar pesos entrenados
    checkpoint_path = Path("checkpoints/best_model.pth")
    if not checkpoint_path.exists():
        print(f"ERROR: No se encontró checkpoint en {checkpoint_path}")
        return
    
    model = load_best_model(checkpoint_path, model)
    
    # Colectar predicciones
    y_true, y_pred = collect_predictions(model, test_loader, DEVICE)
    
    # Nombres de clases (gestos)
    class_names = [f"Gesto {i}" for i in SELECTED_GESTURES]
    
    # Crear directorio de resultados
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Generar y guardar matriz de confusión
    confusion_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, confusion_path)
    
    # Imprimir reporte detallado
    print_detailed_report(y_true, y_pred, class_names)
    
    # Guardar predicciones en CSV
    csv_path = results_dir / "predictions.csv"
    save_predictions_csv(y_true, y_pred, csv_path)
    
    # Estadísticas adicionales
    print(f"\nESTADÍSTICAS ADICIONALES")
    print("-" * 30)
    print(f"Total de muestras evaluadas: {len(y_true)}")
    print(f"Muestras correctamente clasificadas: {(y_true == y_pred).sum()}")
    print(f"Muestras incorrectamente clasificadas: {(y_true != y_pred).sum()}")
    
    # Guardar resumen en archivo
    summary_path = results_dir / "evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=== RESUMEN DE EVALUACIÓN ===\n\n")
        f.write(f"Accuracy General: {(y_true == y_pred).mean():.4f}\n")
        f.write(f"Total muestras: {len(y_true)}\n")
        f.write(f"Correctas: {(y_true == y_pred).sum()}\n")
        f.write(f"Incorrectas: {(y_true != y_pred).sum()}\n\n")
        f.write("Reporte de clasificación:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    print(f"\nResumen guardado en: {summary_path}")
    print(f"\nResultados guardados en: {results_dir}")
    print("Evaluación completada!")

if __name__ == "__main__":
    evaluate_model()
