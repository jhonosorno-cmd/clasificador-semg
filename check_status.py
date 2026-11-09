import torch
import numpy as np
from pathlib import Path

def check_project_status():
    """Verifica estado completo del proyecto"""
    
    print("🔍 ESTADO ACTUAL DEL PROYECTO")
    print("=" * 50)
    
    # 1. Estructura de archivos
    print("\n📁 ESTRUCTURA DE ARCHIVOS:")
    
    required_files = [
        "src/config.py",
        "src/data_loader.py", 
        "src/models.py",
        "src/train.py",
        "src/evaluate.py",
        "requirements.txt",
        "README.md",
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"  ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"  ❌ {file_path} (falta)")
    
    # 2. Datos
    print("\n📊 DATOS:")
    data_files = list(Path("data/raw").glob("*.mat"))
    for data_file in data_files:
        size = data_file.stat().st_size
        print(f"  ✅ {data_file.name} ({size/1024/1024:.1f} MB)")
    
    # 3. Checkpoints
    print("\n💾 MODELOS ENTRENADOS:")
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        for checkpoint in checkpoints:
            size = checkpoint.stat().st_size
            print(f"  ✅ {checkpoint.name} ({size/1024/1024:.1f} MB)")
            
            # Cargar información del checkpoint
            try:
                ckpt = torch.load(checkpoint, map_location='cpu')
                print(f"     Epoch: {ckpt.get('epoch', 'N/A')}")
                print(f"     Val Accuracy: {ckpt.get('val_accuracy', 0):.4f}")
            except Exception as e:
                print(f"     Error: {e}")
    else:
        print("  ❌ No hay checkpoints")
    
    # 4. Configuración
    print("\n⚙️ CONFIGURACIÓN:")
    try:
        from src.config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, SELECTED_GESTURES
        print(f"  Device: {DEVICE}")
        print(f"  Batch Size: {BATCH_SIZE}")
        print(f"  Learning Rate: {LEARNING_RATE}")
        print(f"  Epochs: {EPOCHS}")
        print(f"  Gestos: {SELECTED_GESTURES}")
    except Exception as e:
        print(f"  ❌ Error config: {e}")
    
    # 5. Dataset status
    print("\n📈 DATASET STATUS:")
    try:
        from src.data_loader import create_data_loaders
        train_loader, val_loader, test_loader = create_data_loaders(batch_size=32)
        
        print(f"  Train: {len(train_loader.dataset):,} muestras ({len(train_loader)} batches)")
        print(f"  Val: {len(val_loader.dataset):,} muestras ({len(val_loader)} batches)")
        print(f"  Test: {len(test_loader.dataset):,} muestras ({len(test_loader)} batches)")
        
        # Distribución de clases
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        
        from collections import Counter
        class_dist = Counter(all_labels)
        print(f"  Distribución train:")
        for class_id in sorted(class_dist.keys()):
            count = class_dist[class_id]
            percentage = count / len(all_labels) * 100
            print(f"    Clase {class_id}: {count} ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"  ❌ Error dataset: {e}")
    
    # 6. Modelo
    print("\n🤖 MODELO:")
    try:
        from src.models import HybridCNNLSTM
        model = HybridCNNLSTM()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Modelo: HybridCNNLSTM")
        print(f"  Parámetros totales: {total_params:,}")
        print(f"  Parámetros entrenables: {trainable_params:,}")
        print(f"  Input: (Batch, 1, 50, 12)")
        print(f"  Output: (Batch, 7)")
        
    except Exception as e:
        print(f"  ❌ Error modelo: {e}")
    
    # 7. Estado de entrenamiento
    print("\n🎯 ESTADO DE ENTRENAMIENTO:")
    best_model_path = Path("checkpoints/best_model.pth")
    
    if best_model_path.exists():
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu')
            epoch = checkpoint.get('epoch', 0)
            val_acc = checkpoint.get('val_accuracy', 0)
            
            print(f"  ✅ Modelo entrenado disponible")
            print(f"  Mejor época: {epoch}")
            print(f"  Mejor Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
            
            if val_acc > 0.7972:
                print(f"  🎉 OBJETIVO ALCANZADO: >79.72%!")
            else:
                needed = 0.7972 - val_acc
                print(f"  📈 Necesita +{needed*100:.2f}% para alcanzar objetivo")
                
        except Exception as e:
            print(f"  ❌ Error leyendo checkpoint: {e}")
    else:
        print(f"  ❌ No hay modelo entrenado")
        print(f"  💡 Ejecutar: python -m src.train")
    
    # 8. Resultados
    print("\n📊 RESULTADOS:")
    results_dir = Path("results")
    if results_dir.exists():
        result_files = list(results_dir.glob("*"))
        for result_file in result_files:
            print(f"  ✅ {result_file.name}")
    else:
        print(f"  ❌ No hay resultados")
        print(f"  💡 Ejecutar: python -m src.evaluate")
    
    print("\n" + "=" * 50)
    print("🚀 ESTADO FINAL:")
    
    # Determinar estado general
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location='cpu')
        val_acc = checkpoint.get('val_accuracy', 0)
        
        if val_acc > 0.7972:
            print("🎉 PROYECTO COMPLETADO - OBJETIVO ALCANZADO")
        elif val_acc > 0.7:
            print("📈 PROYECTO EN PROGRESO - BUEN RENDIMIENTO")
        else:
            print("⚠️ PROYECTO NECESITA OPTIMIZACIÓN")
    else:
        print("🔧 PROYECTO LISTO PARA ENTRENAR")
    
    print("\nPróximos pasos:")
    if not best_model_path.exists():
        print("  1. python -m src.train")
    else:
        print("  1. python -m src.evaluate")
        print("  2. Analizar resultados y optimizar si es necesario")

if __name__ == "__main__":
    check_project_status()
