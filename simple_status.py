import torch
from pathlib import Path

print("ESTADO ACTUAL DEL PROYECTO")
print("=" * 50)

# Checkpoints
print("\nMODELOS ENTRENADOS:")
checkpoint_dir = Path("checkpoints")
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    for checkpoint in checkpoints:
        size = checkpoint.stat().st_size
        print(f"  {checkpoint.name} ({size/1024/1024:.1f} MB)")
        
        try:
            ckpt = torch.load(checkpoint, map_location='cpu')
            epoch = ckpt.get("epoch", "N/A")
            val_acc = ckpt.get("val_accuracy", 0)
            print(f"     Epoch: {epoch}")
            print(f"     Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        except Exception as e:
            print(f"     Error: {e}")

# Dataset
print("\nDATASET STATUS:")
try:
    from src.data_loader import create_data_loaders
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=32)
    
    print(f"  Train: {len(train_loader.dataset):,} muestras")
    print(f"  Val: {len(val_loader.dataset):,} muestras")
    print(f"  Test: {len(test_loader.dataset):,} muestras")
    
except Exception as e:
    print(f"  Error dataset: {e}")

# Modelo
print("\nMODELO:")
try:
    from src.models import HybridCNNLSTM
    model = HybridCNNLSTM()
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"  Modelo: HybridCNNLSTM")
    print(f"  Parámetros totales: {total_params:,}")
    print(f"  Input: (Batch, 1, 50, 12)")
    print(f"  Output: (Batch, 7)")
    
except Exception as e:
    print(f"  Error modelo: {e}")

# Estado entrenamiento
print("\nESTADO DE ENTRENAMIENTO:")
best_model_path = Path("checkpoints/best_model.pth")

if best_model_path.exists():
    try:
        checkpoint = torch.load(best_model_path, map_location='cpu')
        epoch = checkpoint.get("epoch", 0)
        val_acc = checkpoint.get("val_accuracy", 0)
        
        print(f"  Modelo entrenado disponible")
        print(f"  Mejor época: {epoch}")
        print(f"  Mejor Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        if val_acc > 0.7972:
            print("  OBJETIVO ALCANZADO: >79.72%!")
        else:
            needed = 0.7972 - val_acc
            print(f"  Necesita +{needed*100:.2f}% para alcanzar objetivo")
            
    except Exception as e:
        print(f"  Error leyendo checkpoint: {e}")
else:
    print(f"  No hay modelo entrenado")
    print(f"  Ejecutar: python -m src.train")

print("\n" + "=" * 50)

# Resultados
print("\nRESULTADOS:")
results_dir = Path("results")
if results_dir.exists():
    result_files = list(results_dir.glob("*"))
    for result_file in result_files:
        print(f"  {result_file.name}")
else:
    print(f"  No hay resultados")
    print(f"  Ejecutar: python -m src.evaluate")

print("\nPROXIMOS PASOS:")
if not best_model_path.exists():
    print("  1. python -m src.train")
else:
    print("  1. python -m src.evaluate")
    print("  2. Analizar resultados y optimizar si es necesario")
