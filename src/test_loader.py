"""Script de prueba para el data loader"""

import torch
from src.data_loader import create_data_loaders

def test_data_loader():
    """Prueba completa del data loader"""
    print("=== PRUEBA DEL DATA LOADER ===\n")
    
    # Crear data loaders
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            subject_id=1, 
            batch_size=8  # Batch pequeño para prueba
        )
    except Exception as e:
        print(f"Error creando data loaders: {e}")
        return
    
    # Resumen de datasets
    print("Resumen de datasets:")
    print(f"  Train: {len(train_loader.dataset)} muestras, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} muestras, {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader.dataset)} muestras, {len(test_loader)} batches")
    print()
    
    # Probar un batch de entrenamiento
    print("Probando batch de entrenamiento...")
    signals, labels = next(iter(train_loader))
    
    print(f"  Signals shape: {signals.shape}")
    print(f"  Signals dtype: {signals.dtype}")
    print(f"  Signals range: [{signals.min():.4f}, {signals.max():.4f}]")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Labels dtype: {labels.dtype}")
    print(f"  Labels únicos: {sorted(labels.unique().tolist())}")
    print()
    
    # Verificar dimensiones esperadas
    expected_signal_shape = (8, 1, 50, 12)
    expected_labels_shape = (8,)
    
    assert signals.shape == expected_signal_shape, f"Shape incorrecto: esperado {expected_signal_shape}, got {signals.shape}"
    assert labels.shape == expected_labels_shape, f"Shape incorrecto: esperado {expected_labels_shape}, got {labels.shape}"
    assert signals.dtype == torch.float32, f"Tipo incorrecto: esperado float32, got {signals.dtype}"
    assert labels.dtype == torch.long, f"Tipo incorrecto: esperado long, got {labels.dtype}"
    
    print("[OK] Todas las verificaciones pasaron!")
    
    # Probar múltiples batches
    print("\nProbando múltiples batches...")
    batch_count = 0
    total_samples = 0
    
    for batch_signals, batch_labels in train_loader:
        batch_count += 1
        total_samples += len(batch_labels)
        
        # Verificar que todas las etiquetas estén en rango 0-6
        assert batch_labels.min() >= 0, f"Etiqueta negativa encontrada: {batch_labels.min()}"
        assert batch_labels.max() <= 6, f"Etiqueta >6 encontrada: {batch_labels.max()}"
        
        if batch_count >= 3:  # Solo probar 3 batches
            break
    
    print(f"  Procesados {batch_count} batches, {total_samples} muestras")
    print("[OK] Múltiples batches funcionan correctamente!")
    
    # Estadísticas finales
    print(f"\n=== ESTADÍSTICAS FINALES ===")
    print(f"Total clases: {len(torch.cat([labels for _, labels in train_loader]).unique())}")
    print(f"Balance de clases (train):")
    
    # Contar muestras por clase
    all_labels = torch.cat([labels for _, labels in train_loader])
    for class_id in range(7):
        count = (all_labels == class_id).sum().item()
        percentage = count / len(all_labels) * 100
        print(f"  Clase {class_id}: {count} muestras ({percentage:.1f}%)")

if __name__ == "__main__":
    test_data_loader()
