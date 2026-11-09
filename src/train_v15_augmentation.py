"""
Entrenamiento v15: Signal Augmentation sobre S1
Estrategia: igual que v10 (CyborgHybrid, S1 solo) + augmentation de senal EMG:
  - Gaussian noise: simula ruido de sensor
  - Amplitude scaling: distintas intensidades de contraccion
  - Channel dropout: simula electrodo ruidoso
Sin Mixup (no ayudo en experimentos previos con esta arquitectura).
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, CHECKPOINTS_PATH, SEED
from src.models_cyborg import CyborgHybrid
from src.data_loader import create_data_loaders

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Augmentation EMG
# ---------------------------------------------------------------------------

def augment_signals(
    x_signal: torch.Tensor,
    noise_std_max: float = 0.05,
    scale_range: tuple = (0.8, 1.2),
    channel_drop_prob: float = 0.1,
) -> torch.Tensor:
    """
    Augmentation de senal EMG especifica para el dominio.

    Aplica tres transformaciones de forma independiente por batch:
      1. Ruido gaussiano: simula ruido electronico del sensor.
      2. Escalado de amplitud: distintas intensidades de contraccion muscular.
      3. Channel dropout: simula electrodo con mala conduccion.

    Args:
        x_signal: Tensor (B, 1, 125, 13) - ventanas de senal EMG.
        noise_std_max: Std maxima del ruido gaussiano (se samplea de [0, noise_std_max]).
        scale_range: Rango (min, max) del factor de escalado de amplitud.
        channel_drop_prob: Probabilidad de anular cada canal por muestra.

    Returns:
        Tensor aumentado de misma forma que x_signal.
    """
    x = x_signal.clone()
    batch_size = x.size(0)
    device = x.device

    # 1. Gaussian noise (std aleatoria por batch)
    noise_std = np.random.uniform(0.0, noise_std_max)
    if noise_std > 0.0:
        x = x + torch.randn_like(x) * noise_std

    # 2. Amplitude scaling (factor independiente por muestra)
    scales = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(*scale_range)
    x = x * scales

    # 3. Channel dropout (mascara binaria por muestra y canal)
    # Shape de mascara: (B, 1, 1, 13) para broadcast sobre (B, 1, 125, 13)
    keep_prob = 1.0 - channel_drop_prob
    channel_mask = torch.bernoulli(
        torch.full((batch_size, 1, 1, 13), keep_prob, device=device)
    )
    x = x * channel_mask

    return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calculate_class_weights(dataset) -> torch.Tensor:
    """Calcula pesos sqrt-inverse de clase."""
    labels = dataset.labels
    counts = np.bincount(labels, minlength=7).astype(np.float64)
    counts[counts == 0] = 1.0
    total = counts.sum()
    weights = np.sqrt(total / (len(counts) * counts))
    print("Class weights (sqrt-inverse):")
    for i in range(len(counts)):
        print(f"  Clase {i}: {int(counts[i])} muestras, peso {weights[i]:.3f}")
    return torch.FloatTensor(weights)


def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """
    Entrena una epoca con augmentation de senal EMG.

    Args:
        model: Modelo a entrenar.
        loader: DataLoader de entrenamiento.
        criterion: Funcion de perdida.
        optimizer: Optimizador.
        device: Dispositivo (cpu/cuda).
        max_grad_norm: Umbral de gradient clipping.

    Returns:
        Tuple (avg_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for signals, features, labels in pbar:
        signals = signals.to(device)
        features = features.to(device)
        labels = labels.to(device)

        # Augmentation solo en train
        signals = augment_signals(signals)

        optimizer.zero_grad()
        outputs = model(signals, features)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total:.3f}"})

    return total_loss / len(loader), correct / total


def validate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, float, list, list]:
    """
    Valida o evalua el modelo sin augmentation.

    Args:
        model: Modelo a evaluar.
        loader: DataLoader de validacion o test.
        criterion: Funcion de perdida.
        device: Dispositivo (cpu/cuda).

    Returns:
        Tuple (avg_loss, accuracy, balanced_accuracy, preds, labels).
    """
    model.eval()
    total_loss = 0.0
    all_preds: list = []
    all_labels: list = []

    with torch.no_grad():
        for signals, features, labels in tqdm(loader, desc="Val", leave=False):
            signals = signals.to(device)
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(signals, features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), accuracy, balanced_acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entrena CyborgHybrid en S1 con augmentation de senal EMG."""
    print("=" * 60)
    print("ENTRENAMIENTO v15: SIGNAL AUGMENTATION (S1)")
    print("Noise + Amplitude Scaling + Channel Dropout")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

    # Datos S1 (igual que v10)
    print("\n[1/5] Cargando datos S1...")
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=BATCH_SIZE)
    print(f"  Train: {len(train_loader.dataset)} muestras")
    print(f"  Val:   {len(val_loader.dataset)} muestras")
    print(f"  Test:  {len(test_loader.dataset)} muestras")

    # Modelo
    print("\n[2/5] Creando modelo CyborgHybrid...")
    model = CyborgHybrid().to(device)
    model.get_model_info()

    # Loss y optimizer
    print("\n[3/5] Configurando optimizer y loss...")
    class_weights = calculate_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Augmentation config (informativo)
    print("\nAugmentation config:")
    print("  Gaussian noise std: [0.0, 0.05]")
    print("  Amplitude scale:    [0.8, 1.2]")
    print("  Channel drop prob:  0.10")

    # Entrenamiento
    print(f"\n[4/5] Iniciando entrenamiento (max {EPOCHS} epochs)...")
    patience_es = 15
    best_val_acc = 0.0
    patience_counter = 0
    checkpoint_path = CHECKPOINTS_PATH / "best_model_v15_augmentation.pth"

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_bal_acc, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | BalAcc: {val_bal_acc*100:.2f}%")
        print(f"LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "val_balanced_accuracy": val_bal_acc,
                    "augmentation": {
                        "noise_std_max": 0.05,
                        "scale_range": (0.8, 1.2),
                        "channel_drop_prob": 0.10,
                    },
                },
                checkpoint_path,
            )
            print(f"[OK] Checkpoint guardado (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience_es:
                print(f"\nEarly stopping (patience={patience_es})")
                break

    # Evaluacion final
    print("\n[5/5] Evaluando en test set (S1)...")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc, test_bal_acc, test_preds, test_labels = validate_epoch(
        model, test_loader, criterion, device
    )

    print(f"\n{'='*60}")
    print("RESULTADOS FINALES - v15 AUGMENTATION")
    print(f"{'='*60}")
    print(f"Test Accuracy:          {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_bal_acc*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print("       " + "".join(f"Pred{i} " for i in range(7)))
    for i in range(7):
        print(f"Real{i}: " + "".join(f"{cm[i,j]:5d} " for j in range(7)))

    print("\nAccuracy por clase:")
    for i in range(7):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        print(f"  Clase {i}: {class_acc*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")

    print(f"\n{'='*60}")
    print("COMPARACION CON BASELINES")
    print(f"{'='*60}")
    print(f"v15 Augmentation (S1): {test_acc*100:.2f}% (BalAcc: {test_bal_acc*100:.2f}%)")
    print(f"v10 Cyborg (S1):       71.32% (BalAcc: 41.92%)")
    print(f"Objetivo paper:        79.72%")

    if test_acc > 0.7972:
        print(f"\n*** OBJETIVO ALCANZADO! (+{(test_acc - 0.7972)*100:.2f}%) ***")
    elif test_acc > 0.7132:
        print(f"\n*** SUPERA v10! (+{(test_acc - 0.7132)*100:.2f}%) ***")
        print(f"Gap al objetivo: {(0.7972 - test_acc)*100:.2f}%")
    else:
        print(f"\nDebajo de v10 por {(0.7132 - test_acc)*100:.2f}%")
        print(f"Gap al objetivo: {(0.7972 - test_acc)*100:.2f}%")

    print(f"\nMejor Val Acc: {best_val_acc*100:.2f}%")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
