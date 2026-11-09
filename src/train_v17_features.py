"""
Entrenamiento v17: Features Manuales Expandidas (132D)
Amplia el vector de features de 48D a 132D añadiendo 7 features por canal:
  Actuales (48D): RMS, ZC, SSC, WL  x 12 canales
  Nuevas   (84D): MAV, VAR, WAMP, AR1, AR2, AR3, AR4  x 12 canales
  Total   (132D): 11 features x 12 canales

Justificacion:
  - MAV: complementa RMS (lineal vs cuadratico)
  - VAR: variabilidad de la señal (potencia)
  - WAMP: cruces con umbral de amplitud (sensible a activacion muscular)
  - AR(4): coeficientes autoregresivos, muy discriminativos para gestos EMG
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, CHECKPOINTS_PATH, SEED
from src.models_cyborg import CyborgHybrid
from src.data_loader import NinaProDB3Dataset, create_data_loaders
from torch.utils.data import DataLoader

torch.manual_seed(SEED)
np.random.seed(SEED)

# Dimension del nuevo vector de features
FEATURES_DIM = 132  # 11 features x 12 canales
WAMP_THRESHOLD = 0.5  # Umbral para WAMP (señal z-scoreada, std≈1)
AR_ORDER = 4


# ---------------------------------------------------------------------------
# Dataset extendido: sobreescribe _extract_manual_features
# ---------------------------------------------------------------------------

class NinaProExtendedFeatures(NinaProDB3Dataset):
    """
    Dataset NinaPro con vector de features expandido a 132D.

    Sobreescribe _extract_manual_features para calcular 11 features por canal
    en lugar de 4, manteniendo todo el resto del pipeline identico.
    """

    def _extract_manual_features(self, emg_window: np.ndarray) -> np.ndarray:
        """
        Extrae 11 features por canal EMG.

        Features por canal:
          0. RMS  - Root Mean Square
          1. ZC   - Zero Crossing Rate
          2. SSC  - Slope Sign Changes
          3. WL   - Waveform Length
          4. MAV  - Mean Absolute Value
          5. VAR  - Varianza de la señal
          6. WAMP - Willison Amplitude (umbral=0.5)
          7. AR1  - Coeficiente AR orden 1
          8. AR2  - Coeficiente AR orden 2
          9. AR3  - Coeficiente AR orden 3
         10. AR4  - Coeficiente AR orden 4

        Args:
            emg_window: (125, 12) - ventana EMG de 12 canales.

        Returns:
            features: (132,) float32 - 11 features x 12 canales.
        """
        n_channels = emg_window.shape[1]
        n_features = 4 + 3 + AR_ORDER  # 11
        features = np.zeros(n_channels * n_features, dtype=np.float32)

        for ch in range(n_channels):
            x = emg_window[:, ch]
            base = ch * n_features

            # --- Features existentes (0-3) ---
            # 0. RMS
            features[base + 0] = np.sqrt(np.mean(x ** 2))
            # 1. ZC
            features[base + 1] = float(np.sum(np.diff(np.sign(x)) != 0))
            # 2. SSC
            dx = np.diff(x)
            features[base + 2] = float(np.sum(np.diff(np.sign(dx)) != 0))
            # 3. WL
            features[base + 3] = np.sum(np.abs(dx))

            # --- Features nuevas (4-10) ---
            # 4. MAV - Mean Absolute Value
            features[base + 4] = np.mean(np.abs(x))
            # 5. VAR - Varianza
            features[base + 5] = np.var(x)
            # 6. WAMP - Willison Amplitude
            features[base + 6] = float(np.sum(np.abs(dx) > WAMP_THRESHOLD))
            # 7-10. AR(4) via Yule-Walker
            ar = _ar_yulewalker(x, order=AR_ORDER)
            features[base + 7:base + 7 + AR_ORDER] = ar

        return features


def _ar_yulewalker(x: np.ndarray, order: int = 4) -> np.ndarray:
    """
    Estima coeficientes AR(order) con el metodo de Yule-Walker.

    Args:
        x: Señal 1D.
        order: Orden del modelo AR.

    Returns:
        Coeficientes AR (order,) float32. Devuelve ceros si la matriz es singular.
    """
    n = len(x)
    # Autocorrelacion normalizada para lags 0..order
    r = np.array([np.dot(x[:n - k], x[k:]) / n for k in range(order + 1)],
                 dtype=np.float64)
    # Matriz de Toeplitz simetrica
    idx = np.arange(order)
    R = r[np.abs(idx[:, None] - idx[None, :])]
    b = r[1:order + 1]
    try:
        coeffs = np.linalg.solve(R, b)
    except np.linalg.LinAlgError:
        coeffs = np.zeros(order, dtype=np.float64)
    return np.clip(coeffs, -10.0, 10.0).astype(np.float32)


# ---------------------------------------------------------------------------
# DataLoaders con dataset extendido
# ---------------------------------------------------------------------------

def create_data_loaders_v17(
    train_subjects: list = None,
    target_subject: int = 1,
    batch_size: int = BATCH_SIZE,
):
    """
    Crea DataLoaders usando NinaProExtendedFeatures (132D features).

    Args:
        train_subjects: Sujetos para train (default: [1]).
        target_subject: Sujeto para val/test.
        batch_size: Tamano de batch.

    Returns:
        train_loader, val_loader, test_loader
    """
    if train_subjects is None:
        train_subjects = [1]

    train_reps = [1, 2, 3, 4]
    val_reps = [5]
    test_reps = [6]

    train_dataset = NinaProExtendedFeatures(
        subjects=train_subjects,
        reps_to_use=train_reps,
        mode="train",
        augment=True,
        target_subject=target_subject,
    )
    val_dataset = NinaProExtendedFeatures(
        subjects=train_subjects,
        reps_to_use=val_reps,
        mode="val",
        target_subject=target_subject,
    )
    test_dataset = NinaProExtendedFeatures(
        subjects=train_subjects,
        reps_to_use=test_reps,
        mode="test",
        target_subject=target_subject,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Helpers (entrenamiento/validacion — identicos a v15)
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


def augment_signals(x_signal: torch.Tensor) -> torch.Tensor:
    """Augmentation EMG: noise + amplitude scaling + channel dropout."""
    x = x_signal.clone()
    device = x.device
    B = x.size(0)
    noise_std = np.random.uniform(0.0, 0.05)
    if noise_std > 0.0:
        x = x + torch.randn_like(x) * noise_std
    scales = torch.empty(B, 1, 1, 1, device=device).uniform_(0.8, 1.2)
    x = x * scales
    mask = torch.bernoulli(torch.full((B, 1, 1, 13), 0.9, device=device))
    x = x * mask
    return x


def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """Entrena una epoca con augmentation de senal."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for signals, features, labels in pbar:
        signals = signals.to(device)
        features = features.to(device)
        labels = labels.to(device)
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
    """Valida el modelo sin augmentation."""
    model.eval()
    total_loss = 0.0
    all_preds: list = []
    all_labels: list = []
    with torch.no_grad():
        for signals, features, labels in tqdm(loader, desc="Val", leave=False):
            signals, features, labels = (
                signals.to(device), features.to(device), labels.to(device)
            )
            outputs = model(signals, features)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), accuracy, bal_acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entrena CyborgHybrid con features expandidas (132D) en S1."""
    print("=" * 60)
    print("ENTRENAMIENTO v17: FEATURES MANUALES EXPANDIDAS")
    print(f"Features: 48D -> {FEATURES_DIM}D (11 features x 12 canales)")
    print("Nuevas: MAV, VAR, WAMP, AR1-AR4")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

    # Datos S1 con features extendidas
    print("\n[1/5] Cargando datos S1 (features 132D)...")
    train_loader, val_loader, test_loader = create_data_loaders_v17()
    print(f"  Train: {len(train_loader.dataset)} muestras")
    print(f"  Val:   {len(val_loader.dataset)} muestras")
    print(f"  Test:  {len(test_loader.dataset)} muestras")

    # Verificar dimension de features
    sample_signals, sample_features, _ = next(iter(train_loader))
    print(f"  Features shape: {sample_features.shape}  (esperado: [64, {FEATURES_DIM}])")
    assert sample_features.shape[1] == FEATURES_DIM, \
        f"Feature dim incorrecta: {sample_features.shape[1]} != {FEATURES_DIM}"

    # Modelo con features_dim=132
    print(f"\n[2/5] Creando CyborgHybrid (manual_features_dim={FEATURES_DIM})...")
    model = CyborgHybrid(manual_features_dim=FEATURES_DIM).to(device)
    model.get_model_info()

    # Loss y optimizer
    print("\n[3/5] Configurando optimizer y loss...")
    class_weights = calculate_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Entrenamiento
    print(f"\n[4/5] Iniciando entrenamiento (max {EPOCHS} epochs, patience=15)...")
    patience_es = 15
    best_val_acc = 0.0
    patience_counter = 0
    checkpoint_path = CHECKPOINTS_PATH / "best_model_v17_features.pth"

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
                    "features_dim": FEATURES_DIM,
                    "features": ["RMS", "ZC", "SSC", "WL", "MAV", "VAR", "WAMP",
                                 "AR1", "AR2", "AR3", "AR4"],
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
    print("RESULTADOS FINALES - v17 FEATURES EXPANDIDAS")
    print(f"{'='*60}")
    print(f"Features: {FEATURES_DIM}D (11 x 12 canales)")
    print(f"Test Accuracy:          {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_bal_acc*100:.2f}%")

    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print("       " + "".join(f"Pred{i} " for i in range(7)))
    for i in range(7):
        print(f"Real{i}: " + "".join(f"{cm[i,j]:5d} " for j in range(7)))

    print("\nAccuracy por clase:")
    for i in range(7):
        acc_i = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        print(f"  Clase {i}: {acc_i*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")

    print(f"\n{'='*60}")
    print("COMPARACION CON BASELINES")
    print(f"{'='*60}")
    print(f"v17 Features 132D (S1): {test_acc*100:.2f}% (BalAcc: {test_bal_acc*100:.2f}%)")
    print(f"v16 Ensemble v10+v15:   72.41%")
    print(f"v10 Cyborg (S1):        71.32% (BalAcc: 41.92%)")
    print(f"v15 Augmentation (S1):  70.74% (BalAcc: 45.16%)")
    print(f"Objetivo paper:         79.72%")

    if test_acc > 0.7972:
        print(f"\n*** OBJETIVO ALCANZADO! (+{(test_acc - 0.7972)*100:.2f}%) ***")
    elif test_acc > 0.7241:
        print(f"\n*** SUPERA ENSEMBLE v16! (+{(test_acc - 0.7241)*100:.2f}%) ***")
        print(f"Gap al objetivo: {(0.7972 - test_acc)*100:.2f}%")
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
