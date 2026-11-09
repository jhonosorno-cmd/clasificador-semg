"""
Entrenamiento v20: CyborgHybrid con features espectrales.

Features (108D = 9 features x 12 canales):
  Dominio temporal (4D/canal — identico a v10):
    RMS, ZC, SSC, WL
  Dominio frecuencial (5D/canal — NUEVO):
    MNF  — Mean Frequency: centroide ponderado de la PSD (Hz)
    MDF  — Median Frequency: frecuencia que divide la potencia al 50% (Hz)
    BP_L — Band Power 20–60  Hz (fraccion de potencia total)
    BP_M — Band Power 60–150 Hz (fraccion de potencia total)
    BP_H — Band Power 150–249 Hz (fraccion de potencia total)

Razonamiento:
  Las clases 1-4 tienen patrones de amplitud similares (gestos de dedos
  adyacentes) pero distintas frecuencias de reclutamiento motor.
  MNF/MDF discriminan bien entre gestos que difieren en la mezcla de
  unidades motoras rapidas (alta freq) y lentas (baja freq).
  Las potencias de banda capturan la estructura espectral completa.

Ventana: 125 samples a 500 Hz -> resolucion freq 4 Hz.
Welch nperseg=64 -> resolucion 500/64 ≈ 7.8 Hz (compromiso bias/varianza).
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.signal import welch
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from tqdm import tqdm

from src.config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, CHECKPOINTS_PATH, SEED
from src.models_cyborg import CyborgHybrid
from src.data_loader import NinaProDB3Dataset, create_data_loaders
from src.train_v15_augmentation import (
    augment_signals,
    calculate_class_weights,
)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

FS = 500          # Hz tras downsampling
WELCH_NPERSEG = 64
FREQ_MIN = 20.0
FREQ_MAX = 249.0
BAND_LOW  = (20.0,  60.0)
BAND_MID  = (60.0, 150.0)
BAND_HIGH = (150.0, 249.0)

N_SPECTRAL = 5    # features espectrales por canal
N_TEMPORAL = 4    # features temporales por canal (igual que v10)
N_CHANNELS = 12   # canales EMG (sin RMS)
FEATURES_DIM = (N_TEMPORAL + N_SPECTRAL) * N_CHANNELS  # 108


# ---------------------------------------------------------------------------
# Dataset con features espectrales
# ---------------------------------------------------------------------------

class NinaProSpectralDataset(NinaProDB3Dataset):
    """
    Subclase de NinaProDB3Dataset que añade 5 features espectrales por canal.

    El vector de features resultante es (108,):
      [RMS, ZC, SSC, WL, MNF, MDF, BP_L, BP_M, BP_H] x 12 canales
    """

    def _extract_manual_features(self, emg_window: np.ndarray) -> np.ndarray:
        """
        Extrae 9 features por canal (4 temporales + 5 espectrales).

        Args:
            emg_window: (125, 12) - ventana EMG de 12 canales normalizados

        Returns:
            features: (108,) float32
        """
        n_channels = emg_window.shape[1]
        n_features = N_TEMPORAL + N_SPECTRAL  # 9
        features = np.zeros(n_channels * n_features, dtype=np.float32)

        for ch in range(n_channels):
            x = emg_window[:, ch]
            base = ch * n_features

            # --- Features temporales (mismas que v10) ---
            dx = np.diff(x)
            features[base + 0] = np.sqrt(np.mean(x ** 2))                        # RMS
            features[base + 1] = float(np.sum(np.diff(np.sign(x)) != 0))         # ZC
            features[base + 2] = float(np.sum(np.diff(np.sign(dx)) != 0))        # SSC
            features[base + 3] = np.sum(np.abs(dx))                              # WL

            # --- Features espectrales ---
            freqs, psd = welch(x, fs=FS, nperseg=WELCH_NPERSEG,
                               noverlap=WELCH_NPERSEG // 2)

            # Restringir al rango EMG util [20, 249] Hz
            mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
            f = freqs[mask]
            p = psd[mask]
            total_power = np.sum(p) + 1e-12

            # MNF — Mean Frequency
            mnf = np.sum(f * p) / total_power

            # MDF — Median Frequency
            cumpower = np.cumsum(p)
            mdf_idx = np.searchsorted(cumpower, total_power / 2)
            mdf = f[min(mdf_idx, len(f) - 1)]

            # Band powers (fraccion de potencia total)
            bp_l = np.sum(p[(f >= BAND_LOW[0])  & (f < BAND_LOW[1])])  / total_power
            bp_m = np.sum(p[(f >= BAND_MID[0])  & (f < BAND_MID[1])])  / total_power
            bp_h = np.sum(p[(f >= BAND_HIGH[0]) & (f <= BAND_HIGH[1])]) / total_power

            features[base + 4] = mnf
            features[base + 5] = mdf
            features[base + 6] = bp_l
            features[base + 7] = bp_m
            features[base + 8] = bp_h

        return features


def create_data_loaders_spectral(batch_size: int = BATCH_SIZE):
    """
    Crea DataLoaders con NinaProSpectralDataset (108D features).

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = NinaProSpectralDataset(
        subjects=[1], reps_to_use=[1, 2, 3, 4], mode="train", target_subject=1
    )
    val_dataset = NinaProSpectralDataset(
        subjects=[1], reps_to_use=[5], mode="val", target_subject=1
    )
    test_dataset = NinaProSpectralDataset(
        subjects=[1], reps_to_use=[6], mode="test", target_subject=1
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Train epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for signals, features, labels in pbar:
        signals  = signals.to(device)
        features = features.to(device)
        labels   = labels.to(device)
        signals  = augment_signals(signals)
        optimizer.zero_grad()
        outputs = model(signals, features)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total:.3f}"})
    return total_loss / len(loader), correct / total


def validate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, float, list, list]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, features, labels in tqdm(loader, desc="Val", leave=False):
            signals, features, labels = (
                signals.to(device), features.to(device), labels.to(device)
            )
            outputs = model(signals, features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    acc = correct / total
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, bal_acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entrena CyborgHybrid con features espectrales (108D) en S1."""
    print("=" * 60)
    print("ENTRENAMIENTO v20: CYBORG + SPECTRAL FEATURES (108D)")
    print("Features: [RMS, ZC, SSC, WL, MNF, MDF, BP_L, BP_M, BP_H] x 12")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

    # Datos S1
    print(f"\n[1/5] Cargando datos S1 con features espectrales ({FEATURES_DIM}D)...")
    train_loader, val_loader, test_loader = create_data_loaders_spectral(batch_size=BATCH_SIZE)
    print(f"  Train: {len(train_loader.dataset)} muestras")
    print(f"  Val:   {len(val_loader.dataset)} muestras")
    print(f"  Test:  {len(test_loader.dataset)} muestras")

    # Modelo CyborgHybrid con features_dim=108
    print(f"\n[2/5] Creando CyborgHybrid (manual_features_dim={FEATURES_DIM})...")
    model = CyborgHybrid(manual_features_dim=FEATURES_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parametros: {total_params:,}")

    # Loss y optimizer
    print("\n[3/5] Configurando optimizer y loss...")
    class_weights = calculate_class_weights(train_loader.dataset).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Entrenamiento
    patience_es      = 15
    best_val_acc     = 0.0
    patience_counter = 0
    checkpoint_path  = CHECKPOINTS_PATH / "best_model_v20_spectral.pth"

    print(f"\n[4/5] Iniciando entrenamiento (max {EPOCHS} epochs, patience={patience_es})...")

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
        print(f"LR: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "val_balanced_accuracy": val_bal_acc,
                    "architecture": "CyborgHybrid",
                    "manual_features_dim": FEATURES_DIM,
                    "features_per_channel": [
                        "RMS", "ZC", "SSC", "WL",
                        "MNF", "MDF", "BP_Low", "BP_Mid", "BP_High"
                    ],
                },
                checkpoint_path,
            )
            print(f"  [OK] Checkpoint guardado (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  [--] Sin mejora ({patience_counter}/{patience_es})")
            if patience_counter >= patience_es:
                print(f"\nEarly stopping activado (patience={patience_es})")
                break

    # Evaluacion final
    print("\n[5/5] Evaluando en test set (S1)...")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_acc, test_bal_acc, test_preds, test_labels = validate_epoch(
        model, test_loader, criterion, device
    )

    cm = confusion_matrix(test_labels, test_preds)

    print(f"\n{'='*60}")
    print("RESULTADOS FINALES - v20 Spectral (108D)")
    print(f"{'='*60}")
    print(f"Test Accuracy:          {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_bal_acc*100:.2f}%")

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
    print(f"v20 Spectral 108D (S1):        {test_acc*100:.2f}% (BalAcc: {test_bal_acc*100:.2f}%)")
    print(f"v17 132D time-domain (S1):     71.89% (BalAcc: 44.10%)")
    print(f"v10 Cyborg 48D (S1):           71.32% (BalAcc: 41.92%)")
    print(f"Ensemble 4M (v10+v15+v17+v18): 72.78%")
    print(f"Objetivo paper:                79.72%")

    if test_acc > 0.7972:
        print(f"\n*** OBJETIVO ALCANZADO! (+{(test_acc - 0.7972)*100:.2f}%) ***")
    elif test_acc > 0.7278:
        print(f"\n*** SUPERA ENSEMBLE 4M! (+{(test_acc - 0.7278)*100:.2f}%) ***")
    elif test_acc > 0.7132:
        print(f"\n*** SUPERA v10! (+{(test_acc - 0.7132)*100:.2f}%) ***")
        print(f"Gap al objetivo: {(0.7972 - test_acc)*100:.2f}%")
    else:
        print(f"\nDebajo de v10 por {(0.7132 - test_acc)*100:.2f}%")

    print(f"\nMejor Val Acc: {best_val_acc*100:.2f}%")
    print(f"Checkpoint:    {checkpoint_path}")


if __name__ == "__main__":
    main()
