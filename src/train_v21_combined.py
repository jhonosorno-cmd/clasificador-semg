"""
Entrenamiento v21: CyborgHybrid con features combinadas (192D).

Features (192D = 16 features x 12 canales):
  Dominio temporal — v17 (11D/canal):
    RMS, ZC, SSC, WL, MAV, VAR, WAMP, AR1, AR2, AR3, AR4
  Dominio frecuencial — v20 (5D/canal):
    MNF  — Mean Frequency normalizada (÷249 → [0,1])
    MDF  — Median Frequency normalizada (÷249 → [0,1])
    BP_L — Band Power 20–60  Hz (fraccion)
    BP_M — Band Power 60–150 Hz (fraccion)
    BP_H — Band Power 150–249 Hz (fraccion)

Mejora clave respecto a v20:
  MNF y MDF se dividen por 249 (Nyquist-1) para escalarlas al mismo
  rango que las features temporales normalizadas. Esto elimina la
  inestabilidad observada en v20 (val 42%→70% oscilante).

Hipotesis:
  Las features temporales (AR, WL, WAMP) capturan la dinamica de
  amplitud/forma de onda. Las frecuenciales (MNF, MDF, BP) capturan
  la mezcla de unidades motoras rapidas vs lentas. Combinadas deberan
  distinguir mejor los gestos de dedos similares (clases 1-4).
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.signal import welch
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from tqdm import tqdm

from src.config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, CHECKPOINTS_PATH, SEED
from src.models_cyborg import CyborgHybrid
from src.data_loader import NinaProDB3Dataset
from src.train_v15_augmentation import augment_signals, calculate_class_weights
from src.train_v17_features import _ar_yulewalker

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

AR_ORDER       = 4
WAMP_THRESHOLD = 0.5
FS             = 500
WELCH_NPERSEG  = 64
FREQ_MAX       = 249.0   # Nyquist - 1

N_TEMPORAL  = 11   # RMS,ZC,SSC,WL,MAV,VAR,WAMP,AR1-4
N_SPECTRAL  = 5    # MNF,MDF,BP_L,BP_M,BP_H
N_CHANNELS  = 12
FEATURES_DIM = (N_TEMPORAL + N_SPECTRAL) * N_CHANNELS   # 192


# ---------------------------------------------------------------------------
# Dataset combinado
# ---------------------------------------------------------------------------

class NinaProCombinedFeatures(NinaProDB3Dataset):
    """
    Dataset NinaPro con 16 features por canal (11 temporales + 5 espectrales).

    Vector de features resultante: (192,) float32.
    MNF y MDF normalizados a [0,1] (÷249) para igualar escala con el resto.
    """

    def _extract_manual_features(self, emg_window: np.ndarray) -> np.ndarray:
        """
        Extrae 16 features por canal: 11 temporales + 5 espectrales.

        Args:
            emg_window: (125, 12) - ventana EMG de 12 canales normalizados.

        Returns:
            features: (192,) float32.
        """
        n_channels = emg_window.shape[1]
        n_features = N_TEMPORAL + N_SPECTRAL  # 16
        features = np.zeros(n_channels * n_features, dtype=np.float32)

        for ch in range(n_channels):
            x = emg_window[:, ch]
            base = ch * n_features
            dx = np.diff(x)

            # --- Features temporales (0-10, identicas a v17) ---
            features[base + 0]  = np.sqrt(np.mean(x ** 2))                       # RMS
            features[base + 1]  = float(np.sum(np.diff(np.sign(x)) != 0))        # ZC
            features[base + 2]  = float(np.sum(np.diff(np.sign(dx)) != 0))       # SSC
            features[base + 3]  = np.sum(np.abs(dx))                             # WL
            features[base + 4]  = np.mean(np.abs(x))                             # MAV
            features[base + 5]  = np.var(x)                                      # VAR
            features[base + 6]  = float(np.sum(np.abs(dx) > WAMP_THRESHOLD))     # WAMP
            ar = _ar_yulewalker(x, order=AR_ORDER)
            features[base + 7:base + 11] = ar                                    # AR1-4

            # --- Features espectrales (11-15) ---
            freqs, psd = welch(x, fs=FS, nperseg=WELCH_NPERSEG,
                               noverlap=WELCH_NPERSEG // 2)

            mask = (freqs >= 20.0) & (freqs <= FREQ_MAX)
            f = freqs[mask]
            p = psd[mask]
            total_power = np.sum(p) + 1e-12

            # MNF y MDF normalizados a [0,1] (÷ FREQ_MAX)
            mnf = np.sum(f * p) / total_power
            cumpower = np.cumsum(p)
            mdf_idx = np.searchsorted(cumpower, total_power / 2)
            mdf = f[min(mdf_idx, len(f) - 1)]

            features[base + 11] = mnf / FREQ_MAX                                 # MNF norm
            features[base + 12] = mdf / FREQ_MAX                                 # MDF norm
            features[base + 13] = np.sum(p[(f >= 20)  & (f < 60)])  / total_power  # BP_L
            features[base + 14] = np.sum(p[(f >= 60)  & (f < 150)]) / total_power  # BP_M
            features[base + 15] = np.sum(p[(f >= 150) & (f <= FREQ_MAX)]) / total_power  # BP_H

        return features


def create_data_loaders_v21(batch_size: int = BATCH_SIZE):
    """Crea DataLoaders con NinaProCombinedFeatures (192D)."""
    train_ds = NinaProCombinedFeatures(
        subjects=[1], reps_to_use=[1, 2, 3, 4], mode="train", target_subject=1
    )
    val_ds = NinaProCombinedFeatures(
        subjects=[1], reps_to_use=[5], mode="val", target_subject=1
    )
    test_ds = NinaProCombinedFeatures(
        subjects=[1], reps_to_use=[6], mode="test", target_subject=1
    )
    make_loader = lambda ds, shuffle: torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        drop_last=shuffle
    )
    return make_loader(train_ds, True), make_loader(val_ds, False), make_loader(test_ds, False)


# ---------------------------------------------------------------------------
# Train / validate
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
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), correct / total, bal_acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entrena CyborgHybrid con features combinadas temporal+espectral (192D)."""
    print("=" * 60)
    print("ENTRENAMIENTO v21: CYBORG + COMBINED FEATURES (192D)")
    print("11 temporales (v17) + 5 espectrales (v20) x 12 canales")
    print("MNF/MDF normalizados ÷249 para estabilidad de entrenamiento")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/5] Cargando datos S1 ({FEATURES_DIM}D features)...")
    train_loader, val_loader, test_loader = create_data_loaders_v21(batch_size=BATCH_SIZE)
    print(f"  Train: {len(train_loader.dataset)} muestras")
    print(f"  Val:   {len(val_loader.dataset)} muestras")
    print(f"  Test:  {len(test_loader.dataset)} muestras")

    print(f"\n[2/5] Creando CyborgHybrid (manual_features_dim={FEATURES_DIM})...")
    model = CyborgHybrid(manual_features_dim=FEATURES_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parametros: {total_params:,}")

    print("\n[3/5] Configurando optimizer y loss...")
    class_weights = calculate_class_weights(train_loader.dataset).to(device)
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    patience_es      = 15
    best_val_acc     = 0.0
    patience_counter = 0
    checkpoint_path  = CHECKPOINTS_PATH / "best_model_v21_combined.pth"

    print(f"\n[4/5] Iniciando entrenamiento (max {EPOCHS} epochs, patience={patience_es})...")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_bal_acc, _, _ = validate_epoch(model, val_loader, criterion, device)
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
                        "RMS","ZC","SSC","WL","MAV","VAR","WAMP",
                        "AR1","AR2","AR3","AR4",
                        "MNF_norm","MDF_norm","BP_Low","BP_Mid","BP_High"
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
    print("RESULTADOS FINALES - v21 Combined (192D)")
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
    print(f"v21 Combined 192D (S1):        {test_acc*100:.2f}% (BalAcc: {test_bal_acc*100:.2f}%)")
    print(f"v20 Spectral  108D (S1):       70.11% (BalAcc: 43.49%)")
    print(f"v17 Temporal  132D (S1):       71.89% (BalAcc: 44.10%)")
    print(f"v10 Cyborg     48D (S1):       71.32% (BalAcc: 41.92%)")
    print(f"Ensemble 4M (v10+v15+v17+v18): 72.78%")
    print(f"Objetivo paper:                79.72%")

    if test_acc > 0.7972:
        print(f"\n*** OBJETIVO ALCANZADO! (+{(test_acc-0.7972)*100:.2f}%) ***")
    elif test_acc > 0.7278:
        print(f"\n*** SUPERA ENSEMBLE 4M! (+{(test_acc-0.7278)*100:.2f}%) ***")
    elif test_acc > 0.7189:
        print(f"\n*** SUPERA v17! (+{(test_acc-0.7189)*100:.2f}%) ***")
        print(f"Gap al objetivo: {(0.7972-test_acc)*100:.2f}%")
    elif test_acc > 0.7132:
        print(f"\n*** SUPERA v10! (+{(test_acc-0.7132)*100:.2f}%) ***")
    else:
        print(f"\nDebajo de v10 por {(0.7132-test_acc)*100:.2f}%")

    print(f"\nMejor Val Acc: {best_val_acc*100:.2f}%")
    print(f"Checkpoint:    {checkpoint_path}")


if __name__ == "__main__":
    main()
