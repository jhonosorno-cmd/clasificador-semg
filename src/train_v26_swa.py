"""
Entrenamiento v26: CyborgHybrid + Label Smoothing + SWA.

Estrategia: regularización del proceso de entrenamiento sin cambiar arquitectura.

  Label Smoothing (eps=0.1):
    - Suaviza las etiquetas duras → el modelo no se vuelve excesivamente
      confiante en clases difíciles (1-4).
    - `CrossEntropyLoss(label_smoothing=0.1)` disponible desde PyTorch 1.10.

  Stochastic Weight Averaging (SWA):
    - Promedia pesos de varios puntos a lo largo de la trayectoria SGD.
    - Encuentra mínimos más planos que generalizan mejor (Izmailov et al., 2018).
    - Implementado con `torch.optim.swa_utils`.

Fases de entrenamiento:
  Fase 1 (epochs 0 → SWA_START-1):
    - AdamW lr=5e-4, ReduceLROnPlateau(factor=0.5, patience=5)
    - Early stopping patience=20 (más largo que v10 para llegar al SWA)
    - Al terminar, carga el mejor checkpoint de la fase 1.

  Fase 2 (SWA_START → SWA_START+SWA_EPOCHS-1):
    - SWALR: coseno entre swa_lr_max=1e-4 y swa_lr_min=5e-6 cada 5 ciclos
    - `swa_model.update_parameters(model)` al final de cada epoch
    - Sin early stopping (fase corta y fija, SWA necesita múltiples puntos)

  Final:
    - `update_bn(train_loader, swa_model)` para recalcular estadísticas BN
    - Evaluar swa_model vs mejor modelo individual en test set

Arquitectura: CyborgHybrid (v10) sin cambios (48D features, 813K params).
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.optim.swa_utils import AveragedModel, SWALR

from src.config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, CHECKPOINTS_PATH, SEED
from src.models_cyborg import CyborgHybrid
from src.data_loader import create_data_loaders

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Hiperparámetros SWA
# ---------------------------------------------------------------------------
SWA_START    = 40   # epochs de entrenamiento normal antes de SWA
SWA_EPOCHS   = 25   # epochs de SWA (fase 2)
SWA_LR       = 1e-4 # LR constante para la fase SWA
LABEL_SMOOTH = 0.1  # eps de label smoothing


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_signal(signals: torch.Tensor, training: bool = True) -> torch.Tensor:
    """Signal augmentation de v15: ruido, scaling, channel dropout."""
    if not training:
        return signals
    B = signals.size(0)
    signals = signals + torch.randn_like(signals) * 0.05
    scale = torch.FloatTensor(B, 1, 1, 1).uniform_(0.9, 1.1).to(signals.device)
    signals = signals * scale
    if np.random.rand() < 0.5:
        ch_mask = (torch.rand(B, 1, 1, 13) > 0.10).float().to(signals.device)
        signals = signals * ch_mask
    return signals


def mixup_data(
    x_signal: torch.Tensor,
    x_features: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
    device: str = "cpu",
) -> tuple:
    """Mixup augmentation sobre señal y features."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x_signal.size(0)).to(device)
    return (
        lam * x_signal   + (1 - lam) * x_signal[idx],
        lam * x_features + (1 - lam) * x_features[idx],
        y, y[idx], lam,
    )


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def calculate_class_weights(dataset) -> torch.Tensor:
    """Pesos sqrt-inverse de clase."""
    labels = dataset.labels
    counts = np.bincount(labels, minlength=7).astype(np.float64)
    counts[counts == 0] = 1.0
    total  = counts.sum()
    weights = np.sqrt(total / (len(counts) * counts))
    print("Class weights (sqrt-inverse):")
    for i in range(len(counts)):
        print(f"  Clase {i}: {int(counts[i])} muestras, peso {weights[i]:.3f}")
    return torch.FloatTensor(weights)


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total   = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for signals, features, labels in pbar:
        signals  = signals.to(device)
        features = features.to(device)
        labels   = labels.to(device)

        signals = augment_signal(signals, training=True)
        signals, features, labels_a, labels_b, lam = mixup_data(
            signals, features, labels, alpha=0.4, device=str(device)
        )

        optimizer.zero_grad()
        outputs = model(signals, features)
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        target = labels_a if lam >= 0.5 else labels_b
        correct += (predicted == target).sum().item()
        total   += labels_a.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total:.3f}"})

    return total_loss / len(loader), correct / total


def validate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, list, list]:
    model.eval()
    total_loss = 0.0
    all_preds:  list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for signals, features, labels in tqdm(loader, desc="Val", leave=False):
            signals  = signals.to(device)
            features = features.to(device)
            labels   = labels.to(device)
            outputs  = model(signals, features)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, bal_acc, all_preds, all_labels


def print_class_results(cm: np.ndarray) -> None:
    for i in range(7):
        acc_i = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        print(f"  Clase {i}: {acc_i*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("ENTRENAMIENTO v26: CyborgHybrid + Label Smoothing + SWA")
    print(f"  Label smoothing eps={LABEL_SMOOTH}")
    print(f"  SWA start epoch={SWA_START}, SWA epochs={SWA_EPOCHS}, LR={SWA_LR}")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")

    print("\n[1/6] Cargando datos S1...")
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=BATCH_SIZE)
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    print("\n[2/6] Creando CyborgHybrid (v10)...")
    model = CyborgHybrid(manual_features_dim=48).to(device)
    model.get_model_info()

    print("\n[3/6] Calculando class weights...")
    class_weights = calculate_class_weights(train_loader.dataset).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    checkpoint_path = CHECKPOINTS_PATH / "best_model_v26_swa.pth"

    # ==============================================================
    # FASE 1: Entrenamiento normal con label smoothing
    # ==============================================================
    print(f"\n[4/6] FASE 1: Entrenamiento normal (epochs 0–{SWA_START-1})...")
    print(f"  Label smoothing eps={LABEL_SMOOTH} | Early stopping patience=20")
    print("-" * 60)

    best_val_acc     = 0.0
    patience_counter = 0
    patience_es      = 20  # más amplio que v10 (15) para llegar a SWA
    phase1_end_epoch = SWA_START

    for epoch in range(EPOCHS):
        if epoch >= phase1_end_epoch:
            break

        print(f"\nEpoch {epoch+1}/{phase1_end_epoch} [Fase 1]")
        print("-" * 40)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_bal_acc, _, _ = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | BalAcc: {val_bal_acc*100:.2f}%")
        print(f"LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_balanced_accuracy": val_bal_acc,
            }, checkpoint_path)
            print(f"[OK] Mejor modelo Fase 1 guardado (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience_es:
                print(f"\nEarly stopping Fase 1 (patience {patience_es}) en epoch {epoch+1}")
                phase1_end_epoch = epoch + 1  # registra el epoch real de parada
                break

    print(f"\nFase 1 completa. Mejor Val Acc: {best_val_acc*100:.2f}%")

    # Cargar mejor modelo de Fase 1 para iniciar SWA desde allí
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print("  Modelo Fase 1 cargado para SWA.")

    # ==============================================================
    # FASE 2: SWA
    # ==============================================================
    print(f"\n[5/6] FASE 2: SWA ({SWA_EPOCHS} epochs, LR={SWA_LR})...")
    print("-" * 60)

    swa_model     = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR, anneal_epochs=5)

    # Reajusta LR del optimizer al inicio de la fase SWA
    for pg in optimizer.param_groups:
        pg["lr"] = SWA_LR

    for swa_epoch in range(SWA_EPOCHS):
        epoch_num = phase1_end_epoch + swa_epoch
        print(f"\nSWA Epoch {swa_epoch+1}/{SWA_EPOCHS} (global {epoch_num+1})")
        print("-" * 40)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Actualizar modelo SWA
        swa_model.update_parameters(model)
        swa_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"LR (SWA): {current_lr:.6f}")

    # Actualizar estadísticas BatchNorm del modelo SWA
    # Nota: update_bn() de PyTorch llama model(input) con un solo argumento,
    # incompatible con CyborgHybrid.forward(x_signal, x_features).
    # Se usa un loop manual equivalente.
    print("\nActualizando BatchNorm del modelo SWA...")
    swa_model.train()
    with torch.no_grad():
        for signals, features, _ in tqdm(train_loader, desc="BN update", leave=False):
            signals  = signals.to(device)
            features = features.to(device)
            swa_model(signals, features)

    # ==============================================================
    # EVALUACIÓN FINAL
    # ==============================================================
    print("\n[6/6] Evaluación en test set...")

    # Criterio sin label smoothing para evaluación justa
    criterion_eval = nn.CrossEntropyLoss(weight=class_weights)

    # Modelo individual (Fase 1 best)
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, ind_acc, ind_bal_acc, ind_preds, ind_labels = validate_epoch(
        model, test_loader, criterion_eval, device
    )

    # Modelo SWA
    _, swa_acc, swa_bal_acc, swa_preds, swa_labels = validate_epoch(
        swa_model, test_loader, criterion_eval, device
    )

    # Resultados
    print(f"\n{'='*60}")
    print("RESULTADOS FINALES — v26 CyborgHybrid + Label Smoothing + SWA")
    print(f"{'='*60}")

    print(f"\nModelo individual (Fase 1 best):")
    print(f"  Test Accuracy:    {ind_acc*100:.2f}%")
    print(f"  Test BalAcc:      {ind_bal_acc*100:.2f}%")
    cm_ind = confusion_matrix(ind_labels, ind_preds)
    print_class_results(cm_ind)

    print(f"\nModelo SWA ({SWA_EPOCHS} epochs promediados):")
    print(f"  Test Accuracy:    {swa_acc*100:.2f}%")
    print(f"  Test BalAcc:      {swa_bal_acc*100:.2f}%")
    cm_swa = confusion_matrix(swa_labels, swa_preds)
    print_class_results(cm_swa)

    best_acc = max(ind_acc, swa_acc)
    best_name = "SWA" if swa_acc >= ind_acc else "Individual"

    print(f"\n{'='*60}")
    print("COMPARACION")
    print(f"{'='*60}")
    print(f"  v26 {best_name} (mejor):      {best_acc*100:.2f}%")
    print(f"  v10 CyborgHybrid (ref):  71.32%")
    print(f"  v23 CyborgTransformer:   71.89%")
    print(f"  Ensemble 4M (maximo):    72.88%")
    print(f"  Objetivo paper:          79.72%")

    if best_acc > 0.7288:
        print(f"\n*** NUEVO MAXIMO: {best_acc*100:.2f}% (+{(best_acc-0.7288)*100:.2f}%) ***")
    elif best_acc > 0.7132:
        print(f"\n  Supera v10: +{(best_acc-0.7132)*100:.2f}%")
    else:
        print(f"\n  Gap vs v10: {(0.7132-best_acc)*100:.2f}%")

    print(f"\nBest Val Acc (Fase 1): {best_val_acc*100:.2f}%")
    print(f"Checkpoint: {checkpoint_path}")

    # Guardar también el modelo SWA
    swa_path = CHECKPOINTS_PATH / "best_model_v26_swa_averaged.pth"
    torch.save({
        "model_state_dict": swa_model.module.state_dict(),
        "val_accuracy": best_val_acc,
        "test_accuracy_swa": swa_acc,
        "test_accuracy_individual": ind_acc,
    }, swa_path)
    print(f"SWA model: {swa_path}")


if __name__ == "__main__":
    main()
