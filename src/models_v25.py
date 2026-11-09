"""
CyborgSTFT (v25): CNN2D sobre espectrograma STFT de señal EMG.

La STFT transforma cada ventana temporal (125 samples) en una imagen
tiempo-frecuencia 2D. Con nperseg=25, hop=5, nfft=32:
  - Frecuencias: 17 bins  (0–250 Hz, resolución ~15.6 Hz)
  - Tramas:      21 columnas
  - Canales:     12 EMG (sin el canal RMS sintético)

El log-espectrograma de cada canal actúa como un "canal de imagen" para
la CNN2D, que procesa una entrada (B, 12, 17, 21).

Flujo de la rama CNN2D:
  (B, 1, 125, 13)
  → STFT x12 canales, log-magnitud
  → (B, 12, 17, 19)
  → CNN2D [Conv(32) → Pool → Conv(64) → Pool]
  → (B, 64, 4, 4)  → Flatten → (B, 1024)
  → Linear(256) + BN + ReLU + Dropout
  → (B, 256)

Fusión:
  (B, 256) + MLP(48D→64D) → (B, 320) → Clasificador → (B, 7)
"""

import math
import torch
import torch.nn as nn
import numpy as np
from src.config import SEED

torch.manual_seed(SEED)


class CyborgSTFT(nn.Module):
    """
    Arquitectura Cyborg con representación STFT + CNN2D.

    La STFT se calcula dentro del forward() con ventana Hann registrada
    como buffer (se mueve automáticamente al dispositivo correcto).

    Args:
        manual_features_dim: dimensión del vector de features manuales (48)
        num_classes:          número de clases de salida (7)
        nperseg:              tamaño de la ventana STFT (25 samples → 50 ms)
        hop_length:           stride de la STFT (5 samples → 10 ms)
        n_fft:                tamaño FFT — n_freq = n_fft//2 + 1 (32 → 17)
        dropout_cnn:          dropout en la rama CNN2D
        dropout_mlp:          dropout en la rama MLP y el clasificador
    """

    # Parámetros STFT fijos (calculados con nperseg=25, hop=5, nfft=32, L=125)
    #   n_freq  = nfft//2 + 1 = 17
    #   n_time  = (L - nfft)//hop + 1 = (125-32)//5 + 1 = 19  (PyTorch usa nfft, no win_length)
    N_FREQ  = 17
    N_TIME  = 19

    def __init__(
        self,
        manual_features_dim: int = 48,
        num_classes: int = 7,
        nperseg: int = 25,
        hop_length: int = 5,
        n_fft: int = 32,
        dropout_cnn: float = 0.25,
        dropout_mlp: float = 0.30,
    ) -> None:
        super().__init__()

        self.manual_features_dim = manual_features_dim
        self.num_classes          = num_classes
        self.nperseg              = nperseg
        self.hop_length           = hop_length
        self.n_fft                = n_fft

        # Ventana Hann registrada como buffer (→ se mueve a GPU automáticamente)
        hann = torch.hann_window(nperseg)
        self.register_buffer("hann_window", hann)

        # ================================================================
        # RAMA CNN2D: procesa log-espectrograma (B, 12, 17, 21)
        # ================================================================
        # Conv1: (B, 12, 17, 21) → (B, 32, 17, 21) → pool → (B, 32, 8, 10)
        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_cnn),
        )
        # Conv2: (B, 32, 8, 10) → (B, 64, 8, 10) → pool → (B, 64, 4, 5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_cnn),
        )

        # Flatten + proyección a 256D
        # Tras 2 MaxPool(2,2):  17→8→4  y  19→9→4  → 64×4×4 = 1024
        cnn_flat_size = 64 * 4 * 4  # 1024
        self.cnn_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_flat_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_cnn),
        )
        self.cnn_output_size = 256

        # ================================================================
        # RAMA MLP: features manuales (48D → 64D)
        # ================================================================
        self.feature_mlp = nn.Sequential(
            nn.Linear(manual_features_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_mlp),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_mlp),
        )
        self.mlp_output_size = 64

        # ================================================================
        # FUSIÓN + CLASIFICADOR
        # ================================================================
        fusion_size = self.cnn_output_size + self.mlp_output_size  # 320
        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_mlp),
            nn.Linear(128, num_classes),
        )

    # ------------------------------------------------------------------
    # STFT helper
    # ------------------------------------------------------------------

    def _compute_log_stft(self, x_signal: torch.Tensor) -> torch.Tensor:
        """
        Calcula log-espectrograma de los 12 canales EMG.

        Args:
            x_signal: (B, 1, 125, 13) — señal EMG multicanal

        Returns:
            log_spec: (B, 12, 17, 21) — log-magnitud del espectrograma
        """
        B = x_signal.size(0)

        # Extraer sólo los 12 canales EMG (no el canal RMS sintético)
        x = x_signal.squeeze(1)      # (B, 125, 13)
        x = x[:, :, :12]             # (B, 125, 12)  — canales EMG
        x = x.permute(0, 2, 1)       # (B, 12, 125)

        # Reshape para stft en batch: (B*12, 125)
        x_flat = x.reshape(B * 12, 125)

        # STFT — return_complex=True → (B*12, n_freq, n_time)
        stft = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.nperseg,
            window=self.hann_window,
            center=False,
            return_complex=True,
        )  # (B*12, 17, 21)

        mag = torch.abs(stft)                      # (B*12, 17, 21)
        log_mag = torch.log(mag + 1e-6)            # log-magnitud

        return log_mag.reshape(B, 12, self.N_FREQ, self.N_TIME)  # (B, 12, 17, 21)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x_signal: torch.Tensor, x_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_signal:   (B, 1, 125, 13) — señal EMG multicanal
            x_features: (B, 48) — features manuales

        Returns:
            logits: (B, 7)
        """
        # --- Rama CNN2D (STFT) ---
        spec = self._compute_log_stft(x_signal)   # (B, 12, 17, 21)
        x = self.conv1(spec)                       # (B, 32, 8, 10)
        x = self.conv2(x)                          # (B, 64, 4, 5)
        cnn_feat = self.cnn_proj(x)                # (B, 256)

        # --- Rama MLP ---
        mlp_feat = self.feature_mlp(x_features)   # (B, 64)

        # --- Fusión y clasificación ---
        fused = torch.cat([cnn_feat, mlp_feat], dim=1)  # (B, 320)
        return self.classifier(fused)                    # (B, 7)

    def get_model_info(self) -> None:
        """Imprime información del modelo."""
        total = sum(p.numel() for p in self.parameters())
        for tag, prefix in [
            ("CNN2D",        "conv"),
            ("Proyección",   "cnn_proj"),
            ("MLP features", "feature_mlp"),
            ("Clasificador", "classifier"),
        ]:
            n = sum(p.numel() for name, p in self.named_parameters() if prefix in name)
            print(f"  {tag}: {n:,} ({n/total*100:.1f}%)")
        print(f"  TOTAL: {total:,}")
        print(f"  STFT: nperseg={self.nperseg}, hop={self.hop_length}, "
              f"n_fft={self.n_fft} -> ({self.N_FREQ} freq x {self.N_TIME} time)")
