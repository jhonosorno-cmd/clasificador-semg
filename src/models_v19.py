"""
Modelo v19: TCNCyborgHybrid
Reemplaza la rama CNN+BiLSTM por una TCN con convoluciones dilatadas no-causales.

Motivacion:
  La CNN+BiLSTM opera en dos fases separadas: extraccion local (CNN con MaxPool)
  seguida de modelado temporal global (LSTM). La TCN integra ambas en un solo
  pipeline de convoluciones dilatadas con campo receptivo exponencial.

  Campo receptivo de la TCN (no-causal, 2 convs/bloque, k=7):
    Bloque 1 (d=1):  2*(7-1)*1*2  =  24 frames
    Bloque 2 (d=2):  2*(7-1)*2*2  =  48 frames
    Bloque 3 (d=4):  2*(7-1)*4*2  =  96 frames
    Bloque 4 (d=8):  2*(7-1)*8*2  = 192 frames
    RF total = 1 + 24 + 48 + 96 + 192 = 361 samples  (ventana = 125)

  Ventajas sobre BiLSTM:
    - Sin cuello de botella secuencial (paralelizable)
    - Gradientes mas estables (no vanishing/exploding en secuencias largas)
    - Campo receptivo controlable y determinista
    - Mismo conteo de parametros, mejor utilizacion

Interfaz identica a CyborgHybrid:
    forward(x_signal, x_features) -> logits
    x_signal: (B, 1, 125, 13)
    x_features: (B, 48)
    logits: (B, 7)
"""

import torch
import torch.nn as nn
from src.config import SEED

torch.manual_seed(SEED)

KERNEL_SIZE = 7
TCN_CHANNELS = [64, 128, 256, 256]   # canales por bloque
DILATIONS    = [1,   2,   4,   8]    # dilataciones por bloque
TCN_DROPOUT  = 0.2


class TCNBlock(nn.Module):
    """
    Bloque residual TCN: dos Conv1d dilatadas no-causales + skip connection.

    Padding simetrico ('same') para preservar la longitud temporal.
    Si in_channels != out_channels, la conexion residual usa una Conv1d 1x1.

    Args:
        in_channels: Canales de entrada.
        out_channels: Canales de salida.
        kernel_size: Tamano del kernel de convolucion.
        dilation: Factor de dilatacion.
        dropout: Probabilidad de dropout entre capas.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Padding simetrico para output = input en longitud temporal
        pad = dilation * (kernel_size - 1) // 2

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      dilation=dilation, padding=pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      dilation=dilation, padding=pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Proyeccion residual (1x1) si los canales cambian
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, T)
        Returns:
            (B, out_channels, T)
        """
        return self.relu(self.conv_block(x) + self.residual(x))


class TCNCyborgHybrid(nn.Module):
    """
    Arquitectura Cyborg con TCN en la rama temporal.

    Ramas:
      - TCN (4 bloques residuales dilatados): procesa la senal EMG -> 256D
      - MLP: procesa features manuales (48D) -> 64D
    Fusion: concat(320D) -> clasificador de 7 clases.

    Args:
        emg_channels: Numero de canales EMG (default 13 = 12 + RMS).
        manual_features_dim: Dimension del vector de features manuales.
        num_classes: Numero de clases de salida.
        dropout_prob: Dropout en MLP y clasificador.
        tcn_channels: Lista de canales de salida por bloque TCN.
        dilations: Lista de dilataciones por bloque TCN.
        kernel_size: Kernel de las Conv1d del TCN.
        tcn_dropout: Dropout dentro de los bloques TCN.
    """

    def __init__(
        self,
        emg_channels: int = 13,
        manual_features_dim: int = 48,
        num_classes: int = 7,
        dropout_prob: float = 0.3,
        tcn_channels: list = None,
        dilations: list = None,
        kernel_size: int = KERNEL_SIZE,
        tcn_dropout: float = TCN_DROPOUT,
    ):
        super().__init__()
        tcn_channels = tcn_channels or TCN_CHANNELS
        dilations     = dilations     or DILATIONS

        assert len(tcn_channels) == len(dilations), \
            "tcn_channels y dilations deben tener la misma longitud"

        self.emg_channels        = emg_channels
        self.manual_features_dim = manual_features_dim
        self.num_classes         = num_classes

        # === RAMA TCN ===
        blocks = []
        in_ch = emg_channels
        for out_ch, dil in zip(tcn_channels, dilations):
            blocks.append(TCNBlock(in_ch, out_ch, kernel_size, dil, tcn_dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)
        tcn_out_dim = tcn_channels[-1]  # 256

        # === RAMA MLP (identica a CyborgHybrid) ===
        self.feature_mlp = nn.Sequential(
            nn.Linear(manual_features_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        mlp_out_dim = 64

        # === FUSION + CLASIFICADOR ===
        fusion_dim = tcn_out_dim + mlp_out_dim  # 320
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x_signal: torch.Tensor,
        x_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_signal:   (B, 1, 125, 13) - ventanas de senal EMG
            x_features: (B, 48)         - features manuales

        Returns:
            logits: (B, 7)
        """
        # (B, 1, 125, 13) -> (B, 13, 125)
        x = x_signal.squeeze(1).permute(0, 2, 1)

        # TCN: (B, 13, 125) -> (B, 256, 125)
        x = self.tcn(x)

        # Global average pooling temporal: (B, 256, 125) -> (B, 256)
        tcn_features = x.mean(dim=-1)

        # Rama MLP: (B, 48) -> (B, 64)
        mlp_features = self.feature_mlp(x_features)

        # Fusion y clasificacion
        fused  = torch.cat([tcn_features, mlp_features], dim=1)  # (B, 320)
        logits = self.classifier(fused)                           # (B, 7)
        return logits
