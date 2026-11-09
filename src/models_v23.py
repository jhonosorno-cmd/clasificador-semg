"""
CyborgTransformer (v23): CNN + Transformer Encoder + Features Manuales.

Misma arquitectura que CyborgHybrid pero reemplaza el BiLSTM por un
Transformer Encoder. Hipótesis: la atención multi-cabeza captura
dependencias no locales en la secuencia temporal mejor que el LSTM
para ventanas cortas de 125 muestras.

Flujo de la rama CNN:
  (B, 1, 125, 13) → CNN(128) → (B, 128, 15)
  → proyección(128→256) → PosEncoding → TransformerEncoder(2L, 8H)
  → GlobalAvgPool → (B, 256)

Fusión con rama MLP:
  (B, 256) + (B, 64) → (B, 320) → Clasificador → (B, 7)
"""

import math
import torch
import torch.nn as nn
from src.config import SEED

torch.manual_seed(SEED)


class PositionalEncoding(nn.Module):
    """
    Codificación posicional sinusoidal estándar (Vaswani et al., 2017).

    Args:
        d_model: dimensión del embedding
        max_len: longitud máxima de secuencia soportada
        dropout: dropout aplicado a la salida
    """

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x + pe: (B, T, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CyborgTransformer(nn.Module):
    """
    Arquitectura Cyborg con Transformer Encoder en lugar de BiLSTM.

    La rama CNN extrae features locales con 3 capas convolucionales 1D.
    El Transformer Encoder captura dependencias globales entre los 15
    frames temporales resultantes. La rama MLP procesa features manuales
    clásicas de EMG (RMS, ZC, SSC, WL).

    Args:
        manual_features_dim: dimensión del vector de features manuales (48 o 132)
        num_classes: número de clases de salida
        d_model: dimensión del Transformer (256)
        nhead: número de cabezas de atención (8)
        num_transformer_layers: número de capas del encoder (2)
        dim_feedforward: dimensión interna del FFN del Transformer (512)
        dropout_cnn: dropout en las capas CNN
        dropout_transformer: dropout en el Transformer
        dropout_mlp: dropout en la rama MLP y clasificador
    """

    def __init__(
        self,
        manual_features_dim: int = 48,
        num_classes: int = 7,
        d_model: int = 256,
        nhead: int = 8,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 512,
        dropout_cnn: float = 0.15,
        dropout_transformer: float = 0.1,
        dropout_mlp: float = 0.3,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.manual_features_dim = manual_features_dim
        self.num_classes = num_classes

        # === RAMA CNN ===
        # Input: (B, 13, 125) — 12 canales EMG + 1 canal RMS
        self.conv1 = nn.Sequential(
            nn.Conv1d(13, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),       # 125 → 62
            nn.Dropout(0.10),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),       # 62 → 31
            nn.Dropout(dropout_cnn),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),       # 31 → 15
            nn.Dropout(dropout_cnn),
        )

        # Proyección 128 → d_model (para alimentar el Transformer)
        self.input_proj = nn.Linear(128, d_model)

        # Codificación posicional
        self.pos_encoding = PositionalEncoding(
            d_model=d_model, max_len=32, dropout=dropout_transformer
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_transformer,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )
        self.transformer_norm = nn.LayerNorm(d_model)
        self.cnn_output_size = d_model  # 256

        # === RAMA MLP: Features Manuales ===
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

        # === FUSIÓN + CLASIFICADOR ===
        # concat: CNN (256) + MLP (64) = 320  — mismo que CyborgHybrid
        fusion_size = self.cnn_output_size + self.mlp_output_size
        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_mlp),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_signal: torch.Tensor, x_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_signal:   (B, 1, 125, 13) — señal EMG multicanal
            x_features: (B, F) — features manuales (F=48 o 132)

        Returns:
            logits: (B, 7)
        """
        # --- Rama CNN ---
        x = x_signal.squeeze(1).permute(0, 2, 1)  # (B, 13, 125)
        x = self.conv1(x)   # (B, 64, 62)
        x = self.conv2(x)   # (B, 128, 31)
        x = self.conv3(x)   # (B, 128, 15)
        x = x.permute(0, 2, 1)          # (B, 15, 128)

        # Proyección y codificación posicional
        x = self.input_proj(x)           # (B, 15, 256)
        x = self.pos_encoding(x)         # (B, 15, 256)

        # Transformer Encoder
        x = self.transformer(x)          # (B, 15, 256)
        x = self.transformer_norm(x)     # (B, 15, 256)

        # Global Average Pooling sobre la dimensión temporal
        cnn_feat = x.mean(dim=1)         # (B, 256)

        # --- Rama MLP ---
        mlp_feat = self.feature_mlp(x_features)  # (B, 64)

        # --- Fusión y clasificación ---
        fused = torch.cat([cnn_feat, mlp_feat], dim=1)  # (B, 320)
        return self.classifier(fused)                    # (B, 7)

    def get_model_info(self) -> None:
        """Imprime información del modelo."""
        total = sum(p.numel() for p in self.parameters())
        for tag, prefix in [
            ("CNN", "conv"), ("Proyección+PosEnc", "input_proj"),
            ("Transformer", "transformer"), ("MLP features", "feature_mlp"),
            ("Clasificador", "classifier"),
        ]:
            n = sum(p.numel() for name, p in self.named_parameters() if prefix in name)
            print(f"  {tag}: {n:,} ({n/total*100:.1f}%)")
        print(f"  TOTAL: {total:,}")
