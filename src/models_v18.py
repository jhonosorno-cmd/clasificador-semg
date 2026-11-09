"""
Modelo v18: CyborgHybrid con Self-Attention sobre LSTM
Reemplaza el Global Average Pooling por MultiheadAttention + residual + LayerNorm.

Motivacion:
  El modelo actual trata todos los frames temporales por igual al promediar
  (B, 15, 256) -> mean -> (B, 256). Un gesto EMG tiene momentos de maxima
  activacion muscular mas informativos que otros. La atencion aprende a
  ponderar esos frames criticos en lugar de promediar uniformemente.

Cambio arquitectonico (solo en la rama CNN):
  ANTES: lstm_out (B,15,256) -> GlobalAvgPool -> (B,256)
  AHORA: lstm_out (B,15,256) -> MultiheadAttention(8 heads) -> residual+LN
                              -> GlobalAvgPool -> (B,256)
"""

import torch
import torch.nn as nn
from src.config import SEED

torch.manual_seed(SEED)

LSTM_HIDDEN = 128   # BiLSTM hidden per direction
LSTM_OUT = LSTM_HIDDEN * 2  # 256 (bidirectional)


class CyborgHybridAttention(nn.Module):
    """
    Arquitectura Cyborg con bloque de Self-Attention sobre la salida del BiLSTM.

    Ramas:
      - CNN + BiLSTM + Self-Attention: procesa la senal EMG temporal.
      - MLP: procesa features manuales (configurable: 48D o 132D).
    Fusion: concatenacion -> clasificador de 7 clases.
    """

    def __init__(
        self,
        emg_channels: int = 13,
        manual_features_dim: int = 48,
        num_classes: int = 7,
        dropout_prob: float = 0.3,
        num_heads: int = 8,
    ):
        super().__init__()

        self.emg_channels = emg_channels
        self.manual_features_dim = manual_features_dim
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.num_heads = num_heads

        # === RAMA CNN (identica a CyborgHybrid) ===
        self.conv1 = nn.Sequential(
            nn.Conv1d(emg_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.15),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.15),
        )

        # BiLSTM (identico a CyborgHybrid)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=LSTM_HIDDEN,
            num_layers=2,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=True,
        )
        # lstm_out: (B, 15, 256)

        # === SELF-ATTENTION sobre salida del BiLSTM ===
        # embed_dim debe ser divisible por num_heads: 256 / 8 = 32 ✓
        self.self_attention = nn.MultiheadAttention(
            embed_dim=LSTM_OUT,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(LSTM_OUT)
        self.attn_dropout = nn.Dropout(0.1)
        self.cnn_output_size = LSTM_OUT  # 256

        # === RAMA MLP (identica a CyborgHybrid) ===
        self.feature_mlp = nn.Sequential(
            nn.Linear(manual_features_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.mlp_output_size = 64

        # === FUSION + CLASIFICADOR ===
        fusion_size = self.cnn_output_size + self.mlp_output_size  # 320
        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_signal: torch.Tensor, x_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass con self-attention sobre BiLSTM.

        Args:
            x_signal: (B, 1, 125, 13) - senal EMG ventaneada.
            x_features: (B, D) - features manuales (48 o 132).

        Returns:
            logits: (B, 7)
        """
        # --- Rama CNN ---
        x = x_signal.squeeze(1).permute(0, 2, 1)  # (B, 13, 125)
        x = self.conv1(x)  # (B, 64, 62)
        x = self.conv2(x)  # (B, 128, 31)
        x = self.conv3(x)  # (B, 128, 15)
        x = x.permute(0, 2, 1)         # (B, 15, 128)

        lstm_out, _ = self.lstm(x)      # (B, 15, 256)

        # --- Self-Attention con residual + LayerNorm ---
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_dropout(attn_out)
        attended = self.attn_norm(lstm_out + attn_out)  # residual (B, 15, 256)

        # Global Average Pooling sobre la secuencia atendida
        cnn_features = attended.mean(dim=1)  # (B, 256)

        # --- Rama MLP ---
        mlp_features = self.feature_mlp(x_features)  # (B, 64)

        # --- Fusion + clasificador ---
        fused = torch.cat([cnn_features, mlp_features], dim=1)  # (B, 320)
        return self.classifier(fused)  # (B, 7)

    def get_model_info(self) -> None:
        """Imprime resumen del modelo."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=== CyborgHybridAttention Model Info ===")
        print(f"Input signal:   (B, 1, 125, {self.emg_channels})")
        print(f"Input features: (B, {self.manual_features_dim})")
        print(f"Attention heads: {self.num_heads}")
        print(f"Total params:    {total:,}")
        print(f"Trainable:       {trainable:,}")
        print("=" * 40)

        # Desglose por componente
        components = [
            ("CNN (conv1+2+3)", "conv"),
            ("BiLSTM", "lstm"),
            ("Self-Attention", "self_attention"),
            ("AttnNorm", "attn_norm"),
            ("MLP features", "feature_mlp"),
            ("Classifier", "classifier"),
        ]
        for label, prefix in components:
            n = sum(p.numel() for n, p in self.named_parameters() if prefix in n)
            print(f"  {label}: {n:,} ({n/total*100:.1f}%)")
