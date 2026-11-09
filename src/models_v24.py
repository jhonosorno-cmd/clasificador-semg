"""
CyborgConformer (v24): CNN + Conformer Encoder + Features Manuales.

El Conformer (Gulati et al., 2020) combina dentro de cada bloque:
  1. Feed-Forward (half-step)
  2. Multi-Head Self-Attention  → dependencias globales
  3. Convolution Module         → patrones locales (depthwise conv)
  4. Feed-Forward (half-step)
  5. LayerNorm

Esta combinación aborda la limitación observada en v23 (Transformer puro):
la convolución captura dependencias locales que la atención sola no modela
eficientemente con sólo 15 frames.

Flujo de la rama CNN:
  (B, 1, 125, 13) → CNN(128) → (B, 128, 15)
  → proyección(128→256) → PosEncoding → ConformerEncoder(2L, 8H, k=7)
  → GlobalAvgPool → (B, 256)

Fusión con rama MLP:
  (B, 256) + (B, 64) → (B, 320) → Clasificador → (B, 7)
"""

import math
import torch
import torch.nn as nn
from src.config import SEED

torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Positional Encoding (igual que v23)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Codificación posicional sinusoidal (Vaswani et al., 2017)."""

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
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Módulos internos del bloque Conformer
# ---------------------------------------------------------------------------

class FeedForwardModule(nn.Module):
    """
    Módulo Feed-Forward del Conformer (macro-residual × 0.5).

    PreNorm → Linear → SiLU → Dropout → Linear → Dropout
    Salida: x + 0.5 * FFN(LayerNorm(x))
    """

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),           # Swish ≈ SiLU
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self.net(self.norm(x))


class MultiHeadSelfAttentionModule(nn.Module):
    """
    Módulo de atención multi-cabeza del Conformer (PreNorm + residual).

    Salida: x + Dropout(MHSA(LayerNorm(x)))
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        return residual + self.dropout(x)


class ConvolutionModule(nn.Module):
    """
    Módulo de convolución del Conformer.

    LayerNorm → PointwiseConv(2d) → GLU → DepthwiseConv(k) → BN → SiLU
               → PointwiseConv(d) → Dropout
    Salida: x + conv_output  (residual)

    Args:
        d_model:     dimensión del modelo
        kernel_size: tamaño del kernel de la depthwise conv (impar)
        dropout:     dropout de salida
    """

    def __init__(
        self, d_model: int, kernel_size: int = 7, dropout: float = 0.1
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size debe ser impar"
        self.norm = nn.LayerNorm(d_model)
        # Pointwise conv: expande a 2*d_model para GLU
        self.pw_expand = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        # Depthwise conv (groups=d_model → un filtro por canal)
        self.dw_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        # Pointwise conv: comprime de vuelta a d_model
        self.pw_contract = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x + conv_out: (B, T, d_model)
        """
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)            # (B, d_model, T)
        x = self.pw_expand(x)             # (B, 2*d_model, T)
        x1, x2 = x.chunk(2, dim=1)       # GLU gate
        x = x1 * torch.sigmoid(x2)       # (B, d_model, T)
        x = self.dw_conv(x)              # (B, d_model, T)
        x = self.bn(x)
        x = self.swish(x)
        x = self.pw_contract(x)          # (B, d_model, T)
        x = self.dropout(x)
        x = x.transpose(1, 2)            # (B, T, d_model)
        return residual + x


class ConformerBlock(nn.Module):
    """
    Bloque Conformer completo: FF1 → MHSA → Conv → FF2 → LayerNorm.

    Args:
        d_model:         dimensión del modelo
        nhead:           cabezas de atención
        conv_kernel_size: kernel de la depthwise conv
        ff_expansion:    factor de expansión del FFN (default 4)
        dropout:         dropout global del bloque
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        conv_kernel_size: int = 7,
        ff_expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff1   = FeedForwardModule(d_model, ff_expansion, dropout)
        self.attn  = MultiHeadSelfAttentionModule(d_model, nhead, dropout)
        self.conv  = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2   = FeedForwardModule(d_model, ff_expansion, dropout)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff1(x)
        x = self.attn(x)
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# CyborgConformer
# ---------------------------------------------------------------------------

class CyborgConformer(nn.Module):
    """
    Arquitectura Cyborg con Conformer Encoder.

    La CNN extrae features locales (igual que CyborgHybrid).
    El Conformer alterna atención global y convolución local sobre los 15
    frames temporales. La rama MLP procesa features manuales (48D).

    Args:
        manual_features_dim:   dimensión del vector de features manuales
        num_classes:           número de clases de salida
        d_model:               dimensión del Conformer (256)
        nhead:                 cabezas de atención (8)
        num_conformer_layers:  capas del encoder (2)
        conv_kernel_size:      kernel depthwise (7, impar)
        ff_expansion:          factor expansión FFN (4)
        dropout_cnn:           dropout en las capas CNN
        dropout_conformer:     dropout en el Conformer
        dropout_mlp:           dropout en la rama MLP y clasificador
    """

    def __init__(
        self,
        manual_features_dim: int = 48,
        num_classes: int = 7,
        d_model: int = 256,
        nhead: int = 8,
        num_conformer_layers: int = 2,
        conv_kernel_size: int = 7,
        ff_expansion: int = 4,
        dropout_cnn: float = 0.15,
        dropout_conformer: float = 0.1,
        dropout_mlp: float = 0.3,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.manual_features_dim = manual_features_dim
        self.num_classes = num_classes

        # === RAMA CNN (idéntica a CyborgHybrid) ===
        self.conv1 = nn.Sequential(
            nn.Conv1d(13, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),        # 125 → 62
            nn.Dropout(0.10),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),        # 62 → 31
            nn.Dropout(dropout_cnn),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),        # 31 → 15
            nn.Dropout(dropout_cnn),
        )

        # Proyección 128 → d_model
        self.input_proj = nn.Linear(128, d_model)
        self.pos_encoding = PositionalEncoding(
            d_model=d_model, max_len=32, dropout=dropout_conformer
        )

        # Conformer Encoder
        self.conformer = nn.ModuleList([
            ConformerBlock(d_model, nhead, conv_kernel_size, ff_expansion, dropout_conformer)
            for _ in range(num_conformer_layers)
        ])
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
        fusion_size = self.cnn_output_size + self.mlp_output_size  # 320
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
            x_features: (B, F) — features manuales (F=48)

        Returns:
            logits: (B, 7)
        """
        # --- Rama CNN ---
        x = x_signal.squeeze(1).permute(0, 2, 1)  # (B, 13, 125)
        x = self.conv1(x)   # (B, 64, 62)
        x = self.conv2(x)   # (B, 128, 31)
        x = self.conv3(x)   # (B, 128, 15)
        x = x.permute(0, 2, 1)             # (B, 15, 128)

        # Proyección + Positional Encoding
        x = self.input_proj(x)             # (B, 15, 256)
        x = self.pos_encoding(x)           # (B, 15, 256)

        # Conformer Encoder
        for block in self.conformer:
            x = block(x)                   # (B, 15, 256)

        # Global Average Pooling
        cnn_feat = x.mean(dim=1)           # (B, 256)

        # --- Rama MLP ---
        mlp_feat = self.feature_mlp(x_features)  # (B, 64)

        # --- Fusión y clasificación ---
        fused = torch.cat([cnn_feat, mlp_feat], dim=1)  # (B, 320)
        return self.classifier(fused)                    # (B, 7)

    def get_model_info(self) -> None:
        """Imprime información del modelo."""
        total = sum(p.numel() for p in self.parameters())
        for tag, prefix in [
            ("CNN",               "conv"),
            ("Proyección+PosEnc", "input_proj"),
            ("Conformer",         "conformer"),
            ("MLP features",      "feature_mlp"),
            ("Clasificador",      "classifier"),
        ]:
            n = sum(p.numel() for name, p in self.named_parameters() if prefix in name)
            print(f"  {tag}: {n:,} ({n/total*100:.1f}%)")
        print(f"  TOTAL: {total:,}")
