"""Modelos de Deep Learning para clasificación EMG"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import SEED

# Para reproducibilidad
torch.manual_seed(SEED)


class HybridCNNLSTM(nn.Module):
    """
    1D CNN + BiLSTM para clasificación de gestos EMG.
    
    Arquitectura:
    - Input: (Batch, 1, 50, 12) → reshape a (B, 12, 50)
    - BiLSTM: Modela dependencias temporales bidireccionales
    - Global pooling: Promedia todos los timesteps del LSTM
    - Clasificador: Mapea a 7 clases de gestos
    """
    
    def __init__(self, input_channels=1, seq_length=50, emg_channels=13,
                 num_classes=7, dropout_prob=0.3):
        super(HybridCNNLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.emg_channels = emg_channels
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        
        # === 1D CNN: convoluciones temporales ===
        # Input: (B, 13, 125) - 12 canales EMG + 1 canal RMS, 125 timesteps (500Hz, 250ms)
        self.conv1 = nn.Sequential(
            nn.Conv1d(emg_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 125 -> 62
            nn.Dropout(0.1),
        )
        # (B, 64, 62)
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 62 -> 31
            nn.Dropout(0.15),
        )
        # (B, 128, 31)
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 31 -> 15
            nn.Dropout(0.15),
        )
        # (B, 128, 15)
        
        self.lstm_input_size = 128
        
        # === BiLSTM ===
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=True
        )
        
        # BiLSTM output = 128*2 = 256
        self.lstm_output_size = 128 * 2
        
        # === CLASIFICADOR ===
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass del modelo.
        
        Args:
            x: Tensor de forma (Batch, 1, 50, 12)
        
        Returns:
            logits: Tensor de forma (Batch, 7)
        """
        batch_size = x.size(0)
        
        # Reshape: (B, 1, 125, 12) -> (B, 12, 125)
        x = x.squeeze(1)          # (B, 125, 12)
        x = x.permute(0, 2, 1)    # (B, 12, 125)
        
        # 1D CNN temporal
        x = self.conv1(x)   # (B, 64, 62)
        x = self.conv2(x)   # (B, 128, 31)
        x = self.conv3(x)   # (B, 128, 15)
        
        # Permute para LSTM: (B, 15, 128)
        x = x.permute(0, 2, 1)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (B, 15, 256)
        
        # Global average pooling sobre todos los timesteps
        pooled = lstm_out.mean(dim=1)  # (B, 256)
        
        # Clasificador
        logits = self.classifier(pooled)  # (B, 7)
        
        return logits
    
    def get_model_info(self):
        """Imprime información detallada del modelo"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"=== 1D-CNN + BiLSTM Model Info ===")
        print(f"Input shape: (Batch, {self.input_channels}, {self.seq_length}, {self.emg_channels})")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"LSTM input features: {self.lstm_input_size}")
        print(f"LSTM output features: {self.lstm_output_size} (bidirectional)")
        print(f"Number of classes: {self.num_classes}")
        print("=" * 35)


def test_model():
    """Prueba del modelo con tensor aleatorio"""
    print("=== PRUEBA DEL MODELO ===\n")
    
    # Crear modelo
    model = HybridCNNLSTM()
    model.get_model_info()
    
    # Crear tensor de prueba
    batch_size = 32
    test_input = torch.randn(batch_size, 1, 125, 13)
    print(f"Input tensor shape: {test_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output tensor shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Verificaciones
    expected_shape = (batch_size, 7)
    assert output.shape == expected_shape, f"Shape incorrecto: esperado {expected_shape}, got {output.shape}"
    assert output.dtype == torch.float32, f"Tipo incorrecto: esperado float32, got {output.dtype}"
    
    print("\n[OK] Todas las verificaciones pasaron!")
    
    # Probar con diferentes batch sizes
    print("\nProbando diferentes batch sizes...")
    for bs in [1, 8, 16, 64]:
        test_tensor = torch.randn(bs, 1, 125, 13)
        out = model(test_tensor)
        assert out.shape == (bs, 7), f"Error con batch size {bs}"
        print(f"  Batch size {bs}: {out.shape} [OK]")
    
    print("\n[OK] Modelo funciona correctamente con diferentes batch sizes!")
    
    # Conteo de parámetros por componente
    print(f"\n=== DISTRIBUCION DE PARAMETROS ===")
    total = sum(p.numel() for p in model.parameters())
    for name_prefix, label in [('conv', 'CNN'), ('lstm', 'LSTM'), ('classifier', 'Classifier'),
                                ('spatial', 'SpatialPool')]:
        params = sum(p.numel() for n, p in model.named_parameters() if name_prefix in n)
        print(f"{label}: {params:,} ({params/total*100:.1f}%)")


if __name__ == "__main__":
    test_model()