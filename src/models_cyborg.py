"""
Modelo Cyborg: Arquitectura Híbrida CNN + Features Manuales
Combina Deep Learning con Feature Engineering clásico
"""

import torch
import torch.nn as nn
from src.config import SEED

# Para reproducibilidad
torch.manual_seed(SEED)


class CyborgHybrid(nn.Module):
    """
    Arquitectura Cyborg: CNN + MLP para features manuales.
    
    Arquitectura:
    - Rama CNN: Procesa señal EMG (1, 125, 13) con 1D CNN + BiLSTM
    - Rama MLP: Procesa features manuales (48,) - RMS, ZC, SSC, WL
    - Fusión: Concatena salidas de ambas ramas
    - Clasificador: Mapea fusión a 7 clases
    """
    
    def __init__(self, input_channels=1, seq_length=125, emg_channels=13,
                 manual_features_dim=48, num_classes=7, dropout_prob=0.3):
        super(CyborgHybrid, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.emg_channels = emg_channels
        self.manual_features_dim = manual_features_dim
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        
        # === RAMA CNN: 1D CNN + BiLSTM ===
        # Input: (B, 13, 125) - 12 canales EMG + 1 canal RMS
        self.conv1 = nn.Sequential(
            nn.Conv1d(emg_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 125 -> 62
            nn.Dropout(0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 62 -> 31
            nn.Dropout(0.15),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 31 -> 15
            nn.Dropout(0.15),
        )
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=True
        )
        self.cnn_output_size = 128 * 2  # BiLSTM output = 256
        
        # === RAMA MLP: Features Manuales ===
        # Input: (B, 48) - 12 canales × 4 features (RMS, ZC, SSC, WL)
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
        self.mlp_output_size = 64
        
        # === FUSIÓN + CLASIFICADOR ===
        # Concatenar: CNN (256) + MLP (64) = 320
        fusion_size = self.cnn_output_size + self.mlp_output_size
        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x_signal, x_features):
        """
        Forward pass del modelo Cyborg.
        
        Args:
            x_signal: Tensor (Batch, 1, 125, 13) - señal EMG
            x_features: Tensor (Batch, 48) - features manuales
        
        Returns:
            logits: Tensor (Batch, 7)
        """
        batch_size = x_signal.size(0)
        
        # === RAMA CNN ===
        # Reshape: (B, 1, 125, 13) -> (B, 13, 125)
        x_cnn = x_signal.squeeze(1)       # (B, 125, 13)
        x_cnn = x_cnn.permute(0, 2, 1)    # (B, 13, 125)
        
        # 1D CNN
        x_cnn = self.conv1(x_cnn)   # (B, 64, 62)
        x_cnn = self.conv2(x_cnn)   # (B, 128, 31)
        x_cnn = self.conv3(x_cnn)   # (B, 128, 15)
        
        # BiLSTM
        x_cnn = x_cnn.permute(0, 2, 1)  # (B, 15, 128)
        lstm_out, _ = self.lstm(x_cnn)  # (B, 15, 256)
        
        # Global average pooling
        cnn_features = lstm_out.mean(dim=1)  # (B, 256)
        
        # === RAMA MLP ===
        mlp_features = self.feature_mlp(x_features)  # (B, 64)
        
        # === FUSIÓN ===
        fused = torch.cat([cnn_features, mlp_features], dim=1)  # (B, 320)
        
        # === CLASIFICADOR ===
        logits = self.classifier(fused)  # (B, 7)
        
        return logits
    
    def get_model_info(self):
        """Imprime información detallada del modelo"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"=== Cyborg Hybrid Model Info ===")
        print(f"Input signal shape: (Batch, {self.input_channels}, {self.seq_length}, {self.emg_channels})")
        print(f"Input features shape: (Batch, {self.manual_features_dim})")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"CNN output size: {self.cnn_output_size}")
        print(f"MLP output size: {self.mlp_output_size}")
        print(f"Fusion size: {self.cnn_output_size + self.mlp_output_size}")
        print(f"Number of classes: {self.num_classes}")
        print("=" * 35)


def test_model():
    """Prueba del modelo Cyborg"""
    print("=== PRUEBA DEL MODELO CYBORG ===\n")
    
    # Crear modelo
    model = CyborgHybrid()
    model.get_model_info()
    
    # Crear tensors de prueba
    batch_size = 32
    test_signal = torch.randn(batch_size, 1, 125, 13)
    test_features = torch.randn(batch_size, 48)
    
    print(f"\nInput signal shape: {test_signal.shape}")
    print(f"Input features shape: {test_features.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(test_signal, test_features)
    
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Verificaciones
    expected_shape = (batch_size, 7)
    assert output.shape == expected_shape, f"Shape incorrecto: esperado {expected_shape}, got {output.shape}"
    assert output.dtype == torch.float32, f"Tipo incorrecto: esperado float32, got {output.dtype}"
    
    print("\n[OK] Todas las verificaciones pasaron!")
    
    # Conteo de parámetros por componente
    print(f"\n=== DISTRIBUCION DE PARAMETROS ===")
    total = sum(p.numel() for p in model.parameters())
    for name_prefix, label in [('conv', 'CNN'), ('lstm', 'LSTM'), 
                                ('feature_mlp', 'MLP Features'),
                                ('classifier', 'Classifier')]:
        params = sum(p.numel() for n, p in model.named_parameters() if name_prefix in n)
        print(f"{label}: {params:,} ({params/total*100:.1f}%)")


if __name__ == "__main__":
    test_model()
