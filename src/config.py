"""Configuración del proyecto de clasificación EMG"""

from pathlib import Path

# Paths
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
CHECKPOINTS_PATH = Path("checkpoints")

# Parámetros de procesamiento de señal
ORIGINAL_FREQUENCY = 2000  # Hz
TARGET_FREQUENCY = 500     # Hz
DOWNSAMPLING_FACTOR = ORIGINAL_FREQUENCY // TARGET_FREQUENCY

# Ventaneo
WINDOW_SIZE_MS = 250      # ms
WINDOW_SIZE_SAMPLES = int((WINDOW_SIZE_MS / 1000) * TARGET_FREQUENCY)  # 125 muestras a 500 Hz
STRIDE_MS = 50           # ms
STRIDE_SAMPLES = int((STRIDE_MS / 1000) * TARGET_FREQUENCY)  # 25 muestras a 500 Hz

# Para compatibilidad con código existente
WINDOW_SIZE = WINDOW_SIZE_SAMPLES
STRIDE = STRIDE_SAMPLES

# Gestos seleccionados (7 clases)
SELECTED_GESTURES = [0, 1, 2, 3, 4, 5, 6]
GESTURE_MAPPING = {old: new for new, old in enumerate(SELECTED_GESTURES)}

# Parámetros de entrenamiento
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 100
SEED = 42

# Dispositivo (definido después de importar torch)
def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = get_device()
