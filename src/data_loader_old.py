# src/data/dataset.py
import scipy.io
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import src.config as cfg

class NinaProDB5Dataset(Dataset):
    def __init__(self, subject_id, reps_to_use, mode='train'):
        """
        Args:
            subject_id (int): ID del sujeto (1-10 para DB5).
            reps_to_use (list): Lista de repeticiones a incluir (ej. [1, 3, 4, 6]).
            mode (str): 'train' o 'test' (para logs).
        """
        self.samples = []
        self.labels = []
        self.mode = mode
        
        # Cargar datos del sujeto
        self._load_subject_data(subject_id, reps_to_use)

    def _load_subject_data(self, subject_id, reps):
        file_name = f"S{subject_id}_E1_A1.mat" # Ajusta según la estructura de archivos de DB5 (E1, E2, E3)
        file_path = Path(cfg.RAW_DATA_PATH) / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

        print(f"[{self.mode.upper()}] Cargando sujeto {subject_id} desde {file_path}...")
        
        mat_data = scipy.io.loadmat(str(file_path))
        
        # En NinaPro DB5:
        # 'emg': señal cruda (muestras x 16)
        # 'restimulus': etiqueta real del movimiento (0-52)
        # 'rerepetition': número de repetición del movimiento
        
        emg_signal = mat_data['emg']
        stimulus = mat_data['restimulus']
        repetition = mat_data['rerepetition']

        # Normalización simple (Min-Max o Z-score es recomendada aquí)
        # Nota: Implementar normalización basada en estadísticas de entrenamiento solamente
        
        # Ventaneo (Windowing)
        self._create_windows(emg_signal, stimulus, repetition, reps)

    def _create_windows(self, emg, stim, rep, target_reps):
        # Esta función recorre la señal y extrae ventanas solo de las repeticiones deseadas
        total_samples = emg.shape[0]
        
        for start in range(0, total_samples - cfg.WINDOW_SIZE, cfg.STRIDE):
            end = start + cfg.WINDOW_SIZE
            
            # Verificar a qué repetición pertenece el centro de la ventana
            mid_point = (start + end) // 2
            current_rep = rep[mid_point][0]
            current_label = stim[mid_point][0]
            
            # Solo agregar si pertenece a las repeticiones seleccionadas
            # Y opcionalmente ignorar la clase 0 (reposo) si quieres balancear
            if current_rep in target_reps:
                window = emg[start:end, :] # (Time, Channels) -> (40, 16)
                
                # Transponer para PyTorch: (Canales, Tiempo) es mejor para Conv1D
                # De (40, 16) a (16, 40)
                window = window.transpose(1, 0)
                
                self.samples.append(window)
                self.labels.append(current_label)

        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.longlong)
        
        print(f" -> Generadas {len(self.samples)} ventanas de tamaño {cfg.WINDOW_SIZE}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Retorna: (señal, etiqueta)
        return torch.from_numpy(self.samples[idx]), torch.tensor(self.labels[idx])
    