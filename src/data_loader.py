"""Data loader para dataset NinaPro DB3 con procesamiento completo"""

import scipy.io
import torch
import numpy as np
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from src.config import *

class NinaProDB3Dataset(Dataset):
    def __init__(self, subjects: list, reps_to_use: list, mode: str = 'train',
                 augment: bool = False, target_subject: int = 1):
        """
        Multi-Subject Training con prevención de data leakage.
        
        Args:
            subjects: Lista de IDs de sujetos para TRAIN (ej: [1, 2, 3]).
            reps_to_use: Lista de repeticiones a incluir.
            mode: 'train', 'val' o 'test'.
            augment: Si aplicar augmentation online (solo train).
            target_subject: Sujeto objetivo para evaluación (default: 1).
        
        Lógica de prevención de data leakage:
        - mode='train': Carga TODOS los sujetos de la lista
        - mode='val' o 'test': Carga SOLO target_subject (para comparación justa)
        """
        self.samples = []
        self.labels = []
        self.mode = mode
        self.augment = augment and mode == 'train'
        self.target_subject = target_subject
        
        # Prevención de data leakage
        if mode == 'train':
            # Entrenar con TODOS los sujetos
            print(f"\n[TRAIN] Multi-Subject Training: {len(subjects)} sujetos")
            for subject_id in subjects:
                self._load_subject_data(subject_id, reps_to_use)
        else:
            # Validar/Testear SOLO en target_subject
            print(f"\n[{mode.upper()}] Single-Subject Evaluation: S{target_subject}")
            self._load_subject_data(target_subject, reps_to_use)
        
        # Convertir listas a arrays numpy después de cargar todos los sujetos
        if self.samples:
            self.samples = np.array(self.samples, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.longlong)
            print(f"\nTotal dataset: {len(self.samples)} ventanas")

    def _load_subject_data(self, subject_id, reps):
        """
        Carga y procesa datos de un sujeto específico.
        Preprocesamiento POR SUJETO para respetar escala individual.
        """
        file_name = f"S{subject_id}_E1_A1.mat"
        file_path = RAW_DATA_PATH / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

        print(f"  Cargando S{subject_id} desde {file_path}...")
        
        # Cargar datos crudos
        mat_data = scipy.io.loadmat(str(file_path))
        
        emg_signal = mat_data['emg']          # (muestras, 12)
        stimulus = mat_data['restimulus']     # (muestras, 1)
        repetition = mat_data['rerepetition'] # (muestras, 1)
        
        print(f"  -> Datos originales: {emg_signal.shape[0]} muestras a {ORIGINAL_FREQUENCY}Hz")
        
        # 1. Downsample de 2000Hz a 500Hz (decimacion: 1 de cada 4 muestras)
        emg_signal = emg_signal[::DOWNSAMPLING_FACTOR]
        stimulus = stimulus[::DOWNSAMPLING_FACTOR]
        repetition = repetition[::DOWNSAMPLING_FACTOR]
        
        print(f"  -> Después de downsampling: {emg_signal.shape[0]} muestras a {TARGET_FREQUENCY}Hz")
        
        # 2. Preprocesamiento: bandpass + Z-score (no se aplica rectificacion)
        emg_signal = self._preprocess_emg(emg_signal)
        
        # 3. Filtrar gestos seleccionados y remapear etiquetas
        self._filter_and_remap_labels(stimulus)
        
        # 4. Crear ventanas
        self._create_windows(emg_signal, self.filtered_stimulus, repetition, reps)

    def _preprocess_emg(self, emg_signal):
        """Preprocesamiento EMG: bandpass filter + Z-score."""
        print(f"  -> Rango crudo: [{emg_signal.min():.6f}, {emg_signal.max():.6f}]")
        
        # 1. Bandpass filter 20-249Hz (Nyquist=250Hz a 500Hz)
        nyquist = TARGET_FREQUENCY / 2.0
        low = 20.0 / nyquist
        high = (nyquist - 1.0) / nyquist  # just below Nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered = np.zeros_like(emg_signal)
        for ch in range(emg_signal.shape[1]):
            filtered[:, ch] = filtfilt(b, a, emg_signal[:, ch])
        print(f"  -> Bandpass 20-{nyquist-1:.0f}Hz aplicado")
        
        # 2. Z-score por canal
        mean = filtered.mean(axis=0, keepdims=True)
        std = filtered.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1e-8
        normalized = (filtered - mean) / std
        
        print(f"  -> Z-score aplicado, rango: [{normalized.min():.2f}, {normalized.max():.2f}]")
        return normalized.astype(np.float32)

    def _filter_and_remap_labels(self, stimulus):
        """Filtra gestos seleccionados y remapea etiquetas a 0-6"""
        # Crear una copia para evitar modificar el original
        filtered_stimulus = stimulus.copy().flatten()
        
        # Reemplazar gestos no seleccionados con -1
        mask = ~np.isin(filtered_stimulus, SELECTED_GESTURES)
        filtered_stimulus[mask] = -1
        
        # Remapear gestos seleccionados a 0-6
        for old_label, new_label in GESTURE_MAPPING.items():
            filtered_stimulus[filtered_stimulus == old_label] = new_label
        
        # Restaurar forma original
        self.filtered_stimulus = filtered_stimulus.reshape(-1, 1)
        
        print(f"  -> Gestos seleccionados: {SELECTED_GESTURES}")
        print(f"  -> Mapeo: {GESTURE_MAPPING}")
        print(f"  -> Etiquetas únicas después de filtrar: {np.unique(self.filtered_stimulus[self.filtered_stimulus >= 0])}")

    def _create_windows(self, emg, stim, rep, target_reps):
        """Crea ventanas solapadas con RMS como canal adicional."""
        total_samples = emg.shape[0]
        valid_windows = 0
        skipped_transition = 0
        
        for start in range(0, total_samples - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            
            # Verificar que TODA la ventana tenga la misma etiqueta y repetición
            window_labels = stim[start:end].flatten()
            window_reps = rep[start:end].flatten()
            
            # Usar la etiqueta del centro como referencia
            mid_label = window_labels[len(window_labels) // 2]
            mid_rep = window_reps[len(window_reps) // 2]
            
            # Filtrar: repetición correcta y etiqueta válida
            if mid_rep not in target_reps or mid_label < 0:
                continue
            
            # Filtrar ventanas de transición (etiqueta no uniforme)
            if not np.all(window_labels == mid_label):
                skipped_transition += 1
                continue
            
            window = emg[start:end, :]  # (125, 12)
            
            # Calcular RMS por timestep como canal adicional (feature de amplitud)
            rms = np.sqrt(np.mean(window**2, axis=1, keepdims=True))  # (125, 1)
            
            # Concatenar: (125, 12) + (125, 1) = (125, 13)
            window_with_rms = np.concatenate([window, rms], axis=1)  # (125, 13)
            window_with_rms = window_with_rms[np.newaxis, :, :]  # (1, 125, 13)
            
            self.samples.append(window_with_rms)
            self.labels.append(mid_label)
            valid_windows += 1
        
        print(f"  -> Generadas {valid_windows} ventanas válidas (descartadas {skipped_transition} transiciones)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retorna:
            signal: Tensor (1, 125, 13) Float32 - señal EMG + RMS
            features: Tensor (48,) Float32 - features manuales (12 canales × 4 features)
            label: Tensor Long con etiqueta 0-6
        """
        signal = self.samples[idx].copy()  # (1, 125, 13)
        label = self.labels[idx]
        
        if self.augment:
            signal = self._online_augment(signal)
        
        # Calcular features manuales (solo sobre los 12 canales EMG, no el RMS)
        emg_window = signal[0, :, :12]  # (125, 12) - solo canales EMG
        features = self._extract_manual_features(emg_window)
        
        return torch.from_numpy(signal), torch.from_numpy(features), torch.tensor(label, dtype=torch.long)

    def _extract_manual_features(self, emg_window: np.ndarray) -> np.ndarray:
        """
        Extrae features manuales clásicos de EMG por cada canal.
        
        Args:
            emg_window: (125, 12) - ventana EMG de 12 canales
        
        Returns:
            features: (48,) - 4 features × 12 canales
        """
        n_channels = emg_window.shape[1]
        features = np.zeros(n_channels * 4, dtype=np.float32)
        
        for ch in range(n_channels):
            signal_ch = emg_window[:, ch]
            
            # 1. RMS (Root Mean Square)
            rms = np.sqrt(np.mean(signal_ch ** 2))
            
            # 2. ZC (Zero Crossing Rate)
            zc = np.sum(np.diff(np.sign(signal_ch)) != 0)
            
            # 3. SSC (Slope Sign Changes)
            diff_signal = np.diff(signal_ch)
            ssc = np.sum(np.diff(np.sign(diff_signal)) != 0)
            
            # 4. WL (Waveform Length)
            wl = np.sum(np.abs(np.diff(signal_ch)))
            
            # Almacenar features: [RMS, ZC, SSC, WL] para cada canal
            features[ch * 4 + 0] = rms
            features[ch * 4 + 1] = zc
            features[ch * 4 + 2] = ssc
            features[ch * 4 + 3] = wl
        
        return features
    
    def _online_augment(self, signal: np.ndarray) -> np.ndarray:
        """Augmentation ligera online: ruido + escalado aleatorio."""
        # signal shape: (1, 125, 13)
        s = signal[0]  # (125, 13)
        
        # 50% probabilidad de añadir ruido gaussiano proporcional
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.05, s.shape).astype(np.float32)
            s = s + noise * s.std()
        
        # 50% probabilidad de escalado de amplitud
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            s = s * scale
        
        # 30% probabilidad de time shift pequeño
        if np.random.random() < 0.3:
            shift = np.random.randint(-3, 4)
            if shift != 0:
                s = np.roll(s, shift, axis=0)
        
        return s[np.newaxis, :, :].astype(np.float32)

def create_data_loaders(train_subjects: list = [1], target_subject: int = 1, 
                        batch_size: int = BATCH_SIZE):
    """
    Crea DataLoaders con Multi-Subject Training.
    
    Args:
        train_subjects: Lista de sujetos para TRAIN (ej: [1, 2, 3])
        target_subject: Sujeto objetivo para VAL/TEST (default: 1)
        batch_size: Tamaño de batch
    
    Prevención de data leakage:
    - TRAIN: Usa TODOS los sujetos de train_subjects
    - VAL/TEST: Usa SOLO target_subject
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_reps = [1, 2, 3, 4]  # 67% para entrenamiento
    val_reps = [5]              # 17% para validación
    test_reps = [6]             # 17% para prueba
    
    # Train: Multi-Subject (todos los sujetos)
    train_dataset = NinaProDB3Dataset(
        subjects=train_subjects, 
        reps_to_use=train_reps, 
        mode='train', 
        augment=True,
        target_subject=target_subject
    )
    
    # Val/Test: Single-Subject (solo target_subject)
    val_dataset = NinaProDB3Dataset(
        subjects=train_subjects,  # Se ignora en mode='val'
        reps_to_use=val_reps, 
        mode='val',
        target_subject=target_subject
    )
    test_dataset = NinaProDB3Dataset(
        subjects=train_subjects,  # Se ignora en mode='test'
        reps_to_use=test_reps, 
        mode='test',
        target_subject=target_subject
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    from collections import Counter
    
    # Test Multi-Subject Training
    print("=== TEST MULTI-SUBJECT TRAINING ===\n")
    train_subjects = [1, 2, 3]  # Entrenar con 3 sujetos
    target_subject = 1           # Evaluar en S1
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_subjects=train_subjects,
        target_subject=target_subject
    )
    
    print(f"\n{'='*60}")
    print(f"RESUMEN DATASETS")
    print(f"{'='*60}")
    print(f"Train: {len(train_loader.dataset)} muestras (S{train_subjects})")
    print(f"Val:   {len(val_loader.dataset)} muestras (S{target_subject})")
    print(f"Test:  {len(test_loader.dataset)} muestras (S{target_subject})")
    
    # Verificar distribución de clases en train
    train_labels = train_loader.dataset.labels
    train_counts = Counter(train_labels)
    print(f"\nDistribución train (multi-subject):")
    for i in range(7):
        print(f"  Clase {i}: {train_counts[i]} muestras")
    
    # Distribucion efectiva en 1 epoch.
    # Nota: el DataLoader usa shuffle=True (muestreo uniforme sin reemplazo),
    # NO un WeightedRandomSampler. El desbalance se corrige en la funcion de
    # perdida mediante class weights sqrt-inverse, no en el muestreo.
    label_counts = Counter()
    for _, _, labels in train_loader:
        for l in labels.numpy():
            label_counts[l] += 1
    print(f"\nDistribucion efectiva en 1 epoch (shuffle=True, drop_last=True):")
    for c in sorted(label_counts):
        print(f"  Clase {c}: {label_counts[c]} muestras")

    signals, features, labels = next(iter(train_loader))
    print(f"\nBatch signals shape:  {signals.shape}")
    print(f"Batch features shape: {features.shape}")
    print(f"Range: [{signals.min():.3f}, {signals.max():.3f}]")
    print(f"Labels: {sorted(set(labels.numpy().tolist()))}")

