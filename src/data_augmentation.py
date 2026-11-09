"""Data augmentation para señales EMG - Balance de clases"""

import numpy as np
import torch
from typing import Tuple, List

class EMGDataAugmentation:
    """
    Clase para aplicar data augmentation a señales EMG.
    Enfocado en balancear clases minoritarias.
    """
    
    def __init__(self, augmentation_factor: float = 2.0, noise_std: float = 0.01):
        """
        Args:
            augmentation_factor: Factor de aumento para clases minoritarias
            noise_std: Desviación estándar para ruido gaussiano
        """
        self.augmentation_factor = augmentation_factor
        self.noise_std = noise_std
        
    def add_gaussian_noise(self, signal: np.ndarray) -> np.ndarray:
        """Añade ruido gaussiano proporcional a la desviación estándar de la señal"""
        signal_std = signal.std()
        if signal_std < 1e-8:
            signal_std = 1.0
        noise = np.random.normal(0, self.noise_std * signal_std, signal.shape)
        return signal + noise
    
    def time_shift(self, signal: np.ndarray, max_shift: int = 5) -> np.ndarray:
        """Desplazamiento temporal de la señal"""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return signal
        
        if shift > 0:
            # Desplazar hacia derecha
            augmented = np.zeros_like(signal)
            augmented[shift:] = signal[:-shift]
        else:
            # Desplazar hacia izquierda
            augmented = np.zeros_like(signal)
            augmented[:shift] = signal[-shift:]
        
        return augmented
    
    def amplitude_scaling(self, signal: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Escalamiento de amplitud"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return signal * scale
    
    def time_warping(self, signal: np.ndarray, warp_factor: float = 0.1) -> np.ndarray:
        """Deformación temporal sutil"""
        length = signal.shape[0]
        warp_steps = int(length * warp_factor)
        
        if warp_steps == 0:
            return signal
        
        # Crear índices deformados
        original_indices = np.arange(length)
        warp_indices = np.arange(length)
        
        # Aplicar deformación aleatoria
        for _ in range(warp_steps):
            pos = np.random.randint(1, length - 1)
            direction = np.random.choice([-1, 1])
            
            if direction == 1 and pos < length - 1:
                warp_indices[pos], warp_indices[pos + 1] = warp_indices[pos + 1], warp_indices[pos]
            elif direction == -1 and pos > 0:
                warp_indices[pos], warp_indices[pos - 1] = warp_indices[pos - 1], warp_indices[pos]
        
        # Asegurar índices válidos
        warp_indices = np.clip(warp_indices, 0, length - 1)
        
        return signal[warp_indices]
    
    def window_slicing(self, signal: np.ndarray, slice_ratio: float = 0.1) -> np.ndarray:
        """Corte y rellenado de ventana"""
        length = signal.shape[0]
        slice_length = int(length * slice_ratio)
        
        if slice_length == 0:
            return signal
        
        # Elegir posición de corte
        start = np.random.randint(0, length - slice_length + 1)
        end = start + slice_length
        
        # Crear señal aumentada
        augmented = signal.copy()
        slice_segment = signal[start:end]
        
        # Reemplazar segmento
        augmented[start:end] = slice_segment[::-1]  # Invertir segmento
        
        return augmented
    
    def augment_signal(self, signal: np.ndarray) -> np.ndarray:
        """Aplica augmentation aleatorio a una señal"""
        # Elegir 1-3 técnicas aleatoriamente
        techniques = [
            self.add_gaussian_noise,
            self.time_shift,
            self.amplitude_scaling,
            self.time_warping,
            self.window_slicing
        ]
        
        num_techniques = np.random.randint(1, 4)
        selected_techniques = np.random.choice(techniques, num_techniques, replace=False)
        
        augmented = signal.copy()
        for technique in selected_techniques:
            augmented = technique(augmented)
        
        return augmented
    
    def balance_dataset(self, signals: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balancea el dataset aplicando augmentation a clases minoritarias.
        
        Args:
            signals: Array de señales (n_samples, channels, time)
            labels: Array de etiquetas (n_samples,)
        
        Returns:
            Tuple[signals_augmented, labels_augmented]
        """
        # Contar muestras por clase
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count = max(counts)
        
        print(f"Distribución original:")
        for label, count in zip(unique_labels, counts):
            print(f"  Clase {label}: {count} muestras")
        
        augmented_signals = [signals]
        augmented_labels = [labels]
        
        # Para cada clase minoritaria
        for label, count in zip(unique_labels, counts):
            if label == 0:  # No aumentar clase mayoritaria
                continue
                
            # Calcular cuántas muestras necesitamos para igualar a clase mayoritaria
            target_count = max_count
            needed_samples = target_count - count
            
            if needed_samples <= 0:
                continue
            
            # Obtener muestras de esta clase
            class_mask = labels == label
            class_signals = signals[class_mask]
            
            print(f"Clase {label}: necesito {needed_samples} muestras adicionales")
            
            # Generar muestras aumentadas
            for i in range(needed_samples):
                # Elegir muestra aleatoria de esta clase
                original_signal = class_signals[i % len(class_signals)]
                
                # Aplicar augmentation
                augmented_signal = self.augment_signal(original_signal)
                
                augmented_signals.append(augmented_signal[np.newaxis, ...])
                augmented_labels.append(np.array([label]))
        
        # Combinar todas las muestras
        final_signals = np.vstack(augmented_signals)
        final_labels = np.hstack(augmented_labels)
        
        # Asegurar tipo float32
        final_signals = final_signals.astype(np.float32)
        
        # Mezclar dataset
        shuffle_indices = np.random.permutation(len(final_labels))
        final_signals = final_signals[shuffle_indices]
        final_labels = final_labels[shuffle_indices]
        
        # Imprimir distribución final
        print(f"\nDistribución después de augmentation:")
        unique_labels_final, counts_final = np.unique(final_labels, return_counts=True)
        for label, count in zip(unique_labels_final, counts_final):
            print(f"  Clase {label}: {count} muestras")
        
        return final_signals, final_labels

def test_augmentation():
    """Prueba las técnicas de augmentation"""
    print("=== PRUEBA DE DATA AUGMENTATION ===\n")
    
    # Crear señal de prueba
    test_signal = np.random.randn(50, 12)  # (time, channels)
    
    augmenter = EMGDataAugmentation()
    
    print("Técnicas de augmentation:")
    
    # Probar cada técnica
    techniques = [
        ("Original", lambda x: x),
        ("Gaussian Noise", augmenter.add_gaussian_noise),
        ("Time Shift", augmenter.time_shift),
        ("Amplitude Scaling", augmenter.amplitude_scaling),
        ("Time Warping", augmenter.time_warping),
        ("Window Slicing", augmenter.window_slicing),
        ("Combined", augmenter.augment_signal)
    ]
    
    for name, technique in techniques:
        augmented = technique(test_signal.copy())
        diff = np.mean(np.abs(augmented - test_signal))
        print(f"  {name}: diferencia promedio = {diff:.6f}")
    
    print("\nPrueba de balanceo:")
    
    # Crear dataset desbalanceado de prueba
    signals = np.random.randn(100, 50, 12)
    labels = np.array([0]*70 + [1]*10 + [2]*10 + [3]*10)  # Desbalanceado
    
    balanced_signals, balanced_labels = augmenter.balance_dataset(signals, labels)
    
    print(f"\nOriginal: {len(signals)} muestras")
    print(f"Balanceado: {len(balanced_signals)} muestras")
    print(f"Aumento: {len(balanced_signals) - len(signals)} muestras adicionales")

if __name__ == "__main__":
    test_augmentation()
