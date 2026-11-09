import scipy.io
import numpy as np
from pathlib import Path

def inspect_mat_file(file_path):
    """Inspecciona un archivo .mat de NinaPro DB"""
    print(f"Inspeccionando archivo: {file_path}")
    print("=" * 50)
    
    # Cargar datos
    mat_data = scipy.io.loadmat(str(file_path))
    
    # Imprimir claves
    print("Claves disponibles:")
    for key in mat_data.keys():
        if not key.startswith('__'):  # Ignorar metadatos de MATLAB
            print(f"  - {key}")
    print()
    
    # Inspeccionar EMG
    if 'emg' in mat_data:
        emg_data = mat_data['emg']
        print(f"EMG shape: {emg_data.shape}")
        print(f"EMG dtype: {emg_data.dtype}")
        print(f"EMG sample range: [{emg_data.min():.3f}, {emg_data.max():.3f}]")
        print()
    
    # Inspeccionar etiquetas
    for label_key in ['restimulus', 'stimulus']:
        if label_key in mat_data:
            labels = mat_data[label_key]
            print(f"{label_key} shape: {labels.shape}")
            print(f"{label_key} dtype: {labels.dtype}")
            unique_labels = np.unique(labels)
            print(f"Valores únicos en {label_key}: {unique_labels}")
            print(f"Total de gestos únicos: {len(unique_labels)}")
            print()
    
    # Inspeccionar repeticiones
    if 'rerepetition' in mat_data:
        reps = mat_data['rerepetition']
        print(f"rerepetition shape: {reps.shape}")
        unique_reps = np.unique(reps)
        print(f"Repeticiones únicas: {unique_reps}")
        print()
    
    # Buscar frecuencia de muestreo
    possible_fs_keys = ['frequency', 'fs', 'samplingrate', 'sampling_rate']
    for key in possible_fs_keys:
        if key in mat_data:
            fs = mat_data[key]
            print(f"Frecuencia de muestreo ({key}): {fs}")
            break
    else:
        print("No se encontró información de frecuencia de muestreo")
    
    print("=" * 50)

if __name__ == "__main__":
    # Especificar archivo a inspeccionar
    file_path = Path("data/raw/S1_E1_A1.mat")
    
    if not file_path.exists():
        print(f"No se encontró el archivo: {file_path}")
    else:
        inspect_mat_file(file_path)
