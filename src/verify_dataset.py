"""Script para verificar qué dataset estamos usando realmente"""

import scipy.io
import numpy as np
from pathlib import Path

def verify_dataset_info():
    """Verifica información detallada del dataset"""
    
    # Analizar ambos archivos
    for exercise in [1, 2]:
        file_path = Path(f"data/raw/S1_E{exercise}_A1.mat")
        
        if not file_path.exists():
            print(f"No existe: {file_path}")
            continue
            
        print(f"\n=== ANÁLISIS S1_E{exercise}_A1.mat ===")
        
        # Cargar datos
        mat_data = scipy.io.loadmat(str(file_path))
        
        # Buscar metadatos
        print("Claves encontradas:")
        for key in mat_data.keys():
            if not key.startswith('__'):
                try:
                    data = mat_data[key]
                    if hasattr(data, 'shape'):
                        print(f"  {key}: {data.shape} {data.dtype}")
                    else:
                        print(f"  {key}: {type(data)} = {data}")
                except:
                    print(f"  {key}: [Error al leer]")
        
        # Análisis específico
        if 'emg' in mat_data:
            emg = mat_data['emg']
            print(f"\nEMG Analysis:")
            print(f"  Shape: {emg.shape}")
            print(f"  Range: [{emg.min():.6f}, {emg.max():.6f}]")
            print(f"  Mean: {emg.mean():.6f}")
            print(f"  Std: {emg.std():.6f}")
        
        if 'restimulus' in mat_data:
            stim = mat_data['restimulus']
            unique_gestures = np.unique(stim)
            print(f"\nGestos únicos: {unique_gestures}")
            print(f"Total gestos: {len(unique_gestures)}")
            
            # Distribución de gestos
            print("Distribución:")
            for gesture in sorted(unique_gestures):
                count = np.sum(stim == gesture)
                percentage = count / len(stim) * 100
                print(f"  Gesto {gesture}: {count} muestras ({percentage:.2f}%)")
        
        if 'rerepetition' in mat_data:
            reps = mat_data['rerepetition']
            unique_reps = np.unique(reps)
            print(f"\nRepeticiones: {unique_reps}")
        
        # Buscar información de frecuencia
        possible_fs_keys = ['frequency', 'fs', 'samplingrate', 'sampling_rate', 'freq']
        for key in possible_fs_keys:
            if key in mat_data:
                fs = mat_data[key]
                print(f"\nFrecuencia ({key}): {fs}")
                break
        else:
            print("\nNo se encontró información de frecuencia")
        
        # Buscar metadatos adicionales
        if 'subject' in mat_data:
            subject = mat_data['subject']
            print(f"Subject: {subject}")
        
        if 'exercise' in mat_data:
            exercise_num = mat_data['exercise']
            print(f"Exercise: {exercise_num}")

def compare_with_ninapro():
    """Compara características con NinaPro DB5 conocido"""
    print("\n" + "="*60)
    print("COMPARACIÓN CON NINAPRO DB5")
    print("="*60)
    
    ninapro_info = {
        "DB5 E1": {
            "gestures": 52,  # 0-51
            "repetitions": 6,
            "channels": 12,
            "frequency": 2000,
            "subjects": 10
        },
        "DB5 E2": {
            "gestures": 40,  # 52-91 (aprox)
            "repetitions": 6, 
            "channels": 12,
            "frequency": 2000,
            "subjects": 10
        }
    }
    
    for exercise, info in ninapro_info.items():
        print(f"\n{exercise}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    verify_dataset_info()
    compare_with_ninapro()
    
    print(f"\n" + "="*60)
    print("CONCLUSIÓN")
    print("="*60)
    print("1. Si los gestos encontrados (18 y 13) coinciden con un subconjunto")
    print("   de NinaPro DB5, entonces es NinaPro DB5.")
    print("2. Si los números no coinciden, podría ser otro dataset")
    print("   con formato similar (quizás una versión reducida).")
    print("3. La estructura (12 canales, 2000Hz, 6 reps) es muy")
    print("   característica de NinaPro DB5.")
