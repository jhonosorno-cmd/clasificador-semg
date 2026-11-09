#!/usr/bin/env bash
# Prepara el entorno para el experimento LOSO (Linux / macOS).
# Uso: bash experimento_loso/setup.sh
# Variables opcionales:
#   TORCH_INDEX_URL  indice de PyTorch para CUDA (por defecto cu126)
#   SKIP_SMOKE=1     omite la prueba rapida del experimento

set -euo pipefail

# Raiz del repositorio = carpeta padre de la carpeta de este script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "============================================================"
echo " Setup del experimento LOSO - Clasificador SEMG (NinaPro DB3)"
echo " Raiz: $REPO_ROOT"
echo "============================================================"

# 2. Buscar Python 3.10 o superior (python3 primero, luego python).
PY_SISTEMA=""
PY_ENCONTRADO=""
for candidato in python3 python; do
    if command -v "$candidato" >/dev/null 2>&1; then
        PY_ENCONTRADO="$candidato"
        if "$candidato" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
            PY_SISTEMA="$candidato"
            break
        fi
    fi
done

if [ -z "$PY_SISTEMA" ]; then
    if [ -n "$PY_ENCONTRADO" ]; then
        echo "[ERROR] Se encontro '$PY_ENCONTRADO' pero su version es menor a 3.10:" >&2
        "$PY_ENCONTRADO" --version >&2
    else
        echo "[ERROR] No se encontro Python. Instala Python 3.10 o superior y reintenta." >&2
    fi
    exit 1
fi
echo "[INFO] Python del sistema: $PY_SISTEMA ($("$PY_SISTEMA" --version 2>&1))"

# 3. Crear el entorno virtual .venv en la raiz y actualizar pip.
if [ ! -x ".venv/bin/python" ]; then
    echo "[INFO] Creando entorno virtual .venv ..."
    "$PY_SISTEMA" -m venv .venv
fi
VPY=".venv/bin/python"
"$VPY" -m pip install --upgrade pip

# 4. Instalar PyTorch: CUDA si hay nvidia-smi, CPU en otro caso.
if "$VPY" -c "import torch" >/dev/null 2>&1; then
    echo "[INFO] PyTorch ya instalado: $("$VPY" -c 'import torch; print(torch.__version__, "| cuda.is_available =", torch.cuda.is_available())')"
else
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        TORCH_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
        echo "[INFO] Variante elegida: CUDA (indice: $TORCH_URL)"
    else
        TORCH_URL="https://download.pytorch.org/whl/cpu"
        echo "[INFO] Variante elegida: CPU (no se detecto nvidia-smi)"
    fi
    "$VPY" -m pip install torch --index-url "$TORCH_URL"
fi

# 5. Dependencias del proyecto y pytest.
"$VPY" -m pip install -r requirements.txt pytest

# 6. Descargar y verificar los 11 archivos .mat de NinaPro DB3.
if [ ! -f "experimento_loso/download_data.py" ]; then
    echo "[ERROR] Falta experimento_loso/download_data.py." >&2
    exit 1
fi
echo "[INFO] Descargando y verificando datos (data/raw) ..."
"$VPY" experimento_loso/download_data.py

# 7. Pruebas unitarias: abortan el setup si fallan.
echo "[INFO] Ejecutando pruebas ..."
"$VPY" -m pytest tests/ -q

# 8. Prueba rapida del experimento (se puede omitir con SKIP_SMOKE=1).
if [ "${SKIP_SMOKE:-0}" = "1" ]; then
    echo "[INFO] Prueba rapida omitida (SKIP_SMOKE=1)."
else
    echo "[INFO] Prueba rapida del experimento ..."
    INICIO=$(date +%s)
    "$VPY" experimento_loso/run_loso.py --models v10 --subjects 1 --conditions zero \
        --epochs 1 --ft-epochs 1 --max-batches 5 --force
    echo "[INFO] Prueba rapida completada en $(( $(date +%s) - INICIO )) s."

    # Eliminar las salidas de la prueba.
    rm -f experimento_loso/resultados/v10_S1_zero_seed42.json \
          experimento_loso/resultados/probs/v10_S1_zero_seed42.npz \
          experimento_loso/checkpoints/v10_S1_pretrain_seed42.pth
fi

# 9. Mensaje final.
echo ""
echo "============================================================"
echo " Listo: entorno preparado para el experimento LOSO"
echo ""
echo " 1. Activa el entorno virtual:"
echo "      source .venv/bin/activate"
echo " 2. Lanza el experimento (con el venv activo), por ejemplo:"
echo "      bash experimento_loso/run_all.sh 1 2 3 4"
echo ""
echo " run_all debe ejecutarse con el venv activo."
echo "============================================================"
