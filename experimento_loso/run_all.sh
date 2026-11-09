#!/usr/bin/env bash
# Ejecuta el experimento leave-one-subject-out por sujeto (reanudable).
#
# Uso (desde cualquier directorio):
#   bash experimento_loso/run_all.sh 1 2 3 4
#   MODELS="v10 v17" bash experimento_loso/run_all.sh 5 6
#   EXTRA_ARGS="--seed 7" bash experimento_loso/run_all.sh 1
#
# Sin argumentos ejecuta los sujetos 1 a 11.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

if [[ $# -gt 0 ]]; then
    SUBJECTS=("$@")
else
    SUBJECTS=(1 2 3 4 5 6 7 8 9 10 11)
fi

read -r -a MODEL_LIST <<< "${MODELS:-v10 v15 v17 v18}"
read -r -a EXTRA_LIST <<< "${EXTRA_ARGS:-}"

echo "Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Directorio: $REPO_ROOT"
echo "Sujetos: ${SUBJECTS[*]}"
echo "Modelos: ${MODEL_LIST[*]}"

for subject in "${SUBJECTS[@]}"; do
    echo "--- Sujeto $subject ($(date '+%H:%M:%S')) ---"
    python experimento_loso/run_loso.py \
        --models "${MODEL_LIST[@]}" \
        --subjects "$subject" \
        --conditions zero ft1 ft2 \
        --seed 42 \
        ${EXTRA_LIST[@]+"${EXTRA_LIST[@]}"}
done

echo "Fin: $(date '+%Y-%m-%d %H:%M:%S')"
