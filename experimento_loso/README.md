# Experimento leave-one-subject-out (LOSO)

Evalúa si los modelos del proyecto generalizan a un sujeto no visto durante el entrenamiento, sobre NinaPro DB3 (11 sujetos transradiales amputados, Ejercicio A, 7 clases).

## Protocolo

Para cada sujeto `i` retenido:

1. **Preentrenamiento** con los otros 10 sujetos: repeticiones 1–4 para entrenamiento y repetición 5 para validación.
2. **Prueba** siempre sobre el sujeto `i`, repetición 6.
3. **Condiciones** de adaptación al sujeto `i`:

| Condición | Fine-tuning | Repeticiones del sujeto `i` usadas | Validación |
|-----------|-------------|-------------------------------------|------------|
| `zero`    | No          | Ninguna con etiquetas. Normalización z-score con la repetición 1 (sin etiquetas) | — |
| `ft1`     | Sí          | Repetición 1 (normalización y ajuste fino) | Repetición 5 |
| `ft2`     | Sí          | Repeticiones 1–2 (normalización y ajuste fino) | Repetición 5 |

La repetición 6 nunca se usa para normalización ni para entrenamiento. Los sujetos de preentrenamiento se normalizan con sus repeticiones 1–4; el filtro pasabanda (filtfilt) se aplica sobre la grabación completa, como en el pipeline original. El ajuste fino entrena todas las capas con tasa de aprendizaje diferencial (clasificador 5e-4, resto 1e-4). Modelos: v10 (CyborgHybrid), v15 (aumentación de señal), v17 (132 características), v18 (autoatención). El ensemble es el promedio uniforme de probabilidades softmax.

## Requisitos

- Python 3.10 o superior.
- Dependencias desde la raíz del repositorio:
  ```bash
  pip install -r requirements.txt
  ```
- GPU con CUDA recomendada. El experimento funciona en CPU, pero es considerablemente más lento.
- Datos NinaPro DB3, Ejercicio A, descargados desde <https://ninapro.hevs.ch/>. Colocar los archivos `S1_E1_A1.mat` … `S11_E1_A1.mat` en `data/raw/`. Los datos no se incluyen en el repositorio.

Cita de la base de datos:

> Atzori, M., Gijsberts, A., Castellini, C., Caputo, B., Hager, A.-G. M., Elsig, S., Giatsidis, G., Bassetto, F., & Müller, H. (2014). Electromyography data for non-invasive naturally-controlled robotic hand prostheses. *Scientific Data*, 1, 140053. https://doi.org/10.1038/sdata.2014.53

Todos los comandos se ejecutan desde la raíz del repositorio (`D:\UV\Clasificador_SEMG` o equivalente).

## Prueba rápida antes del run largo

```bash
python experimento_loso/run_loso.py --models v10 --subjects 1 --conditions zero --epochs 1 --ft-epochs 1 --max-batches 5 --force
```

Después, borrar los resultados de prueba:

- `experimento_loso/resultados/v10_S1_zero_seed42.json`
- El `.npz` con el mismo nombre base en `experimento_loso/resultados/probs/`
- El checkpoint `experimento_loso/checkpoints/v10_S1_pretrain_seed42.pth` (si no se borra, el run real lo reutilizaría en lugar de preentrenar de nuevo)

Las cifras de esta prueba no significan nada: solo verifica que los datos cargan y que el flujo completo corre.

## Ejecución

Cada máquina toma un subconjunto de sujetos. Sin argumentos se ejecutan los 11.

| Máquina | Linux / macOS | Windows |
|---------|---------------|---------|
| A | `bash experimento_loso/run_all.sh 1 2 3 4` | `experimento_loso\run_all.bat 1 2 3 4` |
| B | `bash experimento_loso/run_all.sh 5 6 7 8` | `experimento_loso\run_all.bat 5 6 7 8` |
| C | `bash experimento_loso/run_all.sh 9 10 11` | `experimento_loso\run_all.bat 9 10 11` |

Variables opcionales: `MODELS` (por defecto `v10 v15 v17 v18`) y `EXTRA_ARGS` (por ejemplo `EXTRA_ARGS="--seed 7"`, ver más abajo).

El proceso es **reanudable**: los resultados terminados se omiten. Si se corta, volver a lanzar exactamente el mismo comando.

## Qué devolver

- La carpeta `experimento_loso/resultados/` completa (archivos `.json` y `probs/*.npz`).
- **No** enviar los checkpoints de `experimento_loso/checkpoints/`.

Luego, en una sola máquina, unir las carpetas `resultados/` de todas las máquinas dentro de `experimento_loso/resultados/` y ejecutar:

```bash
python experimento_loso/aggregate.py
```

Esto genera `summary.csv`, `summary_per_subject.csv` y `comparison_vs_intrasubject.csv`.

## Qué reportar en el artículo

- Tabla por sujeto (`summary_per_subject.csv`).
- Media ± desviación estándar entre sujetos, por modelo y condición.
- Comparación pareada con Wilcoxon contra el modelo intra-sujeto v10 de `experimento_multisujeto/resultados` (`comparison_vs_intrasubject.csv`), evaluado también sobre la repetición 6. Solo existe la referencia intra-sujeto de v10; el ensemble se compara contra esa misma referencia.
- Advertencia explícita: con 11 sujetos la potencia estadística de la prueba es baja; los p-valores deben interpretarse con cautela.

## Semilla

La semilla fija es 42. Para estimar la variabilidad del entrenamiento, repetir el experimento con otra semilla, por ejemplo `EXTRA_ARGS="--seed 7"` (en Windows, `set EXTRA_ARGS=--seed 7`). Los resultados quedan separados por semilla en el nombre del archivo.
