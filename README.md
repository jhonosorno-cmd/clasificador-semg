# Clasificador sEMG — reconocimiento de gestos de mano en amputados transradiales

Implementación en PyTorch de **CyborgHybrid**, una arquitectura híbrida que combina una
CNN 1D con BiLSTM y un ramal de características manuales, evaluada sobre **NinaPro DB3**
para distinguir siete clases: reposo y seis gestos de mano.

Este repositorio contiene el código de un trabajo de grado de Ingeniería Electrónica de la
Universidad del Valle (Cali, Colombia). Incluye las arquitecturas evaluadas, el ensemble
final, el análisis de viabilidad de operación en tiempo real y los scripts
que regeneran las figuras del documento.

La demostración interactiva (Streamlit) está en un repositorio aparte:
**[jhonosorno-cmd/clasificador-semg-demo](https://github.com/jhonosorno-cmd/clasificador-semg-demo)**.

---

## Resultado principal

| | Exactitud | Exactitud balanceada |
|---|---|---|
| CyborgHybrid (v10) | 71,32 % | 41,92 % |
| Características extendidas 132D (v17) | 71,89 % | 44,10 % |
| Auto-atención sobre BiLSTM (v18) | 71,63 % | 45,37 % |
| **Ensemble 4M** (argmax) | **72,78 %** | 44,92 % |
| **Ensemble 4M** (umbral θ = 0,36) | **72,88 %** | 44,24 % |

Evaluado sobre el sujeto 1, repetición 6 (1 914 ventanas), con división por repetición y sin
solapamiento entre particiones.

**El trabajo no alcanza la línea base de referencia de 79,72 %** reportada por Arcos et al.
(2024) sobre un protocolo distinto. La brecha y sus causas se analizan en el documento.

---

## Datos

Este repositorio **no distribuye los datos originales**. NinaPro DB3 se obtiene de
<https://ninapro.hevs.ch/> bajo los términos de sus autores.

- **Conjunto:** NinaPro DB3, Ejercicio A
- **Sujetos:** 11 personas con amputación transradial
- **Adquisición:** 12 electrodos Delsys, 2 000 Hz, 6 repeticiones por movimiento
- **Ubicación esperada:** `data/raw/S{n}_E1_A1.mat`

Al usar estos datos debe citarse:

> Atzori, M., Gijsberts, A., Castellini, C., Caputo, B., Hager, A.-G. M., Elsig, S.,
> Giatsidis, G., Bassetto, F., & Müller, H. (2014). *Electromyography data for
> non-invasive naturally-controlled robotic hand prostheses.* Scientific Data, 1, 140053.
> https://doi.org/10.1038/sdata.2014.53

El archivo `data/demo_cache.npz` incluido en el repositorio contiene 1 914 ventanas
derivadas del sujeto 1, repetición 6 —filtradas, normalizadas y segmentadas—, incorporadas
para que los scripts de figuras puedan ejecutarse sin descargar NinaPro. No permite reconstruir el registro
original. Se publica bajo los términos de NinaPro, que exigen citar el artículo anterior.

---

## Procesamiento de la señal

1. Diezmado de 2 000 Hz a 500 Hz (factor 4)
2. Filtro pasabanda Butterworth de orden 4, 20–249 Hz
3. Normalización Z-score por canal
4. Ventaneo deslizante: 125 muestras (250 ms), paso de 25 (50 ms) — 80 % de traslape
5. RMS por instante como canal 13
6. Características manuales: RMS, cruces por cero, cambios de signo de la pendiente y
   longitud de onda, por canal → vector de 48 dimensiones

La división es **por repetición**, no aleatoria: repeticiones 1–4 para entrenamiento
(7 468 ventanas), la 5 para validación (1 900) y la 6 para prueba (1 914). Ninguna ventana
comparte origen temporal entre particiones.

### Desbalance de clases

El reposo domina el conjunto. Sobre el conjunto de prueba:

| Clase | Ventanas | Proporción |
|---|---|---|
| 0 — reposo | 1 161 | 60,66 % |
| 1–6 — gestos | 125–127 cada una | ≈ 6,5 % cada una |

Se compensa con pesos de clase inversos a la raíz de la frecuencia en la función de pérdida.
Este desbalance es la razón de que exactitud global y exactitud balanceada diverjan tanto a
lo largo de todo el trabajo.

---

## Arquitectura

**CyborgHybrid** (`src/models_cyborg.py`) — 813 255 parámetros

```
señal (1, 125, 13) ──> CNN 1D (k = 7, 5, 3) ──> BiLSTM ×2 (h = 128/dir) ──> 256D ─┐
                                                                                   ├─> 320D ─> 128D ─> 7
características (48) ─────────────────> MLP ────────────────────────────> 64D ─────┘
```

El BiLSTM concentra cerca del 81 % de los parámetros. Las dos ramas se concatenan antes de
la clasificación: la convolucional aprende la forma de onda, la manual aporta descriptores
de amplitud y de actividad que la red no tiene que redescubrir.

**Variantes del ensemble final:**

| Componente | Diferencia | Parámetros |
|---|---|---|
| v10 | Modelo base, entrenado con Mixup | 813 255 |
| v15 | Aumento de señal (ruido, escalado, canal suprimido) | 813 255 |
| v17 | 132 características (MAV, VAR, WAMP, AR1–AR4) | 824 007 |
| v18 | Auto-atención multi-cabeza sobre la salida del BiLSTM | 1 076 935 |

El ensemble promedia las cuatro distribuciones softmax con peso uniforme.

---

## Operación en tiempo real

El sistema se evaluó bajo la restricción de operar sobre un flujo continuo de señal.

| | Mediana | p95 |
|---|---|---|
| Preprocesamiento | 2,52 ms | 5,02 ms |
| Inferencia del ensemble | 18,20 ms | 21,05 ms |
| **Total por ventana** | **20,72 ms** | 26,07 ms |

Con un paso de 50 ms, el sistema consume el **41 %** del presupuesto disponible. El retardo
de establecimiento es de **271 ms**, del cual el 92 % corresponde a la longitud de la
ventana y no al cómputo; a partir de ahí la salida se actualiza a **20 Hz**.

El p95 es sensible a la carga del equipo: en corridas repetidas llegó a ser cerca de un 40 %
mayor, sin acercarse al límite de 50 ms. En flujo continuo la decisión se toma por argmax:
el umbral θ = 0,36 se ajustó sobre el pipeline fuera de línea y no se traslada a esta
configuración.

Adaptar el pipeline a flujo continuo exige dos ajustes —aplicar el filtro a cada ventana ya
capturada en lugar de al registro completo, y fijar los estadísticos de normalización en una
calibración previa— con un costo medido de **0,47 puntos porcentuales** de exactitud global
y sin necesidad de reentrenar.

---

## Qué no funcionó

Se evaluaron 26 versiones. La mayoría no superó al modelo base, y los resultados negativos
son parte del aporte:

| Enfoque | Resultado | Lectura |
|---|---|---|
| TCN con dilataciones | 70,38 % | Un campo receptivo mayor que la ventana no aporta |
| Transformer | 71,89 % | Comparable al base, sin superar al ensemble |
| Conformer (3,2 M parámetros) | 71,00 % | Sobreajuste con 7 468 muestras de entrenamiento |
| CNN 2D sobre espectrograma STFT | 68,70 % | Resolución de ~15,6 Hz insuficiente; una clase colapsa a 0 % |
| Características espectrales (MNF, MDF) | 70,32 % | Welch sobre 125 muestras produce estimaciones ruidosas |
| Cascada binario + 6 gestos | 70,27 % | El clasificador binario enruta demasiado hacia reposo |
| Ensemble de 5 modelos | 72,26–72,78 % | Dilución: ninguna combinación de 5 supera al ensemble 4M |

**El patrón dominante:** para secuencias de 15 marcos temporales y datos limitados, el BiLSTM
con compuertas supera consistentemente a los mecanismos sin estado (atención, convoluciones
dilatadas). Aumentar la capacidad del modelo degrada el resultado.

---

## Reproducir

```bash
pip install -r requirements.txt

# Colocar los .mat de NinaPro DB3 en data/raw/

python -m src.inspect_data        # verificar la carga
python -m src.train_cyborg        # modelo base (v10)
python -m src.eval_ensemble_v18   # ensemble de 4 modelos
```

Todos los scripts se ejecutan desde la raíz del proyecto. El dispositivo se detecta
automáticamente (`cuda` si está disponible, `cpu` en caso contrario).

---

## Limitaciones

Conviene tenerlas presentes al interpretar las cifras:

- **Los resultados principales corresponden a un solo sujeto** (S1) y no generalizan a los
  11 sujetos del conjunto. La evaluación sobre los 11 sujetos de `experimento_multisujeto/`
  es un análisis posterior al documento de tesis y no forma parte de él.
- **No se alcanza la línea base de 79,72 %**, obtenida sobre un protocolo y una población
  distintos.
- **v10 se entrenó sin semilla fija.** Reentrenarlo con semilla produce 70,79 % en lugar del
  71,32 % reportado. La diferencia indica la dispersión esperable entre corridas.
- **La normalización Z-score del pipeline original es transductiva**: emplea los estadísticos
  de la grabación completa. El análisis de tiempo real cuantifica el costo de sustituirla por
  estadísticos de calibración en 0,31 pp.
- **La exactitud balanceada permanece cerca del 45 %.** El sistema distingue reposo de gesto
  con solvencia; discriminar entre gestos individuales sigue siendo el problema abierto.

---

## Diferencias con el documento de tesis

El documento de tesis ya fue calificado y no se modifica. Este README sigue lo que hace el
código; donde ambos difieren, la referencia es el código:

| Tema | Documento | Código |
|---|---|---|
| Semilla de v10 | Semilla 42 para los cuatro modelos | v10 se entrenó sin semilla fija (`src/train_cyborg.py`) |
| Mixup en v10 | No figura en la tabla de hiperparámetros | Activo durante el entrenamiento (α = 0,4) |
| Proporción de cada gesto | «apenas un 8 %» (resumen) | ≈ 6,5 % (125–127 de 1 914 ventanas de prueba) |
| Ventanas de entrenamiento de S1 | ≈ 5 700 (metodología) y 7 468 (anexos) | 7 468 |
| Referencia de NinaPro DB3 | Atzori et al. (2015), *IEEE TNSRE* | Se cita además Atzori et al. (2014), *Scientific Data*, artículo que describe DB3 |

Ninguna de estas diferencias altera las cifras de resultados reportadas.

---

## Estructura

```
src/                      Modelos, carga de datos, entrenamiento y evaluación
data/demo_cache.npz       Conjunto de prueba precomputado (usado por los scripts de figuras)
checkpoints/demo/         Pesos de los cuatro modelos del ensemble
experimento_tiempo_real/  Medición de latencia y análisis de operación causal
experimento_multisujeto/  Evaluación sobre los 11 sujetos (posterior a la tesis)
figuras_sustentacion/     Scripts de generación de figuras
```

---

## Cita

Si este trabajo le resulta útil, cite el trabajo de grado:

```bibtex
@mastersthesis{osorno2026semg,
  author      = {Osorno Pacheco, Jhon Alexander},
  title       = {Desarrollo de un sistema de clasificación de movimientos de la
                 mano en personas con amputación de miembro superior usando
                 aprendizaje profundo sobre señales mioeléctricas},
  school      = {Universidad del Valle},
  address     = {Cali, Colombia},
  year        = {2026},
  type        = {Trabajo de grado, Ingeniería Electrónica},
  note        = {Director: Leonardo Antonio Bermeo Varón, Grupo PSI}
}
```

El archivo `CITATION.cff` contiene los mismos metadatos en formato legible por máquina.

---

## Licencia

Código bajo licencia [Apache 2.0](LICENSE); ver también [NOTICE](NOTICE). Los datos de NinaPro DB3 no están cubiertos por esta
licencia y se rigen por los términos de sus autores.
