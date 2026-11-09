# Figuras para la sustentación

Figuras adicionales para la presentación (no forman parte del documento calificado).
Cada una sale de datos que ya existen en el proyecto; ningún valor fue estimado ni simulado.
Todas se generan en PNG (300 dpi) y SVG.

## Regenerar

Desde la raíz del proyecto:

```bash
python figuras_sustentacion/scripts/make_all.py            # 13 figuras + TikZ del documento
python figuras_sustentacion/scripts/make_all.py --skip-tikz
python figuras_sustentacion/scripts/fig05_loro_metricas_macro.py   # una sola figura
```

Requisitos: matplotlib, numpy; para las TikZ, MiKTeX (`pdflatex`, `pdftocairo`).

## Figuras nuevas

| Archivo | Contenido | Fuente de datos |
|---|---|---|
| `fig01_curvas_entrenamiento_v15_v17_v18` | Acc y loss por época, train vs val, con época del checkpoint | `logs/train_v15.log`, `train_v17.log`, `train_v18.log` |
| `fig02_barrido_theta_validacion` | Acc global y balanceada vs θ en validación; θ = 0,36 | `data/demo_cache.npz` (theta_grid, theta_val_acc, theta_val_bal) |
| `fig03_barrido_theta_test` | Efecto de θ en test: global vs balanceada; recall reposo vs gestos | `data/demo_cache.npz` (probs_mean, labels) con la regla de `src/eval_threshold.py`; verificado contra `logs/eval_threshold.log` |
| `fig04_tradeoff_accuracy_vs_balanced` | Scatter acc global vs balanceada de los experimentos con ambas métricas | `latex/capitulos/anexos.tex` (tabla de resultados completos) |
| `fig05_loro_metricas_macro_media_std` | LORO-CV: media ± σ (ddof=1) de acc, precisión, recall y F1 macro | `checkpoints/loro/loro_cv_summary.json` |
| `fig06_loro_dotplot_pliegues` | LORO-CV: los 6 pliegues de v10, v15, v17, v18 y Ensemble 4M | `checkpoints/loro/loro_cv_summary.json`, `v15_fold_k_metrics.json` |
| `fig07_ranking_combinaciones_ensemble` | Las 9 combinaciones de ensemble ensayadas | `logs/eval_ensemble_v18.log` |
| `fig08_correlacion_entre_clases_48D` | Correlación de Pearson entre perfiles medios de features por clase | `data/demo_cache.npz` (class_corr) |
| `fig09_analisis_errores_ensemble_4m` | Destino de los 521 errores: gesto→reposo 82,5 % | `results/ensemble_confusion.json` |
| `fig10_distribucion_clases_test` | Desbalance del test: 1161 reposo, ~125 por gesto | `results/ensemble_confusion.json` |
| `fig11_parametros_cyborghybrid` | Parámetros por bloque de v10 y tamaño por modelo | `INFORME_COMPLETO.md` §3.3, `GUIA_ESTUDIO.md` |
| `fig12_latencia_ensemble_4m` | Latencia p50/p95/media±σ del ensemble por ventana (CPU) | `data/demo_cache.npz` (latency_*) |
| `fig13_variantes_inferencia_final` | v10 → 4M argmax → +θ → +TTA-7, con BalAcc | `resultados.tex`, `logs/eval_threshold.log`, `logs/eval_tta.log` |

## `tikz_documento/`

Las siete figuras TikZ/pgfplots del documento (`latex/figuras/tikz_*.tex`) compiladas
tal cual con el preámbulo de `main.tex`, en PDF, PNG y SVG.

## Limitaciones de los datos

- No existe log de entrenamiento de v10 (ni de v23 a v26); la fig01 solo cubre v15, v17 y v18.
- Solo hay latencia del ensemble completo, no por modelo individual.
- No hay repeticiones por semilla de v10/v17/v18; la única dispersión legítima es la de LORO-CV.
- De la latencia solo se guardaron los estadísticos (p50, p95, media, σ), no las 200 muestras.
- Las figuras LORO usan el ensemble sin umbral θ (argmax), igual que la tabla de la tesis.

## Paleta

v10 `#00539F` (uvblue de la tesis), v17 `#D9730D`, v18 `#8E44AD`, Ensemble 4M `#1B9E77`,
v15/contexto gris, resaltado `#C89600` (uvgold). Validada para daltonismo (deutan/protan/tritan)
y contraste; el par verde/rojo de las TikZ originales no pasa la separación deutan, por eso se
reemplazó en las figuras nuevas.
