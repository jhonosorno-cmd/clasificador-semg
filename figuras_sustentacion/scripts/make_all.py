"""Generate every presentation figure and render the thesis TikZ figures.

Run from the project root:
    python figuras_sustentacion/scripts/make_all.py [--skip-tikz]
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FIGURES: list[str] = [
    "fig01_curvas_entrenamiento",
    "fig02_barrido_theta_validacion",
    "fig03_barrido_theta_test",
    "fig04_tradeoff_acc_balacc",
    "fig05_loro_metricas_macro",
    "fig06_loro_dotplot",
    "fig07_ranking_ensembles",
    "fig08_correlacion_clases",
    "fig09_analisis_errores",
    "fig10_distribucion_clases_test",
    "fig11_parametros_modelo",
    "fig12_latencia",
    "fig13_inferencia_final",
]


def main(argv: list[str]) -> int:
    failures: list[str] = []
    for name in FIGURES:
        try:
            importlib.import_module(name).main()
            print(f"[OK] {name}")
        except Exception:  # noqa: BLE001 - report and continue
            failures.append(name)
            print(f"[FAIL] {name}")
            traceback.print_exc()
    if "--skip-tikz" not in argv:
        import compile_tikz

        if compile_tikz.main() != 0:
            failures.append("compile_tikz")
    print(f"\n{len(FIGURES) - len([f for f in failures if f != 'compile_tikz'])}/{len(FIGURES)} figures generated; failures: {failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
