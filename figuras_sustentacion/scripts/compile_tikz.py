"""Render the thesis TikZ/pgfplots figures (latex/figuras/tikz_*.tex) to PNG and SVG.

Each figure is wrapped in a standalone document that reproduces the preamble
of latex/main.tex (packages, TikZ libraries, colors), compiled with pdflatex,
then rasterized with pdftocairo (MiKTeX). Output goes to
figuras_sustentacion/tikz_documento/.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]
FIG_SRC: Path = ROOT / "latex" / "figuras"
OUT_DIR: Path = ROOT / "figuras_sustentacion" / "tikz_documento"
DPI: int = 300

PREAMBLE: str = r"""
\documentclass[border=6pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish, es-tabla]{babel}
\usepackage{lmodern}
\usepackage{amsmath, amssymb, amsfonts, bm}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{babel, arrows.meta, positioning, shapes.geometric, shapes.multipart,
  fit, calc, backgrounds, decorations.pathreplacing, decorations.markings}
\definecolor{uvblue}{RGB}{0,83,159}
\definecolor{uvgold}{RGB}{200,150,0}
\definecolor{cnnblue}{RGB}{33,150,243}
\definecolor{mlporange}{RGB}{230,120,0}
\definecolor{fusgreen}{RGB}{56,142,60}
\definecolor{outred}{RGB}{211,47,47}
\definecolor{rowgray}{gray}{0.93}
\definecolor{rowgold}{RGB}{255,248,220}
\definecolor{codebg}{gray}{0.96}
\begin{document}
"""
# Some figures use \textwidth; standalone has none, so give it the thesis body width.
DOC_WIDTH: str = r"\setlength{\textwidth}{15cm}" + "\n"


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def compile_one(tex_path: Path, build: Path) -> Path | None:
    """Compile one TikZ file to PDF inside ``build``.

    Args:
        tex_path: Path to latex/figuras/tikz_*.tex.
        build: Scratch directory.

    Returns:
        Path to the PDF, or None when pdflatex fails.
    """
    body = tex_path.read_text(encoding="utf-8")
    wrapper = build / f"{tex_path.stem}.tex"
    wrapper.write_text(PREAMBLE + DOC_WIDTH + body + "\n\\end{document}\n", encoding="utf-8")
    for _ in range(2):  # two passes for pgfplots/label stability
        res = run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", wrapper.name], build)
        if res.returncode != 0:
            log = (build / f"{tex_path.stem}.log")
            tail = log.read_text(encoding="utf-8", errors="replace")[-2500:] if log.exists() else res.stdout[-2500:]
            print(f"[FAIL] {tex_path.name}\n{tail}", file=sys.stderr)
            return None
    return build / f"{tex_path.stem}.pdf"


def rasterize(pdf: Path, stem: str) -> None:
    """Write PNG (300 dpi) and SVG next to each other in OUT_DIR."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_base = OUT_DIR / stem
    res = run(["pdftocairo", "-png", "-r", str(DPI), "-singlefile", str(pdf), str(png_base)], OUT_DIR)
    if res.returncode != 0:
        print(f"[FAIL] pdftocairo png {stem}: {res.stderr}", file=sys.stderr)
    res = run(["pdftocairo", "-svg", str(pdf), str(OUT_DIR / f"{stem}.svg")], OUT_DIR)
    if res.returncode != 0:
        print(f"[FAIL] pdftocairo svg {stem}: {res.stderr}", file=sys.stderr)
    shutil.copy(pdf, OUT_DIR / f"{stem}.pdf")


def main() -> int:
    if shutil.which("pdflatex") is None or shutil.which("pdftocairo") is None:
        print("pdflatex or pdftocairo not found in PATH", file=sys.stderr)
        return 1
    failures = 0
    with tempfile.TemporaryDirectory(prefix="tikz_build_") as tmp:
        build = Path(tmp)
        for tex in sorted(FIG_SRC.glob("tikz_*.tex")):
            pdf = compile_one(tex, build)
            if pdf is None:
                failures += 1
                continue
            rasterize(pdf, tex.stem)
            print(f"[OK] {tex.stem}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
