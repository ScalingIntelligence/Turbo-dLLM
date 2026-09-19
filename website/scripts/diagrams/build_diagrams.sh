#!/usr/bin/env bash
# Compiles the paper's TikZ diagrams (copied verbatim from block_parallel_paper/) into
# tightly cropped SVGs in website/figures/. Needs pdflatex (with tikz, times) and pdftocairo.
# Usage (from website/): bash scripts/diagrams/build_diagrams.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/../../figures"
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT
render() {  # $1 = tikz source, $2 = output stem
  cat > "$build/$2.tex" <<TEX
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc}
\definecolor{BPBlue}{HTML}{356F9F}
\definecolor{BPOrange}{HTML}{B66718}
\pagestyle{empty}
\newsavebox{\fig}
\begin{document}
\sbox{\fig}{\input{$here/$1}}
\pdfpagewidth=\dimexpr\wd\fig+8pt\relax
\pdfpageheight=\dimexpr\ht\fig+\dp\fig+8pt\relax
\hoffset=-1in \voffset=-1in
\shipout\hbox{\kern4pt\vbox{\kern4pt\box\fig\kern4pt}}
\end{document}
TEX
  (cd "$build" && pdflatex -interaction=nonstopmode -halt-on-error "$2.tex" >/dev/null)
  pdftocairo -svg "$build/$2.pdf" "$out/$2.svg"
  echo "wrote $out/$2.svg"
}
render figure_block_parallel.tex diagram-overview
render figure_fused_bp_cp.tex diagram-communication
