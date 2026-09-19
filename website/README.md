# Turbo-dLLM project website

Static project page for Turbo-dLLM and Context-Sharded Block Parallelism (CSBP). It is a
fork of the [LLM-as-a-Verifier](https://llm-as-a-verifier.com/) page (itself adapted from
the [PolaRiS template](https://github.com/polaris-evals/polaris-evals.github.io)),
re-themed to Stanford cardinal red. There is no build step.

## Preview locally

```bash
python3 -m http.server 8000 --directory website
# open http://127.0.0.1:8000/
```

## Layout

| Path | Contents |
| --- | --- |
| `index.html` | The page |
| `static/css/site.css`, `static/js/site.js` | Styles and behavior (copy buttons, scroll reveal, section nav) |
| `static/js/lb-anim.js` | The "Load balancing" animation (`window.lbAnimation.seek(t)`) |
| `static/js/csbp-anim.js` | The "CSBP in motion" animation (inline SVG, no dependencies; `window.csbpAnimation.seek(t)` renders any frame) |
| `static/css/bulma.min.css` | Bulma layout grid, as in the reference page |
| `data/` | Trimmed SpecForge benchmark summary backing the library-comparison table |
| `figures/` | Generated plots (`*.png`, `*-mobile.png`), the social link-preview card, favicon |
| `scripts/figures/render_figures.R` | Renders every plot from the paper's table values |
| `scripts/diagrams/` | Paper TikZ sources and a script that converts them to SVG (not currently shown on the page) |
| `scripts/check_numbers.py` | Checks plotted data and page tables against the paper's LaTeX tables and `data/dflash2_specforge_benchmark.json` |
| `scripts/check_render.mjs` | Headless Chrome check at desktop and phone widths, with full-page and per-step animation screenshots |

## Regenerate figures

Requires R with `ggplot2` and `ragg`, the Lato font, `pdflatex` with TikZ, and `pdftocairo`
(Poppler). From `website/`:

```bash
Rscript scripts/figures/render_figures.R
```

## Verify

From `website/`, with a checkout of the paper sources and the local server running:

```bash
python3 scripts/check_numbers.py --paper path/to/block_parallel_paper
node scripts/check_render.mjs http://127.0.0.1:8000/ /tmp/site-render
```

## Deploy

`.github/workflows/pages.yml` publishes `index.html`, `static/`, `data/` and `figures/`
to GitHub Pages on pushes to `main` that touch `website/`. Enable Pages for this
repository with source "GitHub Actions". The site is served from the lab's custom domain,
`https://scalingintelligence.stanford.edu/Turbo-dLLM/` (the `github.io` URL redirects
there); `og:url`, `og:image` and `twitter:image` in `index.html` must match it.

