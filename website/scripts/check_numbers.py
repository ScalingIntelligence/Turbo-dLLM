#!/usr/bin/env python3
"""Checks every number on the website against the paper's LaTeX tables.

Usage (from website/):
    python3 scripts/check_numbers.py --paper ../../block_parallel_paper

Verifies that
  * the tok/s data in scripts/figures/render_figures.R match the paper tables, and the
    speedup labels the figures print (round(csbp / baseline, 2)) equal the paper's speedups;
  * every row of the website's tables matches the corresponding paper table row;
  * the equal-wall-clock series agree with the paper's stated leads (1.8 / 2 points peak,
    1 point after 12 hours).
"""

from __future__ import annotations

import argparse
import decimal
import html
import json
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
failures: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + message)
    if not ok:
        failures.append(message)


def table_rows(path: Path) -> list[list[str]]:
    """Returns data rows of a booktabs table as cleaned cell lists."""
    text = "\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("%"))
    body = text.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
    body = body.replace("\\midrule", "\\\\").replace("\\addlinespace", "")
    rows = []
    for raw in body.split("\\\\"):
        cell_text = raw.replace("{,}", ",").replace("$\\times$", "×").replace("$\\boldsymbol{\\times}$", "×")
        cell_text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell_text)
        cell_text = re.sub(r"\s+", " ", cell_text).strip()
        if cell_text:
            rows.append([c.strip() for c in cell_text.split("&")])
    return rows


def num(s: str) -> float:
    return float(s.replace(",", "").replace("×", ""))


def pair(cell: str) -> tuple[float, float]:
    a, b = (num(x) for x in cell.split("/"))
    return a, b


def r_series(r_src: str, name: str) -> tuple[list[float], list[float], list[float]]:
    m = re.search(rf"^{name}\s*<-\s*wl\((.*?)\)\s*$", r_src, re.M | re.S)
    if not m:
        raise SystemExit(f"cannot find {name} in render_figures.R")
    vectors = re.findall(r"c\(([^)]*)\)", m.group(1))
    return tuple([float(v) for v in vec.split(",")] for vec in vectors)  # type: ignore[return-value]


def html_rows(page: str) -> list[list[str]]:
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", " ", c)) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        rows.append([re.sub(r"\s+", " ", c).strip() for c in cells])
    return rows


def find_row(rows: list[list[str]], *needles: str) -> list[str] | None:
    for row in rows:
        joined = " | ".join(row)
        if all(n in joined for n in needles):
            return row
    return None


def default_paper_dir() -> Path:
    """Nearest block_parallel_paper checkout above this repository."""
    for parent in SITE.parents:
        candidate = parent / "block_parallel_paper"
        if (candidate / "tables").is_dir():
            return candidate
    return SITE.parents[1] / "block_parallel_paper"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, default=default_paper_dir())
    args = parser.parse_args()
    tables = args.paper / "tables"
    r_src = (SITE / "scripts/figures/render_figures.R").read_text()
    page = (SITE / "index.html").read_text()
    site_rows = html_rows(page)

    ctx_key = {"64K": 64, "128K": 128, "256K": 256, "512K": 512, "1M": 1024}

    def check_series(label: str, var: str, paper: dict[int, tuple[float, float, float]]) -> None:
        ctx, base, csbp = r_series(r_src, var)
        check(sorted(ctx) == sorted(paper), f"{label}: contexts {ctx} match paper {sorted(paper)}")
        for c, b, o in zip(ctx, base, csbp):
            pb, po, ps = paper[int(c)]
            check(round(b) == pb and round(o) == po, f"{label} {int(c)}K tok/s {b:g}/{o:g} = paper {pb:g}/{po:g}")
            check(f"{o / b:.2f}" == f"{ps:.2f}", f"{label} {int(c)}K plotted label {o / b:.2f}× = paper {ps:.2f}×")

    print("Figures vs. two_node_h200_1p2_results.tex (ours / baseline)")
    two_node = table_rows(tables / "two_node_h200_1p2_results.tex")
    for label, var, model in [("DiffusionGemma 26B-A4B", "gemma", "DiffusionGemma 26B-A4B")]:
        paper = {}
        for row in two_node:
            if row[0] == model:
                ours, base = pair(row[5])
                paper[ctx_key[row[1]]] = (base, ours, num(row[9]))
        check_series(label, var, paper)

    print("Figures vs. appendix_fast_dllm_v2_scaling.tex (baseline / ours)")
    paper = {ctx_key[r[1]]: (*pair(r[4]), num(r[5])) for r in table_rows(tables / "appendix_fast_dllm_v2_scaling.tex")
             if r[0] == "Qwen3.8-27B"}
    check_series("Qwen3.8-27B conversion", "qwen_conv", paper)

    print("Figures vs. dflash2_qwen38_h100_scaling.tex (baseline / ours)")
    dflash = table_rows(tables / "dflash2_qwen38_h100_scaling.tex")
    for label, var, model, contexts in [("DFlash2 Qwen3.8-27B", "dflash_qwen", "Qwen3.8-27B", {512, 1024}),
                                        ("DFlash2 Muse-Glimmer-30B", "dflash_muse", "Muse-Glimmer-30B", {256, 512, 1024})]:
        paper = {ctx_key[r[1]]: (*pair(r[4]), num(r[5])) for r in dflash if r[0] == model and ctx_key[r[1]] in contexts}
        check_series(label, var, paper)

    print("Results table vs. main_standard_bdlm_256k.tex")
    for row in table_rows(tables / "main_standard_bdlm_256k.tex"):
        model = row[1]
        base, ours = row[4].split(" / ")
        hbm_base, hbm_ours = row[7].split(" / ")
        topo_base, topo_ours = (t.replace("/TP1", "") for t in (row[2], row[3]))
        site = find_row(site_rows, model, base, ours)
        check(site is not None and site[1] == topo_base and site[3] == topo_ours and row[5] in " ".join(site)
              and f"{hbm_base} → {hbm_ours}" in " ".join(site),
              f"{model}: {topo_base} {base} / {topo_ours} {ours}, {row[5]}, {hbm_base} → {hbm_ours}")

    print("Attention breakdown vs. operator_profile_256k.tex")
    op_text = (tables / "operator_profile_256k.tex").read_text()
    for a, b, delta in re.findall(r"([\d.]+) \$\\rightarrow\$ ([\d.]+) \\textcolor\{[^}]*\}\{\\textbf\{\(\$?([^)$]*)", op_text):
        delta = delta.replace("\\times", "×").replace("\\%", "%").replace("-", "−")
        site = find_row(site_rows, f"{a} → {b}")
        check(site is not None and delta in " ".join(site), f"{a} → {b} ({delta})")

    print("Pure-BP ablation vs. appendix_pure_bp_ablation.tex")
    model = ctx = ""
    pure_rows = [r for r in html_rows(page) if len(r) == 6 and r[2] in ("Best baseline", "Pure BP", "CSBP")]
    site_iter = iter(pure_rows)
    for row in table_rows(tables / "appendix_pure_bp_ablation.tex"):
        model = row[0] or model
        ctx = row[1] or ctx
        method = "CSBP" if row[2] == "Ours" else row[2]
        site = next(site_iter, None)
        topo = row[3].replace("/TP1", "")
        check(site is not None and site[2] == method and site[3] == topo and site[4] == row[5] and site[5] == row[6],
              f"{model} {ctx} {method}: {topo}, {row[5]} tok/s, {row[6]} GiB")

    print("Load balancing vs. appendix_dual_end_scheduling_ablation.tex (contiguous / dual-end)")
    for row in table_rows(tables / "appendix_dual_end_scheduling_ablation.tex"):
        cont, dual = row[4].split(" / ")
        hb, hd = row[6].split(" / ")
        site = find_row(site_rows, row[0], cont, dual)
        check(site is not None and row[7] in " ".join(site) and f"{hb} → {hd}" in " ".join(site),
              f"{row[0]}: {cont} / {dual}, {row[7]}, {hb} → {hd}")

    print("Equal-wall-clock series vs. paper text (peak lead 1.8 / 2 points, +1 after 12 h)")
    for key, peak in [("swe", 1.8), ("tb", 2.0)]:
        m = re.search(rf"{key}\s*=\s*data\.frame\(h = c\(([^)]*)\), base = c\(([^)]*)\),\s*csbp = c\(([^)]*)\)\)", r_src)
        base = [float(x) for x in m.group(2).split(",")]
        csbp = [float(x) for x in m.group(3).split(",")]
        leads = [round(c - b, 1) for b, c in zip(base, csbp)]
        check(max(leads) == peak and leads[-1] == 1.0 and all(l > 0 for l in leads[1:]),
              f"{key}: leads {leads} (peak {peak}, final 1.0, ahead at every trained checkpoint)")

    print("Traffic explanation vs. appendix.tex (Savings Breakdowns)")
    appendix = re.sub(r"\s+", " ", (args.paper / "appendix.tex").read_text()).replace("{,}", ",").replace("\\%", "%")
    page_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", page))
    for page_claim, paper_claim in [
        ("halves attention traffic for NemotronDiffusion 14B, which uses full attention", "uses full attention, so keeping corrupted K/V local halves its logical attention traffic"),
        ("25 of its 30 layers use a 1,024-token sliding window", "25 of its 30 attention layers use a 1,024-token sliding window"),
        ("exchanges full shards in every layer", "uses full-shard K/V exchange in all 30 layers"),
    ]:
        check(page_claim in page_text and paper_claim in appendix, f"'{page_claim}'")

    print("Scaling caption vs. appendix_fast_dllm_v2_scaling.tex and two_node_h200_1p2_results.tex")
    for claim in ["1.26×</span> at 64K to <span class=\"g\">1.33×</span> at 256K", "1.25×</span> at 64K to <span class=\"g\">1.61×</span> at 512K"]:
        check(claim in page, f"scaling caption quotes {re.sub(r'<[^>]+>', '', claim)}")

    print("SpecForge comparison vs. data/dflash2_specforge_benchmark.json and dflash2 paper table")
    bench = json.loads((SITE / "data/dflash2_specforge_benchmark.json").read_text())
    lib_label = {"dllm": "Turbo-dLLM", "specforge": "SpecForge"}
    for key, win in sorted(bench["winners"].items(), key=lambda kv: int(kv[0].split(":")[0])):
        ctx, library = key.split(":")
        tokens = f"{round(win['supervised_tokens_per_second']):,}"
        site = find_row(site_rows, f"{int(ctx) // 1024}K", tokens)
        check(site is not None, f"{lib_label[library]} at {int(ctx) // 1024}K: {tokens} tok/s")
    for ctx, speedup in sorted(bench["speedups"].items(), key=lambda kv: int(kv[0])):
        site = find_row(site_rows, f"{int(ctx) // 1024}K")
        check(site is not None and f"{speedup:.2f}×" in " ".join(site), f"{int(ctx) // 1024}K speedup {speedup:.2f}×")
    oom = bench["specforge_oom_counts"]["262144"]
    check(oom["oom"] == 9 and oom.get("feasible", 0) == 0, f"SpecForge 256K: {oom['oom']}/9 configurations OOM")
    dflash_ours = {ctx_key[r[1]]: pair(r[4])[1] for r in dflash if r[0] == "Qwen3.8-27B"}
    for ctx in (512, 1024):
        label = "512K" if ctx == 512 else "1M"
        site = find_row(site_rows, label, f"{round(dflash_ours[ctx]):,}")
        check(site is not None, f"Turbo-dLLM at {label}: {round(dflash_ours[ctx]):,} tok/s (paper)")

    def gib(value: float) -> str:
        """One decimal, rounded half-up like the page (114.05 -> 114.1)."""
        return str(decimal.Decimal(str(value)).quantize(decimal.Decimal("0.1"), rounding=decimal.ROUND_HALF_UP))

    print("AutoModel comparison vs. data/nemotron14b_automodel_comparison.json")
    auto = json.loads((SITE / "data/nemotron14b_automodel_comparison.json").read_text())
    for row in auto["rows"]:
        label = f"{row['context_tokens'] // 1024}K"
        ours, am = row["ours"], row["automodel"]
        site = find_row(site_rows, label, f"{ours['input_tokens_per_second']:,}", am["topology"])
        joined = " ".join(site) if site else ""
        checks = [ours["topology"] in joined,
                  f"{round(am['input_tokens_per_second']):,}" in joined,
                  f"{gib(ours['peak_hbm_gib'])} GiB" in joined,
                  f"{gib(am['peak_hbm_gib'])} GiB" in joined,
                  f"{row['advantage']:.2f}×" in joined]
        check(site is not None and all(checks),
              f"{label}: {ours['topology']} {ours['input_tokens_per_second']:,} tok/s vs "
              f"{am['topology']} {round(am['input_tokens_per_second']):,} tok/s, {row['advantage']:.2f}×")
    for row in auto["rows"]:
        expected = round(row["ours"]["input_tokens_per_second"] / row["automodel"]["input_tokens_per_second"], 2)
        check(abs(expected - row["advantage"]) < 0.005, f"{row['context_tokens'] // 1024}K advantage is ours/AutoModel ({expected:.2f}×)")
    nemo = {ctx_key[r[1]]: (r[2], pair(r[5])[0], pair(r[7])[0]) for r in two_node if r[0] == "NemotronDiffusion 14B"}
    for row in auto["rows"]:
        topo, tps, hbm = nemo[row["context_tokens"] // 1024]
        ours = row["ours"]
        check(topo.replace("/TP1", "") == ours["topology"] and tps == ours["input_tokens_per_second"] and hbm == ours["peak_hbm_gib"],
              f"{row['context_tokens'] // 1024}K CSBP row matches the paper: {ours['topology']} {tps:,.0f} tok/s, {hbm} GiB")

    print("Headline claims quoted in the page")
    for claim in ["7.59×", "1.61×", "1.18–1.45×", "1.27–1.33×", "4.81×", "+1.8", "+2"]:
        check(claim in page, f"page quotes {claim}")

    print(f"\n{'FAIL' if failures else 'PASS'}: {len(failures)} mismatches")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
