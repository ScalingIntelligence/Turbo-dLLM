#!/usr/bin/env Rscript
# Renders every Turbo-dLLM website plot into website/figures/.
# Style is the CSBP "twitter" figure (block_parallel_paper/output/visualizations/
# render_csbp_twitter.R) on a white background.
#
# Usage (from website/):  Rscript scripts/figures/render_figures.R
#
# Hours per 1B tokens = 1e9 / (cluster tok/s * 3600). Data sources (ICLR 2027 preprint):
#   tables/appendix_fast_dllm_v2_scaling.tex   Qwen3.8-27B AR-to-BDLM conversion
#   tables/two_node_h200_1p2_results.tex       DiffusionGemma 26B-A4B
#   tables/dflash2_qwen38_h100_scaling.tex     DFlash2 drafters (Qwen3.8-27B, Muse-Glimmer-30B)
#   figures/equal_wall_clock_downstream.pdf    DiffusionGemma 26B-A4B equal-wall-clock pass rates

suppressPackageStartupMessages({ library(ggplot2); library(grid) })

# FIG_OUT / FIG_RES let another pipeline (for example the tweet media build) render the
# same figures somewhere else at a higher resolution; defaults reproduce the website assets.
out_dir <- Sys.getenv("FIG_OUT", "figures")
fig_res <- as.numeric(Sys.getenv("FIG_RES", "150"))
bg <- "#FFFFFF"; ink <- "#1C1C1C"; sub_ink <- "#6B6B6B"; grid_col <- "#E3E7EB"
ours <- "#B1040E"; theirs <- "#7D848B"; arrow_col <- "#8C8C8C"; minor_lab <- "#6E6E6E"
font <- "Lato"

hours <- function(tps) 1e9 / (tps * 3600)
wl <- function(ctx, base, csbp) data.frame(ctx = ctx, x = log2(ctx), base = hours(base),
                                           csbp = hours(csbp), speedup = csbp / base)
ctx_label <- function(x) ifelse(x >= 10, sprintf("%gM", 2^(x - 10)), sprintf("%gK", 2^x))

# ---- data (tok/s: best baseline, CSBP) ----
qwen_conv <- wl(c(64, 128, 256), c(4697, 3742, 2623), c(5905, 4853, 3490))
gemma     <- wl(c(64, 128, 256, 512), c(22154.241, 14584.023, 10557.288, 6078.576),
                c(27653.010, 21049.381, 15359.422, 9814.617))
dflash_qwen <- wl(c(512, 1024), c(57027, 18389), c(141222, 139541))
dflash_muse <- wl(c(256, 512, 1024), c(62698, 29493, 12842), c(98046, 98785, 61742))

downstream <- list(
  swe = data.frame(h = c(0, 3, 6, 9, 12), base = c(30.0, 32.2, 33.6, 34.6, 35.4),
                   csbp = c(30.0, 33.8, 35.4, 36.0, 36.4)),
  tb  = data.frame(h = c(0, 3, 6, 9, 12), base = c(26, 26, 27, 28, 29),
                   csbp = c(26, 27, 29, 30, 30))
)

base_theme <- function(title_size, s) {
  theme_minimal(base_family = font, base_size = 18 * s) +
    theme(
      plot.background = element_blank(), panel.background = element_blank(),
      panel.grid.major = element_line(color = grid_col, linewidth = 0.5),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = ink, linewidth = 0.8),
      axis.ticks = element_blank(),
      axis.text = element_text(color = ink, size = 18 * s),
      axis.title = element_text(color = ink, face = "bold", size = 17 * s),
      plot.title = element_text(face = "bold", color = ink, size = title_size * s, hjust = 0,
                                lineheight = 1.0, margin = margin(b = 4)),
      plot.subtitle = element_text(color = sub_ink, size = 15 * s, hjust = 0, margin = margin(b = 16)),
      plot.title.position = "panel",
      plot.margin = margin(8, 22, 6, 8)
    )
}

# Speedup panel: grey baseline (hollow) vs green CSBP (filled), dashed arrow per context.
# label_mode "below": label under-right of the CSBP marker; "side": beside the arrow midpoint.
# s scales all text/marks (mobile renders use a larger s relative to the canvas).
panel <- function(df, title, subtitle, ylim, breaks, xlim, label_mode = "below",
                  stack_last = FALSE, s = 1, axis_titles = FALSE, side_left = NULL, below_ctx = NULL) {
  span <- diff(ylim)
  df$col <- ifelse(df$ctx == max(df$ctx), ink, minor_lab)
  if (label_mode == "below") {
    df$lx <- df$x - 0.1; df$ly <- df$csbp - span * 0.07
    df$lab <- sprintf("%.2f× faster", df$speedup); df$hj <- 0; df$vj <- 1
    if (stack_last) df$lab[df$ctx == max(df$ctx)] <- sprintf("%.2f×\nfaster", df$speedup[df$ctx == max(df$ctx)])
  } else {
    right <- if (is.null(side_left)) df$ctx == max(df$ctx) else !(df$ctx %in% side_left)
    df$lx <- df$x + ifelse(right, 0.1, -0.12); df$ly <- (df$base + df$csbp) / 2
    df$lab <- sprintf("%.2f×\nfaster", df$speedup)
    df$hj <- ifelse(right, 0, 1); df$vj <- 0.5
    # short arrows under a steep baseline: single-line label centered under the CSBP marker
    under <- df$ctx %in% below_ctx
    df$lx[under] <- df$x[under]; df$ly[under] <- df$csbp[under] - span * 0.045
    df$lab[under] <- sprintf("%.2f× faster", df$speedup[under]); df$hj[under] <- 0.5; df$vj[under] <- 1
  }
  # arrows only where the gap is visible; pad keeps heads off the markers
  gap <- df$base - df$csbp
  pad <- pmin(span * 0.033, gap * 0.22)
  arrows <- df[gap > span * 0.06, , drop = FALSE]
  arrows$pad <- pad[gap > span * 0.06]
  p <- ggplot(df)
  if (nrow(arrows)) {
    p <- p + geom_segment(data = arrows, aes(x = x, xend = x, y = base - pad, yend = csbp + pad),
                          color = arrow_col, linewidth = 0.9 * s, linetype = "22",
                          arrow = arrow(length = unit(0.12 * s, "in"), type = "open", angle = 28))
  }
  p <- p +
    geom_line(aes(x, base), color = theirs, linewidth = 1.8 * s, alpha = 0.75, lineend = "round") +
    geom_line(aes(x, csbp), color = ours, linewidth = 2.6 * s, lineend = "round") +
    geom_point(aes(x, base), shape = 21, fill = bg, color = theirs, size = 6 * s, stroke = 1.8 * s) +
    geom_point(aes(x, csbp), color = bg, size = 7.6 * s) +
    geom_point(aes(x, csbp), color = ours, size = 6 * s) +
    geom_text(aes(x = lx, y = ly, label = lab, color = I(col), hjust = hj, vjust = vj),
              family = font, fontface = "bold", size = 6.6 * s, lineheight = 0.88) +
    scale_x_continuous(breaks = df$x, labels = ctx_label, limits = xlim, expand = expansion(0)) +
    scale_y_continuous(breaks = breaks, labels = function(v) paste0(v, "h"),
                       limits = ylim, expand = expansion(0)) +
    labs(title = title, subtitle = subtitle,
         x = if (axis_titles) "Context length" else NULL,
         y = if (axis_titles) "Hours per 1B tokens" else NULL) +
    base_theme(18.5, s) + coord_cartesian(clip = "off")
  p
}

# Equal-wall-clock pass-rate panel (downstream).
downstream_panel <- function(df, title, subtitle, ylim, breaks, s = 1, axis_titles = FALSE, digits = 1) {
  fmt <- function(v) formatC(v, format = "f", digits = digits)
  off <- diff(ylim) * 0.055
  lab_c <- data.frame(h = df$h, y = df$csbp + off, lab = fmt(df$csbp))
  lab_b <- data.frame(h = df$h, y = df$base - off, lab = fmt(df$base))
  # at hour 0 both runs are the same base model: one neutral label
  lab_c <- lab_c[df$h > 0, ]; lab_b <- lab_b[df$h > 0, ]
  zero <- data.frame(h = 0, y = df$base[1] - off, lab = fmt(df$base[1]))
  ggplot(df) +
    geom_line(aes(h, base), color = theirs, linewidth = 1.8 * s, alpha = 0.75, lineend = "round") +
    geom_line(aes(h, csbp), color = ours, linewidth = 2.6 * s, lineend = "round") +
    geom_point(aes(h, base), shape = 21, fill = bg, color = theirs, size = 5.4 * s, stroke = 1.8 * s) +
    geom_point(aes(h, csbp), color = bg, size = 7 * s) +
    geom_point(aes(h, csbp), color = ours, size = 5.4 * s) +
    geom_text(data = lab_c, aes(h, y, label = lab), color = ours, family = font, fontface = "bold",
              size = 5.6 * s, vjust = 0) +
    geom_text(data = lab_b, aes(h, y, label = lab), color = theirs, family = font, fontface = "bold",
              size = 5.6 * s, vjust = 1) +
    geom_text(data = zero, aes(h, y, label = lab), color = minor_lab, family = font, fontface = "bold",
              size = 5.6 * s, vjust = 1, hjust = 0.2) +
    scale_x_continuous(breaks = df$h, labels = function(v) paste0(v, "h"),
                       limits = c(-0.6, 12.8), expand = expansion(0)) +
    scale_y_continuous(breaks = breaks, labels = function(v) paste0(v, "%"),
                       limits = ylim, expand = expansion(0)) +
    labs(title = title, subtitle = subtitle,
         x = if (axis_titles) "Training time" else NULL,
         y = if (axis_titles) "Pass rate" else NULL) +
    base_theme(18.5, s) + coord_cartesian(clip = "off")
}

# ---- composition ----
draw_key <- function(labels, fontsize, y = 0.5, stacked = FALSE) {
  g1 <- gpar(fontfamily = font, fontface = "bold", fontsize = fontsize, col = ours)
  g2 <- gpar(fontfamily = font, fontface = "bold", fontsize = fontsize, col = theirs)
  dot <- unit(fontsize * 0.8, "pt"); gap <- unit(fontsize * 0.45, "pt"); sep <- unit(fontsize * 2.2, "pt")
  inch <- function(u) convertWidth(u, "in", valueOnly = TRUE)
  w1 <- inch(grobWidth(textGrob(labels[1], gp = g1))); w2 <- inch(grobWidth(textGrob(labels[2], gp = g2)))
  d <- inch(dot); gp_ <- inch(gap); sp <- inch(sep)
  entry <- function(x0, yy, lab, filled, g) {
    if (filled) grid.points(x0 + unit(d / 2, "in"), unit(yy, "npc"), pch = 16, size = dot, gp = gpar(col = ours))
    else grid.points(x0 + unit(d / 2, "in"), unit(yy, "npc"), pch = 21, size = dot,
                     gp = gpar(col = theirs, fill = bg, lwd = fontsize / 10))
    grid.text(lab, x = x0 + unit(d + gp_, "in"), y = yy, just = "left", gp = g)
  }
  if (stacked) {
    wmax <- d + gp_ + max(w1, w2)
    x0 <- unit(0.5, "npc") - unit(wmax / 2, "in")
    entry(x0, y + 0.18, labels[1], TRUE, g1); entry(x0, y - 0.18, labels[2], FALSE, g2)
  } else {
    total <- d + gp_ + w1 + sp + d + gp_ + w2
    x0 <- unit(0.5, "npc") - unit(total / 2, "in")
    entry(x0, y, labels[1], TRUE, g1); entry(x0 + unit(d + gp_ + w1 + sp, "in"), y, labels[2], FALSE, g2)
  }
}

aligned_grobs <- function(plots) {
  gs <- lapply(plots, ggplotGrob)
  w <- do.call(grid::unit.pmax, lapply(gs, function(g) g$widths))
  lapply(gs, function(g) { g$widths <- w; g })
}

# Desktop: key on top, panels in a row, shared y and x titles.
compose_row <- function(plots, key, ytitle, xtitle, key_size = if (length(plots) == 3) 24 else 20) {
  n <- length(plots)
  grid.newpage(); grid.rect(gp = gpar(fill = bg, col = NA))
  lay <- grid.layout(3, n + 1, heights = unit(c(0.8, 1, 0.6), c("in", "null", "in")),
                     widths = unit(c(0.45, rep(1, n)), c("in", rep("null", n))))
  pushViewport(viewport(layout = lay, width = 0.965, height = 0.96))
  pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 1:(n + 1))); draw_key(key, key_size); popViewport()
  pushViewport(viewport(layout.pos.row = 2, layout.pos.col = 1))
  grid.text(ytitle, rot = 90, gp = gpar(fontfamily = font, fontface = "bold", fontsize = 21, col = ink))
  popViewport()
  pushViewport(viewport(layout.pos.row = 3, layout.pos.col = 2:(n + 1)))
  grid.text(xtitle, y = 0.55, gp = gpar(fontfamily = font, fontface = "bold", fontsize = 21, col = ink))
  popViewport()
  gs <- aligned_grobs(plots)
  for (i in seq_along(gs)) {
    pushViewport(viewport(layout.pos.row = 2, layout.pos.col = i + 1)); grid.draw(gs[[i]]); popViewport()
  }
  popViewport()
}

# Mobile: stacked key, one panel per row with its own axis titles.
compose_stack <- function(plots, key, key_size = 30) {
  n <- length(plots)
  grid.newpage(); grid.rect(gp = gpar(fill = bg, col = NA))
  lay <- grid.layout(n + 1, 1, heights = unit(c(1.25, rep(1, n)), c("in", rep("null", n))))
  pushViewport(viewport(layout = lay, width = 0.94, height = 0.975))
  pushViewport(viewport(layout.pos.row = 1)); draw_key(key, key_size, stacked = TRUE); popViewport()
  gs <- aligned_grobs(plots)
  for (i in seq_along(gs)) {
    pushViewport(viewport(layout.pos.row = i + 1)); grid.draw(gs[[i]]); popViewport()
  }
  popViewport()
}

save_png <- function(stem, w, h, draw_fn, res = fig_res) {
  path <- file.path(out_dir, paste0(stem, ".png"))
  ragg::agg_png(path, width = w, height = h, units = "in", res = res, background = bg)
  draw_fn(); invisible(dev.off())
  message("wrote ", path)
}

key_parallel <- c("Context-Sharded Block Parallelism (CSBP)", "Best existing parallelism strategies")
dir.create(out_dir, showWarnings = FALSE)

# ---- 1. headline ----
headline_panels <- function(s = 1, axis_titles = FALSE, one_line = TRUE) {
  list(
    panel(qwen_conv, "Autoregressive → block diffusion training", "Qwen3.8-27B · 16× H200",
          ylim = c(30, 112), breaks = c(40, 60, 80, 100), xlim = c(6 - 0.3, 8 + 0.75),
          stack_last = TRUE, s = s, axis_titles = axis_titles),
    panel(gemma, "Block diffusion fine-tuning", "DiffusionGemma 26B-A4B · 16× H200",
          ylim = c(4, 48), breaks = c(10, 20, 30, 40), xlim = c(6 - 0.3, 9 + 0.85),
          stack_last = TRUE, s = s, axis_titles = axis_titles),
    panel(dflash_qwen, "Speculative decoding drafter training", "DFlash2 for Qwen3.8-27B · 8× H100",
          ylim = c(0, 16.5), breaks = c(0, 5, 10, 15), xlim = c(9 - 0.85, 10 + 1.05),
          label_mode = "side", s = s, axis_titles = axis_titles)
  )
}
draw_headline <- function() compose_row(headline_panels(), key_parallel,
                                        "Training time per 1B tokens", "Context length")
save_png("headline", 16, 9, draw_headline)
save_png("social-card", 16, 8.4, draw_headline)  # 2400 x 1260, 1.91:1 link preview
save_png("headline-mobile", 7.2, 17, function() compose_stack(headline_panels(s = 1, axis_titles = TRUE), key_parallel, 17))

# ---- 2. context-length scaling ----
scaling_panels <- function(s = 1, axis_titles = FALSE) {
  list(
    panel(qwen_conv, "Autoregressive → block diffusion training", "Qwen3.8-27B · 16× H200",
          ylim = c(30, 112), breaks = c(40, 60, 80, 100), xlim = c(6 - 0.3, 8 + 0.75),
          stack_last = TRUE, s = s, axis_titles = axis_titles),
    panel(gemma, "Block diffusion fine-tuning", "DiffusionGemma 26B-A4B · 16× H200",
          ylim = c(4, 48), breaks = c(10, 20, 30, 40), xlim = c(6 - 0.3, 9 + 0.85),
          stack_last = TRUE, s = s, axis_titles = axis_titles)
  )
}
save_png("context-scaling", 13, 8, function() compose_row(scaling_panels(), key_parallel,
                                                           "Training time per 1B tokens", "Context length"))
save_png("context-scaling-mobile", 7.2, 12, function() compose_stack(scaling_panels(axis_titles = TRUE), key_parallel, 17))

# ---- 3. DFlash2 drafters (shared 256K-1M axis and hour range) ----
dflash_panels <- function(s = 1, axis_titles = FALSE) {
  xl <- c(8 - 0.75, 10 + 1.0)
  list(
    panel(dflash_qwen, "DFlash2 drafter for Qwen3.8-27B", "Speculative decoding · 8× H100",
          ylim = c(0, 24), breaks = c(0, 5, 10, 15, 20), xlim = xl, label_mode = "side",
          s = s, axis_titles = axis_titles, side_left = 512),
    panel(dflash_muse, "DFlash2 drafter for Muse-Glimmer-30B", "Speculative decoding · 8× H100",
          ylim = c(0, 24), breaks = c(0, 5, 10, 15, 20), xlim = xl, label_mode = "side",
          s = s, axis_titles = axis_titles, side_left = c(256, 512), below_ctx = 512)
  )
}
save_png("dflash2-scaling", 13, 8, function() compose_row(dflash_panels(), key_parallel,
                                                          "Training time per 1B tokens", "Context length"))
save_png("dflash2-scaling-mobile", 7.2, 12, function() compose_stack(dflash_panels(axis_titles = TRUE), key_parallel, 17))

# ---- 4. equal wall clock downstream ----
key_down <- c("Trained with CSBP", "Trained with best existing parallelism")
down_panels <- function(s = 1, axis_titles = FALSE) {
  list(
    downstream_panel(downstream$swe, "SWE-bench Verified", "DiffusionGemma 26B-A4B · 8× H100 · LoRA SFT",
                     ylim = c(28.5, 37.6), breaks = c(30, 32, 34, 36), s = s, axis_titles = axis_titles),
    downstream_panel(downstream$tb, "Terminal-Bench Lite", "DiffusionGemma 26B-A4B · 8× H100 · LoRA SFT",
                     ylim = c(25, 31), breaks = c(26, 27, 28, 29, 30), s = s, axis_titles = axis_titles, digits = 0)
  )
}
save_png("equal-wall-clock", 13, 7.6, function() compose_row(down_panels(), key_down, "Pass rate", "Training time"))
save_png("equal-wall-clock-mobile", 7.2, 12, function() compose_stack(down_panels(axis_titles = TRUE), key_down, 17))
