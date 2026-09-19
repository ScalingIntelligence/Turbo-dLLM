// "Load balancing": contiguous block assignment vs. CSBP's dual-end pairing on 3 GPUs with 6 blocks.
// A block's attention work grows with its position (block b costs b units), so handing out blocks in
// order leaves the last GPU with the most work while the others sit idle. Pairing block b with block
// 7 - b gives every GPU the same work. Frames are a pure function of the clock t (seconds).
(function () {
  'use strict';
  var root = document.getElementById('lb-anim');
  var svg = document.getElementById('lb-svg');
  if (!root || !svg) return;

  var NS = 'http://www.w3.org/2000/svg';
  var C = { corrupt: '#B66718', corruptFill: '#F7EADB', red: '#B1040E', ink: '#1C1C1C', muted: '#7a7a7a', grey: '#9aa0a6' };
  var BLOCKS = 6, GPUS = 3;
  var ASSIGN = [
    { title: 'Blocks in order', sub: 'later GPUs get the expensive blocks', red: false, gpus: [[0, 1], [2, 3], [4, 5]] },
    { title: 'CSBP: pair early and late blocks', sub: 'every GPU gets the same work', red: true, gpus: [[0, 5], [1, 4], [2, 3]] }
  ];
  var MAX_LOAD = 11; // heaviest GPU under contiguous assignment: blocks 5 + 6

  // ---- timeline (seconds) ----
  var FLY = [0.5, 3.1], SWEEP = [3.5, 7.9], HOLD_END = 9.8, DURATION = 10.4;
  var SPEED = 0.5; // playback rate: the timeline above is unchanged, it just advances slower
  var FRAMES = [2.0, 6.4, 9.0]; // mid-assignment, balanced step done, both done

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
  function lerp(a, b, u) { return a + (b - a) * u; }
  function ease(u) { return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; }
  function el(tag, attrs, parent, text) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }

  function geometry(narrow) {
    if (!narrow) {
      return { W: 960, H: 402, stripBase: 96, stripU: 11, barW: 32, barGap: 12, panelTop: 134, panelH: 262,
        panels: [{ x: 0, w: 470 }, { x: 490, w: 470 }], colW: 64, colGap: 46, u: 15, base: 234, titleY: 28, subY: 46 };
    }
    return { W: 360, H: 690, stripBase: 88, stripU: 9, barW: 26, barGap: 10, panelTop: 122, panelH: 276,
      panels: [{ x: 0, y: 0, w: 360 }, { x: 0, y: 288, w: 360 }], colW: 56, colGap: 34, u: 14, base: 248, titleY: 26, subY: 44 };
  }

  var G, built = null, parts;

  function build() {
    var narrow = svg.parentNode.clientWidth < 620;
    if (built === narrow) return;
    built = narrow;
    G = geometry(narrow);
    Array.prototype.slice.call(svg.childNodes).forEach(function (n) { if (n.nodeName !== 'desc') svg.removeChild(n); });
    svg.setAttribute('viewBox', '0 0 ' + G.W + ' ' + G.H);

    var defs = el('defs', {}, svg);
    var pat = el('pattern', { id: 'lb-hatch', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
    el('rect', { width: 6, height: 6, fill: '#f3f3f3' }, pat);
    el('line', { x1: 0, y1: 0, x2: 0, y2: 6, stroke: '#c9c9c9', 'stroke-width': 2.2 }, pat);

    // strip: work per block
    var stripW = BLOCKS * G.barW + (BLOCKS - 1) * G.barGap, sx0 = (G.W - stripW) / 2;
    el('text', { x: G.W / 2, y: 16, 'text-anchor': 'middle', 'font-size': narrow ? 11 : 12, 'font-weight': 600, fill: '#555' }, svg,
      'Attention work per block grows with its position (blocks 1\u20136)');
    var strip = [];
    for (var b = 0; b < BLOCKS; b++) {
      var h = (b + 1) * G.stripU, x = sx0 + b * (G.barW + G.barGap);
      el('rect', { x: x, y: G.stripBase - h, width: G.barW, height: h, rx: 4, fill: C.corruptFill, stroke: C.corrupt, 'stroke-width': 1.2, opacity: 0.4 }, svg);
      el('text', { x: x + G.barW / 2, y: G.stripBase + 15, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 700, fill: C.corrupt, opacity: 0.7 }, svg, String(b + 1));
      strip.push({ x: x, y: G.stripBase - h, h: h });
    }

    parts = ASSIGN.map(function (a, pi) {
      var P = G.panels[pi];
      var top = G.panelTop + (P.y || 0);
      var g = el('g', {}, svg);
      el('rect', { x: P.x, y: top, width: P.w, height: G.panelH, rx: 16, fill: a.red ? '#fcf4f4' : '#f6f6f5',
        stroke: a.red ? 'rgba(177,4,14,0.18)' : 'rgba(0,0,0,0.05)' }, g);
      el('text', { x: P.x + 20, y: top + G.titleY, 'font-size': narrow ? 13.5 : 14.5, 'font-weight': 700, fill: a.red ? C.red : C.ink }, g, a.title);
      el('text', { x: P.x + 20, y: top + G.subY, 'font-size': 11.5, fill: C.muted }, g, a.sub);
      var done = el('text', { x: P.x + P.w - 20, y: top + G.titleY, 'text-anchor': 'end', 'font-size': 12.5, 'font-weight': 700,
        fill: a.red ? C.red : '#555', opacity: 0 }, g, '✓ step done');

      var colsW = GPUS * G.colW + (GPUS - 1) * G.colGap, cx0 = P.x + (P.w - colsW) / 2 + 14;
      var base = top + G.base;
      // step-progress axis on the left
      var axisX = cx0 - 26;
      el('line', { x1: axisX, y1: base, x2: axisX, y2: base - MAX_LOAD * G.u - 6, stroke: '#cfcfcf', 'stroke-width': 1.4 }, g);
      el('text', { x: axisX - 6, y: base - MAX_LOAD * G.u / 2, 'text-anchor': 'middle', 'font-size': 10.5, fill: C.muted,
        transform: 'rotate(-90 ' + (axisX - 6) + ' ' + (base - MAX_LOAD * G.u / 2) + ')' }, g, 'work');

      var cols = a.gpus.map(function (blocks, gi) {
        var x = cx0 + gi * (G.colW + G.colGap);
        el('line', { x1: x - 8, y1: base, x2: x + G.colW + 8, y2: base, stroke: '#bdbdbd', 'stroke-width': 1.4 }, g);
        el('text', { x: x + G.colW / 2, y: base + 17, 'text-anchor': 'middle', 'font-size': 11.5, 'font-weight': 700, fill: '#444' }, g, 'GPU ' + (gi + 1));
        var load = 0;
        // heavier block at the bottom so the stack reads as a pair
        var order = blocks.slice().sort(function (p, q) { return q - p; });
        var tiles = order.map(function (bi) {
          var h = (bi + 1) * G.u, y0 = base - (load + bi + 1) * G.u;
          var start = load;
          load += bi + 1;
          var tg = el('g', { opacity: 0 }, g);
          var r = el('rect', { x: 0, y: 0, width: G.colW, height: h - 2, rx: 5, fill: C.corruptFill, stroke: C.corrupt, 'stroke-width': 1.3 }, tg);
          var fillR = el('rect', { x: 0, y: h - 2, width: G.colW, height: 0, rx: 5, fill: a.red ? C.red : C.grey, opacity: 0.85 }, tg);
          var label = el('text', { x: G.colW / 2, y: Math.min(h / 2 + 4, h - 5), 'text-anchor': 'middle', 'font-size': 11.5, 'font-weight': 700, fill: C.corrupt }, tg, String(bi + 1));
          return { g: tg, rect: r, fill: fillR, label: label, b: bi, h: h, start: start,
            from: strip[bi], to: { x: x, y: y0 } };
        });
        var idle = el('rect', { x: x, y: base - load * G.u, width: G.colW, height: 0, rx: 5, fill: 'url(#lb-hatch)', stroke: '#d0d0d0', 'stroke-width': 1 }, g);
        var idleText = el('text', { x: x + G.colW / 2, y: 0, 'text-anchor': 'middle', 'font-size': 10.5, 'font-weight': 700, fill: '#888', opacity: 0 }, g, 'idle');
        return { x: x, load: load, tiles: tiles, idle: idle, idleText: idleText, base: base };
      });
      var panelMax = Math.max.apply(null, cols.map(function (c) { return c.load; }));
      var sweep = el('line', { x1: cx0 - 12, x2: cx0 + colsW + 12, y1: base, y2: base, stroke: a.red ? C.red : '#666', 'stroke-width': 1.6,
        'stroke-dasharray': '5 4', opacity: 0 }, g);
      return { cols: cols, sweep: sweep, done: done, panelMax: panelMax, base: base };
    });
  }

  function render(t) {
    build();
    var fade = t > HOLD_END ? clamp(1 - (t - HOLD_END) / (DURATION - HOLD_END), 0, 1) : 1;
    // both panels process work at the same rate; the step ends when a panel's heaviest GPU finishes
    var work = clamp((t - SWEEP[0]) / (SWEEP[1] - SWEEP[0]), 0, 1) * MAX_LOAD;

    parts.forEach(function (p, pi) {
      var level = Math.min(work, p.panelMax);
      p.cols.forEach(function (c, gi) {
        c.tiles.forEach(function (ti, k) {
          var start = FLY[0] + (pi * 0.1) + gi * 0.35 + k * 0.28;
          var u = clamp((t - start) / 1.0, 0, 1), e = ease(u);
          var x = lerp(ti.from.x, ti.to.x, e), y = lerp(ti.from.y, ti.to.y, e) - Math.sin(Math.PI * e) * 16;
          var sx = lerp(G.barW / G.colW, 1, e), sy = lerp(ti.from.h / ti.h, 1, e);
          ti.g.setAttribute('transform', 'translate(' + x.toFixed(2) + ' ' + y.toFixed(2) + ') scale(' + sx.toFixed(4) + ' ' + sy.toFixed(4) + ')');
          ti.g.setAttribute('opacity', u > 0 ? fade : 0);
          var doneUnits = clamp(level - ti.start, 0, ti.b + 1);
          var fh = Math.max(0, doneUnits * G.u - 2);
          ti.fill.setAttribute('y', (ti.h - 2 - fh).toFixed(2));
          ti.fill.setAttribute('height', fh.toFixed(2));
          ti.label.setAttribute('fill', doneUnits * G.u - 2 >= ti.h / 2 ? '#fff' : C.corrupt);
        });
        var idleUnits = clamp(level - c.load, 0, MAX_LOAD);
        var ih = idleUnits * G.u;
        c.idle.setAttribute('y', (c.base - c.load * G.u - ih).toFixed(2));
        c.idle.setAttribute('height', Math.max(0, ih - 2).toFixed(2));
        c.idle.setAttribute('opacity', fade);
        c.idleText.setAttribute('y', (c.base - c.load * G.u - ih / 2 + 4).toFixed(2));
        c.idleText.setAttribute('opacity', ih > 22 ? fade : 0);
      });
      var active = t >= SWEEP[0] && t < DURATION;
      p.sweep.setAttribute('y1', (p.base - level * G.u).toFixed(2));
      p.sweep.setAttribute('y2', (p.base - level * G.u).toFixed(2));
      p.sweep.setAttribute('opacity', active && work > 0 ? fade : 0);
      p.done.setAttribute('opacity', work >= p.panelMax ? fade : 0);
    });
    if (progress) progress.style.width = (100 * t / DURATION).toFixed(2) + '%';
  }

  var progress = document.getElementById('lb-progress-fill');
  var t = reduceMotion ? FRAMES[2] : 0, playing = false, userPaused = reduceMotion, rafId = null, last = null;
  var playBtn = document.getElementById('lb-play');

  function setPlaying(p) {
    playing = p;
    playBtn.classList.toggle('paused', !p);
    playBtn.setAttribute('aria-label', p ? 'Pause animation' : 'Play animation');
    if (p && rafId === null) { last = null; rafId = requestAnimationFrame(loop); }
  }
  function loop(ts) {
    if (!playing) { rafId = null; return; }
    if (last !== null) t = (t + Math.min(0.1, (ts - last) / 1000) * SPEED) % DURATION;
    last = ts;
    render(t);
    rafId = requestAnimationFrame(loop);
  }
  function seek(time) { t = clamp(time, 0, DURATION - 0.001); render(t); }

  playBtn.addEventListener('click', function () {
    userPaused = playing;
    if (!playing && t >= HOLD_END) t = 0;
    setPlaying(!playing);
  });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      var inView = entries[0].isIntersecting;
      if (inView && !userPaused) setPlaying(true);
      if (!inView && playing) setPlaying(false);
    }, { threshold: 0.3 }).observe(root);
  }
  window.addEventListener('resize', function () { render(t); });

  setPlaying(false);
  render(t);

  window.lbAnimation = { seek: seek, duration: DURATION, phaseFrames: FRAMES.slice(), pause: function () { userPaused = true; setPlaying(false); } };
})();
