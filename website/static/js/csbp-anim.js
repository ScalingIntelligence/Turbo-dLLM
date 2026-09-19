// "CSBP in motion": conventional context parallelism vs. CSBP on 3 GPUs with 6 blocks.
// Every frame is a pure function of the clock t (seconds), so the animation can be paused,
// jumped to a step, or rendered at a fixed time (window.csbpAnimation.seek) for checks.
(function () {
  'use strict';
  var root = document.getElementById('csbp-anim');
  var svg = document.getElementById('anim-svg');
  if (!root || !svg) return;

  var NS = 'http://www.w3.org/2000/svg';
  var C = { clean: '#356F9F', cleanFill: '#E3ECF4', corrupt: '#B66718', corruptFill: '#F7EADB', red: '#B1040E', ink: '#1C1C1C', muted: '#7a7a7a' };
  var GPUS = 3, BLOCKS = 6;

  // ---- timeline (seconds) ----
  var SHARD = [0.6, 3.4];
  var HOPS = [ // forward ring (+1), then backward ring (-1)
    { start: 3.6, end: 5.5, dir: 1, grad: false },
    { start: 5.8, end: 7.7, dir: 1, grad: false },
    { start: 8.2, end: 10.1, dir: -1, grad: true },
    { start: 10.4, end: 12.3, dir: -1, grad: true }
  ];
  var RESULT = 12.6, RESET = 16.0, DURATION = 16.6;
  var SPEED = 0.5; // playback rate: the timeline above is unchanged, it just advances slower
  var PHASE_START = [0, 3.6, 8.2, RESULT];
  var PHASE_FRAME = [3.2, 4.7, 9.3, 14.5]; // representative still frame per step
  var CAPTIONS = [
    '<b>Shard.</b> Context parallelism cuts both the clean and the corrupted copy by token position. CSBP cuts only the clean context by position and gives each corrupted block, whole, to one GPU, pairing early and late blocks (1+6, 2+5, 3+4) to balance the work.',
    '<b>Forward.</b> Attention passes keys and values around the ring of GPUs. Corrupted K/V are only read by their own block, yet context parallelism ships them to every GPU. CSBP sends clean K/V only.',
    '<b>Backward.</b> Gradients retrace the ring. Under CSBP, each block’s loss and corrupted gradients never leave its GPU; only gradients for the shared clean context travel.',
    '<b>Result.</b> Same attention, same loss and gradients, far less traffic. CSBP only ever moves the shared clean context between GPUs, so each training step spends less time communicating and finishes sooner.'
  ];

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
  function lerp(a, b, u) { return a + (b - a) * u; }
  function ease(u) { return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; }
  function phaseAt(t) { return t < PHASE_START[1] - 0.1 ? 0 : t < PHASE_START[2] - 0.2 ? 1 : t < RESULT - 0.1 ? 2 : 3; }
  function el(tag, attrs, parent, text) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }

  // ---- geometry for wide and narrow layouts ----
  function geometry(narrow) {
    if (!narrow) {
      return {
        W: 960, H: 596, s: 24, gap: 8, stripTitleY: 16, rowY: [30, 60], stripLabel: 12,
        laneTop: [108, 356], laneH: 232, titleX: 24, titleY: 36, subY: [56, 72],
        cardX0: 214, cardW: 224, cardGap: 18, cardDY: 52, cardH: 140, cardS: 28, cardRowDY: [40, 84], cardLabels: true,
        chanTop: 30, chanBot: 208, pktW: 28, pktH: 15, pktText: true,
        meter: { x: 24, labelY: 150, barY: 160, w: 160, valueY: 190, valueX: 24 }, loss: 52, localY: 132, titleSize: 15
      };
    }
    return {
      W: 360, H: 648, s: 20, gap: 6, stripTitleY: 14, rowY: [26, 50], stripLabel: 10,
      laneTop: [86, 370], laneH: 274, titleX: 14, titleY: 24, subY: [41],
      cardX0: 10, cardW: 106, cardGap: 11, cardDY: 76, cardH: 116, cardS: 22, cardRowDY: [30, 66], cardLabels: false,
      chanTop: 62, chanBot: 206, pktW: 20, pktH: 12, pktText: false,
      meter: { x: 14, labelY: 236, barY: 246, w: 332, valueY: 236, valueX: 346, anchor: 'end' }, loss: 34, localY: 108, titleSize: 14
    };
  }

  var G, lanes, stripTiles, built = null;

  function build() {
    var narrow = svg.parentNode.clientWidth < 620;
    if (built === narrow) return;
    built = narrow;
    G = geometry(narrow);
    Array.prototype.slice.call(svg.childNodes).forEach(function (n) {
      if (n.nodeName !== 'desc') svg.removeChild(n);
    });
    svg.setAttribute('viewBox', '0 0 ' + G.W + ' ' + G.H);

    // input strip
    var stripW = BLOCKS * G.s + (BLOCKS - 1) * G.gap;
    var sx0 = (G.W - stripW) / 2;
    el('text', { x: G.W / 2, y: G.stripTitleY, 'text-anchor': 'middle', 'font-size': narrow ? 11 : 12, 'font-weight': 600, fill: '#555' }, svg,
      'One training sequence · 6 blocks');
    el('text', { x: sx0 - 10, y: G.rowY[0] + G.s * 0.68, 'text-anchor': 'end', 'font-size': G.stripLabel, fill: C.muted }, svg, 'clean');
    el('text', { x: sx0 - 10, y: G.rowY[1] + G.s * 0.68, 'text-anchor': 'end', 'font-size': G.stripLabel, fill: C.muted }, svg, 'corrupted');
    stripTiles = [];
    for (var row = 0; row < 2; row++) {
      for (var b = 0; b < BLOCKS; b++) {
        var x = sx0 + b * (G.s + G.gap), y = G.rowY[row];
        tile(svg, row, b, x, y, G.s, 0.35);
        stripTiles.push({ row: row, b: b, x: x, y: y });
      }
    }

    lanes = [makeLane(0, 'Context parallelism', narrow ? 'splits every token by position' : ['splits the clean and corrupted', 'copies by token position']),
             makeLane(1, 'CSBP', narrow ? 'splits the context, keeps blocks whole' : ['splits the clean context;', 'keeps each block on one GPU'])];
  }

  function tile(parent, row, b, x, y, s, opacity) {
    var g = el('g', { transform: 'translate(' + x + ' ' + y + ')', opacity: opacity == null ? 1 : opacity }, parent);
    var r = el('rect', { width: s, height: s, rx: 5, fill: row ? C.corruptFill : C.cleanFill, stroke: row ? C.corrupt : C.clean, 'stroke-width': 1.4 }, g);
    if (row) r.setAttribute('stroke-dasharray', '4 2');
    el('text', { x: s / 2, y: s / 2 + 4, 'text-anchor': 'middle', 'font-size': Math.round(s * 0.46), 'font-weight': 700, fill: row ? C.corrupt : C.clean }, g, String(b + 1));
    return { g: g, rect: r };
  }

  function cardX(i) { return G.cardX0 + i * (G.cardW + G.cardGap); }
  function cardCX(i) { return cardX(i) + G.cardW / 2; }
  // horizontal center of the tile pair inside a card (shifted right when row labels are shown)
  function slotCX(i) { return cardCX(i) + (G.cardLabels ? 18 : 0); }

  function slotFor(lane, row, b) {
    if (lane === 1 && row === 1) { // CSBP: whole corrupted blocks, dual-end pairs (1+6, 2+5, 3+4)
      return { gpu: b < 3 ? b : BLOCKS - 1 - b, k: b < 3 ? 0 : 1 };
    }
    return { gpu: Math.floor(b / 2), k: b % 2 }; // by token position
  }

  function makeLane(idx, title, sub) {
    var top = G.laneTop[idx];
    var lane = { idx: idx, tiles: [], packets: [], cards: [] };
    var g = el('g', {}, svg);
    el('rect', { x: 0, y: top, width: G.W, height: G.laneH, rx: 16, fill: idx ? '#fcf4f4' : '#f6f6f5', stroke: idx ? 'rgba(177,4,14,0.18)' : 'rgba(0,0,0,0.05)' }, g);
    el('text', { x: G.titleX, y: top + G.titleY, 'font-size': G.titleSize, 'font-weight': 700, fill: idx ? C.red : C.ink }, g, title);
    [].concat(sub).forEach(function (line, i) {
      el('text', { x: G.titleX, y: top + G.subY[0] + i * 16, 'font-size': 11.5, fill: C.muted }, g, line);
    });

    // ring channels: hops between neighbors run above the cards; the wrap-around hop runs below
    var yTop = top + G.chanTop, yBot = top + G.chanBot;
    var ring = el('g', { stroke: '#d9d9d9', 'stroke-width': 1.2, fill: 'none', 'stroke-dasharray': '3 4' }, g);
    el('line', { x1: cardCX(0), y1: yTop, x2: cardCX(GPUS - 1), y2: yTop }, ring);
    el('line', { x1: cardCX(0), y1: yBot, x2: cardCX(GPUS - 1), y2: yBot }, ring);
    for (var i = 0; i < GPUS; i++) {
      el('line', { x1: cardCX(i), y1: yTop, x2: cardCX(i), y2: top + G.cardDY }, ring);
      el('line', { x1: cardCX(i), y1: top + G.cardDY + G.cardH, x2: cardCX(i), y2: yBot }, ring);
    }

    // GPU cards
    for (i = 0; i < GPUS; i++) {
      var cx = cardX(i), cy = top + G.cardDY;
      el('rect', { x: cx, y: cy, width: G.cardW, height: G.cardH, rx: 12, fill: '#fff', stroke: 'rgba(0,0,0,0.12)', 'stroke-width': 1.2 }, g);
      el('text', { x: cx + (G.cardLabels ? 14 : 8), y: cy + (G.cardLabels ? 22 : 16), 'font-size': G.cardLabels ? 12 : 11, 'font-weight': 700, fill: '#444' }, g, 'GPU ' + (i + 1));
      if (G.cardLabels) {
        el('text', { x: cx + 14, y: cy + G.cardRowDY[0] + G.cardS * 0.62, 'font-size': 10.5, fill: C.muted }, g, 'clean');
        el('text', { x: cx + 14, y: cy + G.cardRowDY[1] + G.cardS * 0.62, 'font-size': 10.5, fill: C.muted }, g, 'corrupted');
      }
      var local = null, loss = null;
      if (idx === 1) {
        local = el('text', { x: slotCX(i), y: cy + G.localY, 'text-anchor': 'middle', 'font-size': G.cardLabels ? 10.5 : 9, 'font-weight': 600, fill: C.red, opacity: 0 }, g,
          G.cardLabels ? 'blocks computed here' : 'stays on GPU');
        loss = el('text', { x: slotCX(i) + G.loss, y: cy + G.cardRowDY[1] + G.cardS * 0.72, 'text-anchor': 'middle', 'font-size': G.cardLabels ? 15 : 12,
          'font-style': 'italic', 'font-weight': 700, fill: C.red, opacity: 0, 'font-family': 'Georgia, serif' }, g, 'ℒ');
      }
      lane.cards.push({ local: local, loss: loss });
    }

    // tiles that fly from the input strip into their GPU slots
    stripTiles.forEach(function (st) {
      var slot = slotFor(idx, st.row, st.b);
      var pitch = G.cardS + (G.cardLabels ? 12 : 8);
      var tx = slotCX(slot.gpu) - (pitch + G.cardS) / 2 + slot.k * pitch;
      var ty = top + G.cardDY + G.cardRowDY[st.row];
      lane.tiles.push({ node: tile(g, st.row, st.b, st.x, st.y, G.s, 0), row: st.row, b: st.b, from: { x: st.x, y: st.y }, to: { x: tx, y: ty } });
    });

    // packets: 4 hops x 3 GPUs; CP sends clean + corrupted, CSBP sends clean only
    HOPS.forEach(function (hop) {
      for (var src = 0; src < GPUS; src++) {
        var dst = (src + hop.dir + GPUS) % GPUS;
        var wraps = Math.abs(dst - src) > 1;
        var pg = el('g', { opacity: 0 }, g);
        var parts = idx === 0 ? [0, 1] : [0];
        parts.forEach(function (row, pi) {
          var r = el('rect', { x: pi * (G.pktW + 2), y: 0, width: G.pktW, height: G.pktH, rx: 4,
            fill: row ? C.corrupt : C.clean, stroke: row ? C.corrupt : C.clean, 'stroke-width': 1.5 }, pg);
          if (hop.grad) { r.setAttribute('fill-opacity', '0.18'); r.setAttribute('stroke-dasharray', '3 2'); }
          if (G.pktText) {
            el('text', { x: pi * (G.pktW + 2) + G.pktW / 2, y: G.pktH - 4, 'text-anchor': 'middle', 'font-size': 8.5, 'font-weight': 700,
              fill: hop.grad ? (row ? C.corrupt : C.clean) : '#fff' }, pg, hop.grad ? '∇' : 'K/V');
          }
        });
        lane.packets.push({ g: pg, hop: hop, width: parts.length * G.pktW + (parts.length - 1) * 2,
          x0: cardCX(src), x1: cardCX(dst), y: wraps ? yBot : yTop, units: parts.length });
      }
    });

    // traffic meter
    el('text', { x: G.meter.x, y: top + G.meter.labelY, 'font-size': 11, fill: C.muted }, g, 'traffic between GPUs');
    el('rect', { x: G.meter.x, y: top + G.meter.barY, width: G.meter.w, height: 10, rx: 5, fill: '#e9e9e9' }, g);
    lane.meter = el('rect', { x: G.meter.x, y: top + G.meter.barY, width: 0, height: 10, rx: 5, fill: idx ? C.red : '#9aa0a6' }, g);
    lane.meterText = el('text', { x: G.meter.valueX, y: top + G.meter.valueY, 'text-anchor': G.meter.anchor || 'start', 'font-size': G.meter.anchor ? 11 : 12, 'font-weight': 700, fill: idx ? C.red : '#555', opacity: 0 }, g,
      idx ? 'clean K/V only' : 'clean + corrupted K/V');
    return lane;
  }

  // ---- render ----
  var MAX_UNITS = HOPS.length * GPUS * 2;
  var lastPhase = -1;
  var chips = root.querySelectorAll('.anim-chip');
  var caption = document.getElementById('anim-caption');
  var progress = document.getElementById('anim-progress-fill');

  function render(t) {
    build();
    var fade = t > RESET ? clamp(1 - (t - RESET) / (DURATION - RESET), 0, 1) : 1;

    lanes.forEach(function (lane) {
      lane.tiles.forEach(function (ti) {
        var start = SHARD[0] + ti.b * 0.22 + ti.row * 0.12;
        var u = clamp((t - start) / 1.2, 0, 1), e = ease(u);
        var x = lerp(ti.from.x, ti.to.x, e);
        var y = lerp(ti.from.y, ti.to.y, e) - Math.sin(Math.PI * e) * 18;
        var s = lerp(G.s, G.cardS, e);
        ti.node.g.setAttribute('transform', 'translate(' + x.toFixed(2) + ' ' + y.toFixed(2) + ') scale(' + (s / G.s).toFixed(4) + ')');
        ti.node.g.setAttribute('opacity', u > 0 ? fade : 0);
        var glow = lane.idx === 1 && ti.row === 1 && t > HOPS[0].start && t < RESET;
        ti.node.rect.setAttribute('stroke', glow ? C.red : (ti.row ? C.corrupt : C.clean));
        ti.node.rect.setAttribute('stroke-width', glow ? 2.6 : 1.4);
        ti.node.rect.setAttribute('stroke-dasharray', ti.row && !glow ? '4 2' : 'none');
      });

      var units = 0;
      lane.packets.forEach(function (p) {
        var u = (t - p.hop.start) / (p.hop.end - p.hop.start); // same ring and timing; only the payload differs
        if (u >= 1) units += p.units; else if (u > 0) units += p.units * u;
        var e = ease(clamp(u, 0, 1));
        var x = lerp(p.x0, p.x1, e) - p.width / 2;
        p.g.setAttribute('transform', 'translate(' + x.toFixed(2) + ' ' + (p.y - G.pktH / 2) + ')');
        p.g.setAttribute('opacity', u > 0 && u < 1 ? Math.min(1, u * 8, (1 - u) * 8) : 0);
      });
      lane.meter.setAttribute('width', (G.meter.w * units * fade / MAX_UNITS).toFixed(2));
      lane.meterText.setAttribute('opacity', t >= RESULT ? fade : 0);

      lane.cards.forEach(function (c) {
        if (c.local) c.local.setAttribute('opacity', t > HOPS[0].start ? clamp((t - HOPS[0].start) * 3, 0, 1) * fade : 0);
        if (c.loss) c.loss.setAttribute('opacity', t > HOPS[2].start ? fade : 0);
      });
    });

    var phase = phaseAt(t);
    if (phase !== lastPhase) {
      lastPhase = phase;
      caption.innerHTML = CAPTIONS[phase];
      Array.prototype.forEach.call(chips, function (c, i) {
        c.classList.toggle('active', i === phase);
        c.setAttribute('aria-selected', i === phase ? 'true' : 'false');
      });
    }
    if (progress) progress.style.width = (100 * t / DURATION).toFixed(2) + '%';
  }

  // ---- playback ----
  var t = reduceMotion ? PHASE_FRAME[1] : 0, playing = false, userPaused = reduceMotion, rafId = null, last = null;
  var playBtn = document.getElementById('anim-play');

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
    if (!playing && t >= RESET) t = 0;
    setPlaying(!playing);
  });
  Array.prototype.forEach.call(chips, function (chip) {
    chip.addEventListener('click', function () {
      var i = +chip.getAttribute('data-phase');
      seek(playing ? PHASE_START[i] : PHASE_FRAME[i]);
    });
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

  window.csbpAnimation = { seek: seek, duration: DURATION, phaseFrames: PHASE_FRAME.slice(), pause: function () { userPaused = true; setPlaying(false); } };
})();
