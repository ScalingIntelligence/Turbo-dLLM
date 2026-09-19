// Renders the site in headless Chrome over the DevTools protocol and checks it.
//   node scripts/check_render.mjs [url] [outdir]
// For desktop (1440px) and phone (390px, true mobile emulation), it:
//   - fails on console errors, failed/404 requests, or page-level horizontal overflow
//   - checks every <img> decoded (naturalWidth > 0)
//   - opens every <details> and writes a full-page screenshot to outdir
import { spawn } from 'node:child_process';
import { mkdtempSync, writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const url = process.argv[2] || 'http://127.0.0.1:8000/';
const outdir = process.argv[3] || 'render-check';
mkdirSync(outdir, { recursive: true });
const chromePath = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port = 9300 + Math.floor(Math.random() * 500);
const profile = mkdtempSync(join(tmpdir(), 'site-check-'));
const chrome = spawn(chromePath, [
  '--headless=new', '--disable-gpu', '--hide-scrollbars', `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`, 'about:blank',
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function targetWs() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
      const page = list.find((t) => t.type === 'page');
      if (page) return page.webSocketDebuggerUrl;
    } catch { /* not up yet */ }
    await sleep(200);
  }
  throw new Error('Chrome did not start');
}

const ws = new WebSocket(await targetWs());
await new Promise((r) => ws.addEventListener('open', r, { once: true }));
let nextId = 1;
const pending = new Map();
const listeners = [];
ws.addEventListener('message', (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id);
    pending.delete(msg.id);
    msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
  } else if (msg.method) {
    listeners.forEach((fn) => fn(msg));
  }
});
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const id = nextId++;
  pending.set(id, { resolve, reject });
  ws.send(JSON.stringify({ id, method, params }));
});
const evaluate = async (expression) =>
  (await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })).result.value;

const problems = [];
const requests = new Map();
listeners.push((m) => {
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
    problems.push(`console error: ${m.params.args.map((a) => a.value ?? a.description).join(' ')}`);
  }
  if (m.method === 'Runtime.exceptionThrown') problems.push(`exception: ${m.params.exceptionDetails.text}`);
  if (m.method === 'Network.requestWillBeSent') requests.set(m.params.requestId, m.params.request.url);
  if (m.method === 'Network.responseReceived' && m.params.response.status >= 400) {
    problems.push(`HTTP ${m.params.response.status}: ${m.params.response.url}`);
  }
  if (m.method === 'Network.loadingFailed' && !m.params.canceled) {
    problems.push(`request failed (${m.params.errorText}): ${requests.get(m.params.requestId)}`);
  }
});
await send('Runtime.enable');
await send('Network.enable');
await send('Page.enable');

const viewports = [
  { name: 'desktop', width: 1440, height: 900, mobile: false, scale: 1 },
  { name: 'phone', width: 390, height: 844, mobile: true, scale: 2 },
];

for (const vp of viewports) {
  await send('Emulation.setDeviceMetricsOverride', {
    width: vp.width, height: vp.height, deviceScaleFactor: vp.scale, mobile: vp.mobile,
  });
  await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] });
  await send('Page.navigate', { url });
  await sleep(3500);

  const report = await evaluate(`(async () => {
    await document.fonts.ready;
    document.querySelectorAll('details').forEach(d => d.open = true);
    document.querySelectorAll('.reveal').forEach(el => el.classList.add('in'));
    const imgs = [...document.images];
    await Promise.all(imgs.map(i => i.decode().catch(() => null)));
    const broken = imgs.filter(i => !i.naturalWidth).map(i => i.currentSrc || i.src);
    const docW = document.documentElement.scrollWidth, vw = window.innerWidth;
    const wide = [...document.querySelectorAll('body *')].filter(el => {
      const r = el.getBoundingClientRect();
      if (r.right <= vw + 1) return false;
      for (let p = el.parentElement; p; p = p.parentElement) {
        const ox = getComputedStyle(p).overflowX;
        if (ox === 'auto' || ox === 'scroll' || ox === 'hidden') return false;
      }
      return true;
    }).slice(0, 5).map(el => el.tagName + '.' + el.className);
    const sources = imgs.map(i => i.currentSrc.split('/').pop());
    return { vw, docW, broken, wide, sources, height: document.documentElement.scrollHeight,
             fonts: [...document.fonts].filter(f => f.status === 'loaded').map(f => f.family) };
  })()`);

  if (report.docW > report.vw) problems.push(`${vp.name}: page scrolls horizontally (${report.docW} > ${report.vw}): ${report.wide.join(', ')}`);
  report.broken.forEach((b) => problems.push(`${vp.name}: image failed to decode: ${b}`));
  console.log(`${vp.name}: viewport ${report.vw}px, page ${report.height}px, images: ${report.sources.join(' ')}`);
  console.log(`${vp.name}: fonts loaded: ${[...new Set(report.fonts)].join(', ') || '(none)'}`);

  await sleep(400);
  const shot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  const file = join(outdir, `${vp.name}.png`);
  writeFileSync(file, Buffer.from(shot.data, 'base64'));
  console.log(`${vp.name}: screenshot ${file}`);

  // Close-ups of individual cards for visual review.
  for (const [name, selector] of [['hero', '.hero-lead'], ['scaling', 'img[src*="context-scaling"]'], ['results', '.results-table'], ['traffic', '#details ~ .sub-accordion'], ['lib', '.lib-table'], ['why', '.why-table']]) {
    const box = await evaluate(`(() => { const el = document.querySelector('${selector}'); if (!el) return null;
      const r = el.closest('.card-surface, .sub-accordion, .hero-flow').getBoundingClientRect();
      return { x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height }; })()`);
    if (!box) { problems.push(`${vp.name}: ${selector} missing`); continue; }
    const c = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true,
      clip: { x: box.x, y: box.y, width: box.w, height: box.h, scale: 1 } });
    writeFileSync(join(outdir, `${vp.name}-${name}.png`), Buffer.from(c.data, 'base64'));
  }

  // Animation: every step renders, highlights its chip, and fits the viewport.
  const frames = await evaluate('window.csbpAnimation ? window.csbpAnimation.phaseFrames : null');
  if (!frames) { problems.push(`${vp.name}: csbpAnimation did not initialize`); continue; }
  for (let i = 0; i < frames.length; i++) {
    const info = await evaluate(`(() => {
      const a = window.csbpAnimation; a.pause(); a.seek(${frames[i]});
      const card = document.getElementById('csbp-anim'), svg = document.getElementById('anim-svg');
      const r = card.getBoundingClientRect(), s = svg.getBoundingClientRect();
      const active = [...document.querySelectorAll('.anim-chip')].findIndex(c => c.classList.contains('active'));
      const visiblePackets = [...svg.querySelectorAll('g[opacity]')].filter(g => +g.getAttribute('opacity') > 0.5).length;
      return { x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height, svgRight: s.right, vw: innerWidth, active,
               caption: document.getElementById('anim-caption').textContent.slice(0, 40), visiblePackets,
               view: svg.getAttribute('viewBox') };
    })()`);
    if (info.active !== i) problems.push(`${vp.name}: step ${i + 1} highlights chip ${info.active + 1}`);
    if (info.svgRight > info.vw + 1) problems.push(`${vp.name}: animation overflows the viewport at step ${i + 1}`);
    await sleep(150);
    const clip = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true,
      clip: { x: info.x, y: info.y, width: info.w, height: info.h, scale: 1 } });
    const f = join(outdir, `${vp.name}-anim-step${i + 1}.png`);
    writeFileSync(f, Buffer.from(clip.data, 'base64'));
    console.log(`${vp.name}: step ${i + 1} (${info.view}, ${info.visiblePackets} visible groups) "${info.caption}..." -> ${f}`);
  }

  // Load-balancing animation: frames fit, and the balanced panel finishes its step before the in-order one.
  const lbFrames = await evaluate('window.lbAnimation ? window.lbAnimation.phaseFrames : null');
  if (!lbFrames) { problems.push(`${vp.name}: lbAnimation did not initialize`); continue; }
  const expectDone = [[false, false], [false, true], [true, true]]; // [in order, CSBP]
  for (let i = 0; i < lbFrames.length; i++) {
    const info = await evaluate(`(() => {
      const a = window.lbAnimation; a.pause(); a.seek(${lbFrames[i]});
      const card = document.getElementById('lb-anim'), svg = document.getElementById('lb-svg');
      const r = card.getBoundingClientRect(), s = svg.getBoundingClientRect();
      const done = [...svg.querySelectorAll('text')].filter(t => t.textContent.includes('step done')).map(t => +t.getAttribute('opacity') > 0.5);
      return { x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height, svgRight: s.right, vw: innerWidth, done, view: svg.getAttribute('viewBox') };
    })()`);
    if (info.svgRight > info.vw + 1) problems.push(`${vp.name}: load-balancing animation overflows at frame ${i + 1}`);
    if (info.done.join() !== expectDone[i].join()) problems.push(`${vp.name}: load-balancing frame ${i + 1} shows step done = ${info.done.join()}`);
    await sleep(150);
    const clip = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true,
      clip: { x: info.x, y: info.y, width: info.w, height: info.h, scale: 1 } });
    const f = join(outdir, `${vp.name}-lb-frame${i + 1}.png`);
    writeFileSync(f, Buffer.from(clip.data, 'base64'));
    console.log(`${vp.name}: load balancing frame ${i + 1} (${info.view}, done ${info.done.join('/')}) -> ${f}`);
  }
}

ws.close();
chrome.kill();
await sleep(300);
rmSync(profile, { recursive: true, force: true });
if (problems.length) {
  console.error('\nFAIL\n' + [...new Set(problems)].join('\n'));
  process.exit(1);
}
console.log('\nPASS: no console errors, failed requests, broken images, or horizontal overflow');
