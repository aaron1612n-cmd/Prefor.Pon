// Prefor.Pon — GPU-resident particle field.
// Particles never leave the GPU: compute shader integrates, render pass reads the same buffers.

const coarse = matchMedia("(pointer: coarse)").matches;
const MIN_N = 1 << 14;
const MAX_N = 1 << 23; // 8M; 64 MB per vec2f buffer
const DT = 1 / 120; // fixed physics step
const MAX_SUBSTEPS = 4;
const WG = 256;

const SIM_WGSL = /* wgsl */ `
struct U {
  dt: f32, time: f32, aspect: f32, count: u32,
  ptr: vec2f, force: f32, damp: f32,
  seed: u32, _a: u32, _b: u32, _c: u32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> pos: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec2f>;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd(h: ptr<function, u32>) -> f32 {
  *h = pcg(*h);
  return f32(*h) * (1.0 / 4294967296.0);
}

@compute @workgroup_size(${WG})
fn seed(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.count) { return; }
  var h = pcg(i ^ u.seed);
  let a = rnd(&h) * 6.2831853;
  let r = sqrt(rnd(&h)) * 0.015;
  let d = vec2f(cos(a), sin(a));
  let s = 0.25 + pow(rnd(&h), 1.6) * 1.9;
  pos[i] = d * r;
  vel[i] = d * s + vec2f(-d.y, d.x) * s * 0.35;
}

@compute @workgroup_size(${WG})
fn step(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.count) { return; }
  var p = pos[i];
  var v = vel[i];
  let t = u.time;

  // Ambient drift: two octaves of cheap trigonometric flow, divergence-light.
  var a = vec2f(
    sin(p.y * 3.1 + t * 0.31) + 0.5 * sin(p.y * 7.3 - t * 0.23 + p.x),
    cos(p.x * 2.7 - t * 0.27) + 0.5 * cos(p.x * 6.1 + t * 0.17 - p.y)
  ) * 0.22;

  // Pointer: softened inverse-square well (or hill when force < 0).
  if (u.force != 0.0) {
    let d = u.ptr - p;
    let r2 = dot(d, d) + 0.0025;
    a += d * (u.force * inverseSqrt(r2) / r2);
  }

  v = (v + a * u.dt) * u.damp;
  p += v * u.dt;

  // Soft walls.
  let lim = vec2f(u.aspect, 1.0);
  if (abs(p.x) > lim.x) { p.x = sign(p.x) * lim.x; v.x *= -0.4; }
  if (abs(p.y) > lim.y) { p.y = sign(p.y) * lim.y; v.y *= -0.4; }

  pos[i] = p;
  vel[i] = v;
}
`;

const DRAW_WGSL = /* wgsl */ `
struct U {
  dt: f32, time: f32, aspect: f32, count: u32,
  ptr: vec2f, force: f32, damp: f32,
  seed: u32, _a: u32, _b: u32, _c: u32,
};
struct R { gain: f32, exposure: f32, _p: vec2f };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> pos: array<vec2f>;
@group(0) @binding(2) var<storage, read> vel: array<vec2f>;
@group(0) @binding(3) var<uniform> r: R;

struct VO { @builtin(position) p: vec4f, @location(0) c: vec3f };

@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = pos[i];
  let s = length(vel[i]);
  let bone = vec3f(0.91, 0.86, 0.76);
  let sodium = vec3f(1.0, 0.36, 0.06);
  let k = smoothstep(0.15, 1.4, s);
  var o: VO;
  o.p = vec4f(p.x / u.aspect, p.y, 0.0, 1.0);
  o.c = mix(bone * 0.55, sodium, k) * r.gain;
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f { return vec4f(i.c, 1.0); }
`;

const TONE_WGSL = /* wgsl */ `
struct R { gain: f32, exposure: f32, _p: vec2f };
@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var<uniform> r: R;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
  let y = f32(i & 2u) * 2.0 - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let h = textureLoad(hdr, vec2i(fc.xy), 0).rgb;
  let ink = vec3f(0.071, 0.063, 0.051);
  let c = 1.0 - exp(-h * r.exposure);
  return vec4f(ink + c * (1.0 - ink), 1.0);
}
`;

const $ = (id) => document.getElementById(id);
const canvas = $("gl");

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};

async function boot() {
  const adapter = navigator.gpu && (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }));
  if (!adapter) { $("nogpu").hidden = false; return; }
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, MAX_N * 8),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, MAX_N * 8),
    },
  });
  const maxN = Math.min(MAX_N, Math.floor(device.limits.maxStorageBufferBindingSize / 8));

  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  const simMod = device.createShaderModule({ code: SIM_WGSL });
  const drawMod = device.createShaderModule({ code: DRAW_WGSL });
  const toneMod = device.createShaderModule({ code: TONE_WGSL });

  const seedPipe = device.createComputePipeline({ layout: "auto", compute: { module: simMod, entryPoint: "seed" } });
  const stepPipe = device.createComputePipeline({ layout: "auto", compute: { module: simMod, entryPoint: "step" } });
  const additive = { color: { srcFactor: "one", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one", operation: "add" } };
  const drawPipe = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: drawMod, entryPoint: "vs" },
    fragment: { module: drawMod, entryPoint: "fs", targets: [{ format: "rgba16float", blend: additive }] },
    primitive: { topology: "point-list" },
  });
  const tonePipe = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: toneMod, entryPoint: "vs" },
    fragment: { module: toneMod, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const uBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const rBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const uData = new ArrayBuffer(48);
  const uF = new Float32Array(uData);
  const uU = new Uint32Array(uData);

  const state = {
    n: coarse ? 1 << 18 : 1 << 20,
    time: 0,
    repel: false,
    ptr: [0, 0],
    down: false,
    keys: new Set(),
    aspect: 1,
  };

  let pos, vel, simBG = {}, drawBG, hdr, toneBG;

  function alloc() {
    pos?.destroy(); vel?.destroy();
    const size = state.n * 8;
    const usage = GPUBufferUsage.STORAGE;
    pos = device.createBuffer({ size, usage });
    vel = device.createBuffer({ size, usage });
    for (const [k, p] of [["seed", seedPipe], ["step", stepPipe]]) {
      simBG[k] = device.createBindGroup({
        layout: p.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uBuf } }, { binding: 1, resource: { buffer: pos } }, { binding: 2, resource: { buffer: vel } }],
      });
    }
    drawBG = device.createBindGroup({
      layout: drawPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uBuf } }, { binding: 1, resource: { buffer: pos } },
        { binding: 2, resource: { buffer: vel } }, { binding: 3, resource: { buffer: rBuf } },
      ],
    });
    exposure();
    reseed();
    $("n").textContent = fmt(state.n);
  }

  function reseed() {
    writeU(0);
    uU[8] = (Math.random() * 2 ** 32) >>> 0;
    device.queue.writeBuffer(uBuf, 0, uData);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(seedPipe);
    pass.setBindGroup(0, simBG.seed);
    pass.dispatchWorkgroups(Math.ceil(state.n / WG));
    pass.end();
    device.queue.submit([enc.finish()]);
    state.time = 0;
  }

  // Keep total brightness roughly constant regardless of count and screen size.
  function exposure() {
    const px = canvas.width * canvas.height || 1;
    const gain = Math.min(0.6, (0.15 * px) / state.n);
    device.queue.writeBuffer(rBuf, 0, new Float32Array([gain, 1.6, 0, 0]));
  }

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(innerWidth * dpr));
    const h = Math.max(1, Math.round(innerHeight * dpr));
    if (canvas.width === w && canvas.height === h && hdr) return;
    canvas.width = w; canvas.height = h;
    state.aspect = w / h;
    hdr?.destroy();
    hdr = device.createTexture({ size: [w, h], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    toneBG = device.createBindGroup({
      layout: tonePipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: hdr.createView() }, { binding: 1, resource: { buffer: rBuf } }],
    });
    exposure();
  }

  function writeU(force) {
    uF[0] = DT; uF[1] = state.time; uF[2] = state.aspect; uU[3] = state.n;
    uF[4] = state.ptr[0]; uF[5] = state.ptr[1]; uF[6] = force; uF[7] = Math.pow(0.55, DT);
  }

  // ---- input ----
  const toSim = (e) => {
    const r = canvas.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * 2 - 1;
    const y = 1 - ((e.clientY - r.top) / r.height) * 2;
    return [x * state.aspect, y];
  };
  let dragDist = 0, last = null;
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    state.down = true; state.ptr = toSim(e); last = state.ptr;
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!state.down) return;
    state.ptr = toSim(e);
    dragDist += Math.hypot(state.ptr[0] - last[0], state.ptr[1] - last[1]);
    last = state.ptr;
    if (dragDist > 0.8) tutorial.done("drag");
  });
  const up = () => { state.down = false; };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  const ARROWS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
  addEventListener("keydown", (e) => {
    if (e.target.closest?.("button") && (e.key === " " || e.key === "Enter")) return;
    if (ARROWS[e.key]) { state.keys.add(e.key); e.preventDefault(); }
    else if (e.key === " ") { flip(); e.preventDefault(); }
    else if (e.key === "+" || e.key === "=") setN(state.n * 2);
    else if (e.key === "-" || e.key === "_") setN(state.n / 2);
    else if (e.key === "r" || e.key === "R") reseed();
  });
  addEventListener("keyup", (e) => state.keys.delete(e.key));

  function flip() {
    state.repel = !state.repel;
    $("flip").setAttribute("aria-pressed", String(state.repel));
    tutorial.done("flip");
  }
  function setN(n) {
    n = Math.max(MIN_N, Math.min(maxN, Math.round(n)));
    if (n === state.n) return;
    state.n = n; alloc();
  }
  $("flip").onclick = flip;
  $("more").onclick = () => setN(state.n * 2);
  $("less").onclick = () => setN(state.n / 2);
  $("reset").onclick = reseed;
  $("learn").onclick = () => tutorial.start();

  addEventListener("resize", () => { resize(); });
  resize();
  alloc();

  // ---- frame loop ----
  let acc = 0, prev = performance.now(), fpsT = prev, frames = 0, msAvg = 0;
  function frame(now) {
    const dt = Math.min(0.1, (now - prev) / 1000);
    msAvg += ((now - prev) - msAvg) * 0.1;
    prev = now;
    acc += dt;
    let steps = 0;
    while (acc >= DT && steps < MAX_SUBSTEPS) { acc -= DT; steps++; }
    if (steps === MAX_SUBSTEPS) acc = 0; // drop backlog rather than spiral

    // keyboard-driven attractor
    let kx = 0, ky = 0;
    for (const k of state.keys) { kx += ARROWS[k][0]; ky += ARROWS[k][1]; }
    const keyActive = state.keys.size > 0;
    if (keyActive) {
      if (!state.down) state.ptr = [state.ptr[0] + kx * dt * 1.2, state.ptr[1] + ky * dt * 1.2];
      state.ptr[0] = Math.max(-state.aspect, Math.min(state.aspect, state.ptr[0]));
      state.ptr[1] = Math.max(-1, Math.min(1, state.ptr[1]));
      dragDist += Math.hypot(kx, ky) * dt * 1.2;
      if (dragDist > 0.8) tutorial.done("drag");
    }
    const active = state.down || keyActive;
    const force = active ? (state.repel ? -0.5 : 0.35) : 0;

    const enc = device.createCommandEncoder();
    if (steps) {
      const pass = enc.beginComputePass();
      pass.setPipeline(stepPipe);
      pass.setBindGroup(0, simBG.step);
      // Same uniforms for every substep; time advances once per frame block.
      state.time += steps * DT;
      writeU(force);
      device.queue.writeBuffer(uBuf, 0, uData);
      const groups = Math.ceil(state.n / WG);
      for (let s = 0; s < steps; s++) pass.dispatchWorkgroups(groups);
      pass.end();
    } else {
      writeU(force);
      device.queue.writeBuffer(uBuf, 0, uData);
    }

    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: hdr.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
    });
    rp.setPipeline(drawPipe);
    rp.setBindGroup(0, drawBG);
    rp.draw(state.n);
    rp.end();

    const tp = enc.beginRenderPass({
      colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    tp.setPipeline(tonePipe);
    tp.setBindGroup(0, toneBG);
    tp.draw(3);
    tp.end();
    device.queue.submit([enc.finish()]);

    frames++;
    if (now - fpsT > 500) {
      $("fps").textContent = Math.round((frames * 1000) / (now - fpsT));
      $("ms").textContent = msAvg.toFixed(1);
      frames = 0; fpsT = now;
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  if (!store.get("pp.tut")) tutorial.start();
}

function fmt(n) {
  return n >= 1 << 20 ? (n / (1 << 20)).toFixed(n % (1 << 20) ? 1 : 0) + "M" : Math.round(n / 1024) + "K";
}

// ---- wordless tutorial: drag, then flip. One word per step. ----
const tutorial = (() => {
  const el = $("tut"), word = $("word"), finger = el.querySelector(".finger"), trail = $("trail");
  const steps = [
    { id: "drag", word: "Drag" },
    { id: "flip", word: "Flip" },
  ];
  let i = -1, raf = 0;

  function animateFinger() {
    const len = trail.getTotalLength();
    const box = el.querySelector(".ghost").getBoundingClientRect();
    const t0 = performance.now();
    const tick = (now) => {
      const t = ((now - t0) / 1800) % 1;
      const pt = trail.getPointAtLength(len * t);
      finger.style.left = box.left + (pt.x / 100) * box.width + "px";
      finger.style.top = box.top + (pt.y / 100) * box.height + "px";
      finger.style.opacity = String(Math.sin(t * Math.PI));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }

  function show() {
    cancelAnimationFrame(raf);
    document.querySelectorAll(".beckon").forEach((b) => b.classList.remove("beckon"));
    const s = steps[i];
    if (!s) return end();
    el.dataset.step = s.id;
    word.textContent = s.word;
    if (s.id === "drag") animateFinger();
    if (s.id === "flip") $("flip").classList.add("beckon");
  }
  function end() {
    cancelAnimationFrame(raf);
    document.querySelectorAll(".beckon").forEach((b) => b.classList.remove("beckon"));
    el.hidden = true; i = -1;
    store.set("pp.tut", "1");
  }
  $("skip").onclick = end;

  return {
    start() { i = 0; el.hidden = false; show(); },
    done(id) { if (i >= 0 && steps[i]?.id === id) { i++; show(); } },
  };
})();

boot().catch((err) => {
  console.error(err);
  $("nogpu").hidden = false;
});
