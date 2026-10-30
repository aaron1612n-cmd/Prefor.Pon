// Prefor.Pon — GPU-resident particle field.
// Particles never leave the GPU: compute shader integrates, render pass reads the same buffers.

const coarse = matchMedia("(pointer: coarse)").matches;
const CAP = coarse ? 1 << 20 : 1 << 22; // buffer capacity; active count adapts below it
const MIN_N = 1 << 15;
const DT = 1 / 60; // fixed physics step
const MAX_SUBSTEPS = 2; // more substeps on a slow frame only makes the next frame slower
const WG = 256;
const SHOCKS = 4;
const TRAIL = 0.86; // fraction of last frame kept

const U_WGSL = /* wgsl */ `
struct U {
  dt: f32, time: f32, aspect: f32, count: u32,
  seed: u32, _a: u32, _b: u32, _c: u32,
  wells: array<vec4f, 3>,   // xy pos, z pull, w swirl
  shocks: array<vec4f, ${SHOCKS}>, // xy origin, z age, w strength
};
`;

const SIM_WGSL = U_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> pos: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> vel: array<u32>; // f16x2

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
  vel[i] = pack2x16float(d * s + vec2f(-d.y, d.x) * s * 0.35);
}

// Divergence-free flow: curl of a drifting stream function, so particles swirl instead of clumping.
fn flow(p: vec2f, t: f32) -> vec2f {
  let A = 1.7 * p.x + 0.37 * t;
  let B = 2.3 * p.y - 0.29 * t;
  let C = 3.1 * p.x + 3.7 * p.y - 0.41 * t;
  let dx = 1.7 * cos(A) * cos(B) + 1.55 * cos(C);
  let dy = -2.3 * sin(A) * sin(B) + 1.85 * cos(C);
  return vec2f(dy, -dx);
}

@compute @workgroup_size(${WG})
fn step(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.count) { return; }
  var p = pos[i];
  var v = unpack2x16float(vel[i]);

  // Relax toward the flow field.
  var a = (flow(p, u.time) * 0.28 - v) * 0.9;

  // Wells: softened inverse-square pull plus a tangential swirl, so things orbit rather than collapse.
  for (var k = 0u; k < 3u; k++) {
    let w = u.wells[k];
    if (w.z == 0.0 && w.w == 0.0) { continue; }
    let d = w.xy - p;
    let r2 = dot(d, d) + 0.004;
    a += (d * w.z + vec2f(-d.y, d.x) * w.w) * (inverseSqrt(r2) / r2);
  }

  // Shock rings: an expanding front that kicks particles outward as it passes.
  for (var k = 0u; k < ${SHOCKS}u; k++) {
    let s = u.shocks[k];
    if (s.w == 0.0) { continue; }
    let d = p - s.xy;
    let r = length(d) + 1e-4;
    let x = (r - s.z * 1.3) / 0.05;
    a += d / r * (s.w * exp(-x * x) * exp(-s.z * 1.4));
  }

  v += a * u.dt;
  p += v * u.dt;

  // Wrap instead of clamping: clamped walls pile particles onto the same edge pixels,
  // and that blend contention is what made long runs slow down.
  let lim = vec2f(u.aspect, 1.0);
  p = select(p, p - sign(p) * lim * 2.0, abs(p) > lim);

  pos[i] = p;
  vel[i] = pack2x16float(v);
}
`;

const DRAW_WGSL = U_WGSL + /* wgsl */ `
struct R { gain: f32, exposure: f32, _p: vec2f };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> pos: array<vec2f>;
@group(0) @binding(2) var<storage, read> vel: array<u32>;
@group(0) @binding(3) var<uniform> r: R;

struct VO { @builtin(position) p: vec4f, @location(0) c: vec3f };

@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = pos[i];
  let s = length(unpack2x16float(vel[i]));
  let bone = vec3f(0.91, 0.86, 0.76);
  let sodium = vec3f(1.0, 0.36, 0.06);
  var o: VO;
  o.p = vec4f(p.x / u.aspect, p.y, 0.0, 1.0);
  o.c = mix(bone * 0.35, sodium * 1.6, smoothstep(0.25, 1.6, s)) * r.gain;
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f { return vec4f(i.c, 1.0); }
`;

const FULL_VS = /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
  let y = f32(i & 2u) * 2.0 - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}
`;

// Multiplies the HDR target by the blend constant: cheap motion trails, no ping-pong.
const FADE_WGSL = FULL_VS + /* wgsl */ `
@fragment fn fs() -> @location(0) vec4f { return vec4f(0.0); }
`;

const TONE_WGSL = FULL_VS + /* wgsl */ `
struct R { gain: f32, exposure: f32, _p: vec2f };
@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var<uniform> r: R;
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
  const want = CAP * 8;
  const device = await adapter.requestDevice({
    requiredFeatures: adapter.features.has("rg11b10ufloat-renderable") ? ["rg11b10ufloat-renderable"] : [],
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, want),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, want),
    },
  });
  const cap = Math.min(CAP, Math.floor(device.limits.maxStorageBufferBindingSize / 8));

  // 4-byte HDR target when the GPU can render to it: half the fill bandwidth of rgba16float.
  const HDR = device.features.has("rg11b10ufloat-renderable") ? "rg11b10ufloat" : "rgba16float";
  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  const simMod = device.createShaderModule({ code: SIM_WGSL });
  const drawMod = device.createShaderModule({ code: DRAW_WGSL });
  const fadeMod = device.createShaderModule({ code: FADE_WGSL });
  const toneMod = device.createShaderModule({ code: TONE_WGSL });

  const seedPipe = device.createComputePipeline({ layout: "auto", compute: { module: simMod, entryPoint: "seed" } });
  const stepPipe = device.createComputePipeline({ layout: "auto", compute: { module: simMod, entryPoint: "step" } });
  const blend = (srcFactor, dstFactor) => ({ color: { srcFactor, dstFactor, operation: "add" }, alpha: { srcFactor, dstFactor, operation: "add" } });
  const drawPipe = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: drawMod, entryPoint: "vs" },
    fragment: { module: drawMod, entryPoint: "fs", targets: [{ format: HDR, blend: blend("one", "one") }] },
    primitive: { topology: "point-list" },
  });
  const fadePipe = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: fadeMod, entryPoint: "vs" },
    fragment: { module: fadeMod, entryPoint: "fs", targets: [{ format: HDR, blend: blend("zero", "constant") }] },
  });
  const tonePipe = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: toneMod, entryPoint: "vs" },
    fragment: { module: toneMod, entryPoint: "fs", targets: [{ format }] },
  });

  const U_SIZE = 32 + 16 * 3 + 16 * SHOCKS;
  const uBuf = device.createBuffer({ size: U_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const rBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const uData = new ArrayBuffer(U_SIZE);
  const uF = new Float32Array(uData);
  const uU = new Uint32Array(uData);

  const state = {
    n: Math.min(cap, coarse ? 1 << 17 : 1 << 19), // start light, grow to fit the GPU
    time: 0,
    aspect: 1,
    ptr: [0, 0],
    down: false,
    downAt: 0,
    moved: 0,
    keys: new Set(),
    shocks: [], // {x, y, born, s}
  };

  const usage = GPUBufferUsage.STORAGE;
  const pos = device.createBuffer({ size: cap * 8, usage });
  const vel = device.createBuffer({ size: cap * 4, usage });
  const bg = (pipe, extra = []) => device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uBuf } }, { binding: 1, resource: { buffer: pos } }, { binding: 2, resource: { buffer: vel } }, ...extra],
  });
  const seedBG = bg(seedPipe), stepBG = bg(stepPipe), drawBG = bg(drawPipe, [{ binding: 3, resource: { buffer: rBuf } }]);
  let hdr, toneBG;

  function reseed() {
    writeU();
    uU[3] = cap;
    uU[4] = (Math.random() * 2 ** 32) >>> 0;
    device.queue.writeBuffer(uBuf, 0, uData);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(seedPipe);
    pass.setBindGroup(0, seedBG);
    pass.dispatchWorkgroups(Math.ceil(cap / WG));
    pass.end();
    device.queue.submit([enc.finish()]);
    state.time = 0;
  }

  // Keep total brightness roughly constant regardless of count, screen size and trail length.
  function exposure() {
    const px = canvas.width * canvas.height || 1;
    const gain = Math.min(0.35, (0.2 * px * (1 - TRAIL)) / state.n);
    device.queue.writeBuffer(rBuf, 0, new Float32Array([gain, 1.5, 0, 0]));
    $("n").textContent = fmt(state.n);
  }

  function resize() {
    // Points are 1px whatever the density, so render at CSS resolution: on a 2x screen that is a quarter of the fill.
    const dpr = 1;
    const w = Math.max(1, Math.round(innerWidth * dpr));
    const h = Math.max(1, Math.round(innerHeight * dpr));
    if (canvas.width === w && canvas.height === h && hdr) return;
    canvas.width = w; canvas.height = h;
    state.aspect = w / h;
    hdr?.destroy();
    hdr = device.createTexture({ size: [w, h], format: HDR, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    toneBG = device.createBindGroup({
      layout: tonePipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: hdr.createView() }, { binding: 1, resource: { buffer: rBuf } }],
    });
    exposure();
  }

  function writeU(now = performance.now()) {
    uF[0] = DT; uF[1] = state.time; uF[2] = state.aspect; uU[3] = state.n;
    const t = state.time;
    // Pointer well ramps in over a hold, so a tap is a shock and not a tug.
    const held = state.down || state.keys.size ? Math.min(1, (now - state.downAt) / 400) : 0;
    const wells = [
      [state.ptr[0], state.ptr[1], 0.32 * held, 0.22 * held],
      // Two slow wandering vortices keep the field alive with no input.
      [Math.sin(t * 0.13) * state.aspect * 0.55, Math.sin(t * 0.21 + 1) * 0.5, 0.012, 0.05],
      [Math.cos(t * 0.17 + 2) * state.aspect * 0.5, Math.cos(t * 0.11) * 0.55, 0.012, -0.05],
    ];
    wells.forEach((w, k) => uF.set(w, 8 + k * 4));
    state.shocks = state.shocks.filter((s) => now - s.born < 2500);
    for (let k = 0; k < SHOCKS; k++) {
      const s = state.shocks[k];
      uF.set(s ? [s.x, s.y, (now - s.born) / 1000, s.s] : [0, 0, 0, 0], 20 + k * 4);
    }
  }

  function shock(x, y) {
    if (state.shocks.length >= SHOCKS) state.shocks.shift();
    state.shocks.push({ x, y, born: performance.now(), s: 9 });
  }

  // ---- input ----
  const toSim = (e) => {
    const r = canvas.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * 2 - 1;
    const y = 1 - ((e.clientY - r.top) / r.height) * 2;
    return [x * state.aspect, y];
  };
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    state.down = true; state.downAt = performance.now(); state.moved = 0;
    state.ptr = toSim(e);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!state.down) return;
    const p = toSim(e);
    state.moved += Math.hypot(p[0] - state.ptr[0], p[1] - state.ptr[1]);
    state.ptr = p;
  });
  const up = () => {
    if (!state.down) return;
    const held = performance.now() - state.downAt;
    state.down = false;
    if (held < 220 && state.moved < 0.05) { shock(...state.ptr); tutorial.done("tap"); }
    else if (held > 600) tutorial.done("hold");
  };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", () => { state.down = false; });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  const ARROWS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
  addEventListener("keydown", (e) => {
    if (e.target.closest?.("button")) return;
    if (ARROWS[e.key]) {
      if (!state.keys.size) state.downAt = performance.now();
      state.keys.add(e.key); e.preventDefault();
    } else if (e.key === " " || e.key === "Enter") { shock(...state.ptr); tutorial.done("tap"); e.preventDefault(); }
    else if (e.key === "r" || e.key === "R") reseed();
  });
  addEventListener("keyup", (e) => {
    if (!ARROWS[e.key]) return;
    state.keys.delete(e.key);
    if (!state.keys.size && performance.now() - state.downAt > 600) tutorial.done("hold");
  });

  $("learn").onclick = () => tutorial.start();
  addEventListener("resize", resize);
  resize();
  reseed();

  // ---- frame loop ----
  let acc = 0, prev = performance.now(), winT = prev, winF = 0, settled = 0;
  function frame(now) {
    const dt = Math.min(0.1, (now - prev) / 1000);
    prev = now;
    acc += dt;
    let steps = 0;
    while (acc >= DT && steps < MAX_SUBSTEPS) { acc -= DT; steps++; }
    if (steps === MAX_SUBSTEPS) acc = 0; // drop backlog rather than spiral

    if (state.keys.size) {
      let kx = 0, ky = 0;
      for (const k of state.keys) { kx += ARROWS[k][0]; ky += ARROWS[k][1]; }
      state.ptr = [
        Math.max(-state.aspect, Math.min(state.aspect, state.ptr[0] + kx * dt * 1.1)),
        Math.max(-1, Math.min(1, state.ptr[1] + ky * dt * 1.1)),
      ];
    }

    state.time += steps * DT;
    writeU(now);
    device.queue.writeBuffer(uBuf, 0, uData);

    const enc = device.createCommandEncoder();
    if (steps) {
      const pass = enc.beginComputePass();
      pass.setPipeline(stepPipe);
      pass.setBindGroup(0, stepBG);
      const groups = Math.ceil(state.n / WG);
      for (let s = 0; s < steps; s++) pass.dispatchWorkgroups(groups);
      pass.end();
    }

    const rp = enc.beginRenderPass({ colorAttachments: [{ view: hdr.createView(), loadOp: "load", storeOp: "store" }] });
    rp.setPipeline(fadePipe);
    rp.setBlendConstant([TRAIL, TRAIL, TRAIL, TRAIL]);
    rp.draw(3);
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

    // Adaptive load: grow the active count while the frame rate holds, back off when it drops.
    winF++;
    if (now - winT > 1000) {
      const fps = (winF * 1000) / (now - winT);
      winF = 0; winT = now;
      let n = state.n;
      if (fps > 57 && settled < 3) n = Math.min(cap, Math.round(n * 1.5));
      else if (fps < 48) { n = Math.max(MIN_N, Math.round(n * 0.7)); settled++; }
      if (n !== state.n) { state.n = n; exposure(); }
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  if (!store.get("pp.tut2")) tutorial.start();
}

function fmt(n) {
  return n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : Math.round(n / 1e3) + "K";
}

// ---- wordless tutorial: hold, then tap. One word per step. ----
const tutorial = (() => {
  const el = $("tut"), word = $("word"), learn = $("learn");
  const steps = [
    { id: "hold", word: "Hold" },
    { id: "tap", word: "Tap" },
  ];
  let i = -1;

  function show() {
    const s = steps[i];
    if (!s) return end();
    el.dataset.step = s.id;
    word.textContent = s.word;
  }
  function end() {
    el.hidden = true; learn.hidden = false; i = -1;
    store.set("pp.tut2", "1");
  }
  $("skip").onclick = end;

  return {
    start() { i = 0; el.hidden = false; learn.hidden = true; show(); },
    done(id) { if (i >= 0 && steps[i]?.id === id) { i++; show(); } },
  };
})();

boot().catch((err) => {
  console.error(err);
  $("nogpu").hidden = false;
});
