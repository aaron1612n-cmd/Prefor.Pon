// Prefor.Pon — GPU galaxy simulation: particle-mesh self-gravity, millions of stars.
// Particles never leave the GPU: compute shader integrates, render pass reads the same buffers.

const coarse = matchMedia("(pointer: coarse)").matches;
const CAP = coarse ? 1 << 19 : 1 << 21; // buffer capacity; active count adapts below it
const BUDGET = coarse ? 9 : 7; // ms of GPU time per frame we allow ourselves
const MIN_N = 1 << 14;
const DT = 1 / 60; // fixed physics step
const MAX_SUBSTEPS = 2; // more substeps on a slow frame only makes the next frame slower
const WG = 256;
const SHOCKS = 4;
const GALAXIES = 3;
const TRAIL = 0.7; // fraction of last frame kept
const GRID = coarse ? 40 : 56; // gravity grid rows; columns follow the aspect ratio
const G = 0.22;

const U_WGSL = /* wgsl */ `
struct U {
  dt: f32, time: f32, aspect: f32, count: u32,
  seed: u32, gx: u32, gy: u32, ngal: u32,
  well: vec4f,                     // xy pos, z = G*M of the black hole under the finger
  shocks: array<vec4f, ${SHOCKS}>, // xy origin, z age, w strength
  gal: array<vec4f, ${GALAXIES * 2}>, // per galaxy: (cx, cy, vx, vy), (scale, cumulative share, spin, mass)
  grav: vec4f,                     // G, softening^2, cell w, cell h
};
`;

const SIM_WGSL = U_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> pos: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> mass: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> field: array<vec2f>;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd(h: ptr<function, u32>) -> f32 {
  *h = pcg(*h);
  return (f32(*h >> 8u) + 0.5) * (1.0 / 16777216.0);
}

// Spiral discs on circular orbits. The galaxy is picked by hash, not by index,
// so any prefix [0, count) is a fair sample and the adaptive count never skews the mass.
@compute @workgroup_size(${WG})
fn seed(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.count) { return; }
  var h = pcg(i ^ u.seed);
  let pick = rnd(&h);
  var k = 0u;
  for (var j = 0u; j < u.ngal - 1u; j++) { if (pick > u.gal[j * 2u + 1u].y) { k = j + 1u; } }
  let g0 = u.gal[k * 2u];
  let g1 = u.gal[k * 2u + 1u];
  let rd = g1.x;
  let a = rnd(&h) * 6.2831853;
  var r: f32;
  if (rnd(&h) < 0.18) {
    r = rd * 0.35 * sqrt(-2.0 * log(rnd(&h))); // bulge
  } else {
    r = min(-rd * log(rnd(&h) * rnd(&h)), rd * 5.0); // exponential disc
  }
  let d = vec2f(cos(a), sin(a));
  let x = r / rd;
  let enclosed = g1.w * (1.0 - (1.0 + x) * exp(-x));
  let vc = sqrt(u.grav.x * enclosed / sqrt(r * r + u.grav.y));
  let jitter = vec2f(rnd(&h) - 0.5, rnd(&h) - 0.5) * vc * 0.12;
  pos[i] = g0.xy + d * r;
  vel[i] = g0.zw + vec2f(-d.y, d.x) * vc * g1.z + jitter;
}

fn cellOf(p: vec2f) -> u32 {
  let c = vec2i(floor((p + vec2f(u.aspect, 1.0)) / u.grav.zw));
  let cx = u32(clamp(c.x, 0, i32(u.gx) - 1));
  let cy = u32(clamp(c.y, 0, i32(u.gy) - 1));
  return cy * u.gx + cx;
}

@compute @workgroup_size(${WG})
fn clear(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < u.gx * u.gy) { atomicStore(&mass[id.x], 0u); }
}

@compute @workgroup_size(${WG})
fn deposit(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.count) { return; }
  atomicAdd(&mass[cellOf(pos[id.x])], 1u);
}

// Gravity on the grid: every cell sums every other cell, tiled through workgroup memory.
// Cost is cells^2, fixed no matter how many particles there are.
const FT = 64u;
var<workgroup> tile: array<vec3f, FT>;

fn centre(c: u32) -> vec2f {
  return (vec2f(f32(c % u.gx), f32(c / u.gx)) + 0.5) * u.grav.zw - vec2f(u.aspect, 1.0);
}

@compute @workgroup_size(FT)
fn gravity(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
  let cells = u.gx * u.gy;
  let me = centre(min(id.x, cells - 1u));
  var acc = vec2f(0.0);
  for (var t = 0u; t < cells; t += FT) {
    let j = t + li;
    tile[li] = select(vec3f(0.0), vec3f(centre(j), f32(atomicLoad(&mass[j]))), j < cells);
    workgroupBarrier();
    for (var k = 0u; k < FT; k++) {
      let c = tile[k];
      let d = c.xy - me;
      let ir = inverseSqrt(dot(d, d) + u.grav.y);
      acc += d * (c.z * ir * ir * ir);
    }
    workgroupBarrier();
  }
  if (id.x < cells) { field[id.x] = acc * (u.grav.x / f32(u.count)); }
}

fn sampleField(p: vec2f) -> vec2f {
  let g = (p + vec2f(u.aspect, 1.0)) / u.grav.zw - 0.5;
  let hi = vec2f(f32(u.gx) - 1.0, f32(u.gy) - 1.0);
  let gc = clamp(g, vec2f(0.0), hi);
  let i0 = vec2u(floor(gc));
  let i1 = min(i0 + 1u, vec2u(hi));
  let f = gc - vec2f(i0);
  let a = mix(field[i0.y * u.gx + i0.x], field[i0.y * u.gx + i1.x], f.x);
  let b = mix(field[i1.y * u.gx + i0.x], field[i1.y * u.gx + i1.x], f.x);
  return mix(a, b, f.y);
}

@compute @workgroup_size(${WG})
fn step(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.count) { return; }
  var p = pos[i];
  var v = vel[i];

  var a = sampleField(p);

  // Black hole under the finger.
  if (u.well.z > 0.0) {
    let d = u.well.xy - p;
    let ir = inverseSqrt(dot(d, d) + 0.0008);
    a += d * (u.well.z * ir * ir * ir);
  }

  // Supernova: an expanding front that kicks stars outward as it passes.
  for (var k = 0u; k < ${SHOCKS}u; k++) {
    let s = u.shocks[k];
    if (s.w == 0.0) { continue; }
    let d = p - s.xy;
    let r = length(d) + 1e-4;
    let x = (r - s.z * 0.9) / 0.04;
    a += d / r * (s.w * exp(-x * x) * exp(-s.z * 2.0));
  }

  // Soft leash past the screen edge, so escaping stars come home instead of vanishing.
  let lim = vec2f(u.aspect, 1.0);
  a -= sign(p) * max(abs(p) - lim, vec2f(0.0)) * 3.0;

  v += a * u.dt; // kick-drift: symplectic, keeps orbits from spiralling in or out
  p += v * u.dt;
  pos[i] = p;
  vel[i] = v;
}
`;

const DRAW_WGSL = U_WGSL + /* wgsl */ `
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
  const want = Math.max(CAP * 8, GRID * GRID * 4 * 8);
  const device = await adapter.requestDevice({
    requiredFeatures: ["rg11b10ufloat-renderable", "timestamp-query"].filter((f) => adapter.features.has(f)),
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

  const storage = (i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } });
  const simLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, storage(1), storage(2), storage(3), storage(4)],
  });
  const simPL = device.createPipelineLayout({ bindGroupLayouts: [simLayout] });
  const cp = (entryPoint) => device.createComputePipeline({ layout: simPL, compute: { module: simMod, entryPoint } });
  const seedPipe = cp("seed"), clearPipe = cp("clear"), depositPipe = cp("deposit"), gravityPipe = cp("gravity"), stepPipe = cp("step");
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

  // Real GPU time via timestamp queries where supported; otherwise submit-to-done wall time.
  const timed = device.features.has("timestamp-query");
  const qs = timed && device.createQuerySet({ type: "timestamp", count: 2 });
  const qResolve = timed && device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const qRead = timed && device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const U_SIZE = 32 + 16 + 16 * SHOCKS + 32 * GALAXIES + 16;
  const O_WELL = 8, O_SHOCK = 12, O_GAL = O_SHOCK + 4 * SHOCKS, O_GRAV = O_GAL + 8 * GALAXIES;
  const uBuf = device.createBuffer({ size: U_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const rBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const uData = new ArrayBuffer(U_SIZE);
  const uF = new Float32Array(uData);
  const uU = new Uint32Array(uData);

  const state = {
    n: Math.min(cap, coarse ? 1 << 15 : 1 << 16), // start light, grow only while the GPU has headroom
    scale: 1, // internal render resolution, dropped only if the minimum count still misses budget
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
  const vel = device.createBuffer({ size: cap * 8, usage }); // f32: tiny orbital kicks vanish in f16
  const maxCells = GRID * GRID * 4; // up to 4:1 aspect
  const massBuf = device.createBuffer({ size: maxCells * 4, usage });
  const fieldBuf = device.createBuffer({ size: maxCells * 8, usage });
  const base = [{ binding: 0, resource: { buffer: uBuf } }, { binding: 1, resource: { buffer: pos } }, { binding: 2, resource: { buffer: vel } }];
  const simBG = device.createBindGroup({
    layout: simLayout,
    entries: [...base, { binding: 3, resource: { buffer: massBuf } }, { binding: 4, resource: { buffer: fieldBuf } }],
  });
  const drawBG = device.createBindGroup({ layout: drawPipe.getBindGroupLayout(0), entries: [...base, { binding: 3, resource: { buffer: rBuf } }] });
  let gx = GRID;
  let hdr, toneBG;

  // A random encounter: two or three discs on a slow, slightly offset collision course.
  function galaxies() {
    const count = Math.random() < 0.35 ? 3 : 2;
    const wide = state.aspect >= 1;
    const span = wide ? Math.min(state.aspect, 1.6) : 1;
    const unit = Math.min(state.aspect, 1);
    const gs = [];
    let total = 0;
    for (let k = 0; k < count; k++) {
      const t = count === 2 ? (k ? 1 : -1) : k - 1;
      const along = t * span * 0.5, off = (Math.random() - 0.5) * 0.5 * unit;
      const c = wide ? [along, off] : [off, along];
      const speed = 0.1 + Math.random() * 0.08;
      const v = wide ? [-t * speed, (Math.random() - 0.5) * 0.15] : [(Math.random() - 0.5) * 0.15, -t * speed];
      const m = 0.5 + Math.random() * 0.5;
      total += m;
      gs.push({ c, v, m, rd: (0.06 + Math.random() * 0.04) * unit * (0.6 + m * 0.6), spin: Math.random() < 0.5 ? -1 : 1 });
    }
    let cum = 0;
    gs.forEach((g, k) => {
      cum += g.m / total;
      uF.set([...g.c, ...g.v], O_GAL + k * 8);
      uF.set([g.rd, k === count - 1 ? 1 : cum, g.spin, g.m / total], O_GAL + k * 8 + 4);
    });
    uU[7] = count;
  }

  function reseed() {
    writeU();
    galaxies();
    uU[3] = cap;
    uU[4] = (Math.random() * 2 ** 32) >>> 0;
    device.queue.writeBuffer(uBuf, 0, uData);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(seedPipe);
    pass.setBindGroup(0, simBG);
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
    // Points are 1px whatever the density, so render at CSS resolution (or below): on a 2x screen that is a quarter of the fill.
    const dpr = state.scale;
    const w = Math.max(1, Math.round(innerWidth * dpr));
    const h = Math.max(1, Math.round(innerHeight * dpr));
    if (canvas.width === w && canvas.height === h && hdr) return;
    canvas.width = w; canvas.height = h;
    state.aspect = w / h;
    gx = Math.max(8, Math.min(GRID * 4, Math.round(GRID * state.aspect)));
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
    uU[5] = gx; uU[6] = GRID;
    const cw = (2 * state.aspect) / gx, ch = 2 / GRID;
    uF.set([G, (1.2 * ch) ** 2, cw, ch], O_GRAV);
    // The black hole grows over a hold, so a tap is a supernova and not a tug.
    const held = state.down || state.keys.size ? Math.min(1, (now - state.downAt) / 600) : 0;
    uF.set([state.ptr[0], state.ptr[1], 0.12 * held, 0], O_WELL);
    state.shocks = state.shocks.filter((s) => now - s.born < 2000);
    for (let k = 0; k < SHOCKS; k++) {
      const s = state.shocks[k];
      uF.set(s ? [s.x, s.y, (now - s.born) / 1000, s.s] : [0, 0, 0, 0], O_SHOCK + k * 4);
    }
  }

  function shock(x, y) {
    if (state.shocks.length >= SHOCKS) state.shocks.shift();
    state.shocks.push({ x, y, born: performance.now(), s: 14 });
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
    else if (e.key === "r" || e.key === "R") { reseed(); tutorial.done("again"); }
  });
  addEventListener("keyup", (e) => {
    if (!ARROWS[e.key]) return;
    state.keys.delete(e.key);
    if (!state.keys.size && performance.now() - state.downAt > 600) tutorial.done("hold");
  });

  $("learn").onclick = () => tutorial.start();
  $("again").onclick = () => { reseed(); tutorial.done("again"); };
  addEventListener("resize", resize);
  resize();
  reseed();

  // ---- frame loop ----
  // GPU frame time = submit to onSubmittedWorkDone with one frame in flight.
  // rAF rate alone hides a queue of unfinished GPU work, which is what feels like lag.
  let acc = 0, prev = performance.now(), winT = prev, gpuMs = 0, inflight = false;
  function frame(now) {
    requestAnimationFrame(frame);
    if (inflight) return; // never queue frames behind a busy GPU

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
      const pass = enc.beginComputePass(timed ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0 } } : {});
      pass.setBindGroup(0, simBG);
      const groups = Math.ceil(state.n / WG), cells = gx * GRID;
      for (let s = 0; s < steps; s++) {
        pass.setPipeline(clearPipe); pass.dispatchWorkgroups(Math.ceil(cells / WG));
        pass.setPipeline(depositPipe); pass.dispatchWorkgroups(groups);
        pass.setPipeline(gravityPipe); pass.dispatchWorkgroups(Math.ceil(cells / 64));
        pass.setPipeline(stepPipe); pass.dispatchWorkgroups(groups);
      }
      pass.end();
    }

    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: hdr.createView(), loadOp: "load", storeOp: "store" }],
      ...(timed && !steps ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0 } } : {}),
    });
    rp.setPipeline(fadePipe);
    rp.setBlendConstant([TRAIL, TRAIL, TRAIL, TRAIL]);
    rp.draw(3);
    rp.setPipeline(drawPipe);
    rp.setBindGroup(0, drawBG);
    rp.draw(state.n);
    rp.end();

    const tp = enc.beginRenderPass({
      colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      ...(timed ? { timestampWrites: { querySet: qs, endOfPassWriteIndex: 1 } } : {}),
    });
    tp.setPipeline(tonePipe);
    tp.setBindGroup(0, toneBG);
    tp.draw(3);
    tp.end();
    if (timed) {
      enc.resolveQuerySet(qs, 0, 2, qResolve, 0);
      enc.copyBufferToBuffer(qResolve, 0, qRead, 0, 16);
    }
    device.queue.submit([enc.finish()]);
    inflight = true;
    const t0 = performance.now();
    const sample = (ms) => {
      inflight = false;
      if (ms > 0 && ms < 1000) gpuMs = gpuMs ? gpuMs + (ms - gpuMs) * 0.15 : ms;
    };
    if (timed) {
      qRead.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigUint64Array(qRead.getMappedRange());
        const ms = Number(t[1] - t[0]) / 1e6;
        qRead.unmap();
        sample(ms);
      }, () => sample(0));
    } else {
      device.queue.onSubmittedWorkDone().then(() => sample(performance.now() - t0));
    }

    // Adaptive load: grow only with clear headroom, back off fast when over budget.
    if (now - winT > 500) {
      winT = now;
      let n = state.n;
      if (gpuMs > BUDGET) {
        if (n > MIN_N) n = Math.max(MIN_N, Math.round(n * 0.6));
        else if (state.scale > 0.5) { state.scale -= 0.25; resize(); }
      } else if (gpuMs < BUDGET * 0.5 && n < cap) n = Math.min(cap, Math.round(n * 1.25));
      if (n !== state.n) { state.n = n; exposure(); }
      $("n").textContent = `${fmt(state.n)} · ${gpuMs.toFixed(1)}ms`;
    }
  }
  requestAnimationFrame(frame);

  if (!store.get("pp.tut3")) tutorial.start();
}

function fmt(n) {
  return n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : Math.round(n / 1e3) + "K";
}

// ---- wordless tutorial: hold, tap, then a new encounter. One word per step. ----
const tutorial = (() => {
  const el = $("tut"), word = $("word"), learn = $("learn");
  const steps = [
    { id: "hold", word: "Hold" },
    { id: "tap", word: "Tap" },
    { id: "again", word: "Again" },
  ];
  let i = -1;

  function show() {
    const s = steps[i];
    if (!s) return end();
    el.dataset.step = s.id;
    word.textContent = s.word;
    $("again").classList.toggle("beckon", s.id === "again");
  }
  function end() {
    el.hidden = true; learn.hidden = false; i = -1;
    $("again").classList.remove("beckon");
    store.set("pp.tut3", "1");
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
