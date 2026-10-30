// Prefor.Pon — GPU galaxy simulation: particle-mesh self-gravity, dark halos, gas and star formation.
// Particles never leave the GPU: compute shaders integrate, render passes read the same buffers.

const coarse = matchMedia("(pointer: coarse)").matches;
const CAP = coarse ? 1 << 19 : 1 << 21; // buffer capacity; active count adapts below it
const BUDGET = coarse ? 9 : 7; // ms of GPU time per frame we allow ourselves
const MIN_N = 1 << 14;
const DT = 1 / 60; // fixed physics step
const MAX_SUBSTEPS = 2; // more substeps on a slow frame only makes the next frame slower
const WG = 256;
const SHOCKS = 4;
const GALAXIES = 3;
const GRID = coarse ? 40 : 56; // gravity grid rows; columns follow the aspect ratio
const G = 0.5;
// ?n=262144 pins the particle count (for benchmarking and comparing looks); otherwise it adapts.
const FIXED_N = Math.max(0, Math.min(1 << 22, parseInt(new URLSearchParams(location.search).get("n"), 10) || 0));

// Particle kinds. Each is picked by hash, so any prefix [0, count) is a fair sample.
//   0 old star · 1 young star · 2 gas · 3 dark matter
const KIND_SHARE = [0.4, 0.06, 0.24, 0.3];
const DM_WEIGHT = 8; // one dark-matter particle outweighs eight baryonic ones: ~3.4x more dark than visible mass
const MEAN_W = 0.7 + 0.3 * DM_WEIGHT;
const LIFE = 0.45; // young-star phase in sim seconds (~70 Myr)
const MYR_PER_SEC = 150; // 1 sim unit of length = 25 kpc, 1 unit of velocity ~ 165 km/s

const U_WGSL = /* wgsl */ `
struct U {
  dt: f32, time: f32, aspect: f32, count: u32,
  seed: u32, gx: u32, gy: u32, ngal: u32,
  well: vec4f,                     // xy pos, z = G*M of the black hole under the finger
  shocks: array<vec4f, ${SHOCKS}>, // xy origin, z age, w strength
  gal: array<vec4f, ${GALAXIES * 2}>, // per galaxy: (cx, cy, vx, vy), (disc scale, cumulative share, spin, mass)
  grav: vec4f,                     // G, softening^2, cell w, cell h
};
const KIND_SHARE = vec4f(${KIND_SHARE.join(", ")});
const DM_WEIGHT = ${DM_WEIGHT}u;
const MEAN_W = ${MEAN_W.toFixed(4)};
const LIFE = ${LIFE};
`;

const SIM_WGSL = U_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> pos: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> mass: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> field: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> attr: array<vec2f>; // kind, birth time
@group(0) @binding(6) var<storage, read_write> gas: array<atomic<u32>>;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd(h: ptr<function, u32>) -> f32 {
  *h = pcg(*h);
  return (f32(*h >> 8u) + 0.5) * (1.0 / 16777216.0);
}

fn kindOf(x: f32) -> u32 {
  var c = KIND_SHARE.x;
  if (x < c) { return 0u; }
  c += KIND_SHARE.y; if (x < c) { return 1u; }
  c += KIND_SHARE.z; if (x < c) { return 2u; }
  return 3u;
}

// Exponential disc of stars and gas inside a Hernquist dark halo, on near-circular orbits.
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
  let ha = rd * 6.0; // halo scale: diffuse enough that the disc's own gravity can grow spiral arms
  let mb = g1.w * 0.7 / MEAN_W;
  let mh = g1.w - mb;
  var kind = kindOf(rnd(&h));

  let a = rnd(&h) * 6.2831853;
  var r: f32;
  var disp = 0.05; // velocity dispersion as a fraction of circular speed: gas and young stars are cold
  if (kind == 3u) {
    let s = sqrt(rnd(&h) * 0.4); // Hernquist, truncated at 40% of its mass (~1.7 scale radii)
    r = ha * s / (1.0 - s);
  } else if (kind == 0u && rnd(&h) < 0.35) {
    r = rd * 0.3 * sqrt(-2.0 * log(rnd(&h))); // bulge: old and hot
    disp = 0.6;
  } else {
    r = -rd * log(rnd(&h) * rnd(&h));
    if (r > rd * 6.0) { r = rd * 6.0 * rnd(&h); } // re-draw the far tail instead of piling it on a ring
    if (kind == 0u) { disp = 0.12; }
  }
  // Seed a two-armed logarithmic spiral (pitch ~15°) in the disc: real discs carry density waves,
  // and a smooth start would take far longer than the encounter to grow its own.
  var th = a;
  if (kind != 3u && disp < 0.5) {
    let amp = select(select(0.9, 0.7, kind == 1u), 0.3, kind == 0u);
    let phase = log(max(r, 1e-3) / rd) / 0.27;
    th = a - 0.5 * amp * sin(2.0 * (a - phase));
  }
  let d = vec2f(cos(th), sin(th));
  let x = r / rd;
  let enclosed = mb * (1.0 - (1.0 + x) * exp(-x)) + mh * min(1.0, 2.5 * r * r / ((r + ha) * (r + ha)));
  let vc = sqrt(u.grav.x * enclosed / sqrt(r * r + u.grav.y));
  var v = vec2f(0.0); // discs get their rotation from the measured field in spin()
  if (kind == 3u) {
    let b = rnd(&h) * 6.2831853;
    v = vec2f(cos(b), sin(b)) * vc * 0.85; // halo: random orbits, no net spin
  }
  v += vec2f(rnd(&h) - 0.5, rnd(&h) - 0.5) * vc * disp * 2.0;
  pos[i] = g0.xy + d * r;
  vel[i] = g0.zw + v;
  // Young stars start with a spread of ages so the first frame is not one generation.
  attr[i] = vec2f(f32(kind), -rnd(&h) * LIFE);
}

fn cellOf(p: vec2f) -> u32 {
  let c = vec2i(floor((p + vec2f(u.aspect, 1.0)) / u.grav.zw));
  let cx = u32(clamp(c.x, 0, i32(u.gx) - 1));
  let cy = u32(clamp(c.y, 0, i32(u.gy) - 1));
  return cy * u.gx + cx;
}

@compute @workgroup_size(${WG})
fn clear(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < u.gx * u.gy) { atomicStore(&mass[id.x], 0u); atomicStore(&gas[id.x], 0u); }
}

@compute @workgroup_size(${WG})
fn deposit(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.count) { return; }
  let c = cellOf(pos[id.x]);
  let kind = u32(attr[id.x].x);
  atomicAdd(&mass[c], select(1u, DM_WEIGHT, kind == 3u));
  if (kind == 2u) { atomicAdd(&gas[c], 1u); }
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
  if (id.x < cells) { field[id.x] = acc * (u.grav.x / (f32(u.count) * MEAN_W)); }
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
  var at = attr[i];
  let kind = u32(at.x);
  var h = pcg(i ^ u.seed);

  var a = sampleField(p);

  // Black hole under the finger.
  if (u.well.z > 0.0) {
    let d = u.well.xy - p;
    let ir = inverseSqrt(dot(d, d) + 0.0008);
    a += d * (u.well.z * ir * ir * ir);
  }

  // Supernova: an expanding front that kicks stars and gas outward as it passes.
  for (var k = 0u; k < ${SHOCKS}u; k++) {
    let s = u.shocks[k];
    if (s.w == 0.0) { continue; }
    let d = p - s.xy;
    let r = length(d) + 1e-4;
    let x = (r - s.z * 0.9) / 0.04;
    a += d / r * (s.w * exp(-x * x) * exp(-s.z * 2.0));
  }

  if (kind == 2u) {
    // Gas is dissipative: it sheds random motion toward the local flow, which is why
    // real discs keep thin, sharp gas arms while stars stay puffy.
    let c = cellOf(p);
    let dens = f32(atomicLoad(&gas[c]));
    let mean = f32(u.count) * KIND_SHARE.z / f32(u.gx * u.gy);
    // Star formation above a density threshold (Kennicutt–Schmidt-like: faster when denser).
    let over = dens / max(mean, 1e-3);
    if (over > 6.0 && rnd(&h) < (over - 6.0) * 0.004 * u.dt * 60.0) {
      at = vec2f(1.0, u.time);
    }
  } else if (kind == 1u && u.time - at.y > LIFE) {
    // End of the massive stars: a third returns to gas with a supernova kick, the rest
    // are long-lived low-mass stars that join the old population.
    if (rnd(&h) < 0.35) {
      let b = rnd(&h) * 6.2831853;
      v += vec2f(cos(b), sin(b)) * 0.08;
      at = vec2f(2.0, u.time);
    } else {
      at = vec2f(0.0, at.y);
    }
  }

  // Soft leash past the screen edge, so escaping stars come home instead of vanishing.
  let lim = vec2f(u.aspect, 1.0) * 1.3;
  a -= sign(p) * max(abs(p) - lim, vec2f(0.0)) * 3.0;

  v += a * u.dt; // kick-drift: symplectic, keeps orbits from spiralling in or out
  p += v * u.dt;
  pos[i] = p;
  vel[i] = v;
  attr[i] = at;
}

// Circular velocities from the force the grid actually produces, not from a formula:
// mismatched starts make cold gas ring like a struck bell (concentric rings, not spirals).
@compute @workgroup_size(${WG})
fn spin(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.count || u32(attr[i].x) == 3u) { return; }
  var h = pcg(i ^ u.seed);
  let pick = rnd(&h);
  var k = 0u;
  for (var j = 0u; j < u.ngal - 1u; j++) { if (pick > u.gal[j * 2u + 1u].y) { k = j + 1u; } }
  let g0 = u.gal[k * 2u];
  let d = pos[i] - g0.xy;
  let r = max(length(d), 1e-4);
  let inward = max(-dot(sampleField(pos[i]), d / r), 0.0);
  let vc = sqrt(inward * r);
  vel[i] += vec2f(-d.y, d.x) / r * vc * u.gal[k * 2u + 1u].z;
}
`;

const DRAW_WGSL = U_WGSL + /* wgsl */ `
struct R { gain: f32, gasGain: f32, exposure: f32, time: f32, well: vec4f };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> pos: array<vec2f>;
@group(0) @binding(2) var<storage, read> attr: array<vec2f>;
@group(0) @binding(3) var<uniform> r: R;

struct VO { @builtin(position) p: vec4f, @location(0) c: vec4f };

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

// Blackbody colour, linear RGB, normalised to unit luminance (fit to CIE 1931 / sRGB primaries).
fn blackbody(t: f32) -> vec3f {
  let k = clamp(t, 1500.0, 40000.0) / 100.0;
  var c: vec3f;
  c.r = select(1.0, clamp(1.2929 * pow(k - 60.0, -0.1332), 0.0, 1.0), k > 66.0);
  c.g = select(clamp(0.3901 * log(k) - 0.6318, 0.0, 1.0), clamp(1.1299 * pow(k - 60.0, -0.0755), 0.0, 1.0), k > 66.0);
  c.b = select(select(clamp(0.5432 * log(k - 10.0) - 1.1963, 0.0, 1.0), 0.0, k < 19.0), 1.0, k >= 66.0);
  let lin = pow(c, vec3f(2.2));
  return lin / max(dot(lin, vec3f(0.2126, 0.7152, 0.0722)), 1e-3);
}

const OFF = vec4f(2.0, 2.0, 0.0, 1.0); // outside clip space: culled

fn starLight(i: u32, at: vec2f) -> vec3f {
  let kind = u32(at.x);
  let h = pcg(i * 9781u + 17u);
  let x = f32(h & 0xffffu) / 65535.0;
  var t: f32;
  var lum: f32;
  if (kind == 0u) {
    // Old population: K and G dwarfs and red giants. Colour from a skewed temperature spread.
    t = 3400.0 + 2800.0 * x * x;
    lum = 0.6 + 1.4 * x * x * x;
  } else {
    // Young population: O/B stars dominate the light, then fade and redden as they age.
    let age = clamp((u.time - at.y) / LIFE, 0.0, 1.0);
    t = mix(9000.0 + 21000.0 * x, 6500.0, age);
    lum = mix(4.0 + 12.0 * x * x, 1.2, sqrt(age));
  }
  return blackbody(t) * lum * r.gain;
}

// Full resolution: every star as a crisp point.
@vertex fn stars(@builtin(vertex_index) i: u32) -> VO {
  let at = attr[i];
  var o: VO;
  if (u32(at.x) >= 2u) { o.p = OFF; return o; }
  let p = pos[i];
  o.p = vec4f(p.x / u.aspect, p.y, 0.0, 1.0);
  o.c = vec4f(starLight(i, at), 0.0);
  return o;
}

// Quarter resolution: the same starlight (rgb) plus gas (a). Upsampled, this is the smooth
// unresolved glow of a galaxy and the column density for dust and nebulae.
@vertex fn glow(@builtin(vertex_index) i: u32) -> VO {
  let at = attr[i];
  let kind = u32(at.x);
  var o: VO;
  if (kind == 3u) { o.p = OFF; return o; }
  let p = pos[i];
  o.p = vec4f(p.x / u.aspect, p.y, 0.0, 1.0);
  o.c = select(vec4f(starLight(i, at), 0.0), vec4f(0.0, 0.0, 0.0, r.gasGain), kind == 2u);
  return o;
}

@fragment fn fs(i: VO) -> @location(0) vec4f { return i.c; }
`;

const FULL_VS = /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
  let y = f32(i & 2u) * 2.0 - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}
`;

// Compositing: dust extinction and H-II emission from the gas map, a sky of distant
// galaxies, gravitational lensing around the black hole, then a filmic curve.
const TONE_WGSL = FULL_VS + /* wgsl */ `
struct R { gain: f32, gasGain: f32, exposure: f32, time: f32, well: vec4f }; // well: px xy, Einstein radius px, -
@group(0) @binding(0) var light: texture_2d<f32>;
@group(0) @binding(1) var glowMap: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var<uniform> r: R;

fn hash2(p: vec2f) -> vec2f {
  let q = vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)));
  return fract(sin(q) * 43758.5453);
}

// Faint field galaxies: sparse, tiny, mostly yellow-white ellipticals and a few blue spirals.
fn sky(px: vec2f) -> vec3f {
  let cell = 22.0;
  let g = floor(px / cell);
  var col = vec3f(0.0);
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let c = g + vec2f(f32(x), f32(y));
      let h = hash2(c);
      if (h.x > 0.11) { continue; }
      let h2 = hash2(c + 31.7);
      let centre = (c + h2) * cell;
      let ang = h.y * 6.283;
      let d = px - centre;
      let rot = vec2f(d.x * cos(ang) + d.y * sin(ang), -d.x * sin(ang) + d.y * cos(ang));
      let size = 0.6 + h2.x * 1.6;
      let q = rot / vec2f(size, size * (0.35 + h2.y * 0.65));
      let tint = select(vec3f(1.0, 0.85, 0.65), vec3f(0.7, 0.8, 1.0), h.x < 0.03);
      col += tint * exp(-dot(q, q)) * (0.004 + 0.02 * h2.y * h2.y);
    }
  }
  return col;
}

fn filmic(x: vec3f) -> vec3f { // ACES fit (Narkowicz)
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

@fragment fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let size = vec2f(textureDimensions(light));
  var px = fc.xy;
  // Point-mass lens: light from behind bends toward the hole; inside the Einstein ring
  // the image flips, and the shadow at the centre stays black.
  var shadow = 1.0;
  if (r.well.z > 0.0) {
    let d = px - r.well.xy;
    let d2 = max(dot(d, d), 1.0);
    px = px - d * (r.well.z * r.well.z / d2);
    shadow = smoothstep(0.35, 0.55, sqrt(d2) / r.well.z);
  }
  let uv = px / size;
  // Most of a galaxy's light is unresolved stars: mix each crisp point with the quarter-res
  // glow map (tent-filtered upsample), so sparse counts read as smooth light, not glitter.
  let q = 1.0 / vec2f(textureDimensions(glowMap));
  let gl = (textureSampleLevel(glowMap, lin, uv, 0.0) * 2.0
          + textureSampleLevel(glowMap, lin, uv + vec2f(q.x, 0.0), 0.0) + textureSampleLevel(glowMap, lin, uv - vec2f(q.x, 0.0), 0.0)
          + textureSampleLevel(glowMap, lin, uv + vec2f(0.0, q.y), 0.0) + textureSampleLevel(glowMap, lin, uv - vec2f(0.0, q.y), 0.0)) / 6.0;
  let sharp = textureSampleLevel(light, lin, uv, 0.0).rgb;
  var c = sharp * 0.3 + gl.rgb * (0.7 / 16.0);
  let gasD = gl.a / 16.0;
  // H-II regions: gas lit up by hot young stars. H-alpha red plus H-beta/[O III] blue reads pink.
  // Young light is what is bluer than the old population, so ionisation follows the blue excess.
  let young = max(gl.b - gl.r * 0.75, 0.0) / 16.0;
  c += vec3f(1.0, 0.3, 0.5) * young * min(gasD, 3.0) * 0.6;
  // Dust: reddening extinction, stronger in blue (roughly R_V = 3.1).
  c *= exp(-gasD * vec3f(0.5, 0.7, 1.0) * 0.9);
  c += sky(px);
  c *= shadow;
  let mapped = filmic(c * r.exposure);
  return vec4f(pow(mapped, vec3f(1.0 / 2.2)), 1.0);
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
  const toneMod = device.createShaderModule({ code: TONE_WGSL });

  const storage = (i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } });
  const simLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, storage(1), storage(2), storage(3), storage(4), storage(5), storage(6)],
  });
  const simPL = device.createPipelineLayout({ bindGroupLayouts: [simLayout] });
  const cp = (entryPoint) => device.createComputePipeline({ layout: simPL, compute: { module: simMod, entryPoint } });
  const seedPipe = cp("seed"), clearPipe = cp("clear"), depositPipe = cp("deposit"), gravityPipe = cp("gravity"), stepPipe = cp("step"), spinPipe = cp("spin");
  const blend = (srcFactor, dstFactor) => ({ color: { srcFactor, dstFactor, operation: "add" }, alpha: { srcFactor, dstFactor, operation: "add" } });
  const drawLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
    ],
  });
  const drawPL = device.createPipelineLayout({ bindGroupLayouts: [drawLayout] });
  const points = (entryPoint, fmt) => device.createRenderPipeline({
    layout: drawPL,
    vertex: { module: drawMod, entryPoint },
    fragment: { module: drawMod, entryPoint: "fs", targets: [{ format: fmt, blend: blend("one", "one") }] },
    primitive: { topology: "point-list" },
  });
  const starPipe = points("stars", HDR);
  const glowPipe = points("glow", "rgba16float");
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
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
  const rBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const uData = new ArrayBuffer(U_SIZE);
  const uF = new Float32Array(uData);
  const uU = new Uint32Array(uData);

  const state = {
    n: Math.min(cap, FIXED_N || (coarse ? 1 << 15 : 1 << 16)), // start light, grow only while the GPU has headroom
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
  const gasBuf = device.createBuffer({ size: maxCells * 4, usage });
  const attr = device.createBuffer({ size: cap * 8, usage });
  const base = [{ binding: 0, resource: { buffer: uBuf } }, { binding: 1, resource: { buffer: pos } }, { binding: 2, resource: { buffer: vel } }];
  const simBG = device.createBindGroup({
    layout: simLayout,
    entries: [
      ...base, { binding: 3, resource: { buffer: massBuf } }, { binding: 4, resource: { buffer: fieldBuf } },
      { binding: 5, resource: { buffer: attr } }, { binding: 6, resource: { buffer: gasBuf } },
    ],
  });
  const drawBG = device.createBindGroup({
    layout: drawLayout,
    entries: [base[0], base[1], { binding: 2, resource: { buffer: attr } }, { binding: 3, resource: { buffer: rBuf } }],
  });
  let gx = GRID;
  let hdr, glowTex, toneBG;
  const rData = new Float32Array(8);

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
      const along = t * span * 0.6, off = (Math.random() - 0.5) * 0.5 * unit;
      const c = wide ? [along, off] : [off, along];
      const speed = 0.1 + Math.random() * 0.08;
      const v = wide ? [-t * speed, (Math.random() - 0.5) * 0.15] : [(Math.random() - 0.5) * 0.15, -t * speed];
      const m = 0.5 + Math.random() * 0.5;
      total += m;
      gs.push({ c, v, m, rd: (0.07 + Math.random() * 0.04) * unit * (0.6 + m * 0.6), spin: Math.random() < 0.5 ? -1 : 1 });
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
    pass.setBindGroup(0, simBG);
    const groups = Math.ceil(cap / WG), cells = gx * GRID;
    pass.setPipeline(seedPipe); pass.dispatchWorkgroups(groups);
    pass.setPipeline(clearPipe); pass.dispatchWorkgroups(Math.ceil(cells / WG));
    pass.setPipeline(depositPipe); pass.dispatchWorkgroups(groups);
    pass.setPipeline(gravityPipe); pass.dispatchWorkgroups(Math.ceil(cells / 64));
    pass.setPipeline(spinPipe); pass.dispatchWorkgroups(groups);
    pass.end();
    device.queue.submit([enc.finish()]);
    state.time = 0;
  }

  // Keep surface brightness constant regardless of particle count and screen size.
  function exposure() {
    const px = canvas.width * canvas.height || 1;
    rData[0] = (0.25 * px) / state.n;
    rData[1] = (0.6 * px) / state.n;
    rData[2] = 0.32;
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
    hdr?.destroy(); glowTex?.destroy();
    const tex = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    hdr = device.createTexture({ size: [w, h], format: HDR, usage: tex });
    glowTex = device.createTexture({ size: [Math.ceil(w / 4), Math.ceil(h / 4)], format: "rgba16float", usage: tex });
    toneBG = device.createBindGroup({
      layout: tonePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: hdr.createView() }, { binding: 1, resource: glowTex.createView() },
        { binding: 2, resource: sampler }, { binding: 3, resource: { buffer: rBuf } },
      ],
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
    uU[4] = (now * 1000) >>> 0; // per-frame randomness for star formation and supernovae
    // Lens: Einstein radius grows with the hole's mass (sqrt), in screen pixels.
    rData[3] = state.time;
    rData[4] = ((state.ptr[0] / state.aspect + 1) / 2) * canvas.width;
    rData[5] = ((1 - state.ptr[1]) / 2) * canvas.height;
    rData[6] = held > 0 ? Math.sqrt(held) * 0.045 * canvas.height : 0;
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
    else if (e.key === "p" || e.key === "P") time.toggle();
    else if (e.key === "," || e.key === "<") time.nudge(-1);
    else if (e.key === "." || e.key === ">") time.nudge(1);
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
    acc += dt * time.rate();
    // Faster time means more fixed steps, not bigger ones, so orbits stay stable.
    const maxSteps = Math.max(MAX_SUBSTEPS, Math.ceil(MAX_SUBSTEPS * time.rate()));
    let steps = 0;
    while (acc >= DT && steps < maxSteps) { acc -= DT; steps++; }
    if (steps === maxSteps) acc = 0; // drop backlog rather than spiral

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

    device.queue.writeBuffer(rBuf, 0, rData);
    const clear = { r: 0, g: 0, b: 0, a: 0 };
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: hdr.createView(), loadOp: "clear", clearValue: clear, storeOp: "store" }],
      ...(timed && !steps ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0 } } : {}),
    });
    rp.setPipeline(starPipe);
    rp.setBindGroup(0, drawBG);
    rp.draw(state.n);
    rp.end();
    const gp = enc.beginRenderPass({ colorAttachments: [{ view: glowTex.createView(), loadOp: "clear", clearValue: clear, storeOp: "store" }] });
    gp.setPipeline(glowPipe);
    gp.setBindGroup(0, drawBG);
    gp.draw(state.n);
    gp.end();

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
      if (FIXED_N) {
        // pinned: no adaptation
      } else if (gpuMs > BUDGET) {
        if (n > MIN_N) n = Math.max(MIN_N, Math.round(n * 0.6));
        else if (state.scale > 0.5) { state.scale -= 0.25; resize(); }
      } else if (gpuMs < BUDGET * 0.5 && n < cap) n = Math.min(cap, Math.round(n * 1.25));
      if (n !== state.n) { state.n = n; exposure(); }
      $("n").textContent = `${fmt(state.n)} · ${gpuMs.toFixed(1)}ms · ${age(state.time)}`;
    }
  }
  requestAnimationFrame(frame);

  if (!store.get("pp.tut4")) tutorial.start();
}

function age(t) {
  const myr = t * MYR_PER_SEC;
  return myr >= 1000 ? (myr / 1000).toFixed(2) + " Gyr" : Math.round(myr) + " Myr";
}

function fmt(n) {
  return n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : Math.round(n / 1e3) + "K";
}

// ---- time: log-scale rate slider (1/8x to 4x) and a pause toggle ----
const time = (() => {
  const slider = $("rate"), pause = $("pause");
  let paused = false;
  const sync = () => {
    pause.setAttribute("aria-pressed", String(paused));
    const r = 2 ** Number(slider.value);
    const label = `${Math.round(r * MYR_PER_SEC)} Myr/s`;
    slider.setAttribute("aria-valuetext", label);
    $("myr").textContent = paused ? "0 Myr/s" : label;
  };
  slider.addEventListener("input", () => { paused = false; sync(); tutorial.done("time"); });
  pause.onclick = () => api.toggle();
  const api = {
    rate: () => (paused ? 0 : 2 ** Number(slider.value)),
    toggle() { paused = !paused; sync(); tutorial.done("time"); },
    nudge(d) { slider.value = String(Number(slider.value) + d); paused = false; sync(); tutorial.done("time"); },
  };
  sync();
  return api;
})();

// ---- wordless tutorial: hold, tap, time, then a new encounter. One word per step. ----
const tutorial = (() => {
  const el = $("tut"), word = $("word"), learn = $("learn");
  const steps = [
    { id: "hold", word: "Hold" },
    { id: "tap", word: "Tap" },
    { id: "time", word: "Time" },
    { id: "again", word: "Again" },
  ];
  let i = -1;

  function show() {
    const s = steps[i];
    if (!s) return end();
    el.dataset.step = s.id;
    word.textContent = s.word;
    $("again").classList.toggle("beckon", s.id === "again");
    $("clock").classList.toggle("beckon", s.id === "time");
  }
  function end() {
    el.hidden = true; learn.hidden = false; i = -1;
    $("again").classList.remove("beckon");
    $("clock").classList.remove("beckon");
    store.set("pp.tut4", "1");
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
