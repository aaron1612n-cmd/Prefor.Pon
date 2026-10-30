// Prefor.Pon — 3D GPU galaxy simulation.
// Particle-mesh gravity with an FFT Poisson solve, dark halos, gas and star formation,
// rendered like a telescope image. Particles never leave the GPU.

const coarse = matchMedia("(pointer: coarse)").matches;
const CAP = coarse ? 1 << 18 : 1 << 20; // buffer capacity; active count adapts below it
const BUDGET = coarse ? 10 : 8; // ms of GPU time per frame we allow ourselves
const MIN_N = 1 << 14;
const DT = 1 / 60; // fixed physics step
const MAX_SUBSTEPS = 2; // more substeps on a slow frame only makes the next frame slower
const WG = 256;
const SHOCKS = 4;
const GALAXIES = 5;
const SPIRAL = 0, BARRED = 1, ELLIPTICAL = 2, DWARF = 3;
const G = 0.5;
// Gravity grid: N^3 physical cells inside a 2N^3 zero-padded box, so the FFT convolution
// is isolated (no periodic ghost galaxies). Cell size H = L / N.
const N = coarse ? 32 : 64;
const NP = N * 2;
const LOG2NP = Math.log2(NP);
const L = coarse ? 2.4 : 2.6;
const H = L / N;
const FOV = 0.9; // radians, vertical
const HOME = 2.7; // default camera distance
// ?n=262144 pins the particle count (for benchmarking and comparing looks); otherwise it adapts.
const FIXED_N = Math.max(0, Math.min(1 << 21, parseInt(new URLSearchParams(location.search).get("n"), 10) || 0));

// Particle kinds. Each is picked by hash, so any prefix [0, count) is a fair sample.
//   0 old star · 1 young star · 2 gas · 3 dark matter · 4 central black hole · 5 swallowed hole
const KIND_SHARE = [0.4, 0.06, 0.24, 0.3];
const DM_WEIGHT = 8; // one dark-matter particle outweighs eight baryonic ones: ~3.4x more dark than visible mass
const MEAN_W = 0.7 + 0.3 * DM_WEIGHT;
const LIFE = 0.45; // young-star phase in sim seconds (~70 Myr)
const MYR_PER_SEC = 150; // 1 length unit = 25 kpc, 1 velocity unit ~ 165 km/s

const U_WGSL = /* wgsl */ `
struct U {
  dt: f32, time: f32, count: u32, ngal: u32,
  seed: u32, n: u32, np: u32, _p: u32,
  grav: vec4f,                         // G, softening^2, cell size, half box
  well: vec4f,                         // black hole under the finger: xyz, G*M
  shocks: array<vec4f, ${SHOCKS * 2}>, // per shock: (xyz, age), (strength, -, -, -)
  gal: array<vec4f, ${GALAXIES * 4}>,  // per galaxy: (centre, scale), (velocity, mass share), (normal, cumulative share), (type, phase, spin, -)
};
const KIND_SHARE = vec4f(${KIND_SHARE.join(", ")});
const DM_WEIGHT = ${DM_WEIGHT}u;
const MEAN_W = ${MEAN_W.toFixed(4)};
const LIFE = ${LIFE};
const CLUSTER = -100.0; // birth-time marker for globular-cluster stars: they keep their seeded orbit
`;

const SIM_WGSL = U_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> mass: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> force: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> attr: array<vec2f>; // kind, birth time
@group(0) @binding(6) var<storage, read_write> gas: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> spec: array<vec2f>;  // padded complex grid for the FFT
@group(0) @binding(8) var<storage, read_write> green: array<f32>;   // spectrum of the softened 1/r kernel

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd(h: ptr<function, u32>) -> f32 {
  *h = pcg(*h);
  return (f32(*h >> 8u) + 0.5) * (1.0 / 16777216.0);
}
fn gauss(h: ptr<function, u32>) -> f32 {
  return sqrt(-2.0 * log(rnd(h))) * cos(6.2831853 * rnd(h));
}
fn unitVec(h: ptr<function, u32>) -> vec3f {
  let z = rnd(h) * 2.0 - 1.0;
  let a = rnd(h) * 6.2831853;
  let s = sqrt(1.0 - z * z);
  return vec3f(s * cos(a), s * sin(a), z);
}

fn kindOf(x: f32) -> u32 {
  var c = KIND_SHARE.x;
  if (x < c) { return 0u; }
  c += KIND_SHARE.y; if (x < c) { return 1u; }
  c += KIND_SHARE.z; if (x < c) { return 2u; }
  return 3u;
}

fn galaxyOf(pick: f32) -> u32 {
  var k = 0u;
  for (var j = 0u; j < u.ngal - 1u; j++) { if (pick > u.gal[j * 4u + 2u].w) { k = j + 1u; } }
  return k;
}

// Two unit vectors spanning a disc plane with normal nr.
fn basis(nr: vec3f) -> mat2x3f {
  let rf = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(nr.z) > 0.9);
  let e1 = normalize(cross(nr, rf));
  return mat2x3f(e1, cross(nr, e1));
}

fn enclosedMass(r: f32, rd: f32, ha: f32, mb: f32, mh: f32) -> f32 {
  let x = r / rd;
  return mb * (1.0 - (1.0 + x) * exp(-x)) + mh * min(1.0, 2.5 * r * r / ((r + ha) * (r + ha)));
}

// Galaxy types: 0 spiral, 1 barred spiral, 2 elliptical, 3 dwarf irregular.
// Discs are exponential, thin for gas, thicker for old stars, with a bulge, inside a spherical
// Hernquist dark halo. Ellipticals are hot spheroids of old stars; dwarfs are gas-rich and puffy.
@compute @workgroup_size(${WG})
fn seed(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.count) { return; }
  // The first ngal particles are the central supermassive black holes: always inside any prefix.
  if (i < u.ngal) {
    pos[i] = vec4f(u.gal[i * 4u].xyz, 0.0);
    vel[i] = vec4f(u.gal[i * 4u + 1u].xyz, 0.0);
    attr[i] = vec2f(4.0, -1000.0);
    return;
  }
  var h = pcg(i ^ u.seed);
  let k = galaxyOf(rnd(&h));
  let g0 = u.gal[k * 4u];
  let g1 = u.gal[k * 4u + 1u];
  let nrm = u.gal[k * 4u + 2u].xyz;
  let g3 = u.gal[k * 4u + 3u];
  let gtype = u32(g3.x);
  let rd = g0.w;
  let ha = rd * 6.0; // halo scale: diffuse enough that the disc's own gravity can grow spiral arms
  let mb = g1.w * 0.7 / MEAN_W;
  let mh = g1.w - mb;
  let e = basis(nrm);
  var kind = kindOf(rnd(&h));
  if (gtype == 2u && kind != 3u) { kind = 0u; }                  // ellipticals: red and dead
  if (gtype == 3u && kind == 0u && rnd(&h) < 0.4) { kind = 2u; } // dwarfs: gas-rich

  var p: vec3f;
  var v = vec3f(0.0); // discs get their rotation from the measured field in spin()
  var birth = -rnd(&h) * LIFE; // young stars start with a spread of ages
  if (kind == 3u || gtype == 2u) {
    // Spheroids: Hernquist profile, random isotropic orbits.
    let scale = select(rd * 0.5, ha, kind == 3u);
    let s = sqrt(rnd(&h) * select(0.85, 0.4, kind == 3u));
    let r = scale * s / (1.0 - s);
    p = unitVec(&h) * r;
    let vc = sqrt(u.grav.x * enclosedMass(r, rd, ha, mb, mh) / sqrt(r * r + u.grav.y));
    v = unitVec(&h) * vc * 0.7;
  } else if (kind == 0u && gtype != 3u && rnd(&h) < 0.006) {
    // Globular clusters: a dozen tight, old clumps on random halo orbits.
    let c = u32(rnd(&h) * 12.0);
    var hc = pcg(k * 977u + c * 131u + u.seed);
    let cr = rd * (1.5 + 5.0 * rnd(&hc));
    let dir = unitVec(&hc);
    let vc = sqrt(u.grav.x * enclosedMass(cr, rd, ha, mb, mh) / cr);
    p = dir * cr + vec3f(gauss(&h), gauss(&h), gauss(&h)) * rd * 0.03;
    v = normalize(cross(dir, unitVec(&hc))) * vc * 0.8;
    birth = CLUSTER;
  } else if (kind == 0u && gtype != 3u && rnd(&h) < 0.35) {
    // Bulge: old, hot, slightly flattened.
    let b = vec3f(gauss(&h), gauss(&h), gauss(&h)) * rd * 0.3;
    p = e[0] * b.x + e[1] * b.y + nrm * b.z * 0.7;
    let r = length(p);
    let vc = sqrt(u.grav.x * enclosedMass(r, rd, ha, mb, mh) / sqrt(r * r + u.grav.y));
    v = unitVec(&h) * vc * 0.45;
  } else {
    var r = -rd * log(rnd(&h) * rnd(&h));
    if (r > rd * 6.0) { r = rd * 6.0 * rnd(&h); } // re-draw the far tail instead of piling it on a ring
    // Two-armed log spiral (pitch ~15°), strongest in gas and young stars; barred types get a
    // straight bar inside ~1.6 scale lengths. Real discs carry these density waves.
    let a = rnd(&h) * 6.2831853;
    var th = a;
    if (gtype <= 1u) {
      let amp = select(select(0.9, 0.7, kind == 1u), 0.3, kind == 0u);
      var phase = log(max(r, 1e-3) / rd) / 0.27 + g3.y;
      var w = amp;
      if (gtype == 1u && r < rd * 1.6) { phase = g3.y; w = 0.95; }
      th = a - 0.5 * w * sin(2.0 * (a - phase));
    }
    // Scale heights as in the Milky Way: cold gas ~ 1/25 of the scale length, young stars a
    // little thicker, the old disc ~ 1/7. Dwarfs are puffy.
    var zh = select(select(0.15, 0.06, kind == 1u), 0.04, kind == 2u) * rd;
    var disp = select(0.05, 0.12, kind == 0u);
    if (gtype == 3u) { zh = 0.3 * rd; disp = 0.25; }
    p = e[0] * (r * cos(th)) + e[1] * (r * sin(th)) + nrm * (zh * gauss(&h));
    let vc = sqrt(u.grav.x * enclosedMass(r, rd, ha, mb, mh) / sqrt(r * r + u.grav.y));
    v = (e[0] * gauss(&h) + e[1] * gauss(&h) + nrm * gauss(&h) * 0.5) * vc * disp;
  }
  pos[i] = vec4f(g0.xyz + p, 0.0);
  vel[i] = vec4f(g1.xyz + v, 0.0);
  attr[i] = vec2f(f32(kind), birth);
}

fn cellOf(p: vec3f) -> u32 {
  let c = clamp(vec3i(floor((p + u.grav.w) / u.grav.z)), vec3i(0), vec3i(i32(u.n) - 1));
  return (u32(c.z) * u.n + u32(c.y)) * u.n + u32(c.x);
}

@compute @workgroup_size(${WG})
fn clear(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < u.n * u.n * u.n) { atomicStore(&gas[id.x], 0u); }
}

@compute @workgroup_size(${WG})
fn deposit(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.count) { return; }
  let c = cellOf(pos[id.x].xyz);
  let kind = u32(attr[id.x].x);
  var w = select(1u, DM_WEIGHT, kind == 3u);
  // A central black hole weighs ~1% of its galaxy here: heavier than real (~0.1%), so the
  // coarse grid can still sink it to the centre of a merger remnant.
  if (kind == 4u) { w = u32(0.01 * u.gal[id.x * 4u + 1u].w * f32(u.count) * MEAN_W); }
  if (kind == 5u) { w = 0u; }
  atomicAdd(&mass[c], w);
  if (kind == 2u) { atomicAdd(&gas[c], 1u); }
}

// Mass into the padded complex grid (and the counts back to zero for the next step).
@compute @workgroup_size(${WG})
fn pack(@builtin(global_invocation_id) id: vec3u) {
  let np = u.np;
  if (id.x >= np * np * np) { return; }
  let x = id.x % np; let y = (id.x / np) % np; let z = id.x / (np * np);
  var m = 0.0;
  if (x < u.n && y < u.n && z < u.n) {
    m = f32(atomicExchange(&mass[(z * u.n + y) * u.n + x], 0u)) / (f32(u.count) * MEAN_W);
  }
  spec[id.x] = vec2f(m, 0.0);
}

// Softened -1/r kernel on the padded box, with wrapped distances. Its spectrum is kept in green[].
@compute @workgroup_size(${WG})
fn greenInit(@builtin(global_invocation_id) id: vec3u) {
  let np = u.np;
  if (id.x >= np * np * np) { return; }
  let c = vec3u(id.x % np, (id.x / np) % np, id.x / (np * np));
  let w = vec3f(min(c, vec3u(np) - c)) * u.grav.z;
  spec[id.x] = vec2f(-inverseSqrt(dot(w, w) + u.grav.y), 0.0);
}

@compute @workgroup_size(${WG})
fn keep(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < u.np * u.np * u.np) { green[id.x] = spec[id.x].x; }
}

// Convolution theorem: potential spectrum = density spectrum x kernel spectrum.
@compute @workgroup_size(${WG})
fn convolve(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < u.np * u.np * u.np) { spec[id.x] *= green[id.x]; }
}

// Force on the physical grid: central differences of the potential.
@compute @workgroup_size(${WG})
fn gradient(@builtin(global_invocation_id) id: vec3u) {
  let n = i32(u.n);
  if (id.x >= u.n * u.n * u.n) { return; }
  let c = vec3i(i32(id.x % u.n), i32((id.x / u.n) % u.n), i32(id.x / (u.n * u.n)));
  let np = i32(u.np);
  let norm = 1.0 / f32(u.np * u.np * u.np); // the inverse FFT is unnormalised
  var g = vec3f(0.0);
  for (var a = 0; a < 3; a++) {
    var o = vec3i(0);
    o[a] = 1;
    let hi = clamp(c + o, vec3i(0), vec3i(n - 1));
    let lo = clamp(c - o, vec3i(0), vec3i(n - 1));
    let ph = spec[(hi.z * np + hi.y) * np + hi.x].x;
    let pl = spec[(lo.z * np + lo.y) * np + lo.x].x;
    g[a] = (ph - pl) / (f32(hi[a] - lo[a]) * u.grav.z);
  }
  force[id.x] = vec4f(-g * norm * u.grav.x, 0.0);
}

fn sampleForce(p: vec3f) -> vec3f {
  let n = i32(u.n);
  let g = clamp((p + u.grav.w) / u.grav.z - 0.5, vec3f(0.0), vec3f(f32(n) - 1.001));
  let i0 = vec3i(floor(g));
  let f = g - vec3f(i0);
  var acc = vec3f(0.0);
  for (var c = 0; c < 8; c++) {
    let o = vec3i(c & 1, (c >> 1) & 1, (c >> 2) & 1);
    let q = min(i0 + o, vec3i(n - 1));
    let w = mix(1.0 - f, f, vec3f(o));
    acc += force[(q.z * n + q.y) * n + q.x].xyz * (w.x * w.y * w.z);
  }
  return acc;
}

@compute @workgroup_size(${WG})
fn step(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.count) { return; }
  var p = pos[i].xyz;
  var v = vel[i].xyz;
  var at = attr[i];
  let kind = u32(at.x);
  var h = pcg(i ^ u.seed);

  var a = sampleForce(p);

  // Black hole under the finger.
  if (u.well.w > 0.0) {
    let d = u.well.xyz - p;
    let ir = inverseSqrt(dot(d, d) + 0.0008);
    a += d * (u.well.w * ir * ir * ir);
  }

  // Supernova: an expanding spherical front that kicks stars and gas outward as it passes.
  for (var k = 0u; k < ${SHOCKS}u; k++) {
    let s = u.shocks[k * 2u];
    let st = u.shocks[k * 2u + 1u].x;
    if (st == 0.0) { continue; }
    let d = p - s.xyz;
    let r = length(d) + 1e-4;
    let x = (r - s.w * 0.9) / 0.04;
    a += d / r * (st * exp(-x * x) * exp(-s.w * 2.0));
  }

  if (kind == 2u) {
    // Star formation above a gas-density threshold, faster the denser (Kennicutt–Schmidt-like).
    // The reference is the gas of all discs spread over a few percent of one grid layer, so the
    // threshold is crossed in spiral arms, bars and collision shocks, not across whole discs.
    let dens = f32(atomicLoad(&gas[cellOf(p)]));
    let rf = f32(u.count) * KIND_SHARE.z / (f32(u.n * u.n) * 0.06);
    let over = dens / max(rf, 1e-3);
    if (over > 1.5 && rnd(&h) < (over - 1.5) * 0.002 * u.dt * 60.0) {
      at = vec2f(1.0, u.time);
    }
  } else if (kind == 1u && u.time - at.y > LIFE) {
    // End of the massive stars: a third return to gas with a supernova kick; the rest are
    // long-lived low-mass stars that join the old population.
    if (rnd(&h) < 0.35) {
      v += unitVec(&h) * 0.08;
      at = vec2f(2.0, u.time);
    } else {
      at = vec2f(0.0, at.y);
    }
  }

  // Soft leash near the edge of the gravity box, so escaping stars come home.
  let lim = u.grav.w * 0.85;
  a -= sign(p) * max(abs(p) - lim, vec3f(0.0)) * 4.0;

  v += a * u.dt; // kick-drift: symplectic, keeps orbits from spiralling in or out
  p += v * u.dt;
  pos[i] = vec4f(p, 0.0);
  vel[i] = vec4f(v, 0.0);
  attr[i] = at;
}

// Circular velocities from the force the grid actually produces, not from a formula:
// mismatched starts make cold gas ring like a struck bell (concentric rings, not spirals).
@compute @workgroup_size(${WG})
fn spin(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.count || u32(attr[i].x) >= 3u || attr[i].y == CLUSTER) { return; }
  var h = pcg(i ^ u.seed);
  let k = galaxyOf(rnd(&h));
  let g3 = u.gal[k * 4u + 3u];
  if (u32(g3.x) == 2u) { return; } // ellipticals do not rotate
  let nrm = u.gal[k * 4u + 2u].xyz;
  let d = pos[i].xyz - u.gal[k * 4u].xyz;
  let dp = d - nrm * dot(d, nrm);
  let r = max(length(dp), 1e-4);
  let inward = max(-dot(sampleForce(pos[i].xyz), dp / r), 0.0);
  let vc = sqrt(inward * r);
  vel[i] += vec4f(cross(nrm, dp / r) * vc * g3.z, 0.0);
}

// Black-hole mergers: when two central holes fall within a grid cell of each other they
// coalesce (momentum conserved) and the survivor flares as a quasar, stamped with the time.
@compute @workgroup_size(1)
fn holes() {
  for (var a = 0u; a < u.ngal; a++) {
    if (u32(attr[a].x) != 4u) { continue; }
    for (var b = a + 1u; b < u.ngal; b++) {
      if (u32(attr[b].x) != 4u) { continue; }
      let d = pos[b] - pos[a];
      if (dot(d, d) > u.grav.z * u.grav.z) { continue; }
      let ma = u.gal[a * 4u + 1u].w;
      let mb = u.gal[b * 4u + 1u].w;
      vel[a] = (vel[a] * ma + vel[b] * mb) / (ma + mb);
      pos[a] = (pos[a] * ma + pos[b] * mb) / (ma + mb);
      attr[a] = vec2f(4.0, u.time);
      attr[b] = vec2f(5.0, u.time);
    }
  }
}
`;

// One radix-2 FFT line per workgroup, entirely in workgroup memory. AXIS picks the line
// direction through the padded cube; SIGN -1 forward, +1 inverse (normalised in gradient()).
const FFT_WGSL = /* wgsl */ `
override AXIS: u32;
override SIGN: f32;
const NP = ${NP}u;
const LOG2 = ${LOG2NP}u;
@group(0) @binding(0) var<storage, read_write> spec: array<vec2f>;
var<workgroup> sh: array<vec2f, NP>;

fn addr(t: u32, a: u32, b: u32) -> u32 {
  if (AXIS == 0u) { return t + NP * (a + NP * b); }
  if (AXIS == 1u) { return a + NP * (t + NP * b); }
  return a + NP * (b + NP * t);
}

@compute @workgroup_size(NP)
fn fft(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let idx = addr(t, wg.x, wg.y);
  sh[reverseBits(t) >> (32u - LOG2)] = spec[idx];
  workgroupBarrier();
  for (var len = 2u; len <= NP; len <<= 1u) {
    let hl = len >> 1u;
    if (t < NP / 2u) {
      let j = t % hl;
      let i0 = (t / hl) * len + j;
      let i1 = i0 + hl;
      let ang = SIGN * 6.2831853 * f32(j) / f32(len);
      let w = vec2f(cos(ang), sin(ang));
      let x = sh[i1];
      let tw = vec2f(x.x * w.x - x.y * w.y, x.x * w.y + x.y * w.x);
      let y = sh[i0];
      sh[i0] = y + tw;
      sh[i1] = y - tw;
    }
    workgroupBarrier();
  }
  spec[idx] = sh[t];
}
`;

const R_WGSL = /* wgsl */ `
struct R {
  gain: f32, gasGain: f32, exposure: f32, time: f32,
  well: vec4f,  // lens: screen px xy, Einstein radius px, -
  vp: mat4x4f,  // view-projection
  sky: vec4f,   // sky offset in px, splat radius x focal length, glow map height
};
`;

const DRAW_WGSL = U_WGSL + R_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
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

const OFF = vec4f(2.0, 2.0, 2.0, 1.0); // outside clip space: culled

// Inverse-square law. Flux falls as 1/d^2 while a patch of sky shrinks by the same factor, so
// surface brightness, the thing a camera records, is the same at every zoom: true to life.
fn near(w: f32) -> f32 { return min(${HOME * HOME} / max(w * w, 1e-4), 400.0); }

fn starLight(i: u32, at: vec2f) -> vec3f {
  let kind = u32(at.x);
  let h = pcg(i * 9781u + 17u);
  let x = f32(h & 0xffffu) / 65535.0;
  var t: f32;
  var lum: f32;
  if (kind == 0u) {
    // Old population: K and G dwarfs and red giants. Colour from a skewed temperature spread.
    t = 3900.0 + 2600.0 * x * x;
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
  o.p = r.vp * vec4f(pos[i].xyz, 1.0);
  o.c = vec4f(starLight(i, at) * near(o.p.w), 0.0);
  return o;
}

// Half resolution: the same starlight (rgb) plus gas column (a). Upsampled, this is the
// smooth unresolved glow of a galaxy and the column density for dust and nebulae.
// A particle stands for a cloud of stars with a real size, so it is splatted as a Gaussian
// of fixed world radius: a dot from afar, a soft patch up close. The weight is normalised by
// the footprint's area, so a galaxy's surface brightness is the same at every zoom.
struct SO { @builtin(position) p: vec4f, @location(0) c: vec4f, @location(1) q: vec3f };
@vertex fn glow(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> SO {
  let at = attr[i];
  let kind = u32(at.x);
  var o: SO;
  if (kind >= 3u) { o.p = OFF; return o; }
  let corner = vec2f(f32((0x32u >> v) & 1u), f32((0x2cu >> v) & 1u)) * 2.0 - 1.0; // two triangles
  var c = r.vp * vec4f(pos[i].xyz, 1.0);
  if (c.w <= 0.0) { o.p = OFF; return o; }
  let rad = max(1.5 * r.sky.z / c.w * r.sky.w * 0.5, 1.5); // quad radius in glow-map pixels
  let ndc = rad * 2.0 / r.sky.w;
  c = vec4f(c.xy + corner * vec2f(ndc * r.vp[0][0] / r.vp[1][1], ndc) * c.w, c.zw);
  o.p = c;
  o.c = select(vec4f(starLight(i, at), 0.0), vec4f(0.0, 0.0, 0.0, r.gasGain), kind == 2u) * near(c.w);
  o.q = vec3f(corner, 1.0 / (0.6555 * rad * rad)); // 0.6555 R^2 = integral of the kernel below
  return o;
}

@fragment fn fs(i: VO) -> @location(0) vec4f { return i.c; }
// A Gaussian cut to reach exactly zero at the quad's inscribed circle, so no square edges show.
@fragment fn splat(i: SO) -> @location(0) vec4f { return i.c * (max(exp(-4.5 * dot(i.q.xy, i.q.xy)) - 0.011109, 0.0) * i.q.z); }
`;

const FULL_VS = /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
  let y = f32(i & 2u) * 2.0 - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}
`;

// Compositing: dust extinction and H-II emission from the gas map, quasars, a sky of distant
// galaxies, gravitational lensing around the black hole, then a filmic curve.
const TONE_WGSL = FULL_VS + U_WGSL + R_WGSL + /* wgsl */ `
@group(0) @binding(0) var light: texture_2d<f32>;
@group(0) @binding(1) var glowMap: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var<uniform> r: R;
@group(0) @binding(4) var<uniform> u: U;
@group(0) @binding(5) var<storage, read> pos: array<vec4f>;
@group(0) @binding(6) var<storage, read> attr: array<vec2f>;

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

// Nuclei: each central black hole shines faintly by what it swallows (the gas column around it)
// and flares for ~100 Myr after a merger. Seen through a telescope as a point-spread function.
fn nuclei(px: vec2f, size: vec2f) -> vec3f {
  var col = vec3f(0.0);
  for (var k = 0u; k < u.ngal; k++) {
    let at = attr[k];
    if (u32(at.x) != 4u) { continue; }
    let c = r.vp * vec4f(pos[k].xyz, 1.0);
    if (c.w <= 0.0) { continue; }
    let uv = vec2f(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
    let gasHere = textureSampleLevel(glowMap, lin, uv, 0.0).a / 4.0;
    let flare = select(0.0, exp(-(u.time - at.y) / 0.7) * 12.0, at.y > -1.0);
    // Black-hole mass tracks host mass, so a dwarf's nucleus is a faint fraction of a giant's.
    // A nucleus is a point source: its flux falls with distance squared (c.w), unlike surface brightness.
    let l = (0.06 * min(gasHere, 4.0) + flare) * min(u.gal[k * 4u + 1u].w * 2.5, 1.0) * min(${HOME * HOME} / (c.w * c.w), 4.0);
    if (l <= 0.0) { continue; }
    let d = length(px - uv * size) / (1.0 + 0.0015 * size.y);
    col += vec3f(0.85, 0.9, 1.0) * l * (exp(-d * d) + 0.03 / pow(1.0 + d * d * 0.25, 1.5));
  }
  return col;
}

fn filmic(x: vec3f) -> vec3f { // ACES fit (Narkowicz)
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

@fragment fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let size = vec2f(textureDimensions(light));
  var px = fc.xy;
  // Point-mass lens: light from behind bends toward the hole; inside the Einstein ring the
  // image flips, and the shadow at the centre stays black.
  var shadow = 1.0;
  if (r.well.z > 0.0) {
    let d = px - r.well.xy;
    let d2 = max(dot(d, d), 1.0);
    px = px - d * (r.well.z * r.well.z / d2);
    shadow = smoothstep(0.35, 0.55, sqrt(d2) / r.well.z);
  }
  let uv = px / size;
  // Most of a galaxy's light is unresolved stars: mix each crisp point with the half-res
  // glow map (tent-filtered upsample), so sparse counts read as smooth light, not glitter.
  let q = 1.0 / vec2f(textureDimensions(glowMap));
  let gl = (textureSampleLevel(glowMap, lin, uv, 0.0) * 2.0
          + textureSampleLevel(glowMap, lin, uv + vec2f(q.x, 0.0), 0.0) + textureSampleLevel(glowMap, lin, uv - vec2f(q.x, 0.0), 0.0)
          + textureSampleLevel(glowMap, lin, uv + vec2f(0.0, q.y), 0.0) + textureSampleLevel(glowMap, lin, uv - vec2f(0.0, q.y), 0.0)) / 6.0;
  let sharp = textureSampleLevel(light, lin, uv, 0.0).rgb;
  var c = sharp * 0.3 + gl.rgb * (0.7 / 4.0);
  let gasD = gl.a / 4.0;
  // H-II regions: gas lit up by hot young stars. H-alpha red plus H-beta/[O III] blue reads pink.
  // Young light is what is bluer than the old population, so ionisation follows the blue excess.
  let young = max(gl.b - gl.r * 0.75, 0.0) / 4.0;
  c += vec3f(1.0, 0.3, 0.5) * young * min(gasD, 3.0) * 0.6;
  // Dust: reddening extinction along the line of sight, stronger in blue (R_V ~ 3.1).
  // Seen edge-on, a thin gas disc has a huge column: that is the dark lane through the middle.
  // Stars and dust are mixed, not a screen in front: a fifth of the light is a uniform slab, so the
  // thickest lanes still let the near side's stars through.
  let tau = gasD * vec3f(0.5, 0.7, 1.0) * 0.9;
  c *= mix(exp(-tau), (1.0 - exp(-tau)) / max(tau, vec3f(1e-3)), 0.2);
  c += nuclei(px, size);
  c += sky(px + r.sky.xy);
  c *= shadow;
  let mapped = filmic(c * r.exposure);
  return vec4f(pow(mapped, vec3f(1.0 / 2.2)), 1.0);
}
`;

// ---- small vector helpers ----
const v3 = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  norm: (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
};
// Disc normal from an inclination (0 = face-on to +z) and a position angle.
function normalFrom(incl, pa) {
  return [Math.sin(incl) * Math.cos(pa), Math.sin(incl) * Math.sin(pa), Math.cos(incl)];
}

const $ = (id) => document.getElementById(id);
const canvas = $("gl");

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};

async function boot() {
  const adapter = navigator.gpu && (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }));
  if (!adapter) { $("nogpu").hidden = false; return; }
  const want = Math.max(CAP * 16, NP * NP * NP * 8);
  const device = await adapter.requestDevice({
    requiredFeatures: ["rg11b10ufloat-renderable", "timestamp-query"].filter((f) => adapter.features.has(f)),
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, want),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, want),
    },
  });
  const cap = Math.min(CAP, Math.floor(device.limits.maxStorageBufferBindingSize / 16));

  // 4-byte HDR target when the GPU can render to it: half the fill bandwidth of rgba16float.
  const HDR = device.features.has("rg11b10ufloat-renderable") ? "rg11b10ufloat" : "rgba16float";
  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  const simMod = device.createShaderModule({ code: SIM_WGSL });
  const fftMod = device.createShaderModule({ code: FFT_WGSL });
  const drawMod = device.createShaderModule({ code: DRAW_WGSL });
  const toneMod = device.createShaderModule({ code: TONE_WGSL });

  const storage = (i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } });
  const simLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, ...[1, 2, 3, 4, 5, 6, 7, 8].map(storage)],
  });
  const simPL = device.createPipelineLayout({ bindGroupLayouts: [simLayout] });
  const cp = (entryPoint) => device.createComputePipeline({ layout: simPL, compute: { module: simMod, entryPoint } });
  const pipes = Object.fromEntries(
    ["seed", "clear", "deposit", "pack", "greenInit", "keep", "convolve", "gradient", "step", "spin", "holes"].map((e) => [e, cp(e)]),
  );
  const fftLayout = device.createBindGroupLayout({ entries: [storage(0)] });
  const fftPL = device.createPipelineLayout({ bindGroupLayouts: [fftLayout] });
  const fft = (axis, sign) => device.createComputePipeline({
    layout: fftPL, compute: { module: fftMod, entryPoint: "fft", constants: { AXIS: axis, SIGN: sign } },
  });
  const fftFwd = [0, 1, 2].map((a) => fft(a, -1));
  const fftInv = [0, 1, 2].map((a) => fft(a, 1));

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
  const glowPipe = device.createRenderPipeline({
    layout: drawPL,
    vertex: { module: drawMod, entryPoint: "glow" },
    fragment: { module: drawMod, entryPoint: "splat", targets: [{ format: "rgba16float", blend: blend("one", "one") }] },
    primitive: { topology: "triangle-list" },
  });
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

  const U_SIZE = 4 * (16 + 8 * SHOCKS + 16 * GALAXIES);
  const O_GRAV = 8, O_WELL = 12, O_SHOCK = 16, O_GAL = O_SHOCK + 8 * SHOCKS;
  const uBuf = device.createBuffer({ size: U_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const rBuf = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const uData = new ArrayBuffer(U_SIZE);
  const uF = new Float32Array(uData);
  const uU = new Uint32Array(uData);
  const rData = new Float32Array(28);

  const state = {
    n: Math.min(cap, FIXED_N || (coarse ? 1 << 15 : 1 << 16)), // start light, grow only while the GPU has headroom
    scale: coarse ? Math.min(2, devicePixelRatio || 1) : 1, // internal render resolution, dropped only if the minimum count still misses budget
    time: 0,
    aspect: 1,
    // The encounter runs along x: a tall screen looks at it from the side, so it runs top to bottom.
    cam: innerHeight > innerWidth ? { yaw: Math.PI / 2, pitch: 1.05, dist: HOME * 1.15, target: [0, 0, 0] } : { yaw: 0, pitch: 1.05, dist: HOME, target: [0, 0, 0] },
    glide: null, // {from, to, t0}: double-tap recentring
    hole: null, // world position of the finger's black hole while held
    holeAt: 0,
    keys: new Set(),
    shocks: [], // {p, born, s}
  };

  const buf = (size) => device.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
  const pos = buf(cap * 16), vel = buf(cap * 16), attr = buf(cap * 8);
  const massBuf = buf(N * N * N * 4), gasBuf = buf(N * N * N * 4), forceBuf = buf(N * N * N * 16);
  const specBuf = buf(NP * NP * NP * 8), greenBuf = buf(NP * NP * NP * 4);
  const simBG = device.createBindGroup({
    layout: simLayout,
    entries: [uBuf, pos, vel, massBuf, forceBuf, attr, gasBuf, specBuf, greenBuf].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  });
  const fftBG = device.createBindGroup({ layout: fftLayout, entries: [{ binding: 0, resource: { buffer: specBuf } }] });
  const drawBG = device.createBindGroup({
    layout: drawLayout,
    entries: [uBuf, pos, attr, rBuf].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  });
  let hdr, glowTex, toneBG;

  const cells = N * N * N, padded = NP * NP * NP;
  // Mass sits only in the first octant of the padded box, and only that octant of the result is
  // read back, so the first forward axis and the last inverse axis skip the all-zero lines.
  const lines = [[N, N], [NP, N], [NP, NP]];
  function transform(pass, inverse, full = false) {
    pass.setBindGroup(0, fftBG);
    for (const a of inverse ? [2, 1, 0] : [0, 1, 2]) {
      pass.setPipeline((inverse ? fftInv : fftFwd)[a]);
      pass.dispatchWorkgroups(...(full ? [NP, NP] : lines[a]));
    }
    pass.setBindGroup(0, simBG);
  }
  // Mass to grid, FFT, multiply by the kernel spectrum, inverse FFT, differentiate: the force field.
  function solve(pass, count) {
    pass.setPipeline(pipes.clear); pass.dispatchWorkgroups(Math.ceil(cells / WG));
    pass.setPipeline(pipes.deposit); pass.dispatchWorkgroups(Math.ceil(count / WG));
    pass.setPipeline(pipes.pack); pass.dispatchWorkgroups(Math.ceil(padded / WG));
    transform(pass, false);
    pass.setPipeline(pipes.convolve); pass.dispatchWorkgroups(Math.ceil(padded / WG));
    transform(pass, true);
    pass.setPipeline(pipes.gradient); pass.dispatchWorkgroups(Math.ceil(cells / WG));
  }

  // Encounters. Units: 1 length unit = 25 kpc. Separations are compressed to fit the view,
  // but sizes, mass ratios, types, tilts and approach geometry follow the real systems.
  const presets = {
    // The Local Group: Milky Way (barred Sbc) and Andromeda (Sb, ~1.5x heavier, seen ~77° from
    // face-on as we see it) on their approach, with the LMC, Triangulum (M33) and M110 in tow.
    local() {
      const mw = { c: [-0.7, -0.15, 0], v: [0.11, 0.03, 0], m: 1.0, rd: 0.075, spin: 1, type: BARRED, n: normalFrom(0.5, 0.3) };
      const m31 = { c: [0.7, 0.15, 0.05], v: [-0.08, -0.02, 0], m: 1.5, rd: 0.1, spin: -1, type: SPIRAL, phase: 1.2, n: normalFrom(1.35, 2.2) };
      return [
        mw, m31,
        satellite(mw, 0.3, 0.4, { m: 0.1, rd: 0.025, type: DWARF, n: normalFrom(0.9, 1.0) }),              // LMC
        satellite(m31, 0.5, 2.6, { m: 0.12, rd: 0.035, type: SPIRAL, spin: 1, n: normalFrom(0.95, 4.0) }), // M33
        satellite(m31, 0.2, 4.4, { m: 0.03, rd: 0.02, type: ELLIPTICAL, n: [0, 0, 1] }),                    // M110
      ];
    },
    random() {
      const count = Math.random() < 0.35 ? 3 : 2;
      const gs = [];
      for (let k = 0; k < count; k++) {
        const t = count === 2 ? (k ? 1 : -1) : k - 1;
        const speed = 0.08 + Math.random() * 0.08;
        const m = 0.5 + Math.random() * 0.5;
        const roll = Math.random();
        const type = roll < 0.15 ? ELLIPTICAL : roll < 0.45 ? BARRED : SPIRAL;
        gs.push({
          c: [t * 0.7, (Math.random() - 0.5) * 0.5, (Math.random() - 0.5) * 0.3],
          v: [-t * speed, (Math.random() - 0.5) * 0.12, (Math.random() - 0.5) * 0.08],
          m, rd: (0.07 + Math.random() * 0.04) * (0.6 + m * 0.6), spin: Math.random() < 0.5 ? -1 : 1, type,
          n: normalFrom(Math.random() * 1.4, Math.random() * 6.28),
        });
      }
      if (Math.random() < 0.6) {
        gs.push(satellite(gs[0], 0.3 + Math.random() * 0.2, Math.random() * 6.28, { m: 0.06, rd: 0.025, type: DWARF, n: normalFrom(Math.random() * 1.5, Math.random() * 6.28) }));
      }
      return gs;
    },
  };

  // A small companion on a roughly circular orbit in its host's disc plane.
  function satellite(host, dist, angle, g) {
    const e1 = v3.norm(v3.cross(host.n, Math.abs(host.n[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1]));
    const e2 = v3.cross(host.n, e1);
    const dir = v3.add(v3.scale(e1, Math.cos(angle)), v3.scale(e2, Math.sin(angle)));
    const tang = v3.cross(host.n, dir);
    const v = Math.sqrt((G * host.m * 0.45) / dist) * (host.spin || 1);
    return { ...g, spin: g.spin || host.spin, c: v3.add(host.c, v3.scale(dir, dist)), v: v3.add(host.v, v3.scale(tang, v)) };
  }

  function galaxies(which) {
    const gs = presets[which]().slice(0, GALAXIES);
    const total = gs.reduce((t, g) => t + g.m, 0);
    let cum = 0;
    gs.forEach((g, k) => {
      cum += g.m / total;
      const o = O_GAL + k * 16;
      uF.set([...g.c, g.rd], o);
      uF.set([...g.v, g.m / total], o + 4);
      uF.set([...v3.norm(g.n), k === gs.length - 1 ? 1 : cum], o + 8);
      uF.set([g.type, g.phase ?? Math.random() * 6.28, g.spin, 0], o + 12);
    });
    uU[3] = gs.length;
  }

  function writeU(now = performance.now()) {
    uF[0] = DT; uF[1] = state.time; uU[2] = state.n;
    uU[4] = (now * 1000) >>> 0; // per-frame randomness for star formation and supernovae
    uU[5] = N; uU[6] = NP;
    uF.set([G, (0.8 * H) ** 2, H, L / 2], O_GRAV);
    // The black hole grows over a hold, so a tap is a supernova and not a tug.
    const held = state.hole ? Math.min(1, (now - state.holeAt) / 600) : 0;
    uF.set(state.hole ? [...state.hole, 0.12 * held] : [0, 0, 0, 0], O_WELL);
    state.shocks = state.shocks.filter((s) => now - s.born < 2000);
    for (let k = 0; k < SHOCKS; k++) {
      const s = state.shocks[k];
      uF.set(s ? [...s.p, (now - s.born) / 1000, s.s, 0, 0, 0] : [0, 0, 0, 0, 0, 0, 0, 0], O_SHOCK + k * 8);
    }
    return held;
  }

  function reseed(which = "random") {
    writeU();
    galaxies(which);
    uU[2] = cap;
    uU[4] = (Math.random() * 2 ** 32) >>> 0;
    device.queue.writeBuffer(uBuf, 0, uData);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, simBG);
    pass.setPipeline(pipes.seed); pass.dispatchWorkgroups(Math.ceil(cap / WG));
    solve(pass, cap);
    pass.setPipeline(pipes.spin); pass.dispatchWorkgroups(Math.ceil(cap / WG));
    pass.end();
    device.queue.submit([enc.finish()]);
    state.time = 0;
  }

  // The kernel's spectrum depends only on the grid: compute it once.
  {
    writeU();
    device.queue.writeBuffer(uBuf, 0, uData);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, simBG);
    pass.setPipeline(pipes.greenInit); pass.dispatchWorkgroups(Math.ceil(padded / WG));
    transform(pass, false, true); // the kernel fills the whole box
    pass.setPipeline(pipes.keep); pass.dispatchWorkgroups(Math.ceil(padded / WG));
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  // ---- camera: orbits a movable target; z is the "up" of the sky ----
  const camera = {
    frame() {
      const { yaw, pitch, dist, target } = state.cam;
      const off = [dist * Math.cos(pitch) * Math.sin(yaw), -dist * Math.cos(pitch) * Math.cos(yaw), dist * Math.sin(pitch)];
      const f = v3.norm(v3.scale(off, -1));
      const s = v3.norm(v3.cross(f, [0, 0, 1]));
      return { eye: v3.add(target, off), f, s, up: v3.cross(s, f) };
    },
    viewProj() {
      const { eye, f, s, up } = this.frame();
      const near = 0.02, far = 20, t = 1 / Math.tan(FOV / 2), a = state.aspect;
      const V = [s[0], up[0], -f[0], 0, s[1], up[1], -f[1], 0, s[2], up[2], -f[2], 0, -v3.dot(s, eye), -v3.dot(up, eye), v3.dot(f, eye), 1];
      const P = [t / a, 0, 0, 0, 0, t, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
      const M = new Array(16).fill(0);
      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) M[c * 4 + r] += P[k * 4 + r] * V[c * 4 + k];
      return M;
    },
    // Where a screen point meets the plane through the target that faces the camera.
    toWorld(ndc) {
      const { s, up } = this.frame();
      const half = state.cam.dist * Math.tan(FOV / 2);
      return v3.add(state.cam.target, v3.add(v3.scale(s, ndc[0] * half * state.aspect), v3.scale(up, ndc[1] * half)));
    },
    // Slide the view by a screen distance in CSS px: the world under the finger follows it.
    pan(dx, dy) {
      const { s, up } = this.frame();
      const k = (2 * state.cam.dist * Math.tan(FOV / 2)) / canvas.getBoundingClientRect().height;
      state.cam.target = v3.add(state.cam.target, v3.add(v3.scale(s, -dx * k), v3.scale(up, dy * k)));
      state.glide = null;
    },
    glideTo(p) { state.glide = { from: state.cam.target, to: p, t0: performance.now() }; },
    toScreen(p, M) {
      const x = M[0] * p[0] + M[4] * p[1] + M[8] * p[2] + M[12];
      const y = M[1] * p[0] + M[5] * p[1] + M[9] * p[2] + M[13];
      const w = M[3] * p[0] + M[7] * p[1] + M[11] * p[2] + M[15];
      return [((x / w + 1) / 2) * canvas.width, ((1 - y / w) / 2) * canvas.height];
    },
    orbit(dx, dy) {
      state.cam.yaw -= dx;
      state.cam.pitch = Math.max(-1.5, Math.min(1.5, state.cam.pitch + dy));
    },
    zoom(f) { state.cam.dist = Math.max(0.5, Math.min(6, state.cam.dist * f)); },
  };

  // Keep surface brightness constant regardless of particle count and screen size; zoom is
  // handled per particle by the inverse-square law in the shaders.
  function exposure() {
    const px = canvas.width * canvas.height || 1;
    rData[0] = (0.25 * px) / state.n;
    rData[1] = (0.6 * px) / state.n;
    rData[2] = 0.32;
  }

  function resize() {
    // Desktop renders at CSS resolution; phones at device resolution, since CSS pixels there are huge.
    const dpr = state.scale;
    const w = Math.max(1, Math.round(innerWidth * dpr));
    const h = Math.max(1, Math.round(innerHeight * dpr));
    if (canvas.width === w && canvas.height === h && hdr) return;
    canvas.width = w; canvas.height = h;
    state.aspect = w / h;
    hdr?.destroy(); glowTex?.destroy();
    const tex = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    hdr = device.createTexture({ size: [w, h], format: HDR, usage: tex });
    glowTex = device.createTexture({ size: [Math.ceil(w / 2), Math.ceil(h / 2)], format: "rgba16float", usage: tex });
    toneBG = device.createBindGroup({
      layout: tonePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: hdr.createView() }, { binding: 1, resource: glowTex.createView() },
        { binding: 2, resource: sampler }, { binding: 3, resource: { buffer: rBuf } },
        { binding: 4, resource: { buffer: uBuf } }, { binding: 5, resource: { buffer: pos } }, { binding: 6, resource: { buffer: attr } },
      ],
    });
  }

  function shock(p) {
    if (state.shocks.length >= SHOCKS) state.shocks.shift();
    state.shocks.push({ p, born: performance.now(), s: 14 });
  }

  // ---- input: drag orbits; pinch / wheel zooms; two fingers, right-drag, Shift-drag or WASD slide;
  // double-tap flies there; hold still = black hole; quick tap = supernova ----
  const toNdc = (x, y) => {
    const r = canvas.getBoundingClientRect();
    return [((x - r.left) / r.width) * 2 - 1, 1 - ((y - r.top) / r.height) * 2];
  };
  const HOLD_MS = 260;
  const touches = new Map();
  let gesture = null; // {mode: "pending" | "orbit" | "pan" | "hole" | "pinch", ...}
  let lastTap = null;

  function startHole() {
    if (gesture?.mode !== "pending") return;
    gesture.mode = "hole";
    state.hole = camera.toWorld(toNdc(gesture.lastX, gesture.lastY));
    state.holeAt = performance.now();
  }
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    touches.set(e.pointerId, [e.clientX, e.clientY]);
    if (touches.size === 2) {
      clearTimeout(gesture?.timer);
      state.hole = null;
      const [a, b] = [...touches.values()];
      gesture = { mode: "pinch", pinch0: Math.hypot(a[0] - b[0], a[1] - b[1]), dist0: state.cam.dist, mid: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] };
      return;
    }
    gesture = { mode: "pending", t0: performance.now(), x0: e.clientX, y0: e.clientY, lastX: e.clientX, lastY: e.clientY };
    // Right button or Shift slides the view instead of orbiting.
    if (e.button === 2 || e.shiftKey) gesture.mode = "pan";
    else gesture.timer = setTimeout(startHole, HOLD_MS);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!touches.has(e.pointerId) || !gesture) return;
    touches.set(e.pointerId, [e.clientX, e.clientY]);
    if (gesture.mode === "pinch") {
      const [a, b] = [...touches.values()];
      if (!b) return;
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      state.cam.dist = gesture.dist0;
      camera.zoom(gesture.pinch0 / Math.max(1, d));
      if (Math.abs(d - gesture.pinch0) > 30) tutorial.done("zoom");
      // Two fingers moving together slide the view.
      const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      camera.pan(mid[0] - gesture.mid[0], mid[1] - gesture.mid[1]);
      if (Math.hypot(mid[0] - gesture.mid0x, mid[1] - gesture.mid0y) > 40) tutorial.done("move");
      gesture.mid0x ??= gesture.mid[0]; gesture.mid0y ??= gesture.mid[1];
      gesture.mid = mid;
      return;
    }
    const dx = e.clientX - gesture.lastX, dy = e.clientY - gesture.lastY;
    gesture.lastX = e.clientX; gesture.lastY = e.clientY;
    if (gesture.mode === "pending" && Math.hypot(e.clientX - gesture.x0, e.clientY - gesture.y0) > 8) {
      clearTimeout(gesture.timer);
      gesture.mode = "orbit";
    }
    if (gesture.mode === "orbit") {
      const k = 3 / canvas.getBoundingClientRect().height;
      camera.orbit(dx * k, dy * k);
      tutorial.done("drag");
    } else if (gesture.mode === "pan") {
      camera.pan(dx, dy);
      if (Math.hypot(e.clientX - gesture.x0, e.clientY - gesture.y0) > 40) tutorial.done("move");
    } else if (gesture.mode === "hole") {
      state.hole = camera.toWorld(toNdc(e.clientX, e.clientY));
    }
  });
  const release = (e) => {
    touches.delete(e.pointerId);
    if (!gesture) return;
    clearTimeout(gesture.timer);
    const g = gesture;
    if (touches.size === 0) gesture = null;
    if (g.mode === "pending" && e.type === "pointerup") {
      const p = camera.toWorld(toNdc(e.clientX, e.clientY));
      const now = performance.now();
      // A second quick tap nearby flies the view there; a single tap is a supernova.
      if (lastTap && now - lastTap.t < 350 && Math.hypot(e.clientX - lastTap.x, e.clientY - lastTap.y) < 30) {
        camera.glideTo(p);
        tutorial.done("move");
        lastTap = null;
      } else {
        shock(p);
        tutorial.done("tap");
        lastTap = { t: now, x: e.clientX, y: e.clientY };
      }
    }
    if (g.mode === "hole") {
      if (performance.now() - state.holeAt > 500) tutorial.done("hold");
      state.hole = null;
    }
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    camera.zoom(Math.exp(e.deltaY * 0.0015));
    tutorial.done("zoom");
  }, { passive: false });

  const ARROWS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
  const WASD = { a: [-1, 0], d: [1, 0], w: [0, 1], s: [0, -1] };
  addEventListener("keydown", (e) => {
    if (e.target.closest?.("button, input")) return;
    if (ARROWS[e.key]) { state.keys.add(e.key); e.preventDefault(); }
    else if (WASD[e.key.toLowerCase()]) state.keys.add(e.key.toLowerCase());
    else if (e.key === " ") { shock([0, 0, 0]); tutorial.done("tap"); e.preventDefault(); }
    else if (e.key === "Enter" && !state.hole) { state.hole = [0, 0, 0]; state.holeAt = performance.now(); }
    else if (e.key === "+" || e.key === "=") { camera.zoom(0.85); tutorial.done("zoom"); }
    else if (e.key === "-" || e.key === "_") { camera.zoom(1 / 0.85); tutorial.done("zoom"); }
    else if (e.key === "r" || e.key === "R") { reseed(); tutorial.done("again"); }
    else if (e.key === "l" || e.key === "L") reseed("local");
    else if (e.key === "p" || e.key === "P") time.toggle();
    else if (e.key === "," || e.key === "<") time.nudge(-1);
    else if (e.key === "." || e.key === ">") time.nudge(1);
  });
  addEventListener("keyup", (e) => {
    state.keys.delete(e.key);
    state.keys.delete(e.key.toLowerCase());
    if (e.key === "Enter" && state.hole) {
      if (performance.now() - state.holeAt > 500) tutorial.done("hold");
      state.hole = null;
    }
  });

  $("learn").onclick = () => tutorial.start();
  $("again").onclick = () => { reseed(); tutorial.done("again"); };
  addEventListener("resize", resize);
  resize();
  reseed("local"); // first light: the Local Group; the again button deals random encounters

  // ---- frame loop ----
  // GPU frame time with one frame in flight: rAF rate alone hides a queue of unfinished GPU work,
  // which is what feels like lag.
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
      let kx = 0, ky = 0, px = 0, py = 0;
      for (const k of state.keys) {
        if (ARROWS[k]) { kx += ARROWS[k][0]; ky += ARROWS[k][1]; }
        if (WASD[k]) { px += WASD[k][0]; py += WASD[k][1]; }
      }
      camera.orbit(kx * dt * 1.2, ky * dt * 1.2);
      if (kx || ky) tutorial.done("drag");
      // Fly forward/back along the view (W/S) and sideways (A/D), at a speed that suits the zoom.
      if (px || py) {
        const { f, s } = camera.frame();
        const v = state.cam.dist * 0.6 * dt;
        state.cam.target = v3.add(state.cam.target, v3.add(v3.scale(s, px * v), v3.scale(f, py * v)));
        state.glide = null;
        tutorial.done("move");
      }
    }
    if (state.glide) {
      // Ease the view onto a double-tapped point: the one deliberate motion in the interface.
      const g = state.glide;
      const t = Math.min(1, (now - g.t0) / 700);
      const e = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
      state.cam.target = v3.add(g.from, v3.scale(v3.add(g.to, v3.scale(g.from, -1)), e));
      if (t === 1) state.glide = null;
    }

    state.time += steps * DT;
    const held = writeU(now);
    device.queue.writeBuffer(uBuf, 0, uData);

    exposure();
    const M = camera.viewProj();
    rData[3] = state.time;
    if (state.hole) {
      const [sx, sy] = camera.toScreen(state.hole, M);
      rData.set([sx, sy, Math.sqrt(held) * 0.045 * canvas.height * (HOME / state.cam.dist), 0], 4);
    } else {
      rData.set([0, 0, 0, 0], 4);
    }
    rData.set(M, 8);
    // The background sky turns with the camera.
    // Splat radius ~250 pc at the default count, shrinking with the particle spacing.
    const splat = 0.012 * Math.cbrt((1 << 18) / state.n) / Math.tan(FOV / 2);
    rData.set([state.cam.yaw * canvas.height * 0.6, -state.cam.pitch * canvas.height * 0.6, splat, Math.ceil(canvas.height / 2)], 24);
    device.queue.writeBuffer(rBuf, 0, rData);

    const enc = device.createCommandEncoder();
    if (steps) {
      const pass = enc.beginComputePass(timed ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0 } } : {});
      pass.setBindGroup(0, simBG);
      const groups = Math.ceil(state.n / WG);
      for (let s = 0; s < steps; s++) {
        solve(pass, state.n);
        pass.setPipeline(pipes.step); pass.dispatchWorkgroups(groups);
        pass.setPipeline(pipes.holes); pass.dispatchWorkgroups(1);
      }
      pass.end();
    }

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
    gp.draw(6, state.n);
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
      } else if (gpuMs < BUDGET * 0.75 && n < cap) n = Math.min(cap, Math.round(n * 1.25));
      state.n = n;
      $("n").textContent = `${fmt(state.n)} · ${gpuMs.toFixed(1)}ms · ${age(state.time)}`;
    }
  }
  requestAnimationFrame(frame);

  if (!store.get("pp.tut6")) tutorial.start();
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

// ---- wordless tutorial: one word per step, learn by doing ----
const tutorial = (() => {
  const el = $("tut"), word = $("word"), learn = $("learn");
  const steps = [
    { id: "drag", word: "Drag" },
    { id: "zoom", word: coarse ? "Pinch" : "Scroll" },
    { id: "move", word: coarse ? "Slide" : "WASD" },
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
    store.set("pp.tut6", "1");
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
