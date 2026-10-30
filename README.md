# Prefor.Pon

A 3D WebGPU galaxy simulation: colliding galaxies under self-gravity, running entirely on the GPU, rendered to look like a telescope image. Orbit the camera to see discs face-on or edge-on, with the dust lane through the middle.

## Physics

- **Four particle kinds:** old stars, young stars, gas and dark matter. Each galaxy is an exponential disc with a bulge, inside a Hernquist dark-matter halo that holds about 3.4x the visible mass.
- **Galaxy types.** There are spirals, barred spirals (a straight bar inside about 1.6 scale lengths), ellipticals (hot spheroids of old stars with no gas or rotation) and dwarf irregulars (gas-rich, puffy, with no bulge). Each big galaxy carries a dozen globular clusters on random halo orbits.
- **Presets.** The first view is the Local Group: the Milky Way (barred) and Andromeda (about 1.5x heavier) on their head-on approach, with the LMC, Triangulum (M33) and M110. Positions are compressed to fit the screen, while sizes, mass ratios and types follow the real galaxies. ↻ deals random encounters, often with a satellite dwarf, and L brings the Local Group back.
- **Disc rotation from the measured field.** After seeding, one gravity solve runs and every disc particle gets the circular speed that the grid force actually gives at its position. A mismatched start makes cold gas ring in concentric waves.
- **Spiral density waves seeded:** a two-armed logarithmic spiral with a pitch of about 15°, strongest in gas and young stars.
- **Star formation and feedback.** Gas well above the typical disc density (spiral arms, bars, collision shocks) turns into young stars, faster the denser it is. After about 70 Myr, a third of the young stars go supernova and return to gas with a kick, and the rest join the old population. Mergers trigger starbursts and use up the gas.
- **Central black holes.** Every galaxy has one, about 1% of its mass, which is heavier than real (about 0.1%) so the coarse grid can still sink it. When two holes come within a grid cell of each other they merge, conserving momentum, and the survivor flares as a quasar for about 100 Myr. Between mergers, each nucleus glows in proportion to the gas around it.
- **Units:** 1 length unit is 25 kpc and 1 sim second is 150 Myr. The time control shows Myr per second, and the readout shows the elapsed time.

## Rendering

- **Star colours** are blackbody colours. Old stars range from 3400 to 6200 K. Young stars start at 9000 to 30000 K, then cool and fade over their lifetime.
- **Two light layers.** Crisp full-resolution points sit over a quarter-resolution glow map, which is the unresolved starlight, so sparse counts read as smooth light.
- **Dust and nebulae.** The gas column density dims the light, more in blue than red (R_V of about 3.1). Pink H-II emission appears where gas overlaps young blue light.
- **Lensing.** The finger black hole lenses everything behind it, with an Einstein radius proportional to the square root of its mass and a black shadow at the centre. Behind everything is a faint sky of distant galaxies.
- **Tone mapping** is the ACES filmic curve, so cores roll off instead of blowing out to flat white. Up to 2M particles (512K on phones). Add `?n=262144` to the URL to pin the count, one compute dispatch per physics substep, one draw call.

## How it stays fast

- **Particle-mesh gravity with an FFT Poisson solve.** Mass goes onto a 64³ grid (32³ on phones) with atomic adds, inside a zero-padded 128³ box so the convolution is isolated. Radix-2 FFTs run one line per workgroup in workgroup memory; the density spectrum is multiplied by the precomputed spectrum of a softened 1/r kernel, transformed back, and differentiated into a force grid that every star samples trilinearly. Lines that are all zero in the padded box are skipped. Cost is independent of the particle count.
- **Thick and thin discs.** Cold gas sits in a layer about 1/25 of the disc scale length thick, young stars a little thicker and old stars about 1/7, as in the Milky Way. Every galaxy has its own tilt; Andromeda is seen nearly edge-on, as from Earth.
- **Symplectic integration.** Kick-drift with f32 velocity keeps orbits stable. In f16, the small per-step kicks would round away.
- **Prefix-safe seeding.** Each star picks its galaxy by hash rather than by index, so the adaptive count always simulates a fair sample with the same total mass.

- **Nothing round-trips to the CPU.** Positions and velocities live in two GPU storage buffers (struct-of-arrays). The compute shader writes them; the vertex shader reads the same buffers by `vertex_index`. No vertex buffers, no readbacks.
- **Fixed timestep.** Physics runs at 60 Hz with at most 2 substeps per frame, batched in one compute pass. Backlog is dropped instead of spiralling.
- **Small footprint.** Velocity is packed as two f16s (12 bytes per particle in total), and the HDR target is `rg11b10ufloat` (4 bytes per pixel) where the GPU supports rendering to it. It renders at CSS-pixel resolution because points are 1px anyway.
- **Adaptive count, measured on the GPU.** Frame cost comes from timestamp queries (or submit-to-done time where those are not available). Only one frame is ever in flight, so GPU work never queues up into input lag. The count starts at 64K, grows only below half of a 7 ms budget, backs off fast above it, and drops the internal resolution if even 16K misses the budget. The count and GPU ms are shown bottom-left. Buffers are allocated once at capacity, so changing the count never resets the scene.
- **One draw call.** Particles render as `point-list` with additive blending, so density reads as brightness. Trails come from fading the previous frame with a blend constant, with no ping-pong buffer, and a fullscreen tonemap pass maps the result to the screen.
- **Constant brightness.** Per-particle gain scales with pixels / particle count, so 16K and 8M look equally exposed.

## Rendering at any zoom

Every particle's light follows the inverse-square law from its own distance to the camera, while its footprint grows as it gets closer, so a galaxy's surface brightness is the same zoomed in or out, as it is through a real telescope. The smooth light is drawn as soft round patches of a fixed size in space (about 250 pc), so close-ups read as unresolved starlight, not scattered dots. Nuclei are point sources, with flux that falls with distance.

## Controls

| Input | Action |
| --- | --- |
| Drag / arrow keys | Orbit the camera |
| Pinch / scroll / + − | Zoom |
| Two-finger slide / right-drag / Shift-drag | Slide the view |
| W A S D | Fly forward, back and sideways |
| Double-tap | Fly to that spot |
| Hold still (touch or mouse) / Enter | A black hole grows under your finger |
| Tap / Space | Supernova: a shock front blows stars outward |
| ↻ / R | A new random encounter of two or three galaxies |
| L | The Local Group preset |
| Pause glyph / P | Pause and resume time |
| Slider / , . | Time rate, from 1/8x to 4x (starts at 1/2x) |

## Run

```sh
npm install
npm run dev      # local, via wrangler
npm run deploy   # Cloudflare Worker (needs CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID)
```

Or open `public/` with any static server. Requires a WebGPU browser (Chrome/Edge 113+, Safari 26+, Firefox 141+ on Windows).

## Next

- Tree or P³M short-range correction, so galaxy cores get sharper than one grid cell.
- WebGL2 transform-feedback fallback.
