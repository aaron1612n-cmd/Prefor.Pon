# Prefor.Pon

A GPU-resident particle field built on WebGPU. Up to 4M particles (1M on phones), one compute dispatch per physics substep, one draw call.

## How it stays fast

- **Nothing round-trips to the CPU.** Positions and velocities live in two GPU storage buffers (struct-of-arrays). The compute shader writes them; the vertex shader reads the same buffers by `vertex_index`. No vertex buffers, no readbacks.
- **Fixed timestep.** Physics runs at 60 Hz with at most 2 substeps per frame, batched in one compute pass. Backlog is dropped instead of spiralling.
- **Small footprint.** Velocity is packed as two f16s (12 bytes per particle in total), and the HDR target is `rg11b10ufloat` (4 bytes per pixel) where the GPU supports rendering to it. It renders at CSS-pixel resolution because points are 1px anyway.
- **No pile-ups.** The flow field is divergence-free and the edges wrap. Clamped walls used to stack particles on the same edge pixels, and that additive-blend contention slowed long runs down.
- **Adaptive count, measured on the GPU.** Frame cost comes from timestamp queries (or submit-to-done time where those are not available). Only one frame is ever in flight, so GPU work never queues up into input lag. The count starts at 64K, grows only below half of a 7 ms budget, backs off fast above it, and drops the internal resolution if even 16K misses the budget. The count and GPU ms are shown bottom-left. Buffers are allocated once at capacity, so changing the count never resets the scene.
- **One draw call.** Particles render as `point-list` with additive blending, so density reads as brightness. Trails come from fading the previous frame with a blend constant, with no ping-pong buffer, and a fullscreen tonemap pass maps the result to the screen.
- **Constant brightness.** Per-particle gain scales with pixels / particle count, so 16K and 8M look equally exposed.

## Controls

| Input | Action |
| --- | --- |
| Hold (touch or mouse) | A vortex grows under your finger and pulls the field in |
| Tap / Space | Sends out a shockwave ring |
| Arrow keys | Move the vortex with the keyboard |
| R | Reset |

## Run

```sh
npm install
npm run dev      # local, via wrangler
npm run deploy   # Cloudflare Worker (needs CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID)
```

Or open `public/` with any static server. Requires a WebGPU browser (Chrome/Edge 113+, Safari 26+, Firefox 141+ on Windows).

## Next

- Spatial hash (GPU counting sort) for particle–particle interaction: SPH fluid, collisions or flocking.
- WebGL2 transform-feedback fallback.
