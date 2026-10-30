# Prefor.Pon

A GPU-resident particle field built on WebGPU. Up to 8M particles, one compute dispatch per physics substep, one draw call.

## How it stays fast

- **Nothing round-trips to the CPU.** Positions and velocities live in two GPU storage buffers (struct-of-arrays). The compute shader writes them; the vertex shader reads the same buffers by `vertex_index`. No vertex buffers, no readbacks.
- **Fixed timestep.** Physics runs at 120 Hz with up to 4 substeps per frame, all batched in a single compute pass. Backlog is dropped instead of spiralling.
- **One draw call.** Particles render as `point-list` into an `rgba16float` target with additive blending, so density reads as brightness; a fullscreen tonemap pass maps it to the screen.
- **Constant brightness.** Per-particle gain scales with pixels / particle count, so 16K and 8M look equally exposed.

## Controls

| Input | Action |
| --- | --- |
| Drag / touch | Pull particles toward the pointer |
| Flip button / Space | Toggle attract ↔ repel |
| + / − | Double / halve particle count |
| R | Reset |
| Arrow keys | Move the attractor with the keyboard |

## Run

```sh
npm install
npm run dev      # local, via wrangler
npm run deploy   # Cloudflare Worker (needs CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID)
npm run deploy:inline   # same Worker with public/ inlined; use if the assets upload returns 401
```

Or open `public/` with any static server. Requires a WebGPU browser (Chrome/Edge 113+, Safari 26+, Firefox 141+ on Windows).

## Next

- Spatial hash (GPU counting sort) for particle–particle interaction: SPH fluid, collisions or flocking.
- WebGL2 transform-feedback fallback.
