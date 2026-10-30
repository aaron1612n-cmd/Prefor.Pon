// Bundles public/ into a single Worker script. Used where the
// assets-upload endpoint is unreachable (its upload JWT gets replaced
// by a proxy that injects the API token), e.g. cloud sessions.
import { readdirSync, readFileSync, mkdirSync, writeFileSync } from "node:fs";

const types = { html: "text/html; charset=utf-8", js: "text/javascript; charset=utf-8", css: "text/css; charset=utf-8" };
const files = {};
for (const name of readdirSync("public")) {
  files["/" + name] = { body: readFileSync(`public/${name}`, "utf8"), type: types[name.split(".").pop()] ?? "application/octet-stream" };
}

const worker = `const FILES = ${JSON.stringify(files)};
export default {
  fetch(req) {
    const path = new URL(req.url).pathname;
    const f = FILES[path] ?? FILES["/index.html"];
    return new Response(f.body, { headers: { "content-type": f.type, "cache-control": "public, max-age=300" } });
  },
};
`;
mkdirSync("dist", { recursive: true });
writeFileSync("dist/worker.js", worker);
console.log(`dist/worker.js: ${Object.keys(files).length} files inlined`);
