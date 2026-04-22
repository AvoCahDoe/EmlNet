/**
 * Copy site data + sweep + assets into public/ for Vite dev/build.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const resultsUiRoot = path.resolve(__dirname, "..");
const emlnetRoot = path.resolve(resultsUiRoot, "..");
const siteDir = path.join(emlnetRoot, "site");
const publicDir = path.join(resultsUiRoot, "public");

function cpDir(name) {
  const src = path.join(siteDir, name);
  const dest = path.join(publicDir, name);
  if (!fs.existsSync(src)) {
    console.warn("[sync-assets] skip (missing):", src);
    return;
  }
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.cpSync(src, dest, { recursive: true });
  console.log("[sync-assets]", name, "→", dest);
}

fs.mkdirSync(publicDir, { recursive: true });
cpDir("data");
cpDir("sweep");
cpDir("assets");
console.log("[sync-assets] done.");
