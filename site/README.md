# EMLnet static results site

Self-contained **HTML + JS + assets** (Tailwind and Chart.js via CDN in `index.html`) for benchmark visuals and tables.

## Preview locally

```bash
cd site
python -m http.server 8080
```

Open `http://localhost:8080` (serving `index.html` is required so `fetch("data/results.json")` works).

## Refresh after new training runs

From the `EMLnet/` directory (parent of `site/`):

```bash
cp plots/benchmarks/summary.json site/data/results.json
for d in plots/benchmarks/*/; do
  bn=$(basename "$d")
  cp "${d}d3_h56_decision.png" "site/assets/${bn}_decision.png"
  cp "${d}d3_h56_loss_linear.png" "site/assets/${bn}_loss.png"
done
```

If you change `--depth` or `--hidden`, filenames from `train_benchmarks` will change (`d*_h*_…`); update the copy commands or adjust `index.html` / asset names accordingly.

## Deploy

Upload the entire `site/` folder to any static host (GitHub Pages, Netlify, Cloudflare Pages, S3 static website, etc.).
