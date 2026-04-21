# EMLnet

PyTorch implementation of a **stabilized EML (exp − log) layer** with a **custom `autograd.Function`**: clamped forward ($\exp(\mathrm{clamp}(x)) - \ln(|y|+\epsilon)$) and **clamped outgoing gradients** in the backward pass, plus training scripts that compare **EML stacks** to **ReLU MLPs** on 2D classification toy datasets.

**Math in Markdown:** `$…$` inline, `$$…$$` display (GitHub / VS Code preview with math enabled).

---

## Requirements

- Python 3.10+
- See [`requirements.txt`](requirements.txt) (`torch`, `numpy`, `matplotlib`, `scikit-learn`).

---

## Install

From the repository root, enter this directory, then install:

```bash
cd EMLnet
pip install -r requirements.txt
```

---

## Usage

All commands assume the working directory is **`EMLnet/`** (this folder).

### Moons baseline (2 EML blocks vs 2-layer MLP)

```bash
python -m emlnet_pkg.train_moons
```

| Flag | Default | Notes |
|------|---------|--------|
| `--epochs` | 15000 | Use `--epochs 500` for a quick run |
| `--hidden` | 32 | Wider hidden = more parameters |
| `--lr` | 1e-4 | Lower if you see NaNs |
| `--no-plots` | off | Skip figure export |

**Outputs:** [`plots/`](plots/) — `moons_loss_linear.png`, `moons_loss_logy.png`, `moons_accuracy.png`, `moons_grad_norm.png`, `moons_decision_eml_mlp.png`, `moons_summary_grid.png` (300 dpi).

### Multi-scenario benchmarks (deeper nets, harder 2D sets)

```bash
python -m emlnet_pkg.train_benchmarks --scenarios all --epochs 6000 --depth 3 --hidden 56
python -m emlnet_pkg.train_benchmarks --scenarios xor_blobs,spiral,circles --epochs 4000
```

- **Plots:** `plots/benchmarks/<scenario>/` (loss, log-loss, accuracy, grad norm, decision regions, summary grid, **val loss gap** = MLP val BCE − EML val BCE).
- **Tables:** `plots/benchmarks/summary.json`, `plots/benchmarks/summary.csv`.

Scenario keys are defined in [`emlnet_pkg/datasets_scenarios.py`](emlnet_pkg/datasets_scenarios.py). Use `--scenarios all` to run every registered scenario.

---

## Project layout

```text
EMLnet/
  README.md
  requirements.txt
  emlnet_pkg/
    eml_function.py      # EMLFunction + EMLFunctionConfig
    layers.py            # EMLLayer
    models.py            # Two-layer + deep EML / MLP classifiers
    init_utils.py        # Small-variance weight init
    training_core.py     # Shared training loop
    plotting.py          # Resolves EMLnet/plots/
    datasets_scenarios.py
    train_moons.py
    train_benchmarks.py
  plots/                 # generated (gitignored recommended)
```

---

## Mathematical background

**Operator (unstable as-is):** $\mathrm{eml}(x,y) = \exp(x) - \ln(y)$. Problems: $\exp(x)$ overflows; $\ln(y)$ is singular for $y \le 0$ and stiff as $y \to 0^+$.

**Stabilized forward** (used in code):

$$
f = \exp(x_c) - \ln(y_a), \quad x_c = \mathrm{clamp}(x),\quad y_a = |y| + \epsilon.
$$

**Gradients** (before extra stabilization):

$$
\frac{\partial f}{\partial x_c} = \exp(x_c), \qquad
\frac{\partial f}{\partial y_a} = -\frac{1}{y_a}.
$$

Chain rule through `clamp` uses the usual gate on $x$; through $|y|$ use $\operatorname{sign}(y)$. This repo **clips** the final `grad_x` and `grad_y` returned from [`EMLFunction.backward`](emlnet_pkg/eml_function.py) (configurable via `EMLFunctionConfig`) to limit explosion.

---

## Training defaults (both trainers)

- **Optimizer:** `AdamW`
- **Loss:** `BCEWithLogitsLoss` (binary 2D tasks)
- **After `backward`:** `clip_grad_norm_(..., max_norm=1.0)` (default)
- **Init:** tight Gaussian on linear weights (`--init-std` in benchmarks; `train_moons` uses `0.01`)

EML stacks use **more parameters** than a width-matched MLP because each `EMLLayer` uses two linear maps into $(x,y)$; scripts log both counts.

---

## Troubleshooting

1. **NaNs in forward:** lower `--lr`, increase $\epsilon$, tighten $x$ clamp range in `EMLFunctionConfig`.
2. **NaNs in backward:** tighten gradient clip constants in `EMLFunctionConfig`, lower `--lr`, keep `clip_grad_norm_`.
3. **Sanity:** both entrypoints log `torch.isfinite` on the first minibatch of logits.

---

## License

Specify a license in a root `LICENSE` file when you publish the repository (this folder does not ship one by default).
