# EMLnet — conserving gradient descent through the EML operator

**Math notation:** inline formulas use `$…$`; display formulas use `$$…$$` on their own lines. This matches **GitHub**, **GitLab**, and **VS Code / Cursor** Markdown preview (with math enabled).

This folder documents a **pivot** from “safe forward only” to **training the real operator** with backpropagation. The goal is not to bypass $\mathrm{eml}(x,y)=\exp(x)-\ln(y)$, but to **force the network to learn through it** while keeping optimization numerically sane. That is the hard part of rigorous AI system optimization: you keep gradient descent, but you **engineer** stability instead of pretending the operator is benign.

---

## The candid reality of the math

### Forward pass pathologies

- $\exp(x)$ grows **extremely** quickly; in float32 it overflows once $x$ is modestly large.
- $\ln(y)$ is **undefined** for $y\le 0$, and **blows up** as $y\to 0^{+}$.

If you wire the raw operator into PyTorch and call `.backward()` without care, **NaNs on the first epoch** are a normal outcome, not an accident.

### Backward pass (exact derivatives)

Treat $y$ as the **pre-activation** argument to $\ln$ (after your forward stabilizations, you may use $|y|+\epsilon$; the chain rule still flows through whatever variable feeds $\ln$).

- **With respect to $x$:**

$$
\frac{\partial}{\partial x}\bigl(\exp(x)-\ln(y)\bigr)=\exp(x).
$$

If $x$ is allowed to drift positive without control, **gradients explode** because $\exp(x)$ is both the derivative and the sensitivity of the forward branch.

- **With respect to $y$ (for $-\ln(y)$):**

$$
\frac{\partial}{\partial y}\bigl(-\ln(y)\bigr)=-\frac{1}{y}.
$$

As $|y|\to 0$, **gradients explode**.

So you fight a **two-front war**: exploding $\exp(x)$ sensitivity in the $x$-branch, and exploding $1/y$ in the $y$-branch.

---

## Design principle: custom `autograd` (not “trust default AD”)

Standard automatic differentiation faithfully implements the chain rule. For EML, that faithfulness is **exactly** what shatters training.

You **subclass** `torch.autograd.Function` and implement:

1. A **stabilized forward** (clamps / floors).
2. A **conservative backward** that computes the *theoretical* partials, then **clamps** the tensors you return to the previous op **before** they multiply into earlier layers.

This is “conserving gradient descent” in the engineering sense: you still optimize with gradients, but you **bound** how violent those gradients may be.

### 1) Forward pass (recommended baseline)

Let `xc = clamp(x, x_min, x_max)` (for example `[-10, 10]`).

Let `ya = abs(y) + eps` with `eps ≈ 1e-6` (float32-friendly).

Return:

$$
f=\exp(x_c)-\ln(y_a).
$$

**Important implementation detail:** if you clamp $x$ in the forward, the derivative w.r.t. the *original* $x$ should respect the clamp via a **hardtanh-style mask** on the backward of the clamp (PyTorch’s `clamp` already does this if you use `torch.clamp` in the forward and differentiate through it). Your **custom** clamps may instead be applied **only** in backward on the $\exp$ branch; pick one story and stick to it:

- **Story A (simplest):** use `torch.clamp` / `torch.log` / `torch.exp` in `forward` and implement **extra** gradient clamping in `backward` on top of autograd’s clamp behavior.
- **Story B (fully manual):** implement forward with raw ops and implement **all** backward pieces yourself (more control, more foot-guns).

This README assumes **Story A** unless you explicitly need Story B.

### 2) Backward pass (the “magic”: clamp outgoing grads)

Let `g` be the incoming gradient `grad_output` (same shape as $f$).

Exact partials of $f=\exp(x_c)-\ln(y_a)$ w.r.t. **intermediate** branches (ignoring clamp masks for a moment):

- $\partial f/\partial x_c = \exp(x_c)$
- $\partial f/\partial y_a = -1/y_a$

With the chain rule:

- `grad_x = g * exp(x_c)` (then multiply by clamp mask to raw `x` if needed)
- `grad_y = g * (-1.0 / ya)` (and map through `abs`/sign if you defined $y_a=|y|+\epsilon$)

**Clamp before returning** (element-wise), for example:

- `grad_x = clamp(grad_x, -gx_max, gx_max)`
- `grad_y = clamp(grad_y, -gy_max, gy_max)`

Choose `gx_max`, `gy_max` by experiment (start conservative, like `1.0` or smaller, then loosen only if training is too slow).

This does **not** replace the need for a sane forward; it is a **second line of defense** against chain-rule explosions.

---

## 2) Ultra-conservative initialization

Xavier / Kaiming presets are tuned for affine maps with activations like ReLU/tanh. They are **not** tuned for an inner $\exp$ channel.

**Initialize weights very small**, for example:

- Gaussian with `std ≤ 0.01` (often `0.005`–`0.01` is a reasonable first search band).

Intent: start in the **flat** part of the exponential curve and let optimization **earn** larger activations, instead of initializing into a regime where one update spikes $\exp(x)$.

Biases often benefit from **zero** initialization in this setting (then tune if needed).

---

## 3) Architecture and optimizer (keep it shallow)

### Model A — 2-layer EML network (the thesis object)

Define an `EMLLayer` that:

1. Applies a linear map (or two linear maps) to produce paired tensors **$X$** and **$Y$** matching shapes for the EML binary operator.
2. Applies your **custom** `EMLFunction.apply(X, Y)` (the stabilized forward + clamped backward).

Stack **exactly two** such layers, then a small classifier head (for example a final `Linear` to logits).

### Model B — baseline 2-layer MLP

For a clean comparison:

- `Linear → ReLU → Linear → logits`

Match widths where possible so the comparison is about **operator geometry**, not raw width alone (exact parameter matching is optional but should be documented if you publish figures).

### Optimizer and safety nets

- **Optimizer:** `AdamW` (momentum helps in awkward loss landscapes; decoupled weight decay helps keep weights from drifting into insane $\exp$ regimes).
- **Gradient clipping:** always call `torch.nn.utils.clip_grad_norm_(parameters, max_norm=...)` after `loss.backward()` as an **absolute safety net**, even with custom backward clamping.

Suggested starting hyperparameters (tune from here):

- `lr = 1e-4` to `3e-4`
- `weight_decay = 1e-4` (tune)
- `max_norm = 1.0` for global grad norm clipping

---

## Master’s-level experiment (what to plot)

**Dataset:** `sklearn.datasets.make_moons` (2D classification), fixed `random_state` for reproducibility.

**Train both models** with the same batching, epochs, optimizer family (`AdamW`), and clipping policy.

**Deliverable figure:** training loss vs epoch (or step) for **both** models on **one** axes (or side-by-side panels with shared y-scale):

- EML-2L (custom autograd)
- MLP-2L (ReLU baseline)

This makes the “we kept gradient descent” story legible: you are comparing **learning dynamics**, not just final accuracy.

---

## Suggested repository layout (inside `EMLnet/`)

```text
EMLnet/
  README.md
  requirements.txt
  emlnet_pkg/
    __init__.py
    eml_function.py         # EMLFunction + EMLFunctionConfig
    init_utils.py           # tight Gaussian init
    layers.py               # EMLLayer
    models.py               # Two/deep EML + MLP classifiers
    datasets_scenarios.py   # moons/circles/spiral/checkerboard/blobs
    training_core.py        # shared train loop + RunHistory
    plotting.py             # EMLnet/plots path helpers
    train_moons.py          # classic moons experiment
    train_benchmarks.py     # multi-scenario + deeper stacks
  plots/                    # created by train_moons (dpi=300)
```

If you implement code, keep imports runnable with:

```bash
cd EMLnet
python -m emlnet_pkg.train_moons
```

---

## Minimal `Function` example (reference implementation)

Forward uses $\exp(\mathrm{clamp}(x))-\ln(|y|+\epsilon)$. Backward applies the usual chain rule for `clamp` and `abs(y)+eps`, then **clamps** the outgoing `grad_x` and `grad_y` (tune `GX`, `GY`).

The shipping implementation lives in [`emlnet_pkg/eml_function.py`](emlnet_pkg/eml_function.py): `EMLFunction.apply(x, y, EMLFunctionConfig(...))` passes a frozen **config** (clamps, `eps`, gradient clip bounds). Backward restores the raw `y` tensor from `ctx.save_for_backward` so the subgradient of $|y|$ (via $\operatorname{sign}(y)$) is applied correctly in the chain rule.

You can still add global `clip_grad_norm_` in the training loop as a second safety net (see `train_moons.py`).

---

## Debugging checklist (read this before you tune “learning rate”)

1. **Verify no NaNs in forward** on the first batch (print `torch.isfinite` flags).
2. **Log** `x_c.max()`, `ya.min()` occasionally.
3. If NaNs appear in **backward first**, tighten `gx_max`, `gy_max`, and/or lower `lr`.
4. If NaNs appear in **forward first**, tighten `x_max`, raise `eps`, or shrink initialization further.
5. Only loosen constraints after you observe stable loss for many epochs.

---

## Relationship to `EMLquant/`

[`../EMLquant/`](../EMLquant/README.md) contains a separate mini-project (stabilized forward, log quantization toy benchmarks). **`EMLnet/`** is the place for the **custom backward / gradient conservation** narrative and the **2-layer moons loss-curve** study.

---

## Implemented (code + plots)

Run from the **`EMLnet/`** directory:

```bash
cd EMLnet
pip install -r requirements.txt
python -m emlnet_pkg.train_moons
```

**Defaults:** `--epochs 15000`, `--hidden 32`, `--lr 1e-4`, `--weight-decay 1e-4`, `--clip-max-norm 1.0`, `--init-std 0.01`, `--n-samples 800`. Use `--no-plots` for a fast smoke run. Shorter run: `python -m emlnet_pkg.train_moons --epochs 500`.

**Modules:** [`emlnet_pkg/eml_function.py`](emlnet_pkg/eml_function.py) (`EMLFunction` + `EMLFunctionConfig`), [`emlnet_pkg/layers.py`](emlnet_pkg/layers.py), [`emlnet_pkg/models.py`](emlnet_pkg/models.py), [`emlnet_pkg/init_utils.py`](emlnet_pkg/init_utils.py), [`emlnet_pkg/train_moons.py`](emlnet_pkg/train_moons.py), [`emlnet_pkg/plotting.py`](emlnet_pkg/plotting.py).

**Figures** (written to [`plots/`](plots/), `dpi=300`, tight bbox):

| File | Contents |
|------|-----------|
| `moons_loss_linear.png` | Train/val BCE for EML and MLP vs epoch |
| `moons_loss_logy.png` | Same with log-scaled loss |
| `moons_accuracy.png` | Train/val accuracy (sigmoid ≥ 0.5) |
| `moons_grad_norm.png` | Post–`clip_grad_norm_` global L2 norm |
| `moons_decision_eml_mlp.png` | Side-by-side decision regions on full moons |
| `moons_summary_grid.png` | 2×2 panel of loss / log-loss / acc / grad norm |

**Note:** the two-branch EML stack has **more parameters** than the 2-linear MLP for the same `--hidden`; the log line reports both counts. Tune `--hidden` or add a wider MLP if you want a stricter capacity match.

### Harder scenarios + deeper stacks (`train_benchmarks`)

Runs **EML vs MLP** on several **non-trivial 2D** sets (noisy moons, concentric circles, two spirals, noisy checkerboard, messy blobs) with **configurable depth** (`--depth`) and width (`--hidden`). Each scenario writes a folder of plots under `plots/benchmarks/<scenario>/` plus `plots/benchmarks/summary.json` and `summary.csv` (final val loss/acc and **MLP−EML val loss gap**).

```bash
cd EMLnet
python -m emlnet_pkg.train_benchmarks --epochs 6000 --depth 3 --hidden 56
python -m emlnet_pkg.train_benchmarks --scenarios spiral,checkerboard,circles --epochs 4000 --depth 4 --hidden 64
```

Scenario keys (see [`emlnet_pkg/datasets_scenarios.py`](emlnet_pkg/datasets_scenarios.py)): `moons_clean`, `moons_hard`, `moons_extra_noise`, `circles`, `circles_tight`, `spiral`, `checkerboard`, `stripes_diagonal`, `messy_blobs`, `xor_blobs`, `gaussian_quantiles`, `annulus_disk`, `sine_boundary`, `sign_product`, `blobs_overlap`, `swiss_roll_slice`, `linear_borderline`, `radial_waves`, `far_blobs`, or `--scenarios all`.

Extra figure per run: **`val_loss_gap`** (MLP val BCE − EML val BCE) to highlight epochs where one model generalizes better.

---

## License

Add a `LICENSE` if you open-source the implementation. Until then, treat this as private research notes.
