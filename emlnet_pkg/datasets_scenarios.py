"""2D classification scenarios (sklearn + custom) for EML vs MLP stress tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_classification,
    make_gaussian_quantiles,
    make_moons,
    make_swiss_roll,
)


@dataclass(frozen=True)
class DatasetBundle:
    """Numpy arrays + suggested axis limits for contour plots."""

    x: np.ndarray
    y: np.ndarray
    lim: float
    name: str


def _lim_from_xy(x: np.ndarray, margin: float = 0.35) -> float:
    lim = float(np.max(np.abs(x))) + margin
    return max(lim, 1.2)


def scenario_moons_hard(n_samples: int = 1200, seed: int = 0) -> DatasetBundle:
    x, y = make_moons(n_samples=n_samples, noise=0.28, random_state=seed)
    return DatasetBundle(x.astype(np.float32), y.astype(np.float32), _lim_from_xy(x), "moons_hard")


def scenario_moons_clean(n_samples: int = 1000, seed: int = 0) -> DatasetBundle:
    x, y = make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
    return DatasetBundle(x.astype(np.float32), y.astype(np.float32), _lim_from_xy(x), "moons_clean")


def scenario_circles(n_samples: int = 1200, seed: int = 1) -> DatasetBundle:
    x, y = make_circles(n_samples=n_samples, factor=0.35, noise=0.12, random_state=seed)
    return DatasetBundle(x.astype(np.float32), y.astype(np.float32), _lim_from_xy(x), "circles")


def scenario_spiral(n_samples: int = 1600, noise: float = 0.18, seed: int = 2) -> DatasetBundle:
    """Two interleaving spirals (classic toy non-linear boundary)."""
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    tt = np.linspace(0.5, 4.5 * np.pi, n)
    r = tt + rng.normal(0.0, noise, size=n)
    x0 = r * np.cos(tt)
    y0 = r * np.sin(tt)
    x1 = -r * np.cos(tt + np.pi)
    y1 = -r * np.sin(tt + np.pi)
    x = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])]).astype(np.float32)
    y = np.concatenate([np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)])
    perm = rng.permutation(len(y))
    return DatasetBundle(x[perm], y[perm], _lim_from_xy(x), "spiral")


def scenario_checkerboard(n_samples: int = 2000, grid: int = 4, seed: int = 3) -> DatasetBundle:
    """Noisy checkerboard in [-1,1]^2."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n_samples, 2)).astype(np.float32)
    cell_x = ((x[:, 0] + 1) * 0.5 * grid).astype(int)
    cell_y = ((x[:, 1] + 1) * 0.5 * grid).astype(int)
    y = ((cell_x + cell_y) % 2).astype(np.float32)
    x = x + rng.normal(0.0, 0.08, size=x.shape).astype(np.float32)
    return DatasetBundle(x, y, 1.35, "checkerboard")


def scenario_messy_blobs(n_samples: int = 1400, seed: int = 4) -> DatasetBundle:
    """Overlapping Gaussian-ish blobs with label noise."""
    x, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=2,
        flip_y=0.12,
        class_sep=0.9,
        random_state=seed,
    )
    return DatasetBundle(x.astype(np.float32), y.astype(np.float32), _lim_from_xy(x, 0.5), "messy_blobs")


def scenario_xor_blobs(n_samples: int = 1600, std: float = 0.16, seed: int = 5) -> DatasetBundle:
    """XOR: Gaussian clouds at the four corners of the unit square (classic non-linear test)."""
    rng = np.random.default_rng(seed)
    n = n_samples // 4
    corners = ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
    labels = (0.0, 1.0, 1.0, 0.0)
    chunks: list[np.ndarray] = []
    labs: list[np.ndarray] = []
    for (cx, cy), lab in zip(corners, labels, strict=True):
        pts = rng.normal(loc=[cx, cy], scale=std, size=(n, 2))
        chunks.append(pts.astype(np.float32))
        labs.append(np.full((n,), lab, dtype=np.float32))
    x = np.vstack(chunks)
    y = np.concatenate(labs)
    perm = rng.permutation(len(y))
    return DatasetBundle(x[perm], y[perm], _lim_from_xy(x, 0.45), "xor_blobs")


def scenario_gaussian_quantiles(n_samples: int = 1600, seed: int = 6) -> DatasetBundle:
    """Sklearn Gaussian quantiles (piecewise smooth boundary in 2D)."""
    x, y = make_gaussian_quantiles(
        n_samples=n_samples,
        n_features=2,
        n_classes=2,
        random_state=seed,
    )
    return DatasetBundle(x.astype(np.float32), y.astype(np.float32), _lim_from_xy(x, 0.45), "gaussian_quantiles")


def scenario_annulus_disk(n_samples: int = 1800, seed: int = 7) -> DatasetBundle:
    """Disk vs thick ring (two concentric decision regions)."""
    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    th0 = rng.uniform(0.0, 2.0 * np.pi, size=n0)
    r0 = np.sqrt(rng.uniform(0.0, 0.82**2, size=n0))
    inner = np.column_stack([r0 * np.cos(th0), r0 * np.sin(th0)]).astype(np.float32)
    th1 = rng.uniform(0.0, 2.0 * np.pi, size=n_samples - n0)
    r1 = rng.uniform(1.02, 1.48, size=n_samples - n0)
    ring = np.column_stack([r1 * np.cos(th1), r1 * np.sin(th1)]).astype(np.float32)
    x = np.vstack([inner, ring])
    x = x + rng.normal(0.0, 0.05, size=x.shape).astype(np.float32)
    y = np.concatenate([np.zeros(n0, dtype=np.float32), np.ones(n_samples - n0, dtype=np.float32)])
    perm = rng.permutation(len(y))
    return DatasetBundle(x[perm], y[perm], _lim_from_xy(x, 0.5), "annulus_disk")


def scenario_sine_boundary(n_samples: int = 2000, seed: int = 8) -> DatasetBundle:
    """Label by which side of a sinusoidal curve each point falls (wavy decision boundary)."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.25, 1.25, size=n_samples).astype(np.float32)
    x2 = rng.uniform(-1.25, 1.25, size=n_samples).astype(np.float32)
    boundary = np.sin(2.6 * np.pi * x1.astype(np.float64)).astype(np.float32)
    noise = rng.normal(0.0, 0.11, size=n_samples).astype(np.float32)
    y = (x2 > boundary + noise).astype(np.float32)
    x = np.column_stack([x1, x2])
    return DatasetBundle(x, y, _lim_from_xy(x, 0.35), "sine_boundary")


def scenario_sign_product(n_samples: int = 2000, seed: int = 9) -> DatasetBundle:
    """Quadrant XOR via sign(x*y) with jitter (four alternating regions)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.55, 1.55, size=(n_samples, 2)).astype(np.float32)
    x = x + rng.normal(0.0, 0.11, size=x.shape).astype(np.float32)
    y = ((x[:, 0] * x[:, 1]) > 0).astype(np.float32)
    return DatasetBundle(x, y, _lim_from_xy(x, 0.4), "sign_product")


def scenario_blobs_overlap(n_samples: int = 1600, seed: int = 10) -> DatasetBundle:
    """Two Gaussian blobs with large std (heavy overlap)."""
    x, y = make_blobs(
        n_samples=n_samples,
        centers=np.array([[-1.2, -0.8], [1.0, 1.0]], dtype=np.float64),
        cluster_std=1.85,
        random_state=seed,
    )
    return DatasetBundle(x.astype(np.float32), y.astype(np.float32), _lim_from_xy(x, 0.55), "blobs_overlap")


def scenario_swiss_roll_slice(n_samples: int = 2000, seed: int = 11) -> DatasetBundle:
    """Swiss roll embedded in 2D (dims 0 vs 2); binary label from roll parameter median split."""
    x3, t = make_swiss_roll(n_samples=n_samples, noise=0.22, random_state=seed)
    x = np.column_stack([x3[:, 0], x3[:, 2]]).astype(np.float32)
    y = (t > float(np.median(t))).astype(np.float32)
    rng = np.random.default_rng(seed + 99)
    perm = rng.permutation(len(y))
    return DatasetBundle(x[perm], y[perm], _lim_from_xy(x, 0.5), "swiss_roll_slice")


def scenario_linear_borderline(n_samples: int = 1600, seed: int = 12) -> DatasetBundle:
    """Almost linearly separable 2D blobs (small overlap)."""
    x, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=1.15,
        flip_y=0.02,
        random_state=seed,
    )
    return DatasetBundle(x.astype(np.float32), y.astype(np.float32), _lim_from_xy(x, 0.45), "linear_borderline")


def scenario_circles_tight(n_samples: int = 1400, seed: int = 13) -> DatasetBundle:
    """Nested circles with smaller gap (harder than default circles)."""
    x, y = make_circles(n_samples=n_samples, factor=0.08, noise=0.11, random_state=seed)
    return DatasetBundle(x.astype(np.float32), y.astype(np.float32), _lim_from_xy(x, 0.4), "circles_tight")


def scenario_moons_extra_noise(n_samples: int = 1400, seed: int = 14) -> DatasetBundle:
    """Moons with very high label noise (harder than moons_hard)."""
    x, y = make_moons(n_samples=n_samples, noise=0.38, random_state=seed)
    return DatasetBundle(x.astype(np.float32), y.astype(np.float32), _lim_from_xy(x), "moons_extra_noise")


def scenario_radial_waves(n_samples: int = 2000, seed: int = 15) -> DatasetBundle:
    """Binary label from alternating radial bands: sin(r*omega) sign."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    r = np.sqrt(rng.uniform(0.15, 3.2**2, size=n_samples)).astype(np.float32)
    x1 = (r * np.cos(th)).astype(np.float32)
    x2 = (r * np.sin(th)).astype(np.float32)
    x = np.column_stack([x1, x2])
    x = x + rng.normal(0.0, 0.07, size=x.shape).astype(np.float32)
    r2 = np.sqrt(np.maximum(x[:, 0] ** 2 + x[:, 1] ** 2, 1e-6))
    y = (np.sin(4.2 * r2.astype(np.float64)) > 0.0).astype(np.float32)
    return DatasetBundle(x, y, _lim_from_xy(x, 0.45), "radial_waves")


def scenario_stripes_diagonal(n_samples: int = 2000, seed: int = 16) -> DatasetBundle:
    """Alternating diagonal stripes: class from floor((x+y)*k)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.6, 1.6, size=(n_samples, 2)).astype(np.float32)
    x = x + rng.normal(0.0, 0.06, size=x.shape).astype(np.float32)
    k = 2.8
    band = np.floor((x[:, 0] + x[:, 1]) * k + 3.0).astype(int)
    y = (band % 2).astype(np.float32)
    return DatasetBundle(x, y, _lim_from_xy(x, 0.35), "stripes_diagonal")


def scenario_far_blobs(n_samples: int = 1400, seed: int = 17) -> DatasetBundle:
    """Well-separated blobs (sanity / easy baseline)."""
    x, y = make_blobs(
        n_samples=n_samples,
        centers=np.array([[-2.5, -2.0], [2.5, 2.0]], dtype=np.float64),
        cluster_std=0.55,
        random_state=seed,
    )
    return DatasetBundle(x.astype(np.float32), y.astype(np.float32), _lim_from_xy(x, 0.35), "far_blobs")


SCENARIOS = {
    "annulus_disk": scenario_annulus_disk,
    "blobs_overlap": scenario_blobs_overlap,
    "checkerboard": scenario_checkerboard,
    "circles": scenario_circles,
    "circles_tight": scenario_circles_tight,
    "far_blobs": scenario_far_blobs,
    "gaussian_quantiles": scenario_gaussian_quantiles,
    "linear_borderline": scenario_linear_borderline,
    "messy_blobs": scenario_messy_blobs,
    "moons_clean": scenario_moons_clean,
    "moons_extra_noise": scenario_moons_extra_noise,
    "moons_hard": scenario_moons_hard,
    "radial_waves": scenario_radial_waves,
    "sign_product": scenario_sign_product,
    "sine_boundary": scenario_sine_boundary,
    "spiral": scenario_spiral,
    "stripes_diagonal": scenario_stripes_diagonal,
    "swiss_roll_slice": scenario_swiss_roll_slice,
    "xor_blobs": scenario_xor_blobs,
}


def load_scenario(key: str, **kwargs) -> DatasetBundle:
    if key not in SCENARIOS:
        raise KeyError(f"Unknown scenario {key!r}. Choose from: {sorted(SCENARIOS)}")
    return SCENARIOS[key](**kwargs)
