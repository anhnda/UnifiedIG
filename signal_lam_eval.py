"""
signal_lam_eval.py — Evaluation Script for Signal-Harvesting Action Methods
=============================================================================

Evaluates attribution methods across multiple images and computes statistics.

Features:
- Path optimization options: Default, Gauss, or Low-rank
- Multi-image evaluation with configurable n_test
- Insertion/deletion metrics (pixel and region-based)
- Statistical reporting (mean ± std)
- JSON export of results
"""

from __future__ import annotations

import json
import time
from typing import Optional
from dataclasses import asdict

import torch
import torch.nn as nn
import numpy as np

from utilss import (
    AttributionResult,
    compute_all_metrics,
    compute_insertion_deletion,
    compute_region_insertion_deletion,
    get_device,
    set_seed,
)

from lam import (
    _forward_scalar,
    _forward_and_gradient_batch,
    _gradient_batch,
    _rescale,
    _build_steps,
    _straight_line_pass,
    _build_spatial_groups,
    _build_path_2d,
    _eval_path_batched,
    standard_ig,
    idgi,
    guided_ig,
    load_image_and_model,
)

from signal_lam import (
    optimize_mu_signal_harvesting,
    compute_signal_harvesting_objective,
    _signal_harvesting_path_obj,
    _pack_result,
)


# ═════════════════════════════════════════════════════════════════════════════
# §1  PATH OPTIMIZATION VARIANTS
# ═════════════════════════════════════════════════════════════════════════════

def optimize_path_default(
    model, x, baseline, mu, N=50, G=16, patch_size=14,
    n_iter=15, lr=0.08, lam=1.0,
):
    """Default path optimization: stochastic FD on velocity schedule."""
    device = x.device
    delta_x = x - baseline
    gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

    V = torch.ones(G, N, device=device)
    best_obj = float("inf")
    best_V = V.clone()

    def _obj_of(Vm):
        gp = _build_path_2d(baseline, delta_x, Vm, gmap, N)
        d_v, df_v = _eval_path_batched(model, gp, N, device)
        return _signal_harvesting_path_obj(d_v, df_v, mu, lam=lam)

    eps = 0.05
    for it in range(n_iter):
        obj = _obj_of(V)
        if obj < best_obj:
            best_obj = obj
            best_V = V.clone()

        grad_V = torch.zeros_like(V)
        for g in range(G):
            k = torch.randint(0, N, (1,)).item()
            V[g, k] += eps
            obj_plus = _obj_of(V)
            grad_V[g, k] = (obj_plus - obj) / eps
            V[g, k] -= eps

        V = V - lr * grad_V
        V = torch.clamp(V, min=0.01)

    return _build_path_2d(baseline, delta_x, best_V, gmap, N)


def optimize_path_lowrank(
    model, x, baseline, mu, N=50, G=16, patch_size=14,
    n_iter=15, lr=0.002, lam=1.0,
    momentum=0.5, n_basis=15,
    early_stop_patience=10, early_stop_rtol=0.01,
):
    """Low-rank basis path optimization."""
    device = x.device
    delta_x = x - baseline
    gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

    basis = torch.stack([
        torch.cos(torch.arange(N, device=device, dtype=torch.float32) * j * 3.14159 / N)
        for j in range(n_basis)
    ])
    basis = basis / basis.norm(dim=1, keepdim=True)

    A = torch.zeros(G, n_basis, device=device)
    A[:, 0] = basis[0].sum()

    best_obj = float("inf")
    best_A = A.clone()
    vel_A = torch.zeros_like(A)

    def _V_from_A(Am):
        return torch.clamp(Am @ basis, min=0.01)

    def _obj_of(Am):
        V = _V_from_A(Am)
        gp = _build_path_2d(baseline, delta_x, V, gmap, N)
        d_v, df_v = _eval_path_batched(model, gp, N, device)
        return _signal_harvesting_path_obj(d_v, df_v, mu, lam=lam)

    eps = 0.01
    stale_count = 0
    prev_best = float("inf")
    restarted = False

    for it in range(n_iter):
        obj = _obj_of(A)
        improved = obj < best_obj
        if improved:
            best_obj = obj
            best_A = A.clone()

        grad_A = torch.zeros_like(A)
        for g in range(G):
            j = torch.randint(0, n_basis, (1,)).item()
            A[g, j] += eps
            obj_plus = _obj_of(A)
            grad_A[g, j] = (obj_plus - obj) / eps
            A[g, j] -= eps

        grad_norm = grad_A.norm()
        if grad_norm > 1e-8:
            grad_A = grad_A / grad_norm

        vel_A = momentum * vel_A + grad_A
        A = A - lr * vel_A

        if abs(prev_best) > 1e-12:
            rel_change = abs(prev_best - best_obj) / abs(prev_best)
        else:
            rel_change = abs(prev_best - best_obj)

        if rel_change < early_stop_rtol:
            stale_count += 1
        else:
            stale_count = 0
        prev_best = best_obj

        if stale_count == early_stop_patience // 2 and not restarted:
            A = best_A.clone()
            vel_A = torch.zeros_like(A)
            stale_count = 0
            restarted = True
            continue

        if stale_count >= early_stop_patience:
            break

    V = _V_from_A(best_A)
    return _build_path_2d(baseline, delta_x, V, gmap, N)


def optimize_path_gauss(
    model, x, baseline, mu, N=50, G=16, patch_size=14,
    n_iter=15, lr=0.08, lam=1.0,
    momentum=0.7, bump_width=None,
    early_stop_patience=10, early_stop_rtol=0.01,
):
    """Gaussian bump path optimization."""
    device = x.device
    delta_x = x - baseline
    gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

    V = torch.ones(G, N, device=device)
    best_obj = float("inf")
    best_V = V.clone()
    vel_V = torch.zeros_like(V)

    if bump_width is None:
        bump_width = max(N // 10, 3)

    t_idx = torch.arange(N, device=device, dtype=torch.float32)

    def _obj_of(Vm):
        gp = _build_path_2d(baseline, delta_x, Vm, gmap, N)
        d_v, df_v = _eval_path_batched(model, gp, N, device)
        return _signal_harvesting_path_obj(d_v, df_v, mu, lam=lam)

    eps = 0.05
    stale_count = 0
    prev_best = float("inf")
    restarted = False

    for it in range(n_iter):
        obj = _obj_of(V)
        improved = obj < best_obj
        if improved:
            best_obj = obj
            best_V = V.clone()

        grad_V = torch.zeros_like(V)
        for g in range(G):
            center = torch.randint(0, N, (1,)).item()
            bump = torch.exp(-0.5 * ((t_idx - center) / bump_width) ** 2)
            bump = bump / bump.norm()

            V[g] += eps * bump
            obj_plus = _obj_of(V)
            V[g] -= eps * bump

            grad_V[g] = ((obj_plus - obj) / eps) * bump

        vel_V = momentum * vel_V + grad_V
        V = V - lr * vel_V
        V = torch.clamp(V, min=0.01)

        if abs(prev_best) > 1e-12:
            rel_change = abs(prev_best - best_obj) / abs(prev_best)
        else:
            rel_change = abs(prev_best - best_obj)

        if rel_change < early_stop_rtol:
            stale_count += 1
        else:
            stale_count = 0
        prev_best = best_obj

        if stale_count == early_stop_patience // 2 and not restarted:
            V = best_V.clone()
            vel_V = torch.zeros_like(V)
            stale_count = 0
            restarted = True
            continue

        if stale_count >= early_stop_patience:
            break

    return _build_path_2d(baseline, delta_x, best_V, gmap, N)


# ═════════════════════════════════════════════════════════════════════════════
# §2  JOINT* WITH SELECTABLE PATH OPTIMIZATION
# ═════════════════════════════════════════════════════════════════════════════

def joint_star_ig_eval(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    N: int = 50,
    G: int = 16,
    patch_size: int = 14,
    n_alternating: int = 2,
    lam: float = 1.0,
    tau: float = 0.01,
    mu_iter: int = 300,
    path_iter: int = 10,
    init_path: Optional[list[torch.Tensor]] = None,
    path_opt: str = "default",
) -> AttributionResult:
    """
    Joint* with selectable path optimization method.

    path_opt: "default", "lowrank", or "gauss"
    """
    t0 = time.time()
    device = x.device
    delta_x = x - baseline
    f_bl = _forward_scalar(model, baseline)
    f_x = _forward_scalar(model, x)
    target = f_x - f_bl

    if init_path is not None:
        assert len(init_path) == N + 1
        gamma_pts = [p.clone() for p in init_path]
    else:
        gamma_pts = [baseline + (k / N) * delta_x for k in range(N + 1)]

    mu = torch.full((N,), 1.0 / N, device=device)
    Q_history = []

    best_obj = float("inf")
    best_Q = -1.0
    best_gamma_pts = gamma_pts
    best_mu = mu
    best_d_list: list[float] = []
    best_df_list: list[float] = []
    best_f_vals: list[float] = []
    best_gnorms: list[float] = []
    best_grads: list[torch.Tensor] = []

    # Select path optimization function
    if path_opt == "lowrank":
        path_optimizer = optimize_path_lowrank
    elif path_opt == "gauss":
        path_optimizer = optimize_path_gauss
    else:
        path_optimizer = optimize_path_default

    def _evaluate_path(gp, mu_vec):
        ap = torch.cat(gp, dim=0)
        with torch.no_grad():
            fa = model(ap)
        pn = ap[:N]
        gb = _gradient_batch(model, pn)
        sb = ap[1:] - ap[:N]
        dt = (gb * sb).view(N, -1).sum(dim=1)
        dl = dt.tolist()

        f0 = fa[0].item()
        fv = [f0] + fa.tolist()
        dfl = [fv[k + 1] - fv[k] for k in range(N)]

        gn = gb.view(N, -1).norm(dim=1).tolist()
        gr = [gb[k:k+1].clone() for k in range(N)]
        da = torch.tensor(dl, device=device)
        dfa = torch.tensor(dfl, device=device)
        return dl, dfl, fv, gn, gr, da, dfa

    for s in range(n_alternating):
        d_list, df_list, f_vals, gnorms, grads, d_arr, df_arr = \
            _evaluate_path(gamma_pts, mu)

        mu = optimize_mu_signal_harvesting(
            d_arr, df_arr, lam=lam, tau=tau, n_iter=mu_iter)

        var_mu, cv2_mu, Q_mu = compute_all_metrics(d_arr, df_arr, mu)
        obj_mu, _, _, _ = compute_signal_harvesting_objective(
            d_arr, df_arr, mu, lam=lam, tau=tau)

        if obj_mu < best_obj or (abs(obj_mu - best_obj) < 1e-8 and Q_mu > best_Q):
            best_obj = obj_mu
            best_Q = Q_mu
            best_gamma_pts = gamma_pts
            best_mu = mu.clone()
            best_d_list, best_df_list = d_list, df_list
            best_f_vals, best_gnorms, best_grads = f_vals, gnorms, grads

        Q_path = Q_mu
        obj_path = obj_mu
        if s < n_alternating - 1:
            new_gamma_pts = path_optimizer(
                model, x, baseline, mu, N=N, G=G,
                patch_size=patch_size, n_iter=path_iter, lam=lam)

            new_d_list, new_df_list, new_f_vals, new_gnorms, new_grads, \
                new_d_arr, new_df_arr = _evaluate_path(new_gamma_pts, mu)
            _, _, Q_new = compute_all_metrics(new_d_arr, new_df_arr, mu)
            obj_new, _, _, _ = compute_signal_harvesting_objective(
                new_d_arr, new_df_arr, mu, lam=lam, tau=tau)

            Q_path = Q_new
            obj_path = obj_new

            if obj_new < best_obj:
                gamma_pts = new_gamma_pts
                d_list, df_list = new_d_list, new_df_list
                f_vals, gnorms, grads = new_f_vals, new_gnorms, new_grads
                d_arr, df_arr = new_d_arr, new_df_arr

                best_obj = obj_new
                best_Q = Q_new
                best_gamma_pts = new_gamma_pts
                best_mu = mu.clone()
                best_d_list, best_df_list = new_d_list, new_df_list
                best_f_vals, best_gnorms, best_grads = new_f_vals, new_gnorms, new_grads
            else:
                gamma_pts = best_gamma_pts
                mu = best_mu.clone()

        Q_history.append({
            "iteration": s,
            "Q_after_mu": float(Q_mu),
            "Q_after_path": float(Q_path),
            "obj_after_mu": float(obj_mu),
            "obj_after_path": float(obj_path),
            "best_Q": float(best_Q),
            "best_obj": float(best_obj),
        })

    gamma_pts = best_gamma_pts
    mu = best_mu
    grads = best_grads

    attr = torch.zeros_like(x)
    for k in range(N):
        attr += mu[k] * grads[k] * (gamma_pts[k + 1] - gamma_pts[k])
    attr = _rescale(attr, target)

    return _pack_result(f"Joint*-{path_opt}", attr, best_d_list, best_df_list,
                        best_f_vals, best_gnorms, mu, N, t0, Q_history)


def mu_optimized_ig(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    N: int = 50,
    lam: float = 1.0,
    tau: float = 0.01,
    n_iter: int = 300,
) -> AttributionResult:
    """Straight line + optimal μ under the signal-harvesting objective."""
    t0 = time.time()
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, x, baseline, N)

    d_arr = torch.tensor(d_list, device=x.device)
    df_arr = torch.tensor(df_list, device=x.device)

    mu = optimize_mu_signal_harvesting(
        d_arr, df_arr, lam=lam, tau=tau, n_iter=n_iter)

    grad_stack = torch.cat(grads, dim=0)
    mu_4d = mu.view(N, 1, 1, 1)
    wg = (mu_4d * grad_stack).sum(dim=0, keepdim=True)
    attr = _rescale(delta_x * wg, target)

    return _pack_result("μ-Optimized*", attr, d_list, df_list, f_vals,
                        gnorms, mu, N, t0)


# ═════════════════════════════════════════════════════════════════════════════
# §3  EVALUATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def run_methods_on_image(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    N: int = 50,
    lam: float = 1.0,
    tau: float = 0.01,
    path_opt: str = "default",
    guided_init: bool = False,
) -> list[AttributionResult]:
    """Run all attribution methods on a single image."""
    results = []

    # 1. Standard IG
    results.append(standard_ig(model, x, baseline, N))

    # 2. IDGI
    results.append(idgi(model, x, baseline, N))

    # 3. Guided IG
    gig = guided_ig(model, x, baseline, N)
    results.append(gig)

    # 4. μ-Optimized*
    results.append(mu_optimized_ig(
        model, x, baseline, N, lam=lam, tau=tau, n_iter=300))

    # 5. Joint*
    init_path = gig.gamma_pts if guided_init else None
    results.append(joint_star_ig_eval(
        model, x, baseline, N,
        G=16, patch_size=14,
        n_alternating=2, lam=lam, tau=tau,
        mu_iter=300, path_iter=10,
        init_path=init_path,
        path_opt=path_opt))

    return results


def compute_metrics_for_methods(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    methods: list[AttributionResult],
    insdel_steps: int = 100,
    patch_size: int = 14,
    use_slic: bool = True,
) -> dict:
    """Compute insertion/deletion metrics for all methods."""
    metrics = {}

    for m in methods:
        # Pixel-based insertion/deletion
        scores = compute_insertion_deletion(
            model, x, baseline, m.attributions, n_steps=insdel_steps)

        # Region-based insertion/deletion
        region_scores = compute_region_insertion_deletion(
            model, x, baseline, m.attributions,
            patch_size=patch_size, use_slic=use_slic)

        metrics[m.name] = {
            "Q": m.Q,
            "CV2": m.CV2,
            "Var_nu": m.Var_nu,
            "insertion_auc": scores.insertion_auc,
            "deletion_auc": scores.deletion_auc,
            "ins_del": scores.insertion_auc - scores.deletion_auc,
            "region_insertion_auc": region_scores.insertion_auc,
            "region_deletion_auc": region_scores.deletion_auc,
            "region_ins_del": region_scores.insertion_auc - region_scores.deletion_auc,
            "elapsed_s": m.elapsed_s,
        }

    return metrics


def evaluate_multiple_images(
    n_test: int = 30,
    N: int = 50,
    lam: float = 1.0,
    tau: float = 0.01,
    path_opt: str = "default",
    device: Optional[torch.device] = None,
    min_conf: float = 0.70,
    insdel_steps: int = 100,
    patch_size: int = 14,
    use_slic: bool = True,
    guided_init: bool = False,
    seed: int = 42,
) -> dict:
    """
    Evaluate attribution methods on multiple images.

    Args:
        n_test: Number of images to evaluate
        N: Number of interpolation steps
        lam: Signal-harvesting strength
        tau: L2 admissibility multiplier
        path_opt: Path optimization method ("default", "lowrank", "gauss")
        device: Torch device
        min_conf: Minimum confidence for image selection
        insdel_steps: Number of steps for insertion/deletion
        patch_size: Patch size for region-based evaluation
        use_slic: Use SLIC superpixels for regions
        guided_init: Initialize Joint* from Guided IG path
        seed: Random seed

    Returns:
        Dictionary with statistics and per-image results
    """
    set_seed(seed)

    if device is None:
        device = get_device()

    print(f"\n{'='*70}")
    print(f"Evaluation Configuration")
    print(f"{'='*70}")
    print(f"Number of test images: {n_test}")
    print(f"Interpolation steps N: {N}")
    print(f"Signal-harvesting λ:   {lam}")
    print(f"L2 admissibility τ:    {tau}")
    print(f"Path optimization:     {path_opt}")
    print(f"Ins/Del steps:         {insdel_steps}")
    print(f"Region mode:           {'SLIC' if use_slic else f'grid-{patch_size}'}")
    print(f"Device:                {device}")
    print(f"{'='*70}\n")

    all_results = []
    method_names = []

    for img_idx in range(n_test):
        print(f"[{img_idx+1}/{n_test}] Processing image {img_idx}...", end=" ", flush=True)

        try:
            # Load image
            model, x, baseline, info = load_image_and_model(
                device, min_conf, skip=img_idx)

            # Run methods
            methods = run_methods_on_image(
                model, x, baseline, N=N, lam=lam, tau=tau,
                path_opt=path_opt, guided_init=guided_init)

            # Compute metrics
            metrics = compute_metrics_for_methods(
                model, x, baseline, methods,
                insdel_steps=insdel_steps,
                patch_size=patch_size,
                use_slic=use_slic)

            all_results.append({
                "image_idx": img_idx,
                "image_info": info,
                "metrics": metrics,
            })

            if img_idx == 0:
                method_names = list(metrics.keys())

            print("✓")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    # Compute statistics
    print(f"\nComputing statistics across {len(all_results)} images...")
    stats = compute_statistics(all_results, method_names)

    return {
        "config": {
            "n_test": n_test,
            "N": N,
            "lam": lam,
            "tau": tau,
            "path_opt": path_opt,
            "insdel_steps": insdel_steps,
            "patch_size": patch_size,
            "use_slic": use_slic,
            "guided_init": guided_init,
            "seed": seed,
            "device": str(device),
        },
        "statistics": stats,
        "per_image_results": all_results,
    }


def compute_statistics(all_results: list, method_names: list) -> dict:
    """Compute mean and std across all images."""
    stats = {}

    for method_name in method_names:
        metric_names = [
            "Q", "CV2", "Var_nu",
            "insertion_auc", "deletion_auc", "ins_del",
            "region_insertion_auc", "region_deletion_auc", "region_ins_del",
            "elapsed_s",
        ]

        method_stats = {}
        for metric_name in metric_names:
            values = []
            for result in all_results:
                if method_name in result["metrics"]:
                    val = result["metrics"][method_name][metric_name]
                    if val is not None and not np.isnan(val):
                        values.append(val)

            if len(values) > 0:
                method_stats[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n": len(values),
                }
            else:
                method_stats[metric_name] = {
                    "mean": None,
                    "std": None,
                    "n": 0,
                }

        stats[method_name] = method_stats

    return stats


def print_statistics(stats: dict, config: dict):
    """Print statistics table to terminal."""
    print(f"\n{'='*70}")
    print(f"Evaluation Results: {config['path_opt'].upper()} Path Optimization")
    print(f"{'='*70}")
    print(f"Configuration: N={config['N']}, λ={config['lam']}, "
          f"τ={config['tau']}, n_test={config['n_test']}")
    print(f"{'='*70}\n")

    # Quality metrics
    print("Quality Metrics (mean ± std):")
    print(f"{'Method':<20} {'Q':>12} {'CV²':>12} {'Var_ν':>12} {'Time(s)':>12}")
    print("-" * 70)

    for method_name, method_stats in stats.items():
        Q = method_stats["Q"]
        CV2 = method_stats["CV2"]
        Var_nu = method_stats["Var_nu"]
        time_s = method_stats["elapsed_s"]

        print(f"{method_name:<20} "
              f"{Q['mean']:>5.4f}±{Q['std']:<5.4f} "
              f"{CV2['mean']:>5.4f}±{CV2['std']:<5.4f} "
              f"{Var_nu['mean']:>5.6f}±{Var_nu['std']:<5.6f} "
              f"{time_s['mean']:>5.2f}±{time_s['std']:<5.2f}")

    # Pixel-based Insertion/Deletion
    print(f"\nPixel-based Insertion/Deletion (mean ± std):")
    print(f"{'Method':<20} {'Ins AUC':>14} {'Del AUC':>14} {'Ins-Del':>14}")
    print("-" * 70)

    for method_name, method_stats in stats.items():
        ins = method_stats["insertion_auc"]
        dels = method_stats["deletion_auc"]
        diff = method_stats["ins_del"]

        print(f"{method_name:<20} "
              f"{ins['mean']:>6.4f}±{ins['std']:<6.4f} "
              f"{dels['mean']:>6.4f}±{dels['std']:<6.4f} "
              f"{diff['mean']:>6.4f}±{diff['std']:<6.4f}")

    # Region-based Insertion/Deletion
    print(f"\nRegion-based Insertion/Deletion (mean ± std):")
    print(f"{'Method':<20} {'R-Ins AUC':>14} {'R-Del AUC':>14} {'R-Ins-Del':>14}")
    print("-" * 70)

    for method_name, method_stats in stats.items():
        r_ins = method_stats["region_insertion_auc"]
        r_del = method_stats["region_deletion_auc"]
        r_diff = method_stats["region_ins_del"]

        print(f"{method_name:<20} "
              f"{r_ins['mean']:>6.4f}±{r_ins['std']:<6.4f} "
              f"{r_del['mean']:>6.4f}±{r_del['std']:<6.4f} "
              f"{r_diff['mean']:>6.4f}±{r_diff['std']:<6.4f}")

    print(f"\n{'='*70}\n")


# ═════════════════════════════════════════════════════════════════════════════
# §4  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Signal-Harvesting IG Evaluation — Multi-image Statistics")

    parser.add_argument("--n-test", type=int, default=30,
                        help="Number of test images (default: 30)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of interpolation steps N (default: 50)")
    parser.add_argument("--lam", type=float, default=1.0,
                        help="Signal-harvesting strength λ (default: 1.0)")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="L2 admissibility multiplier τ (default: 0.01)")
    parser.add_argument("--path-opt", type=str, default="default",
                        choices=["default", "lowrank", "gauss"],
                        help="Path optimization method (default: default)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, default: auto)")
    parser.add_argument("--min-conf", type=float, default=0.70,
                        help="Minimum confidence for image selection (default: 0.70)")
    parser.add_argument("--insdel-steps", type=int, default=100,
                        help="Number of steps for insertion/deletion (default: 100)")
    parser.add_argument("--patch-size", type=int, default=14,
                        help="Grid patch size for region-based evaluation (default: 14)")
    parser.add_argument("--no-slic", action="store_true",
                        help="Use grid patches instead of SLIC superpixels")
    parser.add_argument("--guided-init", action="store_true",
                        help="Initialize Joint* from Guided IG path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--json", type=str, default=None,
                        help="Export results to JSON file")

    args = parser.parse_args()

    device = get_device(force=args.device)

    # Run evaluation
    results = evaluate_multiple_images(
        n_test=args.n_test,
        N=args.steps,
        lam=args.lam,
        tau=args.tau,
        path_opt=args.path_opt,
        device=device,
        min_conf=args.min_conf,
        insdel_steps=args.insdel_steps,
        patch_size=args.patch_size,
        use_slic=not args.no_slic,
        guided_init=args.guided_init,
        seed=args.seed,
    )

    # Print statistics
    print_statistics(results["statistics"], results["config"])

    # Export to JSON
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.json}")
