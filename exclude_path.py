
# def optimize_path_signal_harvesting_no(
#     model, x, baseline, mu, N=50, G=16, patch_size=14,
#     n_iter=15, lr=0.08, lam=1.0,
#     momentum=0.7, block_size=None,
#     early_stop_patience=10, early_stop_rtol=0.01, verbose=True,
# ):
#     device = x.device
#     delta_x = x - baseline
#     gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

#     V = torch.ones(G, N, device=device)
#     best_obj = float("inf")
#     best_V = V.clone()
#     vel_V = torch.zeros_like(V)

#     if block_size is None:
#         block_size = max(N // 10, 3)

#     def _obj_of(Vm):
#         gp = _build_path_2d(baseline, delta_x, Vm, gmap, N)
#         d_v, df_v = _eval_path_batched(model, gp, N, device)
#         return _signal_harvesting_path_obj(d_v, df_v, mu, lam=lam)

#     eps = 0.05
#     stale_count = 0
#     prev_best = float("inf")
#     obj_history = []
#     restarted = False

#     for it in range(n_iter):
#         t_it = time.time()
#         obj = _obj_of(V)
#         improved = obj < best_obj
#         if improved:
#             best_obj = obj
#             best_V = V.clone()

#         grad_V = torch.zeros_like(V)
#         grad_norms_per_group = []
#         for g in range(G):
#             k0 = torch.randint(0, N - block_size + 1, (1,)).item()
#             k1 = k0 + block_size
#             z = torch.randn(block_size, device=device)
#             z = z / z.norm() * block_size**0.5

#             V[g, k0:k1] += eps * z
#             obj_plus = _obj_of(V)
#             V[g, k0:k1] -= eps * z

#             grad_V[g, k0:k1] = ((obj_plus - obj) / eps) * z
#             grad_norms_per_group.append(float(grad_V[g].norm()))

#         # SGD with momentum
#         vel_V = momentum * vel_V + grad_V
#         V = V - lr * vel_V
#         V = torch.clamp(V, min=0.01)

#         dt = time.time() - t_it
#         obj_history.append(best_obj)

#         if verbose:
#             mean_g = sum(grad_norms_per_group) / len(grad_norms_per_group)
#             max_g = max(grad_norms_per_group)
#             print(f"  path_opt iter {it:2d}/{n_iter}  "
#                   f"obj={obj:+.6f}  best={best_obj:+.6f}  "
#                   f"|∇V|={float(grad_V.norm()):.4f}  "
#                   f"|vel|={float(vel_V.norm()):.4f}  "
#                   f"mean/max_g={mean_g:.4f}/{max_g:.4f}  "
#                   f"{'✓' if improved else ' '}  {dt:.2f}s")

#         # Early stopping
#         if abs(prev_best) > 1e-12:
#             rel_change = abs(prev_best - best_obj) / abs(prev_best)
#         else:
#             rel_change = abs(prev_best - best_obj)

#         if rel_change < early_stop_rtol:
#             stale_count += 1
#         else:
#             stale_count = 0
#         prev_best = best_obj

#         # Restart from best_V halfway through patience
#         if stale_count == early_stop_patience // 2 and not restarted:
#             if verbose:
#                 print(f"  🔄 Restart from best_V at iter {it}")
#             V = best_V.clone()
#             vel_V = torch.zeros_like(V)
#             stale_count = 0
#             restarted = True
#             continue

#         if stale_count >= early_stop_patience:
#             if verbose:
#                 print(f"  ⚡ Early stop at iter {it}: "
#                       f"no improvement > {early_stop_rtol:.1%} "
#                       f"for {early_stop_patience} iters")
#             break

#     if verbose:
#         print(f"  path_opt done: {len(obj_history)} iters, "
#               f"obj {obj_history[0]:+.4f} → {best_obj:+.4f}  "
#               f"(Δ={obj_history[0] - best_obj:+.4f})")
#
#    return _build_path_2d(baseline, delta_x, best_V, gmap, N)
# def optimize_path_signal_harvesting_no(
#     model, x, baseline, mu, N=50, G=16, patch_size=14,
#     n_iter=15, lr=0.08, lam=1.0,
#     momentum=0.7, block_size=None,
#     early_stop_patience=10, early_stop_rtol=0.01, verbose=True,
# ):
#     device = x.device
#     delta_x = x - baseline
#     gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

#     V = torch.ones(G, N, device=device)
#     best_obj = float("inf")
#     best_V = V.clone()
#     vel_V = torch.zeros_like(V)

#     if block_size is None:
#         block_size = max(N // 10, 3)

#     def _obj_of(Vm):
#         gp = _build_path_2d(baseline, delta_x, Vm, gmap, N)
#         d_v, df_v = _eval_path_batched(model, gp, N, device)
#         return _signal_harvesting_path_obj(d_v, df_v, mu, lam=lam)

#     eps = 0.05
#     stale_count = 0
#     prev_best = float("inf")
#     obj_history = []

#     for it in range(n_iter):
#         t_it = time.time()
#         obj = _obj_of(V)
#         improved = obj < best_obj
#         if improved:
#             best_obj = obj
#             best_V = V.clone()

#         grad_V = torch.zeros_like(V)
#         grad_norms_per_group = []
#         for g in range(G):
#             k0 = torch.randint(0, N - block_size + 1, (1,)).item()
#             k1 = k0 + block_size
#             z = torch.randn(block_size, device=device)
#             z = z / z.norm() * block_size**0.5

#             V[g, k0:k1] += eps * z
#             obj_plus = _obj_of(V)
#             V[g, k0:k1] -= eps * z

#             grad_V[g, k0:k1] = ((obj_plus - obj) / eps) * z
#             grad_norms_per_group.append(float(grad_V[g].norm()))

#         # SGD with momentum
#         vel_V = momentum * vel_V + grad_V
#         V = V - lr * vel_V
#         V = torch.clamp(V, min=0.01)

#         dt = time.time() - t_it
#         obj_history.append(best_obj)

#         if verbose:
#             mean_g = sum(grad_norms_per_group) / len(grad_norms_per_group)
#             max_g = max(grad_norms_per_group)
#             print(f"  path_opt iter {it:2d}/{n_iter}  "
#                   f"obj={obj:+.6f}  best={best_obj:+.6f}  "
#                   f"|∇V|={float(grad_V.norm()):.4f}  "
#                   f"|vel|={float(vel_V.norm()):.4f}  "
#                   f"mean/max_g={mean_g:.4f}/{max_g:.4f}  "
#                   f"{'✓' if improved else ' '}  {dt:.2f}s")

#         # Early stopping
#         if abs(prev_best) > 1e-12:
#             rel_change = abs(prev_best - best_obj) / abs(prev_best)
#         else:
#             rel_change = abs(prev_best - best_obj)

#         if rel_change < early_stop_rtol:
#             stale_count += 1
#         else:
#             stale_count = 0
#         prev_best = best_obj

#         if stale_count >= early_stop_patience:
#             if verbose:
#                 print(f"  ⚡ Early stop at iter {it}: "
#                       f"no improvement > {early_stop_rtol:.1%} "
#                       f"for {early_stop_patience} iters")
#             break

#     if verbose:
#         print(f"  path_opt done: {len(obj_history)} iters, "
#               f"obj {obj_history[0]:+.4f} → {best_obj:+.4f}  "
#               f"(Δ={obj_history[0] - best_obj:+.4f})")

#     return _build_path_2d(baseline, delta_x, best_V, gmap, N)
#The below seems to be potentiall
def optimize_path_signal_harvesting(
    model, x, baseline, mu, N=50, G=16, patch_size=14,
    n_iter=15, lr=0.08, lam=1.0,
    early_stop_patience=10, early_stop_rtol=0.01, verbose=True,
):
    device = x.device
    delta_x = x - baseline
    gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

    V = torch.ones(G, N, device=device, requires_grad=False)
    best_obj = float("inf")
    best_V = V.clone()

    # Adam state per V[g, k]
    m_V = torch.zeros_like(V)  # first moment
    v_V = torch.zeros_like(V)  # second moment
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    def _obj_of(Vm):
        gp = _build_path_2d(baseline, delta_x, Vm, gmap, N)
        d_v, df_v = _eval_path_batched(model, gp, N, device)
        return _signal_harvesting_path_obj(d_v, df_v, mu, lam=lam)

    eps = 0.05
    block_size = max(N // 10, 3)
    stale_count = 0
    prev_best = float("inf")
    obj_history = []

    for it in range(n_iter):
        t_it = time.time()
        obj = _obj_of(V)
        improved = obj < best_obj
        if improved:
            best_obj = obj
            best_V = V.clone()

        # Block FD gradient estimation
        grad_V = torch.zeros_like(V)
        grad_norms_per_group = []
        for g in range(G):
            k0 = torch.randint(0, N - block_size + 1, (1,)).item()
            k1 = k0 + block_size
            z = torch.randn(block_size, device=device)
            z = z / z.norm() * block_size**0.5

            V[g, k0:k1] += eps * z
            obj_plus = _obj_of(V)
            V[g, k0:k1] -= eps * z

            grad_V[g, k0:k1] = ((obj_plus - obj) / eps) * z
            grad_norms_per_group.append(float(grad_V[g].norm()))

        # Adam update
        m_V = beta1 * m_V + (1 - beta1) * grad_V
        v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
        m_hat = m_V / (1 - beta1 ** (it + 1))
        v_hat = v_V / (1 - beta2 ** (it + 1))

        V = V - lr * m_hat / (v_hat.sqrt() + adam_eps)
        V = torch.clamp(V, min=0.01)

        dt = time.time() - t_it
        obj_history.append(best_obj)

        if verbose:
            mean_g = sum(grad_norms_per_group) / len(grad_norms_per_group)
            max_g = max(grad_norms_per_group)
            print(f"  path_opt iter {it:2d}/{n_iter}  "
                  f"obj={obj:+.6f}  best={best_obj:+.6f}  "
                  f"|∇V|={float(grad_V.norm()):.4f}  "
                  f"mean/max_g={mean_g:.4f}/{max_g:.4f}  "
                  f"{'✓' if improved else ' '}  {dt:.2f}s")

        # Early stopping
        if abs(prev_best) > 1e-12:
            rel_change = abs(prev_best - best_obj) / abs(prev_best)
        else:
            rel_change = abs(prev_best - best_obj)

        if rel_change < early_stop_rtol:
            stale_count += 1
        else:
            stale_count = 0
        prev_best = best_obj

        if stale_count >= early_stop_patience:
            if verbose:
                print(f"  ⚡ Early stop at iter {it}: "
                      f"no improvement > {early_stop_rtol:.1%} "
                      f"for {early_stop_patience} iters")
            break

    if verbose:
        print(f"  path_opt done: {len(obj_history)} iters, "
              f"obj {obj_history[0]:+.4f} → {best_obj:+.4f}  "
              f"(Δ={obj_history[0] - best_obj:+.4f})")

    return _build_path_2d(baseline, delta_x, best_V, gmap, N)


# Block not really good
def optimize_path_signal_harvesting(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    mu: torch.Tensor,
    N: int = 50,
    G: int = 16,
    patch_size: int = 14,
    n_iter: int = 25,
    lr: float = 0.08,
    lam: float = 1.0,
    early_stop_patience: int = 10,
    early_stop_rtol: float = 0.01,
    verbose: bool = True,
):
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
    stale_count = 0
    prev_best = float("inf")
    obj_history = []
    block_size = 5
    for it in range(n_iter):
        t_it = time.time()
        obj = _obj_of(V)
        improved = False

        if obj < best_obj:
            best_obj = obj
            best_V = V.clone()
            improved = True

        # Stochastic FD: perturb one random time step per group
        grad_V = torch.zeros_like(V)
        grad_norms_per_group = []
        # for g in range(G):
        #     k = torch.randint(0, N, (1,)).item()
        #     V[g, k] += eps
        #     obj_plus = _obj_of(V)
        #     grad_V[g, k] = (obj_plus - obj) / eps
        #     V[g, k] -= eps
        #     grad_norms_per_group.append(abs(grad_V[g, k].item()))
        for g in range(G):
                # Random block start
                k0 = torch.randint(0, N - block_size + 1, (1,)).item()
                k1 = k0 + block_size
                
                # Perturb the block with a structured pattern
                z = torch.randn(block_size, device=device)
                z = z / z.norm() * block_size**0.5  # scale so per-element magnitude ~ eps
                
                V[g, k0:k1] += eps * z
                obj_plus = _obj_of(V)
                V[g, k0:k1] -= eps * z
                grad_V[g, k0:k1] = ((obj_plus - obj) / eps) * z
                grad_norms_per_group.append(float(grad_V[g].norm()))


        V = V - lr * grad_V
        V = torch.clamp(V, min=0.01)

        dt = time.time() - t_it
        grad_norm = float(grad_V.norm())
        obj_history.append(best_obj)

        if verbose:
            mean_g = sum(grad_norms_per_group) / len(grad_norms_per_group)
            max_g = max(grad_norms_per_group)
            print(f"  path_opt iter {it:2d}/{n_iter}  "
                  f"obj={obj:+.6f}  best={best_obj:+.6f}  "
                  f"|∇V|={grad_norm:.4f}  "
                  f"mean/max_g={mean_g:.4f}/{max_g:.4f}  "
                  f"{'✓' if improved else ' '}  {dt:.2f}s")

        # Early stopping: check if best_obj has stagnated
        if abs(prev_best) > 1e-12:
            rel_change = abs(prev_best - best_obj) / abs(prev_best)
        else:
            rel_change = abs(prev_best - best_obj)

        if rel_change < early_stop_rtol:
            stale_count += 1
        else:
            stale_count = 0
        prev_best = best_obj

        if stale_count >= early_stop_patience:
            if verbose:
                print(f"  ⚡ Early stop at iter {it}: "
                      f"no improvement > {early_stop_rtol:.1%} "
                      f"for {early_stop_patience} iters")
            break

    if verbose:
        print(f"  path_opt done: {len(obj_history)} iters, "
              f"obj {obj_history[0]:+.4f} → {best_obj:+.4f}  "
              f"(Δ={obj_history[0] - best_obj:+.4f})")
    return _build_path_2d(baseline, delta_x, best_V, gmap, N)

# def optimize_path_signal_harvesting_no(
#     model, x, baseline, mu, N=50, G=16, patch_size=14,
#     n_iter=30, lr=0.02, lam=1.0,
# ):
#     device = x.device
#     delta_x = x - baseline
#     gmap = _build_spatial_groups(model, x, baseline, G, patch_size)
#     gmap_flat = gmap.flatten()
#     _, C, H, W = baseline.shape

#     # Learnable parameter
#     V_logits = torch.zeros(G, N, device=device, requires_grad=True)
#     optimizer = torch.optim.Adam([V_logits], lr=lr)

#     best_obj = float("inf")
#     best_V = None

#     for it in range(n_iter):
#         optimizer.zero_grad()

#         # Softplus ensures V > 0 without hard clamping (differentiable)
#         V = F.softplus(V_logits)
#         v_sums = V.sum(dim=1, keepdim=True).clamp(min=1e-12)
#         W_norm = V / v_sums                          # (G, N)

#         pixel_weights = W_norm.T[:, gmap_flat]        # (N, H*W)
#         pixel_weights = pixel_weights.view(N, 1, H, W)

#         steps = delta_x * pixel_weights               # (N, C, H, W)
#         cum = torch.cumsum(steps, dim=0)
#         gamma_stack = torch.cat([baseline, baseline + cum], dim=0)  # (N+1, C, H, W)

#         # Forward through model — IN the graph
#         with torch.enable_grad():
#             f_all = model(gamma_stack)                 # (N+1,)

#         # Gradients at first N points
#         pts_N = gamma_stack[:N]
#         grads_N = torch.autograd.grad(
#             f_all[:N].sum(), gamma_stack,
#             create_graph=True
#         )[0][:N]                                       # (N, C, H, W)

#         step_vecs = gamma_stack[1:] - gamma_stack[:N]
#         d_v = (grads_N * step_vecs).view(N, -1).sum(dim=1)

#         f_ext = torch.cat([f_all[0:1], f_all])
#         df_v = f_ext[1:N+1] - f_ext[:N]

#         # Objective: MSE_ν(φ,1) - λ Σ μ_k |d_k|
#         valid = df_v.abs() > 1e-12
#         safe_df = torch.where(valid, df_v, torch.ones_like(df_v))
#         phi = torch.where(valid, d_v / safe_df, torch.ones_like(d_v))

#         nu = mu * df_v ** 2
#         nu = nu / nu.sum().clamp(min=1e-15)
#         mse = (nu * (phi - 1.0) ** 2).sum()
#         signal = (mu * d_v.abs()).sum()
#         loss = mse - lam * signal

#         loss.backward()
#         optimizer.step()

#         with torch.no_grad():
#             if loss.item() < best_obj:
#                 best_obj = loss.item()
#                 best_V = F.softplus(V_logits).detach().clone()

#     return _build_path_2d(baseline, delta_x, best_V, gmap, N)
# ═════════════════════════════════════════════════════════════════════════════
# §7  JOINT* OPTIMISATION  (Algorithm 1 — Full Signal-Harvesting Solution)

#     Alternating minimisation of (Eq. 20):
#       Phase 1 (measure): fix γ, optimise μ via Eq. 24
#       Phase 2 (path):    fix μ, optimise γ via velocity scheduling
#       Each phase monotonically decreases the objective.
# ═════════════════════════════════════════════════════════════════════════════
