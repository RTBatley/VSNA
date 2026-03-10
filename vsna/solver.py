import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import time

from .matrix_foundry import precompute_matrices, enforce_dirichlet, enforce_temporal
from .initial_condition import build_source_terms


def compute_interaction_matrix(modes, cache, keys, active_dim):
    R = modes[0].shape[0]
    W = jnp.ones((R, R))
    for d in range(6):
        if d == active_dim:
            continue
        mat = cache[keys[d]]
        md = modes[d]
        W = W * (md @ mat @ md.T)
    return W

def compute_source_interaction(modes, source_cache, keys, active_dim):
    R = modes[0].shape[0]
    gamma = jnp.ones(R)
    for d in range(6):
        if d == active_dim:
            continue
        vec = source_cache[keys[d]]
        gamma = gamma * (modes[d] @ vec)
    return gamma


def assemble_lhs(modes, cache, terms, active_dim, reg=1e-6):
    R = modes[0].shape[0]
    N_d = modes[active_dim].shape[1]
    size = R * N_d
    LHS = jnp.zeros((size, size))

    for term in terms:
        alpha = term[0]
        keys = term[1:]

        W_k = compute_interaction_matrix(modes, cache, keys, active_dim).T
        A_k = cache[keys[active_dim]].T

        LHS = LHS + alpha * jnp.kron(W_k, A_k)

    return LHS

def assemble_rhs(modes, source_cache, source_terms, active_dim):
    R = modes[0].shape[0]
    N_d = modes[active_dim].shape[1]
    rhs = jnp.zeros(R * N_d)

    for term in source_terms:
        alpha = term[0]
        keys = term[1:]

        gamma_k = compute_source_interaction(modes, source_cache, keys, active_dim)
        v_k = source_cache[keys[active_dim]]

        rhs = rhs + alpha * jnp.kron(gamma_k, v_k)
    return rhs

def op_op_interaction(modes, cache, keys_i, keys_j):
    R = modes[0].shape[0]
    W = jnp.ones((R, R))
    for d in range(6):
        A_i = cache[keys_i[d]]
        A_j = cache[keys_j[d]]
        md = modes[d]
        W = W * (md @ (A_i.T @ A_j) @ md.T)
    return W

def op_src_interaction(modes, cache, source_cache, keys_op, keys_src):
    R = modes[0].shape[0]
    g = jnp.ones(R)
    for d in range(6):
        A = cache[keys_op[d]]
        s = source_cache[keys_src[d]]
        g = g * (modes[d] @ (A.T @ s))
    return g

def src_src_interaction(source_cache, keys_i, keys_j):
    val = 1.0
    for d in range(6):
        si = source_cache[keys_i[d]]
        sj = source_cache[keys_j[d]]
        val = val * (si @ sj)
    return val

def compute_residual_norm(modes, cache, terms, source_cache, source_terms):
    total = 0.0


    for ai, *ki in terms:
        for aj, *kj in terms:
            W = op_op_interaction(modes, cache, ki, kj)
            total += ai * aj * jnp.sum(W)


    for ai, *ki in terms:
        for bs, *ks in source_terms:
            g = op_src_interaction(modes, cache, source_cache, ki, ks)
            total += -2 * ai * bs * jnp.sum(g)


    for bi, *ki in source_terms:
        for bj, *kj in source_terms:
            total += bi * bj * src_src_interaction(source_cache, ki, kj)

    return jnp.maximum(total, 0.0)


def solve_als(nx, ny, nz, nt, nw, nD, R, mu=(0.35, 0.5, 0.5), sigma=(0.1, 0.1, 0.1),
              g0=0.0, g1=0.0, max_iter=100, tol=1e-10, reg=1e-6, degree=3):

    dim_names = ["x", "y", "z", "t", "w", "D"]

    print("VSNA: spatiotemporal-parametric solver")


    t0 = time.perf_counter()

    cache, terms = precompute_matrices(nx, ny, nz, nt, nw, nD, degree)
    source_cache, source_terms, ic_coeffs = build_source_terms(
        nx, ny, nz, nt, nw, nD, cache, mu, sigma, degree)
    Wx, Wy, Wz, Wt, Ww, WD, _, _ = ic_coeffs


    Nx = cache["Mx"].shape[0]
    Ny = cache["My"].shape[0]
    Nz = cache["Mz"].shape[0]
    Nt = cache["Mt"].shape[0]
    Nw = cache["Mw"].shape[0]
    ND = cache["MD"].shape[0]
    dims = [Nx, Ny, Nz, Nt, Nw, ND]

    t_pre = time.perf_counter() - t0

    n_op_terms = len(terms)
    n_src_terms = len(source_terms)
    total_dofs = R * sum(dims)

    print(f"\n  Mesh: {nx}x{ny}x{nz}x{nt}x{nw}x{nD} cells")
    print(f"  Basis dims: {Nx}x{Ny}x{Nz}x{Nt}x{Nw}x{ND}")
    print(f"  Rank R: {R}  |  Total DOFs: {total_dofs}")
    print(f"  Operator terms: {n_op_terms}  |  Source terms: {n_src_terms}")
    print(f"  Precompute: {t_pre:.3f}s")


    key = jax.random.PRNGKey(42)

    Wx, Wy, Wz, Wt, Ww, WD, _, _ = ic_coeffs
    ic_list = [Wx, Wy, Wz, Wt, Ww, WD]

    modes = []

    for d, N in enumerate(dims):
        subkey = jax.random.fold_in(key, d)


        M_d = jnp.tile(ic_list[d][None, :], (R, 1))


        noise = 0.01 * jax.random.normal(subkey, (R, N))
        M_d = M_d + noise


        for r in range(R):
            if d < 3:
                M_d = M_d.at[r, :].set(
                    enforce_dirichlet(M_d[r], N, g0, g1)
                )
            elif d == 3:
                M_d = M_d.at[r, :].set(
                    enforce_temporal(M_d[r], N)
                )

        modes.append(M_d)


    for r in range(R):
        norms = [jnp.linalg.norm(modes[d][r]) for d in range(6)]
        g = (norms[0]*norms[1]*norms[2]*norms[3]*norms[4]*norms[5])**(1/6)

        for d in range(6):
            modes[d] = modes[d].at[r,:].set(modes[d][r] / norms[d] * g)



    t_als = time.perf_counter()
    prev_res_norm = None
    alpha = 1.0

    for it in range(max_iter):
        t_it = time.perf_counter()

        res_sq = compute_residual_norm(modes, cache, terms, source_cache, source_terms)
        res_norm = float(jnp.sqrt(jnp.maximum(res_sq, 0.0)))

        weights = jnp.ones(R)
        for d in range(6):
            N_d = dims[d]
            size = R * N_d


            A = assemble_lhs(modes, cache, terms, d)
            b = assemble_rhs(modes, source_cache, source_terms, d)


            sol = jnp.linalg.solve(A + reg * jnp.eye(size), b)
            new_modes = sol.reshape(R, N_d)
            modes[d] = (1.0 - alpha) * modes[d] + alpha * new_modes

            for r in range(R):
                if d < 3:
                    modes[d] = modes[d].at[r,:].set(enforce_dirichlet(modes[d][r], N_d, g0, g1))
                elif d == 3:
                    modes[d] = modes[d].at[r,:].set(enforce_temporal(modes[d][r], N_d))


            for r in range(R):
                nrm = jnp.linalg.norm(modes[d][r])
                modes[d] = modes[d].at[r,:].set(modes[d][r] / nrm)
                for d_other in range(6):
                    if d_other != d:
                        modes[d_other] = modes[d_other].at[r,:].set(
                            modes[d_other][r] * (nrm**(1/5))
                        )




        if prev_res_norm is not None and prev_res_norm > 0:
            rel_change = abs(res_norm - prev_res_norm) / prev_res_norm
        else:
            rel_change = float('inf')

        dt_it = time.perf_counter() - t_it

        print(f"  Iter {it:3d} | Res: {res_norm:.6e} | ΔRes: {rel_change:.3e} | {dt_it:.2f}s")


        if rel_change < tol and it > 0:
            print(f"\n✓ Converged at iter {it} (ΔRes = {rel_change:.2e} < tol = {tol:.2e})")
            break

        prev_res_norm = res_norm

    t_total = time.perf_counter() - t_als
    res_sq = compute_residual_norm(modes, cache, terms, source_cache, source_terms)
    res_norm = float(jnp.sqrt(jnp.maximum(res_sq, 0.0)))
    print(f"  ALS finished in {t_total:.2f}s  |  Final Res at iter {it+1}: {res_norm:.6e}")

    return modes, cache, source_cache, ic_coeffs
