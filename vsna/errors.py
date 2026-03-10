import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from .matrix_foundry import get_basis, get_1d_quadrature


def l2_stp_error(
    modes, cache, ic_coeffs,
    nx, ny, nz, nt, nw, nD,
    mu, sigma,
    degree=3,
    nq_t=60,
    nq_x=None,
    nq_w=20,
    nq_D=20,
):
    from jax import lax


    MAX_ROT  = jnp.pi / 3
    MIN_DIFF = 0.001
    MAX_DIFF = 0.01

    Ax, Ay, Az, At, Aw, AD = modes
    Wx, Wy, Wz, Wt, Ww, WD, _, _ = ic_coeffs

    Mx, My, Mz, Mt = cache["Mx"], cache["My"], cache["Mz"], cache["Mt"]
    Mw, MD = cache["Mw"], cache["MD"]

    R = Ax.shape[0]




    Vx = Ax @ Mx @ Ax.T
    Vy = Ay @ My @ Ay.T
    Vz = Az @ Mz @ Az.T
    Vt = At @ Mt @ At.T
    Vw = Aw @ Mw @ Aw.T
    VD = AD @ MD @ AD.T
    vv = jnp.sum(Vx * Vy * Vz * Vt * Vw * VD)





    u0x2 = (Wx.T @ Mx @ Wx)
    u0y2 = (Wy.T @ My @ Wy)
    u0z2 = (Wz.T @ Mz @ Wz)
    u0w2 = (Ww.T @ Mw @ Ww)
    u0D2 = (WD.T @ MD @ WD)
    u0u0 = u0x2 * u0y2 * u0z2 * u0w2 * u0D2



    if nq_x is None:
        nq_x = max(nx, ny, nz)

    tq, wt = get_1d_quadrature(nq_t, degree + 5)
    tq = tq.reshape(-1)
    wt = jnp.tile(wt, (nq_t, 1)).reshape(-1)

    phi_t_for_int = get_basis(tq, nt, degree)
    b_t = jnp.einsum("tn,t->n", phi_t_for_int, wt)
    t_int = At @ b_t


    mxWx = Mx @ Wx
    myWy = My @ Wy
    mzWz = Mz @ Wz
    mwWw = Mw @ Ww
    mdWD = MD @ WD

    sx = Ax @ mxWx
    sy = Ay @ myWy
    sz = Az @ mzWz
    sw = Aw @ mwWw
    sD = AD @ mdWD

    u0v = jnp.sum(sx * sy * sz * t_int * sw * sD)




    xq, wx = get_1d_quadrature(nq_x, degree + 3)
    xq = xq.reshape(-1)
    wx = jnp.tile(wx, (nq_x, 1)).reshape(-1)

    wq, ww = get_1d_quadrature(nq_w, degree + 3)
    wq = wq.reshape(-1)
    ww = jnp.tile(ww, (nq_w, 1)).reshape(-1)

    Dq, wD = get_1d_quadrature(nq_D, degree + 3)
    Dq = Dq.reshape(-1)
    wD = jnp.tile(wD, (nq_D, 1)).reshape(-1)

    nq_t_total = tq.shape[0]
    nq_x_total = xq.shape[0]
    nq_w_total = wq.shape[0]
    nq_D_total = Dq.shape[0]


    phi_x = get_basis(xq, nx, degree)
    phi_y = get_basis(xq, ny, degree)
    phi_z = get_basis(xq, nz, degree)
    phi_t = phi_t_for_int
    phi_w = get_basis(wq, nw, degree)
    phi_D = get_basis(Dq, nD, degree)

    mu = jnp.asarray(mu)
    sigma = jnp.asarray(sigma)

    centre = jnp.array([0.5, 0.5])
    delta = mu[:2] - centre


    ct = jnp.einsum("rn,tn->rt", At, phi_t)
    cw = jnp.einsum("rn,wn->rw", Aw, phi_w)
    cD = jnp.einsum("rn,Dn->rD", AD, phi_D)


    ic_w_eval = phi_w @ Ww
    ic_D_eval = phi_D @ WD







    def process_wd_pair(wd_idx, accum):
        exact_norm2_sum, v_exact_sum, u0_exact_sum = accum

        wi = wd_idx // nq_D_total
        Di = wd_idx % nq_D_total


        w_norm = lax.dynamic_index_in_dim(wq, wi, axis=0, keepdims=False)
        D_norm = lax.dynamic_index_in_dim(Dq, Di, axis=0, keepdims=False)


        omega = w_norm * MAX_ROT
        diff  = D_norm * (MAX_DIFF - MIN_DIFF) + MIN_DIFF

        weight_w = lax.dynamic_index_in_dim(ww, wi, axis=0, keepdims=False)
        weight_D = lax.dynamic_index_in_dim(wD, Di, axis=0, keepdims=False)
        weight_wd = weight_w * weight_D


        cos_wt = jnp.cos(omega * tq)
        sin_wt = jnp.sin(omega * tq)
        mu_x_rot = centre[0] + cos_wt * delta[0] - sin_wt * delta[1]
        mu_y_rot = centre[1] + sin_wt * delta[0] + cos_wt * delta[1]


        tau_x = sigma[0]**2 + 2.0 * diff * tq
        tau_y = sigma[1]**2 + 2.0 * diff * tq
        tau_z = sigma[2]**2 + 2.0 * diff * tq


        xq_col = xq[:, None]
        gx = jnp.exp(-(xq_col - mu_x_rot[None, :])**2 / (2.0 * tau_x)) * (sigma[0] / jnp.sqrt(tau_x))
        gy = jnp.exp(-(xq_col - mu_y_rot[None, :])**2 / (2.0 * tau_y)) * (sigma[1] / jnp.sqrt(tau_y))
        gz = jnp.exp(-(xq_col - mu[2])**2        / (2.0 * tau_z)) * (sigma[2] / jnp.sqrt(tau_z))


        wx_col = wx[:, None]
        Ix = jnp.sum(gx * gx * wx_col, axis=0)
        Iy = jnp.sum(gy * gy * wx_col, axis=0)
        Iz = jnp.sum(gz * gz * wx_col, axis=0)
        exact_sq_t = Ix * Iy * Iz
        exact_contrib = jnp.sum(exact_sq_t * wt) * weight_wd


        Px = jnp.einsum("xn,xt,x->nt", phi_x, gx, wx)
        Py = jnp.einsum("xn,xt,x->nt", phi_y, gy, wx)
        Pz = jnp.einsum("xn,xt,x->nt", phi_z, gz, wx)

        cx = Ax @ Px
        cy = Ay @ Py
        cz = Az @ Pz

        cw_slice = lax.dynamic_slice_in_dim(cw, wi, 1, axis=1).squeeze(1)
        cD_slice = lax.dynamic_slice_in_dim(cD, Di, 1, axis=1).squeeze(1)

        v_val = cx * cy * cz * ct * cw_slice[:, None] * cD_slice[:, None]
        v_exact_contrib = jnp.sum(v_val * wt[None, :]) * weight_wd


        icw = lax.dynamic_index_in_dim(ic_w_eval, wi, axis=0, keepdims=False)
        icD = lax.dynamic_index_in_dim(ic_D_eval, Di, axis=0, keepdims=False)

        icx = phi_x @ Wx
        icy = phi_y @ Wy
        icz = phi_z @ Wz

        Px0 = jnp.sum((icx[:, None] * gx) * wx_col, axis=0)
        Py0 = jnp.sum((icy[:, None] * gy) * wx_col, axis=0)
        Pz0 = jnp.sum((icz[:, None] * gz) * wx_col, axis=0)

        u0_exact_contrib = jnp.sum((Px0 * Py0 * Pz0) * wt) * (icw * icD) * weight_wd

        return (
            exact_norm2_sum + exact_contrib,
            v_exact_sum + v_exact_contrib,
            u0_exact_sum + u0_exact_contrib,
        )

    exact_norm2, v_exact, u0_exact = lax.fori_loop(
        0, nq_w_total * nq_D_total,
        process_wd_pair,
        (0.0, 0.0, 0.0)
    )




    err2 = (vv + u0u0 + exact_norm2) + 2.0 * u0v - 2.0 * v_exact - 2.0 * u0_exact
    return jnp.sqrt(jnp.maximum(err2, 0.0))
