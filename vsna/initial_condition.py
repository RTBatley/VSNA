import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .matrix_foundry import get_1d_quadrature, get_basis, build_matrix

def gaussian(mu, sigma):
    gx = lambda x: jnp.exp(-0.5 * ((x - mu[0]) / sigma[0])**2)
    gy = lambda y: jnp.exp(-0.5 * ((y - mu[1]) / sigma[1])**2)
    gz = lambda z: jnp.exp(-0.5 * ((z - mu[2]) / sigma[2])**2)
    return gx, gy, gz

def project(func, ncells, mat, degree=3):
    xq, wq = get_1d_quadrature(ncells, degree=3)
    phi = get_basis(xq, ncells, degree)
    vals = jax.vmap(jax.vmap(func))(xq)
    b = jnp.einsum("cqj,cq,cq->j", phi, vals, wq)
    return jnp.linalg.solve(mat, b)

def proj_ic(nx, ny, nz, nt, nw, nD, cache, mu, sigma, degree=3):
    gx, gy, gz = gaussian(mu, sigma)
    Wx = project(gx, nx, cache["Mx"], degree)
    Wy = project(gy, ny, cache["My"], degree)
    Wz = project(gz, nz, cache["Mz"], degree)
    Wt = project(lambda t: 1.0, nt, cache["Mt"], degree)
    Ww = project(lambda w: 1.0, nw, cache["Mw"], degree)
    WD = project(lambda D: 1.0, nD, cache["MD"], degree)
    return Wx, Wy, Wz, Wt, Ww, WD

def build_source_terms(nx, ny, nz, nt, nw, nD, cache, mu, sigma, degree=3):
    Wx, Wy, Wz, Wt, Ww, WD = proj_ic(nx, ny, nz, nt, nw, nD, cache, mu, sigma, degree)

    M_cross = build_matrix(nt, False, False, None, degree)
    C_cross = build_matrix(nt, False, True, None, degree)

    source_cache = {}

    source_cache["vx_M"] = cache["Mx"] @ Wx
    source_cache["vx_Mw"] = cache["M_advx"] @ Wx
    source_cache["vx_C"] = cache["Cx_weak"] @ Wx
    source_cache["vx_K"] = cache["Kx"] @ Wx


    source_cache["vy_M"] = cache["My"] @ Wy
    source_cache["vy_Mw"] = cache["M_advy"] @ Wy
    source_cache["vy_C"] = cache["Cy_weak"] @ Wy
    source_cache["vy_K"] = cache["Ky"] @ Wy


    source_cache["vz_M"] = cache["Mz"] @ Wz
    source_cache["vz_K"] = cache["Kz"] @ Wz


    source_cache["vt_M"] = M_cross @ Wt
    source_cache["vt_C"] = C_cross @ Wt


    source_cache["vw_M"] = cache["Mw"] @ Ww
    source_cache["vw_adv"] = cache["M_advw"] @ Ww


    source_cache["vD_M"] = cache["MD"] @ WD
    source_cache["vD_diff"] = cache["M_diffD"] @ WD

    source_terms = []


    source_terms.append((-1.0, "vx_C", "vy_Mw", "vz_M", "vt_M", "vw_adv", "vD_M"))

    source_terms.append((-1.0, "vx_Mw", "vy_C", "vz_M", "vt_M", "vw_adv", "vD_M"))

    source_terms.append((-1.0, "vx_K", "vy_M", "vz_M", "vt_M", "vw_M", "vD_diff"))

    source_terms.append((-1.0, "vx_M", "vy_K", "vz_M", "vt_M", "vw_M", "vD_diff"))

    source_terms.append((-1.0, "vx_M", "vy_M", "vz_K", "vt_M", "vw_M", "vD_diff"))
    return source_cache, source_terms, (Wx, Wy, Wz, Wt, Ww, WD, source_cache["vt_M"], source_cache["vt_C"])
