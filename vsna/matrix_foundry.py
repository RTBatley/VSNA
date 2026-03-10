import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss

def bspline_3(t):
    at = jnp.abs(t)
    val1 = (2.0/3.0) - at**2 + 0.5 * at**3
    val2 = (1.0/6.0) * (2.0 - at)**3
    return jnp.where(at < 1.0, val1,
            jnp.where(at < 2.0, val2, 0.0))

def d_bspline_3(t):
    at = jnp.abs(t)
    sgn = jnp.sign(t)
    val1 = -2.0 * t + 1.5 * t * at
    val2 = -0.5 * ((2.0 - at)**2) * sgn
    return jnp.where(at < 1.0, val1,
            jnp.where(at < 2.0, val2, 0.0))

def get_basis(x, ncells, degree=3):
    h = 1.0 / ncells
    offset = 1.0
    x_idx = (x / h) + offset
    k = jnp.arange(ncells + degree)
    dist = x_idx[..., None] - k
    return bspline_3(dist)

def get_deriv(x, ncells, degree=3):
    h = 1.0 / ncells
    offset = 1.0
    x_idx = (x / h) + offset
    k = jnp.arange(ncells + degree)
    dist = x_idx[..., None] - k
    scale = 1.0 / h
    return scale * d_bspline_3(dist)

def enforce_dirichlet(w, ncells, g0, g1):
    w = w.at[0].set(6*g0 - 4*w[1] - w[2])
    w = w.at[ncells+2].set(6*g1 - 4*w[ncells+1] - w[ncells])
    return w

def enforce_temporal(w, ncells):
    w = w.at[0].set(-4*w[1] - w[2])
    return w

def get_1d_quadrature(ncells, degree=3):
    xi, wi = leggauss(degree + 2)
    xi = 0.5 * (xi + 1.0); wi = 0.5 * wi
    h = 1.0 / ncells
    offsets = jnp.arange(ncells)
    return (xi[None,:] + offsets[:,None]) * h, wi[None,:] * h

def build_matrix(ncells, deriv_U=False, deriv_V=False, weight_fn=None, degree=3):
    xq, wq = get_1d_quadrature(ncells, degree)
    phi_U = get_deriv(xq, ncells, degree) if deriv_U else get_basis(xq, ncells, degree)
    phi_V = get_deriv(xq, ncells, degree) if deriv_V else get_basis(xq, ncells, degree)
    eff_weights = wq if weight_fn is None else jax.vmap(weight_fn)(xq) * wq
    return jnp.einsum('cqU,cqV,cq->UV', phi_U, phi_V, eff_weights)

def get_mass_matrix(ncells, degree=3):
    return build_matrix(ncells, False, False, None, degree)

def get_stiffness_matrix(ncells, degree=3):
    return build_matrix(ncells, True, True, None, degree)

def get_convection_matrix(ncells, degree=3):
    return build_matrix(ncells, False, True, None, degree)

def get_weighted_mass(ncells, weight_fn, degree=3):
    return build_matrix(ncells, False, False, weight_fn, degree)

def get_boundary_matrix(ncells, t, degree=3):
    x = jnp.array([t])
    phi = get_basis(x, ncells, degree)[0]
    return jnp.outer(phi, phi)

def precompute_matrices(nx, ny, nz, nt, nw, nD, degree=3):

    MIN_ROT, MAX_ROT = 0.0, jnp.pi/3
    MIN_DIFF, MAX_DIFF = 0.001, 0.01

    cache = {}

    cache["Mx"] = get_mass_matrix(nx, degree)
    cache["Kx"] = get_stiffness_matrix(nx, degree)
    cache["Cx"] = get_convection_matrix(nx, degree)

    cache["My"] = get_mass_matrix(ny, degree)
    cache["Ky"] = get_stiffness_matrix(ny, degree)
    cache["Cy"] = get_convection_matrix(ny, degree)

    cache["Mz"] = get_mass_matrix(nz, degree)
    cache["Kz"] = get_stiffness_matrix(nz, degree)

    cache["Mt"] = get_mass_matrix(nt, degree)
    cache["Ct"] = get_convection_matrix(nt, degree)
    cache["Kt"] = get_stiffness_matrix(nt, degree)
    cache["Bt"] = get_boundary_matrix(nt, 1.0, degree)


    uvel = lambda y: -(y - 0.5)
    vvel = lambda x: (x - 0.5)
    omega_fn = lambda w: (1 - w) * MIN_ROT + w * MAX_ROT
    diff_fn = lambda D: (1 - D) * MIN_DIFF + D * MAX_DIFF

    cache["M_advx"] = get_weighted_mass(ny, uvel, degree)
    cache["M_advy"] = get_weighted_mass(nx, vvel, degree)


    cache["Mw"] = get_mass_matrix(nw, degree)
    cache["MD"] = get_mass_matrix(nD, degree)
    cache["M_advw"] = get_weighted_mass(nw, omega_fn, degree)
    cache["M_diffD"] = get_weighted_mass(nD, diff_fn, degree)


    cache["Ct_weak"] = cache["Bt"] - cache["Ct"]
    cache["Cx_weak"] = -cache["Cx"]
    cache["Cy_weak"] = -cache["Cy"]

    terms = []

    terms.append((1.0, 'Mx', 'My', 'Mz', 'Ct_weak', 'Mw', 'MD'))

    terms.append((1.0, 'Cx_weak', 'M_advx', 'Mz', 'Mt', 'M_advw', 'MD'))

    terms.append((1.0, 'M_advy', 'Cy_weak', 'Mz', 'Mt', 'M_advw', 'MD'))

    terms.append((1.0, 'Kx', 'My', 'Mz', 'Mt', 'Mw', 'M_diffD'))

    terms.append((1.0, 'Mx', 'Ky', 'Mz', 'Mt', 'Mw', 'M_diffD'))

    terms.append((1.0, 'Mx', 'My', 'Kz', 'Mt', 'Mw', 'M_diffD'))
    return cache, terms
