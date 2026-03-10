import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from .matrix_foundry import get_basis


def reconstruct_slice(modes, cache,
                      Wx, Wy, Wz, Wt, Ww, WD,
                      nx, ny, nz, nt, nw, nD,
                      degree=3,
                      z_frac=0.5, t_frac=0.5,
                      w_val=1.0, D_val=1.0,
                      n_plot=100):


    x_ev = jnp.linspace(0, 1, n_plot)
    y_ev = jnp.linspace(0, 1, n_plot)

    z_pt = jnp.array([z_frac])
    t_pt = jnp.array([t_frac])
    w_pt = jnp.array([w_val])
    D_pt = jnp.array([D_val])


    phi_x = get_basis(x_ev, nx, degree)
    phi_y = get_basis(y_ev, ny, degree)
    phi_z = get_basis(z_pt, nz, degree)
    phi_t = get_basis(t_pt, nt, degree)
    phi_w = get_basis(w_pt, nw, degree)
    phi_D = get_basis(D_pt, nD, degree)

    Ax, Ay, Az, At, Aw, AD = modes
    R = Ax.shape[0]


    val_x = phi_x @ Ax.T
    val_y = phi_y @ Ay.T
    val_z = (phi_z @ Az.T)[0, :]
    val_t = (phi_t @ At.T)[0, :]


    val_w = (phi_w @ Aw.T)[0, :]
    val_D = (phi_D @ AD.T)[0, :]

    scale = val_z * val_t * val_w * val_D

    v_slice = jnp.einsum('ir,jr,r->ij', val_x, val_y, scale)


    ic_x = phi_x @ Wx
    ic_y = phi_y @ Wy
    ic_z = float((phi_z @ Wz)[0])
    ic_w = float((phi_w @ Ww)[0])
    ic_D = float((phi_D @ WD)[0])

    u0_slice = jnp.outer(ic_x, ic_y) * ic_z * ic_w * ic_D

    u_slice = u0_slice + v_slice

    X, Y = jnp.meshgrid(x_ev, y_ev, indexing='ij')
    return X, Y, u_slice



def plot_solution(modes, cache,
                  Wx, Wy, Wz, Wt, Ww, WD,
                  nx, ny, nz, nt, nw, nD,
                  mu, sigma, degree=3,
                  t_fracs=(0.0, 0.25, 0.5, 0.75, 1.0),
                  w_val=1.0, D_val=1.0, n_plot=100,
                  filename="solution.png"):

    MAX_ROT  = jnp.pi/3
    MIN_DIFF = 0.001
    MAX_DIFF = 0.01

    def exact_slice(x, y, z, t, mu, sigma, diff, omega):

        omega_phys = omega * MAX_ROT
        diff_phys  = diff * (MAX_DIFF - MIN_DIFF) + MIN_DIFF

        cx, cy = 0.5, 0.5
        dx0 = mu[0] - cx
        dy0 = mu[1] - cy

        th = omega_phys * t
        muxt = cx + jnp.cos(th)*dx0 - jnp.sin(th)*dy0
        muyt = cy + jnp.sin(th)*dx0 + jnp.cos(th)*dy0
        muzt = mu[2]

        tau_x = sigma[0]**2 + 2*diff_phys*t
        tau_y = sigma[1]**2 + 2*diff_phys*t
        tau_z = sigma[2]**2 + 2*diff_phys*t

        gx = jnp.exp(-(x-muxt)**2/(2*tau_x)) * (sigma[0]/jnp.sqrt(tau_x))
        gy = jnp.exp(-(y-muyt)**2/(2*tau_y)) * (sigma[1]/jnp.sqrt(tau_y))
        gz = jnp.exp(-(z-muzt)**2/(2*tau_z)) * (sigma[2]/jnp.sqrt(tau_z))

        return gx * gy * gz

    n_t = len(t_fracs)
    fig, axes = plt.subplots(n_t, 3,
                             figsize=(15, 4*n_t),
                             squeeze=False,
                             layout="constrained")

    als_slices = []
    exact_slices = []
    diff_slices = []

    vmin, vmax = None, None
    diffmax = 0.0


    for t_f in t_fracs:
        X, Y, u_als = reconstruct_slice(
            modes, cache,
            Wx, Wy, Wz, Wt, Ww, WD,
            nx, ny, nz, nt, nw, nD,
            degree,
            z_frac=0.5,
            t_frac=t_f,
            w_val=w_val,
            D_val=D_val,
            n_plot=n_plot
        )

        Xnp = np.array(X)
        Ynp = np.array(Y)

        u_ex = exact_slice(Xnp, Ynp, 0.5, t_f, mu, sigma, D_val, w_val)
        u_ex = np.array(u_ex)

        u_diff = np.abs(u_als - u_ex)

        als_slices.append(u_als)
        exact_slices.append(u_ex)
        diff_slices.append(u_diff)

        if vmin is None:
            vmin = min(u_als.min(), u_ex.min())
            vmax = max(u_als.max(), u_ex.max())
        else:
            vmin = min(vmin, u_als.min(), u_ex.min())
            vmax = max(vmax, u_als.max(), u_ex.max())

        diffmax = max(diffmax, u_diff.max())


    for i, t_f in enumerate(t_fracs):
        X, Y = np.array(X), np.array(Y)

        pcm0 = axes[i,0].pcolormesh(X, Y, als_slices[i],
                                    shading="gouraud",
                                    cmap="RdYlBu_r",
                                    vmin=vmin, vmax=vmax)
        axes[i,0].set_title(f"ALS  t={t_f:.2f}")
        axes[i,0].set_aspect("equal")

        axes[i,1].pcolormesh(X, Y, exact_slices[i],
                             shading="gouraud",
                             cmap="RdYlBu_r",
                             vmin=vmin, vmax=vmax)
        axes[i,1].set_title("Exact")
        axes[i,1].set_aspect("equal")

        pcm2 = axes[i,2].pcolormesh(X, Y, diff_slices[i],
                                    shading="gouraud",
                                    cmap="inferno",
                                    vmin=0, vmax=diffmax)
        axes[i,2].set_title("|Diff|")
        axes[i,2].set_aspect("equal")

        for j in range(3):
            axes[i,j].set_xlabel("x")
            axes[i,j].set_ylabel("y")

    cbar1 = fig.colorbar(pcm0, ax=axes[:,0:2].ravel().tolist())
    cbar1.set_label("u")

    cbar2 = fig.colorbar(pcm2, ax=axes[:,2].ravel().tolist())
    cbar2.set_label("|error|")

    fig.suptitle("VSNA 6D: ALS vs Exact vs Abs Diff", fontsize=14)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"saved plot to {filename}")
