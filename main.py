import os
import jax; jax.config.update("jax_enable_x64", True)

from vsna.solver import solve_als
from vsna.utils import plot_solution
from vsna.errors import l2_stp_error


def run():
    os.makedirs("output", exist_ok=True)
    mu = (0.65, 0.5, 0.5)
    sigma = (0.045, 0.045, 0.045)
    ncells = 8
    ncellt = 8
    ncellp = 8
    rank = 16

    modes, cache, _, ic_coeffs = solve_als(
        nx=ncells, ny=ncells, nz=ncells, nt=ncellt, nw=ncellp, nD=ncellp,
        R=rank, mu=mu, sigma=sigma, max_iter=3, tol=1e-4, reg=1e-12
    )

    Wx, Wy, Wz, Wt, Ww, WD, _, _ = ic_coeffs
    plot_w, plot_D = 1.0, 0.0
    plot_solution(
        modes, cache, Wx, Wy, Wz, Wt, Ww, WD,
        nx=ncells, ny=ncells, nz=ncells, nt=ncellt, nw=ncellp, nD=ncellp,
        mu=mu, sigma=sigma, w_val=plot_w, D_val=plot_D,
        t_fracs=(0.0, 0.25, 0.5, 0.75, 1.0), n_plot=100,
        filename="output/solution.png",
    )

    l2_err = l2_stp_error(
        modes, cache, ic_coeffs,
        nx=ncells, ny=ncells, nz=ncells, nt=ncellt, nw=ncellp, nD=ncellp,
        mu=mu, sigma=sigma, degree=3, nq_x=12, nq_t=12, nq_w=12, nq_D=12,
    )
    print(f"L2 error: {l2_err:.6e}")


run()
