"""
This script demonstrates the effect of aliasing error on the accuracy of the Koopman-Hill projection (KHP) for the Meissner equation by varying the Discrete Fourier Transform (DFT) length. It compares the accuracy of computed Floquet multipliers for different DFT lengths and visualizes the results.

Run the script directly to generate plots and optionally save results to a CSV file. The main function :py:func:`analyze_N_meissner_aliasing` is called with example parameters to showcase the influence of aliasing on prediction accuracy.

* The Meissner equation's Fourier coefficients decay as ``O(1/|k|)``, with every even coefficient being zero and odd ones alternating in sign.
* Aliasing errors affect the accuracy of the computed Floquet multipliers.
* Knowledge about the sign of the (true) Fourier coefficients can significantly improve the results: In the case of Meissner choosing a DFT length that is divisible by 2 but not by 4
"""

import numpy as np
import matplotlib.pyplot as plt
from skhippr.odes.ltp import SmoothedMeissner, TruncatedMeissner
from skhippr.Fourier import Fourier
from skhippr.solvers.newton import NewtonSolver

from demo_mathieu_N_convergence import (
    analyze_N_convergence,
    initialize_csv,
    setup_plot,
)


def analyze_N_meissner_aliasing(N_max=60, Ls_DFT=(1024,), csv_path=None):
    """
    Showcase the effect of aliasing error on KHP accuracy for the Meissner equation by varying the DFT length.
    See notes for an explanation of the observations.

    This function initializes a CSV file for results, sets up system parameters, computes the reference monodromy matrix,
    and iterates over different DFT lengths to analyze the impact on the accuracy of the computed Floquet multipliers.
    Results are plotted and optionally saved to a CSV file.
    Args:
        N_max (int, optional): Maximum order of the Hill determinant to use. Default is 60.
        Ls_DFT (tuple of int, optional): Tuple of DFT lengths to analyze. Default is (1024,).
        csv_path (str or None, optional): Path to the CSV file for saving results. If None, results are not saved.
    Returns:
        None
    Notes:
        The Fourier coefficients of the Meissner equation decay with O(1/|k|).
        Every even Fourier coefficient is zero, and the odd ones have alternating sign.

        When  L_DFT is approximately 1e3, the aliasing errors in the Fourier coefficients are approx. 1e-3.
        Hence, the error incurred by aliasing can be at most approx. 1e-3 (case L_DFT=125).
        If L_DFT is uneven, nonzero Fourier coefficients with k > L_DFT are added as aliasing error
        to the even Fourier coefficients which are supposed to be zero, leading to the largest possible error.
        If L_DFT is divisible by 4, positive Fourier coefficients with k > L_DFT are added as aliasing error to positive Fourier coefficients,
        and vice versa. Hence, the total aliasing error is given by a series whose summands all have the same sign (case L_DFT = 1024).
        In contrast, if L_DFT is divisible by 2 but not by 4, the series for the total aliasing error is alternating in sign, making the resulting aliasing error negligible (case L_DFT=1026).
    """
    initialize_csv(csv_path, N_max, key_param="L_DFT")
    ode = SmoothedMeissner(
        t=0, x=np.array([0.0, 0.0]), smoothing=0, a=4, b=0.2, omega=1, damping=0.0
    )
    solver = NewtonSolver()

    params_plot = dict()
    T = 2 * np.pi / ode.omega
    Phi_T_ref = ode.fundamental_matrix(t_end=T, t_0=0)
    lambdas_ref = np.linalg.eig(Phi_T_ref).eigenvalues
    ax_conv, axs, _ = setup_plot(None, lambdas_ref)
    ax_time = axs[1]
    ax_time.clear()
    ax_time.set_aspect("auto")

    # Plot true square function into right figure
    ts = np.linspace(0, T, max(Ls_DFT) + 1, endpoint=True)
    g = ode.a + ode.b * ode.g_fcn(ts)
    ax_time.plot(ts, g, "k", label="True square function")
    ax_time.set_xlim((0.2 * T, 0.3 * T))
    ax_time.set_ylim((ode.a - 1.2 * ode.b, ode.a + 1.2 * ode.b))

    # Perform Koopman-Hill projection on Meissner equation for various L_DFT
    for k, L_DFT in enumerate(Ls_DFT):
        params_plot["color"] = f"C{k}"
        if k >= 4:
            params_plot["marker"] = "o"
            params_plot["fillstyle"] = "none"
        else:
            params_plot["marker"] = "."
        params_plot["label"] = f"L_DFT={L_DFT}"
        fourier_ref = Fourier(1, L_DFT, n_dof=ode.n_dof, real_formulation=True)
        ode.L_DFT = L_DFT  # for the csv file, has no effect on the dynamics
        hbm = analyze_N_convergence(
            solver=solver,
            ode=ode,
            fourier_ref=fourier_ref,
            Phi_T_ref=Phi_T_ref,
            N_max=N_max,
            params_plot=params_plot,
            ax_conv=ax_conv,
            csv_path=csv_path,
            parameter="L_DFT",
        )

        # Plot the identified "square" function
        del params_plot["marker"]
        ts = np.linspace(0, T, L_DFT, endpoint=False)
        J_time = hbm.equations[0].ode_samples()
        ax_time.plot(ts, -J_time[1, 0, :].squeeze(), "--", **params_plot)

    # Perform Koopman-Hill projection without any aliasing (i.e., on the function given by the truncated Fourier series of the square function)
    params_plot["color"] = f"C{k+1}"
    params_plot["marker"] = "x"
    params_plot["fillstyle"] = "none"
    params_plot["label"] = "No aliasing"

    ode_trunc = TruncatedMeissner(
        t=0,
        x=np.array([0.0, 0.0]),
        N_harmonics=N_max,
        a=ode.a,
        b=ode.b,
        omega=ode.omega,
        damping=ode.damping,
    )
    g = ode_trunc.a + ode_trunc.b * ode_trunc.g_fcn(fourier_ref.time_samples_normalized)
    ax_time.plot(ts, g, "r:", label="No aliasing")

    ode_trunc.L_DFT = f"truncated N = {ode_trunc.N_harmonics}"

    analyze_N_convergence(
        solver,
        ode_trunc,
        fourier_ref=fourier_ref,
        Phi_T_ref=Phi_T_ref,
        N_max=N_max,
        csv_path=csv_path,
        parameter="L_DFT",
        params_plot=params_plot,
        ax_conv=ax_conv,
    )

    ax_conv.legend()
    ax_conv.set_title("Aliasing error in J_k affects KHP accuracy")
    ax_time.legend()


if __name__ == "__main__":
    # Showcase the influence of L_DFT (i.e., the influence of aliasing) on the prediction accuracy of the Meissner eq.

    analyze_N_meissner_aliasing(
        N_max=20,
        csv_path="data_meissner.csv",
        Ls_DFT=(512,),
    )
    plt.show()
