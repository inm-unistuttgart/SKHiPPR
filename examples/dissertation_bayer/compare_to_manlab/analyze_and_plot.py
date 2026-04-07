import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import time
import tqdm

from import_reference import import_reference, iterate_from_reference
from skhippr.solvers.continuation import pseudo_arclength_continuator
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.stability.AbstractStabilityHBM import AbstractStabilityHBM


def compute_step_1(
    ode, filename, Ns_HBM, L_DFT, stability_method_generator, solver, early_break=np.inf
):
    data = import_reference(filename, ode)
    real_formulation = data["real_formulation"]
    error_stats = np.zeros((5, len(Ns_HBM)))

    for k, N in enumerate(Ns_HBM):
        print(f"{k}/{len(Ns_HBM)}: N_HBM={N}")

        errors_FM_before, errors_FM_after, comptimes = compute_config_1_and_2(
            ode=ode,
            data=data,
            N_HBM=N,
            L_DFT=L_DFT,
            stability_method_generator=stability_method_generator,
            solver=solver,
            real_formulation=real_formulation,
            early_break=early_break,
        )

        error_stats[0, k] = np.median(errors_FM_before)
        error_stats[1, k] = np.min(errors_FM_before)
        error_stats[2, k] = np.max(errors_FM_before)
        error_stats[3, k] = np.nanmedian(
            errors_FM_after,
        )
        error_stats[4, k] = np.median(comptimes)

    return error_stats


def compute_config_1_and_2(
    ode,
    data,
    N_HBM,
    L_DFT,
    stability_method_generator,
    solver,
    real_formulation=True,
    early_break=np.inf,
    k_init=0,
):
    comptimes = []
    errors_FM_before = []
    errors_FM_after = []

    for l, bp in tqdm.tqdm(
        enumerate(
            iterate_from_reference(
                ode=ode,
                data=data,
                N_HBM=N_HBM,
                L_DFT=L_DFT,
                real_formulation=real_formulation,
                stability_method=stability_method_generator,
                k_init=k_init,
            )
        ),
        total=min(early_break, len(data["param"]) - k_init),
    ):
        bp.omega = data["param"][l + k_init]
        FMs_ref = data["FMs"][l + k_init, :]
        FMs_before = bp.equations[0].determine_stability(update=True)[1]
        errors_FM_before.append(FM_error_measure(FMs_before, FMs_ref))

        solved = True
        start = time.perf_counter_ns()
        try:
            solver.solve_equation(bp.equations[0], "X")
        except RuntimeError:
            solved = False
        stop = time.perf_counter_ns()
        if solved:
            errors_FM_after.append(FM_error_measure(bp.eigenvalues, FMs_ref))
        else:
            errors_FM_after.append(np.nan)
        comptimes.append((stop - start) * 1e-9)

        # # DEBUG
        if l > early_break:
            print(f"DEBUGGING: stopped after {l + k_init} points on branch")
            break

    return errors_FM_before, errors_FM_after, comptimes


def plot_step_1(error_stats, Ns_HBM=None):
    _, ax = plt.subplots(1, 1)
    labels = ["median", "min", "max", "HBM + stab median"]
    for k in range(len(labels)):
        ax.plot(error_stats[-1, :], error_stats[k, :], label=f"{labels[k]} error")
    ax.set_xlabel("HBM + stab comp time")
    ax.set_ylabel("max FM error")
    ax.set_yscale("log")
    ax.legend()
    return ax


def compute_step_2(
    ode,
    data,
    N_HBM,
    L_DFT,
    stability_method_generator,
    solver,
    early_break=np.inf,
    continuation_verbose=False,
    stepsize_range=(0.001, 0.1),
    k_init=0,
):
    real_formulation = data["real_formulation"]

    # Determine step 2 error
    _, errors_FM_after, _ = compute_config_1_and_2(
        ode,
        data,
        N_HBM,
        L_DFT,
        stability_method_generator,
        solver,
        real_formulation=real_formulation,
        early_break=early_break,
        k_init=k_init,
    )

    error_median_after = np.nanmedian(errors_FM_after)

    bp = next(
        iterate_from_reference(
            ode=ode,
            data=data,
            N_HBM=N_HBM,
            L_DFT=L_DFT,
            real_formulation=real_formulation,
            stability_method=stability_method_generator,
        )
    )
    # Create EquationSystem from the branch point, removing the previous anchor equation.
    initial_system = EquationSystem(
        equations=bp.equations[:-1],
        unknowns=bp.unknowns[:-1],
        equation_determining_stability=bp.equation_determining_stability,
    )

    # Determine step 3 time and error
    start = time.perf_counter_ns()
    direction = np.sign(data["param"][-1] - data["param"][0])

    frc = []
    for bp in pseudo_arclength_continuator(
        initial_system=initial_system,
        solver=solver,
        stepsize=0.1,
        stepsize_range=stepsize_range,
        continuation_parameter="omega",
        initial_direction=direction,
        verbose=continuation_verbose,
        num_steps=early_break,
    ):
        frc.append(bp)
        if direction * bp.omega > direction * data["param"][-1]:
            break
    stop = time.perf_counter_ns()
    comptime_total = (stop - start) * 1e-9
    comptime_per_bp = comptime_total / len(frc)

    return error_median_after, comptime_total, comptime_per_bp, len(frc)


def iterate_and_plot_step_2(
    ode,
    filename,
    dict_Ns: dict[str, tuple[AbstractStabilityHBM, int]],
    L_DFT,
    solver,
    early_break=np.inf,
    axs=None,
    continuation_verbose=False,
    stepsize_range=(0.001, 0.1),
    k_init=0,
):

    if axs is None:
        axs = []
        _, ax1 = plt.subplots(1, 1)
        ax1.set_xlabel("comp. time per bp")
        ax1.set_ylabel("median FM error (CII)")
        axs.append(ax1)
        _, ax2 = plt.subplots(1, 1)
        ax2.set_xlabel("total comp. time")
        ax2.set_ylabel("median FM error (CII)")
        axs.append(ax2)

    data = import_reference(filename, ode)

    for label, (stability_method_generator, N_HBM) in dict_Ns.items():
        print(label)
        error_median_after, comptime_total, comptime_per_bp, num_points = (
            compute_step_2(
                ode=ode,
                data=data,
                N_HBM=N_HBM,
                L_DFT=L_DFT,
                stability_method_generator=stability_method_generator,
                solver=solver,
                early_break=early_break,
                continuation_verbose=continuation_verbose,
                stepsize_range=stepsize_range,
                k_init=k_init,
            )
        )
        axs[0].plot(
            comptime_per_bp,
            error_median_after,
            "*",
            label=f"{label}, N = {N_HBM}, {num_points} points",
        )
        axs[1].plot(comptime_total, error_median_after, "*", label=label)

    for ax in axs:
        # ax.set_yscale("log")
        # ax.set_xscale("log")
        ax.legend()

    return axs


def FM_error_measure(FMs, FMs_ref):
    if len(FMs) != len(FMs_ref):
        raise ValueError(
            f"FMs and FMs_ref have lengths {len(FMs)} and {len(FMs_ref)}, but should be equal."
        )

    FMs_ref_pos = FMs_ref[np.imag(FMs_ref) >= 0]

    idx_max = np.argmax(np.abs(FMs_ref_pos))

    err = np.min(np.abs(FMs_ref_pos[idx_max] - FMs))
    if err > 0.1:
        pass
    return err
