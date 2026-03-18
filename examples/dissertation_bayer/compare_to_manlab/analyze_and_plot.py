import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import time

from import_reference import import_reference, iterate_from_reference


def compute_step_1(ode, filename, Ns_HBM, L_DFT, stability_method_generator, solver):
    data = import_reference(filename, ode)
    real_formulation = data["real_formulation"]
    error_stats = np.zeros((5, len(Ns_HBM)))

    for k, N in enumerate(Ns_HBM):
        print(f"{k}/{len(Ns_HBM)}: N_HBM={N}")
        errors_FM_before = []
        errors_FM_after = []
        comptimes = []
        for l, bp in enumerate(
            iterate_from_reference(
                ode=ode,
                data=data,
                N_HBM=N,
                L_DFT=L_DFT,
                real_formulation=real_formulation,
                stability_method=stability_method_generator,
            )
        ):
            bp.omega = data["param"][l]
            FMs_ref = data["FMs"][l, :]
            FMs_before = bp.equations[0].determine_stability(update=True)[1]
            errors_FM_before.append(FM_error_measure(FMs_before, FMs_ref))

            solved = True
            start = time.monotonic_ns()
            try:
                solver.solve_equation(bp.equations[0], "X")
            except RuntimeError:
                solved = False
            stop = time.monotonic_ns()
            if solved:
                errors_FM_after.append(FM_error_measure(bp.eigenvalues, FMs_ref))
            else:
                errors_FM_after.append(np.nan)
            comptimes.append((stop - start) * 1e-9)

            # DEBUG
            # if l > 100:
            #     print(f"DEBUGGING: stopped after {l} points on branch")
            #     break

        error_stats[0, k] = np.median(errors_FM_before)
        error_stats[1, k] = np.min(errors_FM_before)
        error_stats[2, k] = np.max(errors_FM_before)
        error_stats[3, k] = np.nanmedian(
            errors_FM_after,
        )
        error_stats[4, k] = np.median(comptimes)

    return error_stats


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


def FM_error_measure(FMs, FMs_ref):
    if len(FMs) != len(FMs_ref):
        raise ValueError(
            f"FMs and FMs_ref have lengths {len(FMs)} and {len(FMs_ref)}, but should be equal."
        )

    FMs_ref_pos = FMs_ref[np.imag(FMs_ref) >= 0]
    FMs_pos = FMs[np.imag(FMs) >= 0]

    idx_max_FM = np.argmax(np.abs(FMs_pos))
    idx_max_FM_ref = np.argmax(np.abs(FMs_ref_pos))
    return np.abs(FMs_ref_pos[idx_max_FM_ref] - FMs_pos[idx_max_FM])
