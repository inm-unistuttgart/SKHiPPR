import os
import csv
from collections.abc import Generator
import numpy as np

from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMEquation
from skhippr.solvers.continuation import BranchPoint, pseudo_arclength_continuator


def generate_stability_data_branch(
    filename,
    initial_system,
    solver,
    stepsize,
    stepsize_range,
    initial_direction,
    continuation_parameter,
    verbose,
    num_steps,
) -> Generator[BranchPoint, None, None]:

    if os.path.exists(filename):
        answer = input(f"File {filename} already exists. Overwrite? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            raise RuntimeError(f"File {filename} already exists")
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")

        init_csv(initial_system.equations[0].fourier, writer, continuation_parameter)
        arclength = 0
        num_stable = 0
        num_ustbl = 0

        for k, bp in enumerate(
            pseudo_arclength_continuator(
                initial_system,
                solver,
                stepsize,
                stepsize_range,
                initial_direction,
                continuation_parameter,
                verbose,
                num_steps,
            )
        ):
            if k > 0:
                arclength += np.linalg.norm(X_prev - bp.X)
            X_prev = bp.X

            if bp.stable:
                num_stable += 1
            else:
                num_ustbl += 1

            stab_info = extract_info_from_hbm(bp.equations[0], continuation_parameter)
            to_csv(writer, arclength=arclength, **stab_info)

            yield bp

        print(
            f"ratio stable/unstable: {num_stable}/{num_ustbl} = {num_stable/(num_stable+num_ustbl):.2f}"
        )


def init_csv(fourier: Fourier, writer, name_param: str):

    params = [
        name_param,
        "arclength",
    ]
    X_labels = [f"X0 [{l}]" for l in range(fourier.n_dof)]

    if fourier.real_formulation:
        for sc in ["c", "s"]:
            for k in range(1, fourier.N_HBM + 1):
                X_labels = X_labels + [f"X{sc},{k} [{l}]" for l in range(fourier.n_dof)]

    else:
        for k in range(1, fourier.N_HBM + 1):
            X_labels = X_labels + [f"X {k}, {l}" for l in range(fourier.n_dof)]
            X_labels = [f"X {-k}, {l}" for l in range(fourier.n_dof)] + X_labels

    FM_labels = [f"FM [{k}]" for k in range(fourier.n_dof)]

    hill_labels = []
    for k in range(fourier.n_dof * (2 * fourier.N_HBM + 1)):
        for j in range(fourier.n_dof * (2 * fourier.N_HBM + 1)):
            hill_labels = hill_labels + [f" hill [{k}, {j}]"]

    header = params + FM_labels + X_labels + hill_labels
    writer.writerow(header)
    return header


def extract_info_from_hbm(hbm: HBMEquation, name_param: str):
    _, FMs = hbm.determine_stability(update=True)
    param = np.squeeze(getattr(hbm, name_param))
    X = hbm.X
    hill_mat = hbm.hill_matrix(real_formulation=True, update=True)
    return {"param": param, "FMs": FMs, "X": X, "hill_mat": hill_mat}


def to_csv(
    writer,
    X: np.ndarray,
    hill_mat: np.ndarray,
    FMs: np.ndarray,
    param: float,
    arclength: float,
):

    # sort the FMs if n = 2
    if len(FMs) == 2:
        if np.abs(np.imag(FMs[0])) > 1e-10:
            # sort by imaginary part, if the FMs are complex conjugate pairs
            idx = np.argsort(np.imag(FMs))
            FMs = FMs[idx]
        else:
            # otherwise sort by real part
            idx = np.argsort(np.real(FMs))
            FMs = FMs[idx]

    row = [param, arclength] + list(FMs) + list(X) + list(hill_mat.flatten(order="C"))
    writer.writerow(row)
