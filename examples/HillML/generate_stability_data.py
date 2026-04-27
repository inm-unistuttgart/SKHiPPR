import os
import csv
from collections.abc import Generator
import numpy as np

from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMEquation
from skhippr.solvers.continuation import BranchPoint, pseudo_arclength_continuator


def generate_stability_data(
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
        raise RuntimeError(f"File {filename} already exists")
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")

        init_csv(initial_system.equations[0].fourier, writer, continuation_parameter)
        arclength = 0

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

            to_csv(writer, bp.equations[0], continuation_parameter, arclength)

            yield bp


def init_csv(fourier: Fourier, writer, name_param: str):

    params = [
        name_param,
        "arclength",
    ]
    X_labels = [f"X0 [{l}]" for l in range(fourier.n_dof)]
    J_labels = []
    for l in range(fourier.n_dof):
        for j in range(fourier.n_dof):
            J_labels = J_labels + [f"J0 [{j},{l}]"]

    # CAUTION - J counts in Fortran-style, i.e., across the columns first, not lexicographically!
    # This makes it more difficult to read, but it is more efficient to write and read from the csv file.
    if fourier.real_formulation:
        for sc in ["c", "s"]:
            for k in range(1, fourier.N_HBM + 1):
                X_labels = X_labels + [f"X{sc},{k} [{l}]" for l in range(fourier.n_dof)]

                for l in range(fourier.n_dof):
                    for j in range(fourier.n_dof):
                        J_labels = J_labels + [f"J{sc},{k}, [{j}],{l}]"]
    else:
        for k in range(1, fourier.N_HBM + 1):
            X_labels = X_labels + [f"X {k}, {l}" for l in range(fourier.n_dof)]
            X_labels = [f"X {-k}, {l}" for l in range(fourier.n_dof)] + X_labels

            for l in range(fourier.n_dof):
                for j in range(fourier.n_dof):
                    J_labels = J_labels + [f"J,{k}, [{l}],{j}]"]
                    J_labels = [f"J,{-k}, [{j}],{l}]"] + J_labels

    FM_labels = [f"FM [{k}]" for k in range(fourier.n_dof)]

    header = params + FM_labels + X_labels + J_labels
    writer.writerow(header)
    return header


def to_csv(
    writer,
    hbm: HBMEquation,
    name_param,
    arclength,
):
    _, FMs = hbm.determine_stability(update=True)

    # sort the FMs if n = 2
    if hbm.fourier.n_dof == 2:
        if np.abs(np.imag(FMs[0])) > 1e-10:
            # sort by imaginary part, if the FMs are complex conjugate pairs
            idx = np.argsort(np.imag(FMs))
            FMs = FMs[idx]
        else:
            # otherwise sort by real part
            idx = np.argsort(np.real(FMs))
            FMs = FMs[idx]

    param = np.squeeze(getattr(hbm, name_param))
    X = hbm.X
    hill_mat = hbm.hill_matrix(real_formulation=True, update=True)
    Js = hbm.fourier.matrix_inv_DFT(hill_mat)

    row = [param, arclength] + list(FMs) + list(X) + list(Js.flatten(order="F"))
    writer.writerow(row)
