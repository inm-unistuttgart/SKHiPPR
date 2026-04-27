from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMEquation


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
    param = getattr(hbm, name_param)
    X = hbm.X
    hill_mat = hbm.hill_matrix(real_formulation=True, update=True)
    Js = hbm.fourier.matrix_inv_DFT(hill_mat)

    row = [param, arclength] + list(FMs) + list(X) + list(Js.flatten(order="F"))
    writer.writerow(row)
