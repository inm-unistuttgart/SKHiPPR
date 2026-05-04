import numpy as np
import os


def save_result_with_hill_matrix(hbm, description, path=""):
    hill_matrix = hbm.hill_matrix(update=False)
    filename = f"{path}hill_matrix_{description}.csv"
    res = np.hstack((hbm.X[:, np.newaxis], hill_matrix))
    header = ["X"] + [f"H[:,{j}]" for j in range(hill_matrix.shape[1])]
    np.savetxt(filename, res, delimiter=";", header=";".join(header))


def to_csv(hbms, filename: str, N_max=None, **kwargs):

    # parse file name
    if not filename.endswith(".csv"):
        filename = f"{filename}.csv"

    # guard against accidental overwriting of existing files
    # if os.path.exists(filename):
    #     answer = (
    #         input(f"File '{filename}' already exists. Overwrite? [y/N]: ")
    #         .strip()
    #         .lower()
    #     )
    #     if answer not in ("y", "yes"):
    #         raise RuntimeError(f"File '{filename}' already exists.")

    # prepare the iteration and header
    if N_max is None:
        N_max = max(hbm.fourier.N_HBM for hbm in hbms)
    n_dof = hbms[0].fourier.n_dof
    fourier_max = hbms[0].fourier.__replace__(N_HBM=N_max)
    header = csv_header(N_max, n_dof)

    # Create and populate result table
    results_table = np.zeros((len(hbms), n_dof * (2 * N_max + 1) + 2))

    for k, hbm in enumerate(hbms):
        results_table[k, 0] = hbm.fourier.N_HBM
        results_table[k, 1] = np.linalg.norm(hbm.residual(update=False))
        results_table[k, 2:] = fourier_max.resize_coefficients(hbm.X)

    # store the result
    np.savetxt(filename, results_table, delimiter=";", header=header)


def csv_header(N_max, n_dof):
    header_components = lambda text: [f"X{text} [l]" for l in range(n_dof)]
    header_const = header_components("0")
    header_cos = []
    header_sin = []
    for k in range(1, N_max + 1):
        header_cos += header_components(f"c,{k}")
        header_sin += header_components(f"s,{k}")
    header = ["N_HBM", "HBM residual"] + header_const + header_cos + header_sin
    header = ";".join(header)
    return header
