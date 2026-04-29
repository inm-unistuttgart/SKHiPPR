import numpy as np


def to_csv(hbms, N_max=None):

    if N_max is None:
        N_max = max(hbm.fourier.N_HBM for hbm in hbms)
    n_dof = hbms[0].fourier.n_dof

    header = csv_header(N_max, n_dof)

    results_to_csv = np.zeros((len(hbms), n_dof * (2 * N_max + 1) + 2))


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
