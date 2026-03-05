import csv
import pandas as pd
import numpy as np
import tikzplotlib

import matplotlib.pyplot as plt

from create_reference import init_csv

from skhippr.solvers.continuation import BranchPoint
from skhippr.cycles.hbm import HBMSystem
from skhippr.Fourier import Fourier


def import_reference(filename, ode):
    df = pd.read_csv(filename, delimiter=";", dtype=complex, engine="python")
    columns = df.columns

    param = np.real(df[columns[0]].values)
    arclength = np.real(df[columns[1]].values)
    error_init = np.real(df[columns[2]].values)
    error_time = np.real(df[columns[3]].values)
    error_FMs = df[columns[4]].values
    FMs = np.array([df[columns[4 + k]].values for k in range(ode.n_dof)]).T
    X = df[[col for col in columns if col.startswith("X")]].values
    real_formulation = "c" in columns[5 + 2 * ode.n_dof]

    return {
        "name_param": columns[0],
        "param": param,
        "arclength": arclength,
        "error_init": error_init,
        "error_time": error_time,
        "error_FMs": error_FMs,
        "FMs": FMs,
        "X": X,
        "real_formulation": real_formulation,
    }


def iterate_from_reference(
    ode, data, N_HBM, L_DFT, real_formulation, stability_method=None
):

    fourier_new = Fourier(N_HBM, L_DFT, ode.n_dof, real_formulation=real_formulation)
    if stability_method is not None:
        stability_method = stability_method(fourier_new)
    initial_guess = np.zeros((2 * N_HBM + 1) * ode.n_dof)
    hbm = HBMSystem(
        ode, ode.omega, fourier_new, initial_guess, stability_method=stability_method
    )
    bp = BranchPoint(hbm, data["name_param"], 1)

    for X, param in zip(data["X"], data["param"]):
        bp = bp.duplicate()
        bp.X = change_N_HBM(X, fourier_new=fourier_new)
        setattr(bp, data["name_param"], param)
        yield bp


def change_N_HBM(X, fourier_new):
    N_ref = int((len(X) / fourier_new.n_dof - 1) / 2)
    fourier_old = Fourier(
        N_HBM=N_ref,
        L_DFT=fourier_new.L_DFT,
        n_dof=fourier_new.n_dof,
        real_formulation=True,
    )
    return fourier_new.DFT(fourier_old.inv_DFT(X))


def plot_reference_data(data, filename):

    _, ax = plt.subplots(1, 1)
    ax.plot(data["arclength"], data["param"])
    ax.set_xlabel("arclength")
    ax.set_ylabel(data["name_param"])
    ax.set_title(f"{data["name_param"]} vs arclength")
    tikzplotlib.save(f"{filename}_arclength.tikz")

    _, ax = plt.subplots(nrows=1, ncols=1)
    ax.semilogy(data["arclength"], data["error_time"], label="error time")
    ax.semilogy(data["arclength"], data["error_FMs"], label="error FMs")
    ax.legend()
    ax.set_xlabel("arclength")
    ax.set_ylabel("error")
    tikzplotlib.save(f"{filename}_error.tikz")
