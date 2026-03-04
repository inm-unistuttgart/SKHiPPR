import csv
import pandas as pd
import numpy as np
import tikzplotlib

import matplotlib.pyplot as plt

from create_reference import init_csv


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

    return {
        "name_param": columns[0],
        "param": param,
        "arclength": arclength,
        "error_init": error_init,
        "error_time": error_time,
        "error_FMs": error_FMs,
        "FMs": FMs,
        "X": X,
    }


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
