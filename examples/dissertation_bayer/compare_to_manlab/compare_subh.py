from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from analyze_and_plot import iterate_from_reference, import_reference
from analyze_hinged import init_hinged

from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillSubharmonic,
    KoopmanHillProjection,
)
from skhippr.stability.ClassicalHill import ClassicalHill

from skhippr.Fourier import Fourier
from skhippr.solvers.newton import NewtonSolver

k_max = 20

ode = init_hinged(n_modes=10, omega_0_normalized=0.1)

fourier = Fourier(n_dof=ode.n_dof, N_HBM=10, L_DFT=1024, real_formulation=True)

solver = NewtonSolver(tolerance=1e-8, verbose=False)

dir = KoopmanHillProjection(fourier)
subh = KoopmanHillSubharmonic(fourier)
classical = ClassicalHill(fourier, "imaginary")


data = import_reference(
    f"examples/dissertation_bayer/compare_to_manlab/hinged_nmodes_{ode.n_modes}_N_40_tol_1e-14.csv",
    ode,
)
comptimes = [[], [], [], [], []]
for k, bp in enumerate(
    tqdm(
        iterate_from_reference(
            ode=ode,
            data=data,
            N_HBM=fourier.N_HBM,
            L_DFT=fourier.L_DFT,
            real_formulation=data["real_formulation"],
            stability_method=None,
        ),
        total=min(k_max, len(data["arclength"])),
    )
):
    for j, stabmethod in enumerate([dir, subh, classical]):
        start = time.perf_counter_ns()
        stabmethod.determine_eigenvalues(hbm=bp.equations[0])
        stop = time.perf_counter_ns()
        comptimes[j].append((stop - start) * 1e-9)

    hill_mat = bp.equations[0].hill_matrix()
    start = time.perf_counter_ns()
    np.linalg.eig(hill_mat)
    stop = time.perf_counter_ns()
    comptimes[j + 1].append((stop - start) * 1e-9)

    start = time.perf_counter_ns()
    expm(hill_mat)
    stop = time.perf_counter_ns()
    comptimes[j + 2].append((stop - start) * 1e-9)

    if k >= k_max:
        break

plt.figure()
for j, label in enumerate(
    ["Direct", "KoopmanHillSubharmonic", "ClassicalHill", "eig", "expm"]
):
    plt.plot(comptimes[j], label=label)
plt.legend()
plt.show()
