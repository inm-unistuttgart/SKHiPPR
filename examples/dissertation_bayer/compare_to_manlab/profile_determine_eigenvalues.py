import cProfile
import io
import pstats
import time

from analyze_and_plot import iterate_from_reference, import_reference
from analyze_hinged import init_hinged

from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillProjection,
    KoopmanHillSubharmonic,
)
from skhippr.stability.ClassicalHill import ClassicalHill
from skhippr.Fourier import Fourier


def get_first_branch_equations(ode, fourier, data, n_points):
    equations = []
    iterator = iterate_from_reference(
        ode=ode,
        data=data,
        N_HBM=fourier.N_HBM,
        L_DFT=fourier.L_DFT,
        real_formulation=data["real_formulation"],
        stability_method=None,
    )

    for k, bp in enumerate(iterator):
        if k >= n_points:
            break
        equations.append(bp.equations[0])
    return equations


def profile_method(name, method, equations, top_n=20):
    # Warm-up call to reduce one-time overhead from first invocation.
    method.determine_eigenvalues(hbm=equations[0])

    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.enable()
    for equation in equations:
        method.determine_eigenvalues(hbm=equation)
    profile.disable()
    stop = time.perf_counter()

    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream).sort_stats("cumulative")
    stats.print_stats(top_n)

    print("=" * 24)
    print(f"Method: {name}")
    print(f"Branch points: {len(equations)}")
    print(f"Total wall time [s]: {stop - start:.6f}")
    print("=" * 24)
    print(stream.getvalue())


def main():
    ode = init_hinged(n_modes=10, omega_0_normalized=0.1)
    fourier = Fourier(n_dof=ode.n_dof, N_HBM=10, L_DFT=1024, real_formulation=True)

    data = import_reference(
        "examples/dissertation_bayer/compare_to_manlab/hinged_nmodes_10_N_40_tol_1e-14.csv",
        ode,
    )
    equations = get_first_branch_equations(
        ode=ode, fourier=fourier, data=data, n_points=20
    )

    methods = [
        ("KoopmanHillProjection", KoopmanHillProjection(fourier)),
        ("KoopmanHillSubharmonic", KoopmanHillSubharmonic(fourier)),
        ("ClassicalHill(imaginary)", ClassicalHill(fourier, "imaginary")),
    ]

    for name, method in methods:
        profile_method(name=name, method=method, equations=equations, top_n=20)


if __name__ == "__main__":
    main()
