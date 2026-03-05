import numpy as np
import time

from skhippr.solvers.continuation import BranchPoint
from skhippr.solvers.newton import NewtonSolver

from import_reference import change_N_HBM


def measure_time_to_hill(bp: BranchPoint, X_ext_ref, X_ext_prev, solver: NewtonSolver):

    # extract Hill matrix from bp
    start = time.monotonic_ns()
    hill_mat_last = bp.equations[0].hill_matrix(update=False)
    stop = time.monotonic_ns()
    time_hill_read = (stop - start) * 1e-9

    start = time.monotonic_ns()
    hill_mat_comp = bp.equations[0].hill_matrix(update=True)
    stop = time.monotonic_ns()
    time_hill_comp = (stop - start) * 1e-9

    # extract Hill matrix at ref, avoid modifying original bp
    bp = bp.duplicate()

    X_ref = change_N_HBM(X_ext_ref[:-1], bp.equations[0].fourier)
    param_ref = np.real(X_ext_ref[-1])

    bp.X = X_ref
    setattr(bp, bp.unknowns[-1], param_ref)
    start = time.monotonic_ns()
    hill_mat_ref = bp.equations[0].hill_matrix(update=True)
    stop = time.monotonic_ns()
    time_hill_ref = (stop - start) * 1e-9

    # analyze tangential prediction and correction steps
    X_prev = change_N_HBM(X_ext_prev[:-1], bp.equations[0].fourier)
    param_prev = np.real(X_ext_prev[-1])

    bp.X = X_prev
    setattr(bp, bp.unknowns[-1], param_prev)
    solver.solve(bp)

    X_ext_ref = np.hstack((X_ref, param_ref))
    X_ext_prev = np.hstack((X_prev, param_prev))

    if bp.solved:
        # Determine tangent at previous point
        start = time.monotonic_ns()
        bp.determine_tangent(update=True)
        stop = time.monotonic_ns()
        time_tangent = (stop - start) * 1e-9

        # determine step size such that corrections are orthogonal to next point
        stepsize = np.inner(bp.tangent, X_ext_ref - X_ext_prev)
        bp_next = bp.predict(stepsize)

        # Solve and time
        start = time.monotonic_ns()
        solver.solve(bp_next)
        stop = time.monotonic_ns()
        time_corr = (stop - start) * 1e-9
        num_iter = solver.num_iter - 1

        # error measures
        distance_to_ref = np.linalg.norm(bp_next.X - X_ref)
        residual = np.linalg.norm(bp.residual_function(update=True))
        solved = bp.solved
    else:
        time_tangent = np.nan
        time_corr = np.nan
        solved = False
        residual = np.inf
        distance_to_ref = np.nan
        num_iter = np.inf

    ########### Results ###################

    times = {
        "tangent": time_tangent,
        "corr": time_corr,
        "hill_read": time_hill_read,
        "hill_comp": time_hill_comp,
        "hill_ref": time_hill_ref,
        "total": time_tangent + time_corr + time_hill_read,
    }

    hill_matrices = {"ref": hill_mat_ref, "comp": hill_mat_comp, "last": hill_mat_last}

    other = {
        "num_iter": num_iter,
        "distance_to_ref": distance_to_ref,
        "residual": residual,
        "solved": solved,
    }
    return hill_matrices, times, other
