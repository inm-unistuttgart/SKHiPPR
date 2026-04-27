from skhippr.Fourier import Fourier


def init_csv(fourier: Fourier, writer, name_param: str):

    errors = [
        name_param,
        "arclength",
        "shoot error init",
        "shoot error max",
        "shoot error FMs",
    ]
    X_labels = [f"X 0, {l}" for l in range(fourier.n_dof)]

    if fourier.real_formulation:
        for sc in ["c", "s"]:
            for k in range(1, fourier.N_HBM + 1):
                X_labels = X_labels + [f"X{sc} {k}, {l}" for l in range(fourier.n_dof)]

    else:
        for k in range(1, fourier.N_HBM + 1):
            X_labels = X_labels + [f"X {k}, {l}" for l in range(fourier.n_dof)]
            X_labels = [f"X {-k}, {l}" for l in range(fourier.n_dof)] + X_labels

    FM_labels = [f"FM {k}" for k in range(fourier.n_dof)]

    writer.writerow(errors + FM_labels + X_labels)
    return errors + FM_labels + X_labels


def to_csv(
    writer,
    hbm: HBMEquation,
    name_param,
    arclength,
    solver,
    FM_error_measure=None,
    **kwargs_odesolver,
):
    _, FMs = hbm.determine_stability(update=True)
    param = getattr(hbm, name_param)
    X = hbm.X
    errors, _ = determine_ode_accuracy(
        hbm,
        solver,
        visualize=False,
        FM_error_measure=FM_error_measure,
        **kwargs_odesolver,
    )
    # errors = (0, 0, 0)
    row = np.hstack(
        (np.atleast_1d(param), np.atleast_1d(arclength), np.atleast_1d(errors), FMs, X),
        dtype=complex,
    )

    writer.writerow([str(complex(val)).strip("()") for val in row])
