import numpy as np

from skhippr.odes.daes import FrictionOscillator, SmoothedFrictionOscillator


def init_oscillator(name_case="A", smoothing=np.inf):
    """Caution: Oscillator is excited with sin, not cos! --> phase must be pi/2 + phase_cos."""
    masses = [1, 1]
    stiffnesses = [1, 1]
    dampings = [0.02, 0.02]
    omega = 0.299
    phases = [0.5 * np.pi, 0]
    mu = 0.9
    prox_parameter = 1

    match name_case:
        case "A":
            omega = 0.618
            phases = [0.5 * np.pi, 0]
            forcings = [20, 0]
            normal_force = 8
            g = normal_force / masses[1]
        case "B":
            omega = 0.293
            phases = [0.5 * np.pi, 0]
            forcings = [20, 0]
            normal_force = 8
            g = normal_force / masses[1]
        case "C":
            omega = 0.299
            phases = [0.5 * np.pi, 0]
            forcings = [20, 0]
            normal_force = 10.5
            g = normal_force / masses[1]
        case "D":
            omega = 0.308
            phases = [0.5 * np.pi, 0]
            forcings = [20, 0]
            normal_force = 10.5
            g = normal_force / masses[1]

        case "Schuetz1":
            g = 10
            omega = 2 * np.pi
            forcings = [20, 50]
            phases = [-0.5 * np.pi, np.pi]  # TODO passt das?? sin vs cos
            mu = 4

        case "Schuetz2":
            g = 10
            omega = 2 * np.pi
            forcings = [20, 10]
            phases = [0.4398, 2.0106]  # TODO passt das?? sin vs cos
            prox_parameter = 10

        case _:
            raise ValueError(f"Case {name_case} not defined!")

    if smoothing == np.inf:
        dae = FrictionOscillator(
            stiffnesses=stiffnesses,
            dampings=dampings,
            masses=masses,
            g=g,
            mu=mu,
            forcing_amplitudes=forcings,
            forcing_phases=phases,
            prox_parameter=prox_parameter,
        )
    else:
        dae = SmoothedFrictionOscillator(
            stiffnesses=stiffnesses,
            dampings=dampings,
            masses=masses,
            g=g,
            mu=mu,
            forcing_amplitudes=forcings,
            forcing_phases=phases,
            smoothing=smoothing,
        )

    dae.omega = omega
    return dae
