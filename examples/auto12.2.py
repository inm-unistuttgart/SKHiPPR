"""
Notes: the goal is to implement the autonomous ODE System as defined in equation 12.2 
in the auto07p document. 

(12.2)  u_1_dot = -u_1 + p_1*(1-u_1)e**u_2
        u_2_dot = -u_2 + p_1*p_2*(1-u_1)e**u_2 - p_3*u_2

TODO: check equilibriums, find periodic solutions by varying parameters(autonomous?) and check stability.
"""

import numpy as np
import matplotlib.pyplot as plt
from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMSystem
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.equations.Equilibrium12_2 import Equilibrium12_2
from skhippr.solvers.newton import NewtonSolver
from skhippr.odes.ode12_2 import ODE12_2
from skhippr.stability.ClassicalHill import ClassicalHill

def determine_equilibrium(ode):
    equation_system = EquationSystem(
        equations = [ode],
        unknowns =['x'],
        equation_determining_stability = ode, # schatzsuche in der doku, stab dazu plotten..
    )
    newton_solver = NewtonSolver(verbose=True)
    newton_solver.solve(equation_system)
    if not equation_system.solved:
        raise RuntimeError("No Equilibrium was found")
    stable, eigs = equation_system.determine_stability()
    print("Equilibrium stabiliy: ", stable, eigs)
    u1s, u2s = equation_system.x[0], equation_system.x[1]
    return u1s,u2s

def create_HBMSystem(ode, x0, fourier, stability_method):
    omega_start = 1.0
    x_sample_init = np.tile(x0[:, None], (1,fourier.L_DFT))
    X0 = fourier.DFT(x_sample_init)
    hbm_system = HBMSystem(
        ode = ode,
        omega = omega_start,
        fourier = fourier,
        initial_guess = X0,
        period_k = 1.0,
        stability_method = stability_method
    )
    return hbm_system

def plot_results(hbm):
    x_time = hbm.x_time()
    ts = hbm.fourier.time_samples(hbm.omega_solution)
    u1_time = x_time[0, :]
    u2_time = x_time[1, :]
    print(u1_time[-1])
    plt.figure()
    plt.plot(ts, u1_time, label="u1")
    plt.plot(ts, u2_time, label="u2")
    plt.xlabel("t")
    plt.legend()
    plt.grid(True)
    plt.figure()
    plt.plot(u1_time, u2_time)
    plt.xlabel("u1")
    plt.ylabel("u2")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def plot_floquet(hbm, hill_method):
    hill_method.hill_EVP(hbm = hbm, visualize = True)   # returns floquet exponents
    t = np.linspace(0, 2*np.pi, 400)
    x = np.cos(t)
    y = np.sin(t)
    plt.figure()
    plt.plot(x, y, label = "Unit Circle")
    plt.title("FLoquet Multipliers")
    plt.gca().set_aspect('equal', adjustable='box')
    stable, eigs = hbm.determine_stability(update= True) # returns floquet multipliers
    # print(np.abs(eigs[0]))
    print("HB solution stability: ", stable, eigs)
    plt.plot(np.real(eigs), np.imag(eigs), "+", label="Multipliers")
    plt.legend()
    plt.grid(True)
    plt.ylabel("Imag part")
    plt.xlabel("Real part")


def main():
    # set parameters
    p1, p2, p3 = 1.0, 2.0, 0.5
    u10, u20 = 0.0, 0.0
    x0 = np.array([u10,u20])

    # setup a fourier system, the classical Hill stability method as well as an instance of the ode
    fourier = Fourier(n_dof=2, N_HBM=50, L_DFT=102, real_formulation = True)
    hill_method = ClassicalHill(fourier = fourier, sorting_method = "imaginary", tol = 1e-8, autonomous = True)
    ode = ODE12_2(p1=p1, p2=p2, p3=p3, x0=x0, autonomous=True)

    # determine the equilibrium/stationary state of the ode
    u1s, u2s = determine_equilibrium(ode)
    print(u1s, u2s)
    x0 = np.array([u1s, u2s])

    # setup for the HB-method as well as solving with Newton-method
    hbm_system = create_HBMSystem(ode, x0, fourier, hill_method)
    newton_solver = NewtonSolver(verbose=True)
    newton_solver.solve(hbm_system)
    if not hbm_system.solved:
        raise RuntimeError("HBM Newton didnt converge")
    hbm = hbm_system.equations[0]
    # print(hill_method.hill_EVP(hbm = hbm, visualize = True))
    plot_floquet(hbm, hill_method)
    plot_results(hbm)


if __name__ == "__main__":
    main()