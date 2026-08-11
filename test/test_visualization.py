"""Tests for the plotting/animation functions in :py:mod:`skhippr.visualization.cycles`.

The functions under test only ever read already-computed data from a solved
:py:class:`~skhippr.cycles.hbm.HBMEquation` (``X``, ``fourier``, ``omega_solution``,
``eigenvalues``, ``T_solution``). None of that requires an actual Newton solve, so the
fixtures below build a cheap, deterministic ``HBMEquation``/``HBMSystem`` "as if" it were
solved (an arbitrary but consistent initial guess, plus manually assigned Floquet
multipliers) instead of paying for a real solve in every test.

Three behaviors are exercised for (almost) every visualization function:

* Either a :py:class:`~skhippr.cycles.hbm.HBMEquation` or an
  :py:class:`~skhippr.equations.EquationSystem.EquationSystem` (e.g. a
  :py:class:`~skhippr.cycles.hbm.HBMSystem`) may be passed.
* If ``ax=None``, a new axis is created (and auto-titled/labelled); if an existing axis is
  passed, it is reused as-is (and never auto-titled/labelled).
* Keyword arguments that are valid properties of the underlying artist are applied;
  keyword arguments that are not valid are dropped with a ``UserWarning`` instead of
  raising, while any other, valid keyword arguments passed alongside them still take
  effect.
"""

import warnings

import numpy as np
import pytest
import matplotlib.pyplot as plt

from skhippr.Fourier import Fourier
from skhippr.odes.nonautonomous import Duffing
from skhippr.cycles.hbm import HBMEquation, HBMSystem

from skhippr.visualization.cycles import (
    plot_period,
    animate_period,
    plot_phase,
    animate_phase,
    plot_floquet_multipliers,
    animate_floquet_multipliers,
    plot_floquet_exponents,
    animate_floquet_exponents,
    plot_hill_matrix_blocks,
    plot_matrix_block_norm,
)

# --------------------------------------------------------------------------- #
# Helpers and fixtures
# --------------------------------------------------------------------------- #

# Small Fourier discretization: plotting tests only care about the shape/type
# of the resulting artists, not numerical accuracy, so keep this cheap.
_N_HBM = 3
_L_DFT = 32

# Fixed, deterministic "Floquet multipliers" (never actually computed by a
# stability method) - one inside, one outside and a complex-conjugate pair
# straddling the unit circle, which is enough to exercise the plotting code.
_EIGENVALUES = np.array([0.9, -1.3, 0.2 + 0.5j, 0.2 - 0.5j])


def _make_fourier():
    return Fourier(N_HBM=_N_HBM, L_DFT=_L_DFT, n_dof=2, real_formulation=True)


def _make_ode(omega):
    return Duffing(t=0, x=np.array([1.0, 0.0]), alpha=1, beta=3, F=1, delta=1, omega=omega)


def _initial_guess(fourier, omega):
    ts = fourier.time_samples(omega)
    x0_samples = np.array([np.cos(omega * ts), -omega * np.sin(omega * ts)])
    return fourier.DFT(x0_samples)


def _make_hbm_equation(omega=1.3):
    """A cheap, deterministic HBMEquation that behaves like a solved one for plotting."""
    fourier = _make_fourier()
    ode = _make_ode(omega)
    equation = HBMEquation(
        ode=ode,
        omega=omega,
        fourier=fourier,
        initial_guess=_initial_guess(fourier, omega),
        stability_method=None,
    )
    equation.eigenvalues = _EIGENVALUES
    return equation


def _make_hbm_system(omega=1.3):
    """A HBMSystem wrapping an equation built the same way as `_make_hbm_equation`."""
    fourier = _make_fourier()
    ode = _make_ode(omega)
    system = HBMSystem(
        ode=ode,
        omega=omega,
        fourier=fourier,
        initial_guess=_initial_guess(fourier, omega),
        stability_method=None,
    )
    system.equations[0].eigenvalues = _EIGENVALUES
    return system


def _make_hbm(kind, omega=1.3):
    if kind == "equation":
        return _make_hbm_equation(omega)
    elif kind == "system":
        return _make_hbm_system(omega)
    else:
        raise ValueError(kind)


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    """Avoid letting figures pile up across the many small tests in this module."""
    yield
    plt.close("all")


@pytest.fixture(params=["equation", "system"])
def hbm_kind(request):
    """Parametrizes tests over both supported argument types."""
    return request.param


@pytest.fixture
def hbm_arg(hbm_kind):
    """A single HBMEquation or HBMSystem, ready to be passed to a plot_* function."""
    return _make_hbm(hbm_kind)


@pytest.fixture
def hbm_set(hbm_kind):
    """A small list of HBMEquation/HBMSystem objects, for the animate_* functions."""
    return [_make_hbm(hbm_kind, omega=omega) for omega in (1.1, 1.3, 1.5)]


@pytest.fixture
def hbm_system_set_for_animate_period():
    """
    `animate_period` (unlike the other animate_* functions) reaches directly into
    `hbm_set[0].equations[-1].continuation_parameter`, so its items must be
    EquationSystem-like (a bare HBMEquation has no `.equations` attribute). This
    fixture builds the HBMSystem set it actually supports and sets a valid
    `continuation_parameter` so the per-frame title can be formatted.
    """
    systems = [_make_hbm_system(omega=omega) for omega in (1.1, 1.3, 1.5)]
    for system in systems:
        system.equations[0].continuation_parameter = "omega"
    return systems


# Functions that add a Line2D to `ax.lines` and forward plot_kwargs to `ax.plot()`.
LINE_PLOT_FUNCS = [plot_period, plot_phase]

# Functions that add a PathCollection to `ax.collections` and forward plot_kwargs to
# `ax.scatter()`.
SCATTER_PLOT_FUNCS = [plot_floquet_multipliers, plot_floquet_exponents]

ALL_PLOT_FUNCS = LINE_PLOT_FUNCS + SCATTER_PLOT_FUNCS


def _ids(funcs):
    return [f.__name__ for f in funcs]


# --------------------------------------------------------------------------- #
# 1. Either a HBMEquation or a HBMSystem (EquationSystem) may be passed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("plot_func", ALL_PLOT_FUNCS, ids=_ids(ALL_PLOT_FUNCS))
def test_plot_accepts_hbm_equation_or_hbm_system(plot_func, hbm_arg):
    """Every plot_* function under test must accept both supported argument types."""
    ax = plot_func(hbm_arg)
    assert isinstance(ax, plt.Axes)


def test_hbm_system_and_its_hbm_equation_produce_the_same_plot():
    """Passing a HBMSystem must be equivalent to passing the HBMEquation it wraps."""
    equation = _make_hbm_equation()
    system = _make_hbm_system()
    # Make the system's equation numerically identical to the standalone one.
    system.equations[0].X = equation.X

    ax_from_equation = plot_phase(equation)
    ax_from_system = plot_phase(system)

    np.testing.assert_allclose(
        ax_from_equation.lines[0].get_xdata(), ax_from_system.lines[0].get_xdata()
    )
    np.testing.assert_allclose(
        ax_from_equation.lines[0].get_ydata(), ax_from_system.lines[0].get_ydata()
    )


@pytest.mark.parametrize("plot_func", ALL_PLOT_FUNCS, ids=_ids(ALL_PLOT_FUNCS))
def test_plot_rejects_unrelated_object(plot_func):
    """Anything that is neither a HBMEquation nor an EquationSystem must be rejected."""
    with pytest.raises(ValueError):
        plot_func(object())


@pytest.mark.parametrize(
    "animate_func", [animate_phase, animate_floquet_multipliers, animate_floquet_exponents]
)
def test_animate_accepts_hbm_equation_or_hbm_system_set(animate_func, hbm_set):
    """The animate_* functions (besides animate_period) accept either item type too."""
    ax, animation = animate_func(hbm_set)
    assert isinstance(ax, plt.Axes)


def test_animate_period_accepts_hbm_system_set(hbm_system_set_for_animate_period):
    # animate_period specifically requires EquationSystem-like items (see the
    # `hbm_system_set_for_animate_period` fixture docstring for why).
    ax, animation = animate_period(hbm_system_set_for_animate_period)
    assert isinstance(ax, plt.Axes)


# --------------------------------------------------------------------------- #
# 2. ax=None creates a new axis; a passed-in axis is reused as-is
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("plot_func", ALL_PLOT_FUNCS, ids=_ids(ALL_PLOT_FUNCS))
def test_ax_none_creates_and_auto_titles_a_new_axis(plot_func, hbm_arg):
    n_figures_before = len(plt.get_fignums())

    ax = plot_func(hbm_arg)

    assert isinstance(ax, plt.Axes)
    assert len(plt.get_fignums()) == n_figures_before + 1
    # Auto-titling/labelling only happens for an axis the function created itself.
    assert ax.get_title() != ""
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""


@pytest.mark.parametrize("plot_func", ALL_PLOT_FUNCS, ids=_ids(ALL_PLOT_FUNCS))
def test_existing_ax_is_reused_and_left_untouched(plot_func, hbm_arg):
    fig, ax = plt.subplots()
    ax.set_title("pre-existing title")
    ax.set_xlabel("pre-existing xlabel")
    n_figures_before = len(plt.get_fignums())

    returned_ax = plot_func(hbm_arg, ax=ax)

    assert returned_ax is ax
    assert len(plt.get_fignums()) == n_figures_before  # no extra figure created
    # The function must not override an axis that was handed to it.
    assert ax.get_title() == "pre-existing title"
    assert ax.get_xlabel() == "pre-existing xlabel"


@pytest.mark.parametrize(
    "animate_func", [animate_phase, animate_floquet_multipliers, animate_floquet_exponents]
)
def test_animate_ax_none_creates_new_axis(animate_func, hbm_set):
    ax, animation = animate_func(hbm_set)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() != ""


@pytest.mark.parametrize(
    "animate_func", [animate_phase, animate_floquet_multipliers, animate_floquet_exponents]
)
def test_animate_existing_ax_is_reused(animate_func, hbm_set):
    fig, ax = plt.subplots()
    ax.set_title("pre-existing title")

    returned_ax, animation = animate_func(hbm_set, ax=ax)

    assert returned_ax is ax
    assert ax.get_title() == "pre-existing title"


# --------------------------------------------------------------------------- #
# 3. Keyword arguments: valid ones are used, invalid ones warn and are dropped
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("plot_func", LINE_PLOT_FUNCS, ids=_ids(LINE_PLOT_FUNCS))
def test_valid_kwarg_is_applied_line_plot(plot_func, hbm_arg):
    ax = plot_func(hbm_arg, label="my label", linestyle="--")
    line = ax.lines[-1]
    assert line.get_label() == "my label"
    assert line.get_linestyle() == "--"


@pytest.mark.parametrize("plot_func", SCATTER_PLOT_FUNCS, ids=_ids(SCATTER_PLOT_FUNCS))
def test_valid_kwarg_is_applied_scatter_plot(plot_func, hbm_arg):
    ax = plot_func(hbm_arg, label="my label")
    collection = ax.collections[-1]
    assert collection.get_label() == "my label"


@pytest.mark.parametrize("plot_func", ALL_PLOT_FUNCS, ids=_ids(ALL_PLOT_FUNCS))
def test_invalid_kwarg_warns_and_is_ignored(plot_func, hbm_arg):
    with pytest.warns(UserWarning, match="not_a_real_kwarg"):
        ax = plot_func(hbm_arg, not_a_real_kwarg="nonsense")
    # The plot must still have been produced despite the invalid keyword argument.
    # (plot_floquet_multipliers/exponents also draw an extra decorative line -
    # a unit circle resp. a zero axvline - when ax is auto-generated, so this only
    # checks that at least the main artist is present, not an exact count.)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) + len(ax.collections) >= 1


@pytest.mark.parametrize("plot_func", LINE_PLOT_FUNCS, ids=_ids(LINE_PLOT_FUNCS))
def test_invalid_kwarg_does_not_suppress_valid_sibling_kwargs_line_plot(plot_func, hbm_arg):
    """A single bad keyword argument must not prevent the *other* valid ones from applying."""
    with pytest.warns(UserWarning, match="not_a_real_kwarg"):
        ax = plot_func(hbm_arg, label="keep me", not_a_real_kwarg="nonsense")
    assert ax.lines[-1].get_label() == "keep me"


@pytest.mark.parametrize("plot_func", SCATTER_PLOT_FUNCS, ids=_ids(SCATTER_PLOT_FUNCS))
def test_invalid_kwarg_does_not_suppress_valid_sibling_kwargs_scatter_plot(
    plot_func, hbm_arg
):
    with pytest.warns(UserWarning, match="not_a_real_kwarg"):
        ax = plot_func(hbm_arg, label="keep me", not_a_real_kwarg="nonsense")
    assert ax.collections[-1].get_label() == "keep me"


@pytest.mark.parametrize("plot_func", ALL_PLOT_FUNCS, ids=_ids(ALL_PLOT_FUNCS))
def test_title_kwarg_applied_only_when_ax_is_generated(plot_func, hbm_arg):
    # ax=None: the special "title" keyword argument fits and is used.
    ax_generated = plot_func(hbm_arg, title="Custom title")
    assert ax_generated.get_title() == "Custom title"

    # ax=<existing>: "title" is recognized (consumed, so it does not reach
    # ax.plot()/ax.scatter() and therefore never raises or warns) but is not
    # applied to an axis that was handed in, so it is effectively ignored.
    fig, ax_existing = plt.subplots()
    ax_existing.set_title("original title")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        returned_ax = plot_func(hbm_arg, ax=ax_existing, title="Custom title")
    assert returned_ax.get_title() == "original title"


# --------------------------------------------------------------------------- #
# animate_* keyword argument handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "animate_func,artist_container",
    [
        (animate_phase, "lines"),
        (animate_floquet_multipliers, "collections"),
        (animate_floquet_exponents, "collections"),
    ],
    ids=["animate_phase", "animate_floquet_multipliers", "animate_floquet_exponents"],
)
def test_animate_valid_kwarg_is_applied(animate_func, artist_container, hbm_set):
    ax, animation = animate_func(hbm_set, label="my label")
    artist = getattr(ax, artist_container)[-1]
    assert artist.get_label() == "my label"


@pytest.mark.parametrize(
    "animate_func",
    [animate_phase, animate_floquet_multipliers, animate_floquet_exponents],
    ids=["animate_phase", "animate_floquet_multipliers", "animate_floquet_exponents"],
)
def test_animate_invalid_kwarg_warns_and_is_ignored(animate_func, hbm_set):
    with pytest.warns(UserWarning, match="not_a_real_kwarg"):
        ax, animation = animate_func(hbm_set, not_a_real_kwarg="nonsense")
    assert isinstance(ax, plt.Axes)


@pytest.mark.parametrize(
    "animate_func",
    [animate_phase, animate_floquet_multipliers, animate_floquet_exponents],
    ids=["animate_phase", "animate_floquet_multipliers", "animate_floquet_exponents"],
)
def test_animate_frame_update_does_not_raise(animate_func, hbm_set):
    """
    Regression test: animate_phase/animate_floquet_multipliers/animate_floquet_exponents
    used to reference an undefined `scaling` variable while updating a frame (or, for
    animate_phase, even before the first frame). Actually advancing a frame here would
    have raised a NameError before that was fixed.
    """
    ax, animation = animate_func(hbm_set)
    for frame_idx in range(len(hbm_set)):
        animation._draw_frame(frame_idx)


def test_animate_period_frame_update_does_not_raise(hbm_system_set_for_animate_period):
    ax, animation = animate_period(hbm_system_set_for_animate_period)
    for frame_idx in range(len(hbm_system_set_for_animate_period)):
        animation._draw_frame(frame_idx)


# --------------------------------------------------------------------------- #
# plot_hill_matrix_blocks / plot_matrix_block_norm
# --------------------------------------------------------------------------- #


def test_plot_hill_matrix_blocks_accepts_hbm_equation_or_hbm_system(hbm_arg):
    ax = plot_hill_matrix_blocks(hbm_arg)
    assert isinstance(ax, plt.Axes)
    assert len(ax.collections) == 1


def test_plot_hill_matrix_blocks_ax_none_vs_existing():
    equation = _make_hbm_equation()

    ax_generated = plot_hill_matrix_blocks(equation)
    assert ax_generated.get_title() != ""

    fig, ax_existing = plt.subplots()
    ax_existing.set_title("kept")
    returned_ax = plot_hill_matrix_blocks(equation, ax=ax_existing)
    assert returned_ax is ax_existing


def test_plot_hill_matrix_blocks_invalid_kwarg_warns_and_is_ignored():
    equation = _make_hbm_equation()
    with pytest.warns(UserWarning, match="not_a_real_kwarg"):
        ax = plot_hill_matrix_blocks(equation, not_a_real_kwarg="nonsense")
    assert isinstance(ax, plt.Axes)


@pytest.fixture
def random_square_matrix():
    rng = np.random.default_rng(0)
    block_size = 2
    n_blocks = 4
    return rng.random((block_size * n_blocks, block_size * n_blocks)), block_size


def test_plot_matrix_block_norm_ax_none_creates_new_axis(random_square_matrix):
    matrix, block_size = random_square_matrix
    ax, sc = plot_matrix_block_norm(matrix, block_size)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() != ""


def test_plot_matrix_block_norm_existing_ax_is_reused(random_square_matrix):
    matrix, block_size = random_square_matrix
    fig, ax = plt.subplots()

    returned_ax, sc = plot_matrix_block_norm(matrix, block_size, ax=ax)

    assert returned_ax is ax


def test_plot_matrix_block_norm_valid_kwarg_is_applied(random_square_matrix):
    matrix, block_size = random_square_matrix
    ax, sc = plot_matrix_block_norm(matrix, block_size, label="my label")
    assert sc.get_label() == "my label"


def test_plot_matrix_block_norm_invalid_kwarg_warns_and_is_ignored(random_square_matrix):
    matrix, block_size = random_square_matrix
    with pytest.warns(UserWarning, match="not_a_real_kwarg"):
        ax, sc = plot_matrix_block_norm(matrix, block_size, not_a_real_kwarg="nonsense")
    assert isinstance(ax, plt.Axes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
