Visualization
================

The following modules offer SKHiPPR internal methods for visualizing equilibria, periodic solutions of solved equations and continuation branches.

Cycle and equilibria visualization functions accept either a certain :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` object (depending on the visualization function) or an :py:class:`~skhippr.equations.EquationSystem.EquationSystem` containing one as the first argument.
When an :py:class:`~skhippr.equations.EquationSystem.EquationSystem` is passed, the first applicable :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` within it is visualized.

Calling these visualization functions directly with only an :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` or :py:class:`~skhippr.equations.EquationSystem.EquationSystem` as input creates and
returns a :py:class:`matplotlib.axes.Axes` object for each plot. 
Alternatively, passing an existing :py:class:`~matplotlib.axes.Axes` object uses that axes for plotting and returns it. 

The continuation visualization functions require a collection of :py:class:`skhippr.solvers.continuation.BranchPoint` objects, which can be generated with SKHiPPR continuation functions.

The supported equations and equation systems are indicated in the documentation of each respective class.
Standard plotting variables like ``title``, ``xlabel`` and ``ylabel`` may be passed to each visualization function as a keyword argument.

All plots created with the provided functions including 2D-plots, 3D-plots and animations can be exported with functions provided in the :py:mod:`~skhippr.visualization.data_export` module.

.. contents::

visualization.equilibria
------------------------

.. automodule:: skhippr.visualization.equilibria
    :members:

visualization.cycles
----------------------------

.. automodule:: skhippr.visualization.cycles
    :members:

visualization.continuation
----------------------------

.. automodule:: skhippr.visualization.continuation
    :members:

visualization.data_export
--------------------------

.. automodule:: skhippr.visualization.data_export
    :members: 