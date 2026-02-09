Visualization
================

The following modules offer SKHiPPR internal methods for visualizing equilibria, periodic solutions of solved equations and continuation branches.

Cycle and equilibria visualization functions accept as first argument either a certain :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` object (depending on the visualization function) or an :py:class:`~skhippr.equations.EquationSystem.EquationSystem` containing one. When an :py:class:`~skhippr.equations.EquationSystem.EquationSystem` is passed, the first applicable :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` within it is visualized.

Calling these visualization functions directly with only an :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` or :py:class:`~skhippr.equations.EquationSystem.EquationSystem` as input creates and
returns a :py:class:`matplotlib.axes.Axes` object for each plot. 
Alternatively, passing an existing :py:class:`~matplotlib.axes.Axes` object uses that axes for plotting and returns it. 

The continuation visualization functions require a collection of :py:class:`skhippr.solvers.continuation.BranchPoint` objects, which can be generated with SKHiPPR continuation functions.

The supported equations and equation systems are indicated in the documentation of each respective class.


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