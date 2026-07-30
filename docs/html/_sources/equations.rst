Formalizing equations and equation systems
=================================================================

.. contents::

Algebraic equations
--------------------

.. autoclass:: skhippr.equations.AbstractEquation.AbstractEquation
    :members:

Equation systems
-----------------

Equation systems collect a set of :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` objects, together with their unknowns. Notable subclasses of :py:class:`~skhippr.equations.EquationSystem.EquationSystem` are documented elsewhere:

    * :py:class:`~skhippr.solvers.continuation.BranchPoint`
    * :py:class:`~skhippr.cycles.hbm.HBMSystem`
    * :py:class:`~skhippr.cycles.shooting.ShootingSystem`

.. autoclass:: skhippr.equations.EquationSystem.EquationSystem
    :members:

Concrete subclasses of :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation`
-----------------------------------------------------------------------------------------

The following subclasses of :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` are documented elsewhere:

    * :py:class:`~skhippr.odes.AbstractODE.AbstractODE`,
    * :py:class:`~skhippr.solvers.continuation.ContinuationAnchor`, 
    * :py:class:`~skhippr.cycles.AbstractCycleEquation.AbstractCycleEquation`,
    * :py:class:`~skhippr.cycles.hbm.HBMEquation` and :py:class:`~skhippr.cycles.hbm.HBMPhaseAnchor`,
    * :py:class:`~skhippr.cycles.shooting.ShootingBVP` and :py:class:`~skhippr.cycles.shooting.ShootingPhaseAnchor`

Two concrete subclass for demonstration purposes encode radius and angle conditions on a circle.

.. autoclass:: skhippr.equations.Circle.CircleEquation

.. autoclass:: skhippr.equations.Circle.AngleEquation