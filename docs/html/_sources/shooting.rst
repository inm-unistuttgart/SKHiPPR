Addendum: Shooting method
=========================

As a reference to compare/check against the HBM methods which are the focus of the toolbox, the shooting method is also implemented as a subclass of :py:class:`~skhippr.cycles.AbstractCycleEquation.AbstractCycleEquation`. Similarly as with the :py:class:`~skhippr.cycles.hbm.HBMEquation` and the :py:class:`~skhippr.cycles.hbm.HBMSystem`, there also exists a corresponding :py:class:`~skhippr.cycles.shooting.ShootingSystem` class which may have a :py:class:`~skhippr.cycles.shooting.ShootingPhaseAnchor`.

.. autoclass:: skhippr.cycles.shooting.ShootingBVP
    :members:

.. autoclass:: skhippr.cycles.shooting.ShootingSystem
    :members:

.. autoclass:: skhippr.cycles.shooting.ShootingPhaseAnchor
    :members: