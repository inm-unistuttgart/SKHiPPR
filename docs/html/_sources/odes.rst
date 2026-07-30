ODEs
=====

Ordinary differential equations are formalized as subclasses of :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation`. The :py:class:`~skhippr.odes.AbstractODE.AbstractODE` formalizes the syntax, and many concrete subclasses are provided.

.. autoclass:: skhippr.odes.AbstractODE.AbstractODE
    :members:

Nonautonomous ODEs
-----------------------
.. autoclass:: skhippr.odes.nonautonomous.Duffing

Autonomous ODEs
---------------------
.. autoclass:: skhippr.odes.autonomous.Vanderpol

.. autoclass:: skhippr.odes.autonomous.Truss

.. autoclass:: skhippr.odes.autonomous.BlockOnBelt

Linear time-periodic ODEs
------------------------------

To investigate the properties of the stability methods without influence of a periodic solution, the module ``skhippr.odes.ltp`` provides linear time-periodic ordinary differential equations with an equilibrium at zero. Hill stability methods are applicable by searching for the trivial periodic solution ``X = 0`` of the corresponding :py:class:`~skhippr.cycles.hbm.HBMEquation`. 

.. autoclass:: skhippr.odes.ltp.HillODE

.. autoclass:: skhippr.odes.ltp.HillLTI
    :members: fundamental_matrix

.. autoclass:: skhippr.odes.ltp.SmoothedMeissner
    :members: fundamental_matrix

.. autoclass:: skhippr.odes.ltp.MathieuODE

.. autoclass:: skhippr.odes.ltp.TruncatedMeissner

.. autoclass:: skhippr.odes.ltp.ShirleyODE