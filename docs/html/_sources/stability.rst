Stability methods
=================

.. contents::  
    :depth: 2

The stability methods are implemented as subclasses of abstract :py:class:`~skhippr.stability.AbstractStabilityMethod.AbstractStabilityMethod`. Each such class defines a method which returns stability-defining eigenvalues from the derivative information of a given :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation`, and a method to assert stability based on these eigenvalues.  

Abstract parent class
----------------------------------------------------------------------------------------------------------------------------------------------------------------------

.. autoclass:: skhippr.stability.AbstractStabilityMethod.AbstractStabilityMethod
    :members:

Stability of equilibria 
-----------------------------------------------------------------------------------------------

.. autoclass:: skhippr.stability.AbstractStabilityMethod.StabilityEquilibrium
    :members:


Abstract parent class for Hill matrix methods
-------------------------------------------------------------------------------------------------------------

.. autoclass:: skhippr.stability.AbstractStabilityHBM.AbstractStabilityHBM
    :members:

Koopman-Hill projection methods
-----------------------------------

.. autoclass:: skhippr.stability.KoopmanHillProjection.KoopmanHillProjection
    :members:

.. autoclass:: skhippr.stability.KoopmanHillProjection.KoopmanHillSubharmonic
    :members:

Classical sorting-based Hill methods
------------------------------------

.. autoclass:: skhippr.stability.ClassicalHill.ClassicalHill
    :members:

Explicit Runge Kutta single-pass time integration method
---------------------------------------------------------

.. autoclass:: skhippr.stability.SinglePass.SinglePassRK
    :members:

.. autoclass:: skhippr.stability.SinglePass.SinglePassRK4
    :members:

.. autoclass:: skhippr.stability.SinglePass.SinglePassRK38
    :members:





