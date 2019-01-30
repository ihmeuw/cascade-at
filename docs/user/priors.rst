.. _prior-specification:

Priors
------

These are classes for the priors.

.. py:class:: cascade.model.priors._Prior

   All priors have these methods.

   .. function:: parameters()

      Returns a dictionary of all parameters for this prior,
      including the prior type as "density".

   .. function:: assign(parameter=value, parameter=value...)

      Creates a new Prior object with the same parameters
      as this Prior, except for the requested changes.


.. autoclass:: cascade.model.Uniform
   :members:

.. autoclass:: cascade.model.Constant
   :members:

.. autoclass:: cascade.model.Gaussian
   :members:

.. autoclass:: cascade.model.Laplace
   :members:

.. autoclass:: cascade.model.StudentsT
   :members:

.. autoclass:: cascade.model.LogGaussian
   :members:

.. autoclass:: cascade.model.LogLaplace
   :members:

.. autoclass:: cascade.model.LogStudentsT
   :members:
