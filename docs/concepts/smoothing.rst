
.. _dismod-smoothing:

Smoothing Continuous Functions
------------------------------

We said that rates and :ref:`covariate multipliers <covariates>` are continuous functions of age and time.
It takes a little work to parametrize an interpolated function of age and time.

 * You have to tell it where the control points are. In Cascade, we call this
   the :py:class:`AgeTimeGrid <cascade.model.grids.AgeTimeGrid>`.
   It's a list of ages and a list of times
   that define a rectangular grid.

 * At each of the control points of the age time grid, Dismod-AT will evaluate
   how close the rate or covariate multiplier is to some reference value. At these
   points, we define prior distributions. Cascade makes these
   :ref:`value priors <prior-specification>`
   part of the :py:class:`PriorGrid <cascade.model.grids.PriorGrid>`.

 * It's rare to have data points that are dense across all of age and time.
   Dismod-AT needs to take a data point at one end, a data point at the other
   end, and draw a line that connects them. We help it by introducing constraints
   on how quickly a value can change over age and time. These are a kind of
   regularization of the problem, called *age-time difference priors*. They apply
   to the difference in value between one age-time point and the next greater
   in age and the next-greater in time. As with value priors, these are specified
   in the Cascade as part of the :py:class:`PriorGrid <cascade.model.grids.PriorGrid>`.

The random effect for locations is also a continuous quantity.
