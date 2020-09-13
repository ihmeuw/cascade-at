
.. _dismod-hierarchical:

Hierarchical Model
------------------

The hierarchical part of Dismod-AT does one thing, estimate how
locations affect rates. If the rate at grid point :math:`(i,k)`
is :math:`q_{ik}(a,t)`, and the covariate
multiplier is :math:`\alpha_{ik}(a,t)`, then the adjusted rate is

.. math::

    r_{ik}(a,t) = q_{ik}(a,t) \exp\left(u_{ik}(a,t) + \sum_j x_{ikj}\alpha_{jik}(a,t)\right).

The offset, :math:`u`, is linear with the covariates, but it is inside the exponential,
which guarantees that all rates remain positive. This offset is the only
random effect in the problem, and it is called the
*child rate effect* because each location, or node in Dismod-AT's language,
is considered a child of a parent.

Because the child rate effect is continuous, you can conclude that it must be
defined on a smoothing grid. Dismod-AT will either define one smoothing grid
for each child rate effect (one for each of the five rates) or let you define
a smoothing grid for every location and every child rate effect, should that
be necessary.

