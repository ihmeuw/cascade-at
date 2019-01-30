.. _model-class:

Model Class
-----------

The Model holds all of the SmoothGrids that define priors on
rates, random effects, and covariates. It also has a few other
properties necessary to define a complete model.

 * Which of the rates are nonzero. This is a list of, for instance,
   `["iota", "omega", "chi"]`.

 * The parent location as an integer ID. These correspond to the IDs
   supplied to the Dismod-AT session.

 * A list of child locations. Not children and grandchildren, but the
   direct child locations as integer IDs.

 * A list of covariates, supplied as :py:class:`Covariate` objects.

 * Weight functions, that are used to compute integrands. Each weight
   function is a Var.

 * A scaling function, which sets the scale for every model variable.
   If this isn't set, it will be calculated by Dismod-AT from the
   mean of value priors. It is used to ensure different terms in
   the likelihood have similar importance.



.. autoclass:: cascade.model.Model
   :members: __init__, scale, from_var
