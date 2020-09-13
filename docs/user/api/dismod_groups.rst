.. _dismod-groups-class:

DismodGroups Class
------------------

.. py:class:: cascade.model.DismodGroups

   A DismodGroups contains Var instances or contains SmoothGrid instances.
   It gives them the shape of the whole model, so it expresses what
   rates are nonzero, what random effects are defined, and on which
   of these there are covariate multipliers.

   The DismodGroups structure will appear in lots of places.
   The fit returned by Dismod will be a DismodGroups containing Var objects.
   The Model itself is a DismodGroups containing SmoothGrid objects.

   A classic use of this is to create a new ``DismodGroups`` of ``Var``.
   The first loop is over the rate, random effect, and covariate group
   names. The inner loop is over particular sets of keys, which are
   composed of tuples of the primary rate, covariate name, and location IDs.

   .. code::

       var_groups = DismodGroups()
       for group_name, group in var_ids.items():
           for key, var_id_mapping in group.items():
               var_groups[group_name][key] = var_builder(table, var_id_mapping)

   .. py:attribute:: rate[primary_rate]

      This is a dictionary of rates. They are always one of the
      five underlying rates: iota, chi, omega, rho, pini::

          dg = DismodGroups()
          dg.rate["iota"] = Var([0, 1, 50], [2000])

   .. py:attribute:: random_effect[(primary_rate, child_location)]

      A dictionary of random effects on the rates, so the keys
      are a rate and the ID of the child for which this is a random
      effect. When constructing a :py:class:`Model`, we typically want
      to make one :py:class:`SmoothGrid` of priors for all child random
      effects on a particular rate. In that case, specify the child
      ID as ``None``::

          model = Model()  # A Model is a DismodGroups object, too.
          model.random_effect[("iota", None)] = SmoothGrid([0, 100], [1990])

          scale = DismodGroups()
          scale.random_effect[("omega", 2)] = Var([0, 100], [1990, 2000])
          scale.random_effect[("omega", 3)] = Var([0, 100], [1990, 2000])

   .. py:attribute:: alpha[(covariate_name, rate_name)]

      Alpha are covariate multipliers on the rates. The key is the
      name of the Covariate, which should match the name in the class
      given as an argument to the Session object. The rate name is one
      of the five underlying rates.

   .. py:attribute:: beta[(covariate_name, integrand_name)]

      Beta are covariate multipliers on the measured value of the
      integrands. The integrand name is one of the canonical values.

   .. py:attribute:: gamma[(covariate_name, integrand_name)]

      Gamma are covariate multipliers on the measured standard
      deviation of the integrands.
