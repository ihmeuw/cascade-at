.. _interpolation-country-covariates:

Interpolation of Country Covariates for EpiViz-AT
-------------------------------------------------

The Country Covariate tab in EpiViz-AT lets you specify multiple
Covariate Multipliers, each of which implies use of a covariate.
The two functions that read the EpiViz-AT settings and construct
data for Dismod-AT are ``assign_covariates`` and ``create_covariate_multipliers``.


.. autofunction:: cascade.input_data.configuration.construct_country.covariate_to_measurements_nearest_favoring_same_year
    :noindex:

.. autofunction:: cascade.input_data.configuration.construct_country.reference_value_for_covariate_mean_all_values
    :noindex:
