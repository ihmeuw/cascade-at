.. _study-country-covariates:

Study and Country Covariates
----------------------------

There are three reasons to use a covariate.

*Country Covariate*
    We believe this covariate predicts disease behavior.

*Study Covariate*
    The covariate marks a set of studies that behave differently.
    For instance, that set may have different criteria for incipience
    of a disease. We assign a covariate to the set of studies
    to account for bias from study design.

*Sex Covariate*
    This is usually used to select a subset of data by sex,
    but this could be done based on any covariate associated
    with observation data. In addition to being used to subset
    data, the sex covariate is a covariate multiplier applied
    the same way as a study covariate.

A covariate column that is used just for exclusion doesn't need
a covariate multiplier. In practice, the sex covariate is used
at global or super-region level as a study covariate. Then the
adjustments determined at the upper level are applied as constraints
down the hierarchy. This means there is a covariate multiplier
for sex, and its smooth is a grid of constraints, not typical
priors.

Dismod-AT applies covariate effects to one of three different variables.
It either uses the covariate to `predict the underlying rate`_,
or it applies the covariate to predict the measured data. It can
be an effect on either the `measured data value`_ or the
observation data `standard deviation`_. Dismod-AT calls these
the alpha, beta, and gamma covariates.

As a rule of thumb, the three uses of covariates apply
to different variables, as shown in the table below.

====================  =======  ================ ===============
Use of Covariate      Rate     Measured Value   Measured Stddev
====================  =======  ================ ===============
Country               Yes      Maybe            Maybe
Study                 Maybe    Yes              Yes
Sex (exclusion)       No       Yes              No
====================  =======  ================ ===============


Country and study covariates can optionally use outliering.
The sex covariate is defined by its use of regular outliering.
Male and female data is assigned a value of -0.5 and 0.5, and
the mean and maximum difference are adjusted to include one,
the other, or both sexes.


.. _predict the underlying rate:
    https://bradbell.github.io/dismod_at/doc/avg_integrand.htm#Rate%20Functions.Rate%20Covariate%20Multiplier,%20alpha_jk

.. _measured data value:
    https://bradbell.github.io/dismod_at/doc/avg_integrand.htm#Measurement%20Value%20Covariates.Multiplier,%20beta_j

.. _standard deviation:
    https://bradbell.github.io/dismod_at/doc/data_like.htm#Measurement%20Standard%20Deviation%20Covariates.gamma_j
