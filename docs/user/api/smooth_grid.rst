.. _SmoothGrid:

SmoothGrid Class
----------------

A SmoothGrid represents model priors (as opposed to data priors) in
a Dismod-AT model. A :py:class:`Model <cascade.model.Model>` is a bunch of
SmoothGrids, one for each rate, random effect, and covariate multiplier.

For instance, in order to set priors on underlying incidence rate, iota,
create a SmoothGrid, set its priors, and add it to the Model::

    smooth = SmoothGrid([0, 5, 10, 50, 100], [1990, 2015])
    smooth.value[:, :] = Uniform(mean=0.01, lower=1e-6, upper=5)
    smooth.dage[:, :] = Gaussian(mean=0, standard_deviation=10)
    smooth.dtime[:, :] = Gaussian(mean=0, standard_deviation=0.1)

All of the priors in a SmoothGrid need to be defined. There is a value
prior at each age and time, but the prior for difference in age and time
are *forward differences,* so there is no prior for the largest age
point and largest time point. That means you'll notice examples
with no dtime priors when the underlying grid is defined for only
one year.

If you want more control over exact priors, iterate over them. The
`age_time_diff` iterator returns the age and time at the age points
but also the difference in age and time to the next age point::

    for age, time, age_diff, time_diff in smooth.age_time_diff():
        if not isinf(age_diff):
            smooth.dage[age, time] = \
                Gaussian(mean=0, standard_deviation=1 + 5 * age_diff)

This would change the standard deviation as the age interval changes,
which could be helpful when age intervals change greatly. The check
for `isinf` catches the last age difference, which returns the value
`inf` because there is no next age point.

It is also possible to see what priors are set. This gets the prior
at each age and time. Then it sets a new value for the prior with
twice-as-large a standard deviation but the same density::

    for age, time in smooth.age_time():
        prior = smooth.value[age, time]
        print(f"prior mean {prior.mean} std {prior.standard_deviation}")
        smooth.value[age, time] = prior.update(standard_deviation=2 * prior.standard_deviation)


.. autoclass:: cascade.model.SmoothGrid
   :members: value, dage, dtime, variable_count
