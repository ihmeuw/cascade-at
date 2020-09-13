.. _smooth-grid-priors:


Smooth Grid Priors
==================

For the Rate tab, Random Effect tab, Study tab and Country Covariate tab,
the interface sets priors. This describes how those settings are interpreted.
Most of this work happens in the function
:py:func:`cascade.executor.construct_model.smooth_grid_from_smoothing_form`,
and you can check its source there.

The default value, dage, and dtime priors are used to initialize those parts
of the smooth grid. For smooth grids with only one age, the dage priors
aren't meaningful, and the same is true for dtime priors when there is
only one year.

After that, the detailed priors are applied in the order they appear in
the settings, and note that the order may or may not reflect the order
in the user interface. There are three ways to specify which age
and time points each detailed prior applies to:

 *  ``age_lower`` and ``age_upper`` - A missing value here (one that's not
    filled-in in the UI) is treated as -infinity or infinity, respectively.

 *  ``time_lower`` and ``time_upper`` - Missing values similarly set to
    include all points on that side.

 *  ``born_lower`` and ``born_upper`` - Each line for the born limit
    corresponds to :math:`a \le t - b` or :math:`a \ge t - b`, respectively.

For each prior, all three of these sieves are applied to the grid of
ages and times defined by the age values and time values for that smooth grid.
If a detailed prior doesn't match any of the age and time points in this grid,
there will be a statement in the log that says "No ages and times match prior
with extents <lower and upper extents>."
