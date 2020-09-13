.. _epiviz-bundle:

Bundle
======

Measurements on populations are stored in the *bundle.*
Each time EpiViz-AT runs, it copies the current bundle
for this modelable entity ID to a permanent storage, called Tier 3.

.. _input-bundle-columns:

Input Columns
^^^^^^^^^^^^^

Command-line EpiViz-AT uses the following columns in the bundle.

====================        ======================================
``bundle_id``               used to select bundle
``seq``                     unique ID for each record in bundle
``input_type_id``           exclude 5, 6 from usage
``outlier_type_id``         Keep 0 always. Exclude 1.
``mean``                    mean value of measurement
``lower``                   lower bound of 95% confidence interval
``upper``                   upper bound of 95% confidence interval
``location_id``             Integer location ID
``sex_id``                  1 male, 2 female, 3 both, 4 neither
``year_start``              first year in which data collected
``year_end``                last year in which data collected
``age_start``               real for youngest age
``age_end``                 real for oldest age
``measure_id``              measure ID, integer
====================        ======================================


The following are ignored:
``standard_error`` (standard error of the mean),
``nid``, ``underlying_nid``, ``source_type_id``,
``sampling_type_id``, ``representative_id``, ``urbanicity_type_id``,
``recall_type_id``, ``recall_type_value``, ``unit_type_id``,
``unit_type_value``, ``unit_value_as_published``,
``uncertainty_type_id``, ``uncertainty_type_value``,
``effective_sample_size``, ``sample_size``, ``cases``, ``design_effect``.


.. _demographic-interval-transformation:

Demographic Interval Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There has been discussion about whether bundles can use demographic
notation. There is a somewhat complicated formula on the Hub about
implied demographic notation for bundle data. As shown in
:py:func:`bundle_to_observations <cascade.input_data.configuration.construct_bundle.bundle_to_observations>`,
there is no fix applied for demographic intervals.


.. _bundle-measures-used:

Recognized Measures in the Bundle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dismod-AT only recognizes certain IHME measures. The mapping between
IHME measures and Dismod-AT data is in
`id_map.py code <https://github.com/ihmeuw/cascade/blob/develop/src/cascade/input_data/configuration/id_map.py>`_.

.. literalinclude:: ../../src/cascade/input_data/configuration/id_map.py
    :lines: 8-62

In particular, measure ID 17, for case fatality rate, is deleted from the
bundle before it's given to Dismod-AT.


.. _standard-deviation-of-bundle:

Conversion to Standard Deviation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dismod-AT describes uncertainty in data using a density, a mean, a standard
deviation, and possibly other parameters for the density. That standard
deviation will come from the upper and lower confidence intervals, not from
standard error in the mean.

Each bundle measurement is converted to have a standard deviation
using lower = :math:`x_l` and upper = :math:`x_u`,

.. math::
    :label: convert-stdev

    \sigma = \frac{x_u - x_l}{2 (1.96)}

where 1.96 is :math:`z^*` for the 95% confidence interval.

The equation is in :py:func:`bounds_to_stdev <cascade.stats.estimation.bounds_to_stdev>`
and it is applied to the bundle in
:py:func:`bundle_to_observations <cascade.input_data.configuration.construct_bundle.bundle_to_observations>`.
