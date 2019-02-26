.. _epiviz-bundle:

Bundle
======

Measurements on populations are stored in the *bundle.*
Each time EpiViz-AT runs, it copies the current bundle
for this modelable entity ID to a permanent storage, called Tier 3.

Command-line EpiViz-AT uses the following columns in the bundle.

====================        ======================================
``bundle_id``               used to select bundle
``seq``                     unique ID for each record in bundle
``input_type_id``           exclude 5, 6 from usage
``outlier_type_id``         Keep 0 always. Exclude 1.
``mean``                    mean value of measurement
``lower``                   lower error on measurement
``upper``                   upper error on measurement
``standard_error``
``location_id``
``sex_id``
``year_start``
``year_end``
``age_start``
``age_end``
``measure_id``
====================        ======================================


The following are ignored:
``nid``, ``underlying_nid``, ``source_type_id``,
``sampling_type_id``, ``representative_id``, ``urbanicity_type_id``,
``recall_type_id``, ``recall_type_value``, ``unit_type_id``,
``unit_type_value``, ``unit_value_as_published``,
``uncertainty_type_id``, ``uncertainty_type_value``,
``effective_sample_size``, ``sample_size``, ``cases``, ``design_effect``.

