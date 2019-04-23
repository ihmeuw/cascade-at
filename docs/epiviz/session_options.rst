.. _session-options:

Dismod-AT Session Options
=========================

Dismod-AT accepts options, as described in
https://bradbell.github.io/dismod_at/doc/option_table.htm.


The application selects options based on a set of layers.
At the top layer are choices in the EpiViz-AT UI.
Most of these are set in the EpiViz-AT settings and documented in
:mod:`configuration.Form <cascade.input_data.configuration.Form>`.
In particular, policies are set in
:class:`cascade.input_data.db.configuration.form:Policies`.

======================================  ==================================================
``random_seed``                         Set by EpiViz-AT UI for each run.
``meas_noise_effect``                   ``add_var_scale_log``, set in policies
``zero_sum_random``                     In EV settings. Currently ignored.
``warn_on_stderr``
``ode_step_size``                       Set by modeler in EpiViz-AT.
``age_avg_split``                       Set as part of setting ages in EpiViz-AT.
``derivative_test_fixed``
``derivative_test_random``
``max_num_iter_fixed``
``max_num_iter_random``
``print_level_fixed``
``print_level_fixed``
``accept_after_max_steps_fixed``
``accept_after_max_steps_random``
``tolerance_fixed``
``tolerance_random``
``quasi_fixed``
``bound_frac_fixed``
``limited_memory_max_history_fixed``
``bound_random``
======================================  ==================================================


The following options in Dismod-AT are part of model specification
and set by the model: ``parent_node_id``, ``parent_node_name``,
``data_extra_columns``, ``avgint_extra_columns``, ``rate_case``.

At the lowest layer is the object wrapper selection, these are the defaults
if nothing is set by policy.

.. literalinclude:: ../../src/cascade/model/object_wrapper.py
   :pyobject: ObjectWrapper._create_options_table
