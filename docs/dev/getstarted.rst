.. _get-started:

Get Started
===========

Dear Developer,

This code runs Dismod-AT within the IHME environment.
As a first step, consider the problem domain for
Dismod-AT so that you know terminology.

 * :ref:`dismod-description`

 * :ref:`global-model-draft`

Once you have skimmed that, maybe you want to see the scope
of what is installed and with which teams this development
effort communicates.

 * :ref:`cascade-architecture`. Notice the separate
   installations.

 * :ref:`map-of-the-code`

How do you install?

 * On your machine or under your account on the cluster:
   :ref:`install-api`.

 * For IHME, as infrastructure, :ref:`operations-and-maintenance`.

From here, I would turn to look at inputs in more detail.

 * The input data is read in a function in the
   module called ``cascade.executor.estimate_locations``.

 * That input data comes from what people call the
   "Epi DB." The key for that database is referenced
   by the execution context in ``cascade.executor.execution_context``,
   and that key points to an IHME-specified database
   using our internal ``db_tools``.

 * EpiViz-AT describes a model by sending a JSON object
   called the settings. The code to interpret that
   is in the ``cascade.input_data.configuration.form`` module.

 * There are tools to get and examine data from the
   Epi DB. They are listed in the environment's setup.py,
   and there are more in IHME's code filesystem, under
   ``dismod_at/bin``.

From there, the developer section of the manual,
the section you are reading, should be of help.
