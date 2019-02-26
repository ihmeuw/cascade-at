.. _session-class:

Session
-------

For all Session commands that accept data, the format is a Pandas
DataFrame with the columns:
``integrand``, ``location``,
``name``, ``hold_out``,
``age_lower``, ``age_upper``, ``time_lower``, ``time_upper``,
``density``, ``mean``, ``std``, ``eta``, ``nu``.
The ``name`` is optional and will be assigned from the index.
In addition, covariate columns are included. If ``hold_out``
is missing, it will be assigned ``hold_out=0`` for not held out.
If nu or eta aren't there, they will be added. If ages
or times are listed as ``age`` and ``time``, they will be
considered point values and expanded into upper and lower.


.. autoclass:: cascade.model.Session
   :members: __init__, fit, fit_fixed, fit_random, predict, simulate, sample, set_option

.. autoclass:: cascade.model.session.FitResult
   :members:

.. autoclass:: cascade.model.session.SimulateResult
   :members:
