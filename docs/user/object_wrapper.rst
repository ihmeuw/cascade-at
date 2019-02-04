.. _object-wrapper-class:

ObjectWrapper Class
-------------------

.. py:class:: cascade.model.ObjectWrapper

    The Session class uses an object-oriented interface to Dismod-AT Sqlite
    database files. This makes it possible to read and write
    Model, Var, and residuals from the Dismod-AT db files.

    **Writing a Model object deletes the db file and makes a new one.**

    For instance::

        wrapper = ObjectWrapper(locations, parent_location, filename)
        wrapper.model = my_model_object
        wrapper.data = data_df
        # Maybe run dismod init and fit by hand
        fit_var = wrapper.fit_var

    The var tables should all show up as properties.

    .. py:method:: set_option(option_name=option_value, ...)

       Set options. Setting an option to None or nan disables it.

    .. py:attribute:: model

       Set the model in order to make a new database, overwriting
       whatever was there. Cannot currently read a model.

    .. py:attribute:: data

       Set a Pandas dataframe with data.

    .. py:attribute:: avgint

       Creates an avgint table.

    .. py:attribute:: start_var

       Initial guess for fit.

    .. py:attribute:: scale_var

       The value at which to find the derivative of each term
       in the log-likelihood to use for rescaling.

    .. py:attribute:: truth_var

       Truth around which to simulate.

    .. py:attribute:: prior_residuals

       Residuals on priors, as an AgeTimeGrid.

    .. py:attribute:: data_residuals

       Dataframe of residuals on data values.

    .. py:attribute:: predict

       Results of a predict. Returns (predicted, not_predicted) because
       avgints with covariates out of bounds cannot be predicted, so they
       are returned as a separate list.

    .. py:attribute:: locations

       Set locations from a dataframe.

    .. py:attribute:: samples

       Results of a sample.

    .. py:method:: read_simulation_model_and_data(model, data, index)

       Returns a model and a data corresponding to that simulation index.

    .. py:method:: refresh()

       Reread all tables from disk.

    .. py:method:: close()

       Close the db file. Dispose of the sqlite3 connection.