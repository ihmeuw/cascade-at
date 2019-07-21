from collections import Iterable
from contextlib import contextmanager
from math import nan, isnan
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.core import getLoggers
from cascade.core.subprocess_utils import (
    add_gross_timing, read_gross_timing, processor_type
)
from cascade.core.subprocess_utils import run_with_logging
from cascade.dismod.constants import COMMAND_IO
from cascade.dismod.constants import DensityEnum, RateEnum, IntegrandEnum
from cascade.dismod.db.wrapper import DismodFile, get_engine
from cascade.dismod.metrics import gather_metrics
from cascade.dismod.process_behavior import check_command
from cascade.model.data_read_write import (
    write_data, avgint_to_dataframe, read_avgint, read_data_residuals,
    read_simulation_data, amend_data_input, point_age_time_to_interval
)
from cascade.model.grid_read_write import (
    read_var_table_as_id, read_vars, write_vars, read_prior_residuals, read_samples,
    read_simulation_model
)
from cascade.model.model_writer import ModelWriter
from cascade.model.serialize import default_integrand_names, make_log_table

CODELOG, MATHLOG = getLoggers(__name__)


class ObjectWrapper:
    """
    An I/O layer on top of the Dismod db file that presents Model objects.
    It sets and gets models, vars, data, and residuals. Set the locations
    before anything else. In order to make a model, you must set:

    1. Locations
    2. Parent location ID
    3. Model
    4. Anything else.

    It takes work to ensure that all the database records are
    consistent. This class groups sets of tables and columns
    within those tables into higher-level objects that are then
    easier to reason about.
    """
    def __init__(self, filename):
        """
        Args:
            filename (Path|str|None): Path to filename or None if this
                DismodFile should be in memory, as used for testing.
        """
        if filename is not None:
            assert isinstance(filename, (Path, str))
            self._filename = Path(filename)
        else:
            self._filename = filename
        self.dismod_file = DismodFile()
        self.dismod_file.engine = get_engine(self._filename)
        self.ensure_dismod_file_has_default_tables()

    @property
    def db_filename(self):
        """pathlib.Path: path to the file. Read-only. Make
        a different `ObjectWrapper` for a different file."""
        return self._filename

    @property
    def locations(self):
        """
        Returns a dataframe of locations, built from the node table.
        Input locations have ``location_id``, ``parent_id`` and optional
        ``name``. This output table also has a ``node_id``, which was the node
        used by Dismod-AT.
        """
        node = self.dismod_file.node
        assert not ({"node_id", "node_name", "parent"} - set(node.columns))
        if "c_location_id" not in node.columns:
            node = node.assign(c_location_id=node.node_id)
        location_map = node[["node_id", "c_location_id"]].rename(
            columns={"node_id": "parent", "c_location_id": "parent_location_id"})
        parent_location = node.merge(
            location_map, on="parent", how="left")
        missing = parent_location[parent_location.parent_location_id.isna()]
        if len(missing) > 1:  # Root will have nan for parent.
            raise ValueError(f"parent location IDs unknown {missing}")
        return parent_location.rename(columns=dict(
            parent_location_id="parent_id", c_location_id="location_id",
            node_name="name"
        ))[["parent_id", "location_id", "name", "node_id"]]

    @locations.setter
    def locations(self, locations):
        for required_column in ["parent_id", "location_id"]:
            if required_column not in locations.columns:
                raise ValueError(f"Locations should be a DataFrame with location_id and parent_id, "
                                 f"and optional name, not {locations.columns}.")
        if "name" not in locations:
            locations = locations.assign(name=locations.location_id.astype(str))
        node = locations.rename(columns={"name": "node_name", "location_id": "c_location_id"})
        node = node.reset_index(drop=True).assign(node_id=node.index)

        def location_to_node_func(location_id):
            if np.isnan(location_id):
                return np.nan
            return np.where(node.c_location_id == location_id)[0][0]

        node = node.assign(parent=node.parent_id.apply(location_to_node_func)).drop(columns=["parent_id"])
        self.dismod_file.node = node

    @property
    def parent_location_id(self):
        """int: location ID for the parent location. This is not the node ID,
        which is an internal Dismod-AT representation that has to be
        zero-indexed."""
        location_df = self.locations
        if location_df.empty:
            raise RuntimeError("Cannot get parent location until locations exist")
        node_id = self.get_option("parent_node_id")
        node_name = self.get_option("parent_node_name")
        if node_id is not None:
            parent_location = location_df[location_df.node_id == node_id].location_id.iloc[0]
        elif node_name is not None:
            parent_location = location_df[location_df.name == node_name].location_id.iloc[0]
        else:
            parent_location = None
        return parent_location

    @parent_location_id.setter
    def parent_location_id(self, value):
        location_df = self.locations
        if location_df.empty:
            raise RuntimeError("Cannot set parent location until locations exist")
        parent_node_id = location_df[location_df.location_id == value].node_id.iloc[0]
        self.set_option(
            parent_node_id=parent_node_id,
            parent_node_name=nan
        )

    @property
    def model(self):
        """cascade.model.Model: The main model object. Can write but
        not read."""
        raise NotImplementedError("Reading a model is not implemented.")

    @model.setter
    def model(self, new_model):
        """When you write a model, it deletes the file."""
        if self.locations.empty:
            raise RuntimeError("Cannot create a model until locations exist")
        writer = ModelWriter(self, self.dismod_file)
        new_model.write(writer)
        writer.close()

    def set_option(self, **kwargs):
        """Erase an option by setting it to None or nan.

        * Setting a list of items sets it to a space-separated list of strings.
        * Setting a bool sets it to "true" or "false", lower-case.
        * Setting None or NaN, sets it to NaN.
        * All other ``x`` become ``str(x)``.

        The exact allowed options are described here:

        .. literalinclude:: ../../../src/cascade/model/object_wrapper.py
           :pyobject: ObjectWrapper._create_options_table

        """
        option = self.dismod_file.option
        unknowns = list()
        for name in kwargs.keys():
            if not (option.option_name == name).any():
                unknowns.append(name)
        if unknowns:
            raise KeyError(f"Unknown options {unknowns}")

        for name, value in kwargs.items():
            if isinstance(value, str):
                str_value = value
            elif isinstance(value, Iterable):
                str_value = " ".join(str(x) for x in value)
            elif isinstance(value, bool):
                str_value = str(value).lower()
            elif value is None or isnan(value):
                str_value = None
            else:
                str_value = str(value)
            if str_value is not None:
                option.loc[option.option_name == name, "option_value"] = str_value
            else:
                option.loc[option.option_name == name, "option_value"] = nan
        option = option.reset_index(drop=True)
        option = option.assign(option_id=option.index)
        self.dismod_file.option = option

    def get_option(self, name):
        """Gets an option, as set with ``set_option``."""
        option_df = self.dismod_file.option
        records = option_df[option_df.option_name == name]
        if len(records) == 1:
            return records.option_value.iloc[0]
        else:
            raise KeyError(f"Option {name} not found in options")

    @property
    def data(self):
        """
        pd.DataFrame: Data can be set but not read.
        A Pandas DataFrame with columns:

         * ``location`` - an integer location.
         * ``integrand`` - string from the integrand names.
         * ``age_lower`` and ``age_upper`` - float ages. If only ``age`` is
           present, then both lower and upper will be equal to ``age``.
         * ``time_lower`` and ``time_upper`` - float ages. If only ``time`` is
           present, then both lower and upper will be equal to ``time``.
         * ``density`` - string from density names.
         * ``mean`` - float value
         * ``std`` - float value
         * ``name`` - Optional. Name for this record, as a string.
         * ``hold_out`` - Optional. 0 for don't hold out, 1 for do.
         * ``nu`` - Optional. Float parameter for distributions.
         * ``eta`` - Optional. Float parameter for distributions.
         * ``node_id`` - Optional. On read, this is the node that was used
           internally by Dismod-AT.
         * Covariate columns are floats and depend on covariates already
           defined.
        """
        raise NotImplementedError("Reading data is not implemented.")

    @data.setter
    def data(self, data):
        with_missing_columns = amend_data_input(data)
        write_data(self.dismod_file, with_missing_columns, self.covariate_rename)

    @property
    def avgint(self):
        """pd.DataFrame: The average integrands have the same shape as
        the data and can be set but not read."""
        raise NotImplementedError("Reading avgints is not implemented.")

    @avgint.setter
    def avgint(self, avgint):
        avgint = point_age_time_to_interval(avgint)
        self.dismod_file.avgint = avgint_to_dataframe(
            self.dismod_file, avgint, self.covariate_rename)

    @property
    def start_var(self):
        """cascade.model.DismodGroups: A set of Var with
        starting values for variables. Can read or write."""
        return self.get_var("start")

    @start_var.setter
    def start_var(self, new_vars):
        self.set_var("start", new_vars)

    @property
    def scale_var(self):
        """cascade.model.DismodGroups: A set of Var with scaling
        values for variables. Can read or write."""
        return self.get_var("scale")

    @scale_var.setter
    def scale_var(self, new_vars):
        self.set_var("scale", new_vars)

    @property
    def fit_var(self):
        """cascade.model.DismodGroups: A set of Var that
        is the result of a fit. Can read or write."""
        return self.get_var("fit")

    @fit_var.setter
    def fit_var(self, new_vars):
        self.set_var("fit", new_vars)

    @property
    def truth_var(self):
        """cascade.model.DismodGroups: A set of Var
        to use as truth for simulation or prediction. Can read or write."""
        return self.get_var("truth")

    @truth_var.setter
    def truth_var(self, new_vars):
        self.set_var("truth", new_vars)

    def get_var(self, name):
        var_id = read_var_table_as_id(self.dismod_file)
        return read_vars(self.dismod_file, var_id, name)

    def set_var(self, name, new_vars):
        var_id = read_var_table_as_id(self.dismod_file)
        write_vars(self.dismod_file, new_vars, var_id, name)
        self.flush()

    def set_minimum_meas_cv(self, integrand, value):
        """The minimum measured CV is neither an option nor a dataset,
        so this sets those values.

        Args:
            integrand (str): One of the integrands.
            value (float): The CV value, greater than or equal to zero.
        """
        if value is not None:
            fvalue = float(value)
            assert fvalue >= 0.0
        else:
            fvalue = 0
        integrand = IntegrandEnum[integrand]
        dmf = self.dismod_file
        if dmf:
            dmf.integrand.loc[dmf.integrand.integrand_name == integrand.name, "minimum_meas_cv"] = fvalue
        else:
            CODELOG.info(f"minimum_meas_cv not set because dismod_file is None.")

    @property
    def prior_residuals(self):
        """pd.DataFrame: Reads prior residuals into a dataframe."""
        var_id = read_var_table_as_id(self.dismod_file)
        return read_prior_residuals(self.dismod_file, var_id)

    @property
    def data_residuals(self):
        """pd.DataFrame: Reads data residuals into a dataframe."""
        return read_data_residuals(self.dismod_file)

    @property
    def samples(self):
        """pd.DataFrame: These are generated by the simulate command."""
        var_id = read_var_table_as_id(self.dismod_file)
        return read_samples(self.dismod_file, var_id)

    def read_simulation_model_and_data(self, model, data, index):
        """The Dismod-AT simulate command creates a new model and new data
        for each simulation. This reads one of the models generated.

        Args:
            model (Model): This is the model that generated the simulations.
            data (pd.DataFrame): This is the data that generated the simulations.
            index (int): Which simulation to read.

        Returns:
            (Model, pd.DataFrame): the model and data for a single simulation.

        """
        var_id = read_var_table_as_id(self.dismod_file)
        sim_model = read_simulation_model(self.dismod_file, model, var_id, index)
        sim_data = read_simulation_data(self.dismod_file, data, index)
        return sim_model, sim_data

    def refresh(self, list_of_tables):
        """Asks the wrapper to re-read data out of Dismod-AT. Accepts
        a list of which tables to read.

        Args:
            list_of_tables (List[str]): Names of tables.
        """
        self.dismod_file.refresh(list_of_tables)

    def flush(self):
        """Writes all modified data to disk. This asks the caching
        mechanism to stop caching and ensure data is in the file."""
        self.dismod_file.flush()

    def close(self):
        """Closes the database engine. This makes sure that the running
        Python application isn't connected to the file Dismod-AT has
        to read because Dismod-AT can't read it while the application
        reads it."""
        self.flush()
        if self.dismod_file.engine is not None:
            self.dismod_file.engine.dispose()
            self.dismod_file.engine = None

    @contextmanager
    def close_db_while_running(self):
        """
        A context manager to make it easier to work with this object
        and run Dismod-AT. It closes the file, flushes it, and then
        refreshes the file afterwards.

        .. code::

            with dismod_objects.close_db_while_running():
                dismod_objects.run_dismod(["fit"])
        """
        self.close()
        try:
            yield
        finally:
            self.dismod_file.engine = get_engine(self._filename)

    def run_dismod(self, command):
        """Pushes tables to the db file, runs Dismod-AT, and refreshes
        tables written. This flushes the in-memory objects before
        running Dismod.

        Args:
            command (List[str]|str): Command to run as a list of strings
                or a single string.

        Returns:
            (str, str, Dict): Stdout and stderr as strings, not bytes.
        """
        if isinstance(command, str):
            command = command.split()
        self.flush()
        CODELOG.debug(f"Running Dismod-AT {command}")
        str_command = [str(word) for word in command]
        include_dismod = ["dmdismod", str(self.db_filename)] + str_command
        timed_command, timing_out_file = add_gross_timing(include_dismod)
        metrics = gather_metrics(self.dismod_file)
        metrics["dismod_at command"] = " ".join(str(x) for x in command)
        metrics.update({"processor type": processor_type()})
        try:
            with self.close_db_while_running():
                str_command = [str(c) for c in command]
                return_code, stdout, stderr = run_with_logging(timed_command)
        finally:
            metrics.update(read_gross_timing(timing_out_file))
        log = self.log
        check_command(str_command[0], log, return_code, stdout, stderr)
        if command[0] in COMMAND_IO:
            self.refresh(COMMAND_IO[command[0]].output)
        return stdout, stderr, metrics

    @property
    def log(self):
        """pd.DataFrame: Each record is a log entry indicating what
        the Dismod-AT executable did."""
        self.dismod_file.refresh(["log"])
        return self.dismod_file.log

    @property
    def predict(self):
        """
        Returns two dataframes, one of the points predicted and one of the
        points excluded from prediction because their covariates were out
        of bounds.

        Columns are ``sample_index``, ``integrand``, ``location``,
        ``age_lower``, ``age_upper``, ``time_lower``, ``time_upper``,
        ``mean``, and any covariates. If you drop the sample index,
        and add standard deviation, this can serve as data values.
        """
        avgint = read_avgint(self.dismod_file)
        raw = self.dismod_file.predict.merge(avgint, on="avgint_id", how="left")
        normalized = raw.drop(columns=["avgint_id", "predict_id"]).rename(columns={"avg_integrand": "mean"})
        not_predicted = avgint[~avgint.avgint_id.isin(raw.avgint_id)].drop(columns=["avgint_id"])
        return normalized, not_predicted

    @property
    def age_extents(self):
        """This ensures the ages and times for integration cover the given
        list of ages and times. If you ask Dismod-AT to fit data, and some
        of that data is outside the range of times in the model, it asks
        you to insert two data points into its age and time tables as
        reassurance you intend to integrate over a longer region.
        With this function, you can give it a list of lots of ages and
        times, and it ensures they are all within bounds, or it will
        increase those bounds.

        Args:
            ages (List[float]): List of ages
        """
        age_df = self.dismod_file.age
        return age_df.age.min(), age_df.age.max()

    @age_extents.setter
    def age_extents(self, ages):
        ages_df = self.dismod_file.age
        if ages_df.age.min() > min(ages):
            ages_df = ages_df.append(
                dict(age_id=len(ages_df), age=min(ages)), ignore_index=True)
        if ages_df.age.max() < max(ages):
            ages_df = ages_df.append(
                dict(age_id=len(ages_df), age=max(ages)), ignore_index=True)
        self.dismod_file.age = ages_df

    @property
    def time_extents(self):
        """This ensures the ages and times for integration cover the given
        list of ages and times. If you ask Dismod-AT to fit data, and some
        of that data is outside the range of times in the model, it asks
        you to insert two data points into its age and time tables as
        reassurance you intend to integrate over a longer region.
        With this function, you can give it a list of lots of ages and
        times, and it ensures they are all within bounds, or it will
        increase those bounds.

        Args:
            times (List[float]): List of times
        """
        time_df = self.dismod_file.time
        return time_df.time.min(), time_df.time.max()

    @time_extents.setter
    def time_extents(self, times):
        times_df = self.dismod_file.time
        if times_df.time.min() > min(times):
            times_df = times_df.append(
                dict(time_id=len(times_df), time=min(times)), ignore_index=True)
        if times_df.time.max() < max(times):
            times_df = times_df.append(
                dict(time_id=len(times_df), time=max(times)), ignore_index=True)
        self.dismod_file.time = times_df

    @property
    def covariates(self):
        """
        Sets covariates. Must call before setting data. Cannot read
        covariates.

        Args:
            covariate (List[Covariate]): A list of covariate objects.
        """
        return None

    @covariates.setter
    def covariates(self, value):
        null_references = list()
        for check_ref_col in value:
            if not isinstance(check_ref_col.reference, Real):
                null_references.append(check_ref_col.name)
        if null_references:
            raise ValueError(f"Covariate columns without reference values {null_references}")

        # Dismod requires us to rename covariates from names like sex, and "one"
        # to x_0, x_1,... They must be "x_<digit>".
        covariate_rename = dict()
        for covariate_idx, covariate_obj in enumerate(value):
            covariate_rename[covariate_obj.name] = f"x_{covariate_idx}"

        self._ensure_schema_has_covariates(covariate_rename.values())
        self.dismod_file.covariate = pd.DataFrame(
            {
                "covariate_id": np.arange(len(value)),
                "covariate_name": [col.name for col in value],
                "reference": np.array([col.reference for col in value], dtype=np.float),
                "max_difference": np.array([col.max_difference for col in value], dtype=np.float),
            }
        )

    @property
    def covariate_rename(self):
        """Covariates are stored in columns numbered ``x_0, x_1``, not by name.
        This is a dictionary from name to ``x_0``, ``x_1``, and so on.
        """
        covariate_df = self.dismod_file.covariate
        id_name = dict(covariate_df[["covariate_id", "covariate_name"]].to_records(index=False))
        return {name: f"x_{idx}" for (idx, name) in id_name.items()}

    @property
    def covariate_to_index(self):
        """Returns a dictionary from covariate name to its integer index."""
        covariate_df = self.dismod_file.covariate
        return dict(covariate_df[["covariate_name", "covariate_id"]].to_records(index=False))

    def _ensure_schema_has_covariates(self, x_underscore_columns):
        """Both the data and avgints need to have extra columns for covariates.
        Dismod-AT wants these defined, and at least an empty data and avgint
        table, before it will write the model. This step updates the list
        of covariates in the database schema before creating empty tables
        if necessary."""
        previous_rename = self.covariate_rename
        if set(x_underscore_columns) == set(previous_rename.values()):
            return
        # Only rewrite schema if the x_<integer> list has changed.
        # because the schema depends on the number of covariates, not
        # their names.
        covariate_columns = list(x_underscore_columns)
        # ASCII sorting isn't correct b/c x_11 is before x_2.
        covariate_columns.sort(key=lambda x: int(x[2:]))
        for create_name in ["data", "avgint"]:
            empty = self.dismod_file.empty_table(create_name)
            without = [c for c in empty.columns if not c.startswith("x_")]
            # The wrapper needs these columns to have a dtype of Real.
            empty = empty[without].assign(**{cname: np.empty((0,), dtype=np.float) for cname in covariate_columns})
            self.dismod_file.update_table_columns(create_name, empty)
            if getattr(self.dismod_file, create_name).empty:
                CODELOG.debug(f"Writing empty {create_name} table with columns {covariate_columns}")
                setattr(self.dismod_file, create_name, empty)
            else:
                CODELOG.debug(f"Adding to {create_name} table schema the columns {covariate_columns}")

    def ensure_dismod_file_has_default_tables(self):
        """
        There are a number of tables that a DismodFile should always have.
        If they aren't there, this makes them. If they are there, this
        doesn't change them.
        """
        db_file = self.dismod_file
        # Density table does not depend on model.
        if db_file.density.empty:
            db_file.density = pd.DataFrame({"density_name": [x.name for x in DensityEnum]})

        # Standard integrand naming scheme.
        all_integrands = default_integrand_names()
        if db_file.integrand.empty:
            self.dismod_file.integrand = all_integrands
            # Fill in the min_meas_cv later if required. Ensure integrand kinds have
            # known IDs early. Not nan because this "is non-negative and less than
            # or equal to one."
            self.dismod_file.integrand["minimum_meas_cv"] = 0

        if db_file.rate.empty:
            db_file.rate = pd.DataFrame(dict(
                rate_id=[rate.value for rate in RateEnum],  # Will be 0-4.
                rate_name=[rate.name for rate in RateEnum],
                parent_smooth_id=nan,
                child_smooth_id=nan,
                child_nslist_id=nan,
            ))

        # Defaults, empty, b/c Brad makes them empty even if there are none.
        for create_name in ["nslist", "nslist_pair", "mulcov", "smooth_grid", "smooth", "data", "avgint"]:
            if getattr(db_file, create_name).empty:
                setattr(db_file, create_name, self.dismod_file.empty_table(create_name))
        if db_file.log.empty:
            db_file.log = make_log_table()
        if db_file.option.empty:
            self._create_options_table()

    def _create_options_table(self):
        # https://bradbell.github.io/dismod_at/doc/option_table.htm
        # Only options in this list can be set.
        option = pd.DataFrame([
            dict(option_name="parent_node_id", option_value=nan),
            dict(option_name="parent_node_name", option_value=nan),
            dict(option_name="meas_noise_effect", option_value="add_var_scale_log"),
            dict(option_name="zero_sum_random", option_value=nan),
            dict(option_name="data_extra_columns", option_value=nan),
            dict(option_name="avgint_extra_columns", option_value=nan),
            dict(option_name="warn_on_stderr", option_value="true"),
            dict(option_name="ode_step_size", option_value="5.0"),
            dict(option_name="age_avg_split", option_value=nan),
            dict(option_name="random_seed", option_value="0"),
            dict(option_name="rate_case", option_value="iota_pos_rho_zero"),
            dict(option_name="derivative_test_fixed", option_value="none"),
            dict(option_name="derivative_test_random", option_value="none"),
            dict(option_name="max_num_iter_fixed", option_value="100"),
            dict(option_name="max_num_iter_random", option_value="100"),
            dict(option_name="print_level_fixed", option_value=5),
            dict(option_name="print_level_random", option_value=5),
            dict(option_name="accept_after_max_steps_fixed", option_value="5"),
            dict(option_name="accept_after_max_steps_random", option_value="5"),
            dict(option_name="tolerance_fixed", option_value="1e-8"),
            dict(option_name="tolerance_random", option_value="1e-8"),
            dict(option_name="quasi_fixed", option_value="false"),
            dict(option_name="bound_frac_fixed", option_value="1e-2"),
            dict(option_name="limited_memory_max_history_fixed", option_value="30"),
            dict(option_name="bound_random", option_value=nan),
        ], columns=["option_name", "option_value"])
        self.dismod_file.option = option.assign(option_id=option.index)
