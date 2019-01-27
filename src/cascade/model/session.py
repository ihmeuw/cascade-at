from collections import Iterable
from contextlib import contextmanager
from math import nan
from pathlib import Path
from subprocess import run, PIPE

import numpy as np
import pandas as pd

from cascade.core import getLoggers
from cascade.dismod.constants import DensityEnum, RateEnum, COMMAND_IO
from cascade.dismod.db.wrapper import DismodFile, get_engine
from cascade.dismod.serialize import default_integrand_names, make_log_table
from cascade.model import Model
from cascade.model.data_read_write import (
    write_data, write_avgint, read_avgint, read_data_residuals, read_simulation_data
)
from cascade.model.grid_read_write import (
    read_var_table_as_id, read_vars, write_vars, read_prior_residuals, read_samples,
    read_simulation_model
)
from cascade.model.model_writer import ModelWriter

CODELOG, MATHLOG = getLoggers(__name__)


class Session:
    def __init__(self, locations, parent_location, filename):
        """
        A session represents a connection with a Dismod-AT backend through
        a single Dismod-AT db file, the sqlite file it uses for input and
        output.

        Args:
            locations (pd.DataFrame): Initialize here because data refers to this.
            parent_location (int): The session uses parent location to subset
                                   data, but it isn't in the model.
            filename (str|Path): Location of the Dismod db to overwrite.
        """
        self.dismod_file = DismodFile()
        self._filename = Path(filename)
        if self._filename.exists():
            MATHLOG.info(f"{self._filename} exists so overwriting it.")
            self._filename.unlink()
        self.dismod_file.engine = get_engine(self._filename)
        self.parent_location = parent_location

        self._basic_db_setup(locations)
        # From covariate name to the x_<number> name that is used internally.
        # The session knows this piece of information but not the covariate
        # reference values. This is here because the columns of avgint and data
        # need to be renamed before writing, and they aren't part of the model.
        self._covariate_rename = dict()
        for create_name in ["data", "avgint"]:
            setattr(self.dismod_file, create_name, self.dismod_file.empty_table(create_name))

    def fit(self, model, data, initial_guess=None):
        """This is a fit without a predict. If the model
        has random effects, this optimizes over both fixed
        and random effects.

        Args:
            model (Model): A model, possibly without scale vars.
            data (pd.DataFrame): Data to fit.
            initial_guess (Var): Starting point to look for solutions. If not
                given, then the mean of the priors is taken as the initial
                guess.

        Returns:
            DismodGroups[Var]: A set of fit var.
        """
        if model.random_effect:
            MATHLOG.info(f"Running fit both.")
            return self._fit("both", model, data, initial_guess)
        else:
            MATHLOG.info(f"Running fit fixed.")
            return self._fit("fixed", model, data, initial_guess)

    def fit_fixed(self, model, data, initial_guess=None):
        """Fits a model without optimizing over any random effects.
        It does apply constant child value priors, but other random effects
        are constrained to zero. (This is equivalent to fitting with
        ``bound_random`` equal to zero.) This is useful when one uses fitting
        with no random effects as a starting point for fitting with
        random effects.

        Args:
            model (Model): A model, possibly without scale vars.
            data (pd.DataFrame): Data to fit.
            initial_guess (Var): Starting point to look for solutions. If not
                given, then the mean of the priors is taken as the initial
                guess.

        Returns:
            DismodGroups[Var]: A set of fit var.
        """
        return self._fit("fixed", model, data, initial_guess)

    def fit_random(self, model, data, initial_guess=None):
        """
        Fits the data with the model.
        This optimizes the random effects with the fixed effects set to their
        starting values. The fixed effects are unchanged.

        Args:
            model (Model): A model, possibly without scale vars.
            data (pd.DataFrame): Data to fit.
            initial_guess (Var): Starting point to look for solutions. If not
                given, then the mean of the priors is taken as the initial
                guess.

        Returns:
            DismodGroups[Var]: A set of fit var.
        """
        return self._fit("random", model, data, initial_guess)

    def _fit(self, fit_level, model, data, initial_guess):
        self._setup_model_for_fit(model, data)
        if initial_guess is not None:
            MATHLOG.info(f"Setting initial value for search from user argument.")
            self.set_var("start", initial_guess)
        # else use the one generated by the call to init, coming from the mean.
        self._run_dismod(["fit", fit_level])
        return FitResult(self, self.get_var("fit"))

    def _setup_model_for_fit(self, model, data):
        extremal = list()
        if data is not None and not data.empty:
            for dimension in ["age", "time"]:
                cols = [ac for ac in data.columns if ac.startswith(dimension)]
                if not cols:
                    raise ValueError(f"Dataframe must have age and time columns but has {data.columns}.")
                extremal.append({data[cols].min().min(), data[cols].max().max()})
        self.write_model(model, extremal)
        write_data(self.dismod_file, data, self._covariate_rename)
        self._run_dismod(["init"])
        if model.scale_set_by_user:
            self.set_var("scale", model.scale)
        else:
            # Assign to the private variable because setting the property
            # indicates that the user of the API wants to set their own scale
            # instead of using the one Dismod-AT calculates during init.
            model._scale = self.get_var("scale")

    def predict(self, var, avgint, parent_location, weights=None, covariates=None):
        """Given rates, calculated the requested average integrands.

        Args:
            var (DismodGroups): Var objects with rates.
            avgint (pd.DataFrame): Request data in these ages, times, and
                locations. Columns are ``integrand`` (str), ``location``
                (location_id), ``age_lower`` (float), ``age_upper`` (float),
                ``time_lower`` (float), ``time_upper`` (float). The integrand
                should be one of the names in IntegrandEnum.
            parent_location: The id of the parent location.
            weights (Dict[Var]): Weights are estimates of ``susceptible``,
                ``with_condition``, and ``total`` populations, used to bias
                integrands with age or time extent. Each one is a single
                Var object.
            covariates (List[Covariate]): A list of Covariates, so that we know
                the name and reference value for each.

        Returns:
            (pd.DataFrame, pd.DataFrame): The predicted avgints, and a dataframe
            of those not predicted because their covariates are greater than
            ``max_difference`` from the ``reference`` covariate value.
            Columns in the ``predicted`` are ``predict_id``, ``sample_index``,
            ``avg_integrand`` (this is the value), ``location``, ``integrand``,
            ``age_lower``, ``age_upper``, ``time_lower``, ``time_upper``.
        """
        self._check_vars(var)
        model = Model.from_var(var, parent_location, weights=weights, covariates=covariates)
        extremal = ({avgint.age_lower.min(), avgint.age_upper.max()},
                    {avgint.time_lower.min(), avgint.time_upper.max()})
        self.write_model(model, extremal)
        self.dismod_file.avgint = write_avgint(self.dismod_file, avgint, self._covariate_rename)
        self._run_dismod(["init"])
        self.set_var("truth", var)
        self._run_dismod(["predict", "truth_var"])
        predicted, not_predicted = self.get_predict()
        return predicted, not_predicted

    def simulate(self, model, data, fit_var, simulate_count):
        """Simulates posterior distribution for model variables.

        This is described in several places:
        https://bradbell.github.io/dismod_at/doc/posterior.htm
        https://bradbell.github.io/dismod_at/doc/simulate_command.htm
        https://bradbell.github.io/dismod_at/doc/user_posterior.py.htm

        Args:
            model (Model): A model. The mean of the prior is ignored.
            data (DataFrame): Same format as for a fit.
            fit_var (Var): A set of model variables around which to simulate.
            simulate_count (int): Number of simulations to generate.

        Returns:
            (DataFrame, Groups of SmoothGrids): These are the data simulations
            and the prior simulations. The former are stacked in a dataframe
            with an index, and the latter are in a DismodGroups container
            of SmoothGrids.
        """
        # The data is ordered and stays ordered through
        # to construction of SimulateResult.
        data = data.reset_index(drop=True)
        self._setup_model_for_fit(model, data)
        self.set_var("truth", fit_var)
        self._run_dismod(["simulate", simulate_count])
        return SimulateResult(self, simulate_count, model, data)

    def sample(self, simulate_result):
        """Given that a simulate has been run, make samples.

        Args:
            simulate_result (SimulateResult): Output of a simulate command.

        Returns:
            DismodGroups[Var] with multiple samples.
        """
        self._run_dismod(["sample", "simulate", simulate_result.count])
        # Sample creates a sample table.
        var_id = read_var_table_as_id(self.dismod_file)
        return read_samples(self.dismod_file, var_id)

    def get_var(self, name):
        var_id = read_var_table_as_id(self.dismod_file)
        return read_vars(self.dismod_file, var_id, name)

    def set_var(self, name, new_vars):
        var_id = read_var_table_as_id(self.dismod_file)
        write_vars(self.dismod_file, new_vars, var_id, name)
        self.flush()

    def get_prior_residuals(self):
        var_id = read_var_table_as_id(self.dismod_file)
        return read_prior_residuals(self.dismod_file, var_id)

    def get_data_residuals(self):
        return read_data_residuals(self.dismod_file)

    def set_option(self, **kwargs):
        option = self.dismod_file.option
        unknowns = list()
        for name, value in kwargs.items():
            if not (option.option_name == name).any():
                unknowns.append(name)
            if isinstance(value, str):
                str_value = value
            elif isinstance(value, Iterable):
                str_value = " ".join(str(x) for x in value)
            elif isinstance(value, bool):
                str_value = str(value).lower()
            else:
                str_value = str(value)
            option.loc[option.option_name == name, "option_value"] = str_value
        if unknowns:
            raise KeyError(f"Unknown options {unknowns}")

    @property
    def covariate_rename(self):
        return self._covariate_rename

    @covariate_rename.setter
    def covariate_rename(self, rename_dict):
        """Both the data and avgints need to have extra columns for covariates.
        Dismod-AT wants these defined, and at least an empty data and avgint
        table, before it will write the model. This step updates the list
        of covariates in the database schema before creating empty tables
        if necessary."""
        if set(rename_dict.values()) == set(self._covariate_rename.values()):
            self._covariate_rename = rename_dict
            return
        else:
            # Only rewrite schema if the x_<integer> list has changed.
            self._covariate_rename = rename_dict
        covariate_columns = list(sorted(self._covariate_rename.values()))
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

    def _run_dismod(self, command):
        """Pushes tables to the db file, runs Dismod-AT, and refreshes
        tables written."""
        self.flush()
        with self._close_db_while_running():
            str_command = [str(c) for c in command]
            completed_process = run(["dmdismod", str(self._filename)] + str_command, stdout=PIPE, stderr=PIPE)
            if completed_process.returncode != 0:
                MATHLOG.error(completed_process.stdout.decode())
                MATHLOG.error(completed_process.stderr.decode())
            assert completed_process.returncode == 0, f"return code is {completed_process.returncode}"
        if command[0] in COMMAND_IO:
            self.dismod_file.refresh(COMMAND_IO[command[0]].output)

    @contextmanager
    def _close_db_while_running(self):
        self.dismod_file.engine.dispose()
        try:
            yield
        finally:
            self.dismod_file.engine = get_engine(self._filename)

    @staticmethod
    def _check_vars(var):
        for group_name, group in var.items():
            for key, one_var in group.items():
                one_var.check(f"{group_name}-{key}")

    def write_model(self, model, extremal_age_time):
        writer = ModelWriter(self, extremal_age_time)
        model.write(writer)
        writer.close()
        self.flush()

    def flush(self):
        self.dismod_file.flush()

    def get_predict(self):
        avgint = read_avgint(self.dismod_file)
        raw = self.dismod_file.predict.merge(avgint, on="avgint_id", how="left")
        not_predicted = avgint[~avgint.avgint_id.isin(raw.avgint_id)].drop(columns=["avgint_id"])
        return raw.drop(columns=["avgint_id", "predict_id"]), not_predicted

    def _basic_db_setup(self, locations):
        """These things are true for all databases."""
        self._create_node_table(locations)

        # Density table does not depend on model.
        self.dismod_file.density = pd.DataFrame({"density_name": [x.name for x in DensityEnum]})

        # Standard integrand naming scheme.
        all_integrands = default_integrand_names()
        self.dismod_file.integrand = all_integrands
        # Fill in the min_meas_cv later if required. Ensure integrand kinds have
        # known IDs early. Not nan because this "is non-negative and less than
        # or equal to one."
        self.dismod_file.integrand["minimum_meas_cv"] = 0

        self.dismod_file.rate = pd.DataFrame(dict(
            rate_id=[rate.value for rate in RateEnum],  # Will be 0-4.
            rate_name=[rate.name for rate in RateEnum],
            parent_smooth_id=nan,
            child_smooth_id=nan,
            child_nslist_id=nan,
        ))

        # Defaults, empty, b/c Brad makes them empty even if there are none.
        for create_name in ["nslist", "nslist_pair", "mulcov", "smooth_grid", "smooth"]:
            setattr(self.dismod_file, create_name, self.dismod_file.empty_table(create_name))
        self.dismod_file.log = make_log_table()
        self._create_options_table()

    def _create_options_table(self):
        # https://bradbell.github.io/dismod_at/doc/option_table.htm
        option = pd.DataFrame([
            dict(option_name="parent_node_id", option_value=str(self.location_func(self.parent_location))),
            dict(option_name="parent_node_name", option_value=nan),
            dict(option_name="meas_std_effect", option_value="add_std_scale_all"),
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

    def _create_node_table(self, locations):
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
        self.location_func = location_to_node_func


class FitResult:
    """Outcome of a Dismod-AT fit"""
    def __init__(self, session, fit_var):
        self._session = session
        self._fit_var = fit_var

    @property
    def fit(self):
        """All model variables. This is a DismodGroups instance."""
        return self._fit_var

    @property
    def prior_residuals(self):
        """The difference between model variables and their prior means.
        Prior residuals in a DismodGroups instance."""
        return self._session.get_prior_residuals()

    @property
    def data_residuals(self):
        """The difference between input data and output estimates of data.
        A DataFrame of residuals, identified by name from input data."""
        return self._session.get_data_residuals()

    @property
    def fit_data(self):
        """Which of the data points were fit."""
        raise NotImplementedError(f"Cannot retrieve fit data subset.")

    @property
    def excluded_data(self):
        """Which of the data points were excluded due
        to hold outs or covariates."""
        raise NotImplementedError(f"Cannot retrieve excluded data points.")


class SimulateResult:
    """Outcome of a Dismod-AT Simulate."""
    def __init__(self, session, count, model, data):
        self._session = session
        self._count = count
        self._model = model
        self._data = data

    @property
    def count(self):
        return self._count

    def simulation(self, index):
        """Retrieve one of the simulations.

        Args:
            index (int): Which simulation to retrieve, zero-based.

        Returns:
            Model, Data: A new model and data, modified to be
            the Nth simulation.
        """
        var_id = read_var_table_as_id(self._session.dismod_file)
        sim_model = read_simulation_model(self._session.dismod_file, self._model, var_id, index)
        sim_data = read_simulation_data(self._session.dismod_file, self._data, index)
        return sim_model, sim_data

    @property
    def data(self):
        """Simulation of the data.
        This is a DataFrame identified by the name of the input data.
        It is restricted to the fit data subset. It has an extra
        ``index`` column to identify the simulation index."""
        raise NotImplementedError(f"Cannot retrieve data simulation table.")

    @property
    def prior(self):
        """Simulation of the prior.
        This is an AgeTimeGrid table where each entry is a set of three
        priors, and there are sets of values, identified by an index."""
        raise NotImplementedError(f"Cannot retrieve the prior simulation table.")
