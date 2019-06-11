from pathlib import Path

import pandas as pd

from cascade.core import getLoggers
from cascade.model import Model
from cascade.model.data_read_write import amend_data_input, point_age_time_to_interval
from cascade.model.object_wrapper import ObjectWrapper

CODELOG, MATHLOG = getLoggers(__name__)


class Session:
    """
    A Session interacts with Dismod-AT. It estimates fits,
    predicts rates, and simulates. Collaborates with the ObjectWrapper
    to manipulate the DismodFile.
    """
    def __init__(self, locations, parent_location, filename):
        """
        A session represents a connection with a Dismod-AT backend through
        a single Dismod-AT db file, the sqlite file it uses for input and
        output.

        Args:
            locations (pd.DataFrame): Both the model and data refer to a
                hierarchy of locations. Supply those as a DataFrame
                with ``location_id`` as an integer, ``parent_id`` as an integer,
                and an optional ``name`` as a string.
            parent_location (int): The session uses parent location to subset
                data, but it isn't in the model. This is a location ID supplied
                in the locations argument.
            filename (str|Path): Location of the Dismod db to overwrite.
        """
        assert isinstance(locations, pd.DataFrame)
        assert isinstance(parent_location, int)
        assert isinstance(filename, (Path, str))

        self._filename = Path(filename)
        if self._filename.exists():
            CODELOG.info(f"{self._filename} exists so overwriting it.")
            self._filename.unlink()
        self._objects = ObjectWrapper(filename)
        # Every time a new file is made, these local objects are set again
        # in the dismod objects.
        self._locations = locations
        self._parent_location = parent_location
        self._options = dict()
        self._age_extents = None

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
        if initial_guess:
            misalignment = model.check_alignment(initial_guess)
            if misalignment:
                raise ValueError(f"Model and initial guess are misaligned: {misalignment}.")
        self.setup_model_for_fit(model, data, initial_guess)
        # else use the one generated by the call to init, coming from the mean.
        dm_out, dm_err = self._objects.run_dismod(["fit", fit_level])
        return FitResult(self._objects, self._objects.fit_var, dm_out, dm_err)

    def setup_model_for_fit(self, model, data=None, initial_guess=None):
        """Writes a model and options to a db file and runs init on it.
        This isn't normally run in the course of work but can be helpful
        if you want to tweak the db file before running a fit.

        Args:
            model (Model): The model object.
            data (pd.DataFrame|None): Can be None.
            initial_guess (Var|None): Initial values, can be None.
        """
        self._objects.locations = self._locations
        self._objects.parent_location_id = self._parent_location
        self._objects.model = model
        # The data has to be in there before init in order to build
        # the data subset table.
        self._objects.data = data
        self._objects.set_option(**self._options)
        if self._age_extents:
            self._objects.age_extents = self._age_extents
        self._objects.run_dismod("init")
        if model.scale_set_by_user:
            self._objects.scale_var = model.scale
        # else Dismod uses the initial guess as the scale.
        if initial_guess is not None:
            self._objects.start_var = initial_guess
        # else Dismod-AT uses the distribution means as the start_var.

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
            Columns in the ``predicted`` are ``sample_index``,
            ``mean`` (this is the value), ``location``, ``integrand``,
            ``age_lower``, ``age_upper``, ``time_lower``, ``time_upper``.
        """
        self._check_vars(var)
        model = Model.from_var(var, parent_location, weights=weights, covariates=covariates)
        self.setup_model_for_fit(model)
        avgint = point_age_time_to_interval(avgint)
        self._objects.avgint = avgint
        self._objects.truth_var = var
        self._objects.run_dismod(["predict", "truth_var"])
        predicted, not_predicted = self._objects.predict
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
        # Ensure data has name, nu, eta, time_upper and lower.
        if fit_var:
            misalignment = model.check_alignment(fit_var)
            if misalignment:
                raise ValueError(f"Model and fit var are misaligned: {misalignment}.")
        self.setup_model_for_fit(model, data, fit_var)
        self._objects.truth_var = fit_var
        self._objects.run_dismod(["simulate", simulate_count])
        amended_data = amend_data_input(data)
        return SimulateResult(self._objects, simulate_count, model, amended_data)

    def sample(self, simulate_result):
        """Given that a simulate has been run, make samples.

        Args:
            simulate_result (SimulateResult): Output of a simulate command.

        Returns:
            DismodGroups[Var] with multiple samples.
        """
        self._objects.run_dismod(["sample", "simulate", simulate_result.count])
        return self._objects.samples

    def set_option(self, **kwargs):
        self._options.update(kwargs)
        if self._objects.dismod_file:
            self._objects.set_option(**self._options)

    @property
    def age_extents(self):
        return self._age_extents

    @age_extents.setter
    def age_extents(self, ages):
        self._age_extents = ages

    def set_minimum_meas_cv(self, **kwargs):
        """Sets the minimum coefficient of variation for this integrand.
        The name is one of :py:class:`cascade.dismod.constants.IntegrandEnum`.
        integrand_name (str) The canonical Dismod-AT name for the integrand.
        value (float) A value greater-than or equal to zero. If it is
        zero, then there is no coefficient of variation for this integrand.

        Args:
            name-value pairs: This is a set of integrand=value pars.
        """
        for integrand_name, value in kwargs.items():
            self._objects.set_minimum_meas_cv(integrand_name, value)

    @staticmethod
    def _check_vars(var):
        for group_name, group in var.items():
            for key, one_var in group.items():
                one_var.check(f"{group_name}-{key}")


class FitResult:
    """Outcome of a Dismod-AT fit"""
    def __init__(self, file_objects, fit_var, dm_out, dm_err):
        self._file_objects = file_objects
        self._fit_var = fit_var
        self.dismod_out = dm_out
        self.dismod_err = dm_err

    @property
    def success(self):
        return "Optimal Solution Found" in self.dismod_out

    @property
    def fit(self):
        """All model variables. This is a DismodGroups instance."""
        return self._fit_var

    @property
    def prior_residuals(self):
        """The difference between model variables and their prior means.
        Prior residuals in a DismodGroups instance."""
        return self._file_objects.prior_residuals

    @property
    def data_residuals(self):
        """The difference between input data and output estimates of data.
        A DataFrame of residuals, identified by name from input data."""
        return self._file_objects.data_residuals

    @property
    def fit_data(self):
        """Which of the data points were fit."""
        raise NotImplementedError(f"Retrieve fit data subset not implemented.")

    @property
    def excluded_data(self):
        """Which of the data points were excluded due
        to hold outs or covariates."""
        raise NotImplementedError(f"Retrieve excluded data points not implemented.")


class SimulateResult:
    """Outcome of a Dismod-AT Simulate."""
    def __init__(self, file_objects, count, model, data):
        self._file_objects = file_objects
        self._count = count
        self._model = model
        self._data = data

    @property
    def count(self):
        return self._count

    def simulation(self, index):
        """Retrieve one of the simulations as a model and data.

        Args:
            index (int): Which simulation to retrieve, zero-based.

        Returns:
            Model, Data: A new model and data, modified to be
            the Nth simulation.
        """
        return self._file_objects.read_simulation_model_and_data(self._model, self._data, index)
