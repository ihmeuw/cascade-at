from collections import Iterable
from contextlib import contextmanager
from math import nan, isnan
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.core import getLoggers
from cascade.dismod.constants import DensityEnum, RateEnum
from cascade.dismod.db.wrapper import DismodFile, get_engine
from cascade.model.serialize import default_integrand_names, make_log_table
from cascade.model.data_read_write import (
    write_data, avgint_to_dataframe, read_avgint, read_data_residuals, read_simulation_data
)
from cascade.model.grid_read_write import (
    read_var_table_as_id, read_vars, write_vars, read_prior_residuals, read_samples,
    read_simulation_model
)
from cascade.model.model_writer import ModelWriter

CODELOG, MATHLOG = getLoggers(__name__)


class ObjectWrapper:
    """
    An I/O layer on top of the Dismod db file that presents Model objects.
    It sets and gets models, vars, data, and residuals.
    """
    def __init__(self, locations, parent_location, filename):
        assert isinstance(locations, pd.DataFrame)
        assert isinstance(parent_location, int)
        assert isinstance(filename, (Path, str))

        self._filename = Path(filename)
        self.dismod_file = None
        self.parent_location = parent_location
        self.location_func = None
        self._locations_df = locations

        # From covariate name to the x_<number> name that is used internally.
        # The session knows this piece of information but not the covariate
        # reference values. This is here because the columns of avgint and data
        # need to be renamed before writing, and they aren't part of the model.
        self._covariate_rename = dict()

    @property
    def model(self):
        raise NotImplementedError("Cannot read a model.")

    @model.setter
    def model(self, new_model):
        """When you write a model, it deletes the file."""
        self.make_new_dismod_file(self._locations_df)
        writer = ModelWriter(self)
        new_model.write(writer)
        writer.close()

    def set_option(self, **kwargs):
        """Erase an option by setting it to None or nan."""
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

    @property
    def data(self):
        raise NotImplementedError("Cannot read data.")

    @data.setter
    def data(self, data):
        write_data(self.dismod_file, data, self._covariate_rename)

    @property
    def avgint(self):
        raise NotImplementedError("Cannot read avgint.")

    @avgint.setter
    def avgint(self, avgint):
        self.dismod_file.avgint = avgint_to_dataframe(self.dismod_file, avgint, self.covariate_rename)

    @property
    def start_var(self):
        return self.get_var("start")

    @start_var.setter
    def start_var(self, new_vars):
        self.set_var("start", new_vars)

    @property
    def scale_var(self):
        return self.get_var("scale")

    @scale_var.setter
    def scale_var(self, new_vars):
        self.set_var("scale", new_vars)

    @property
    def fit_var(self):
        return self.get_var("fit")

    @fit_var.setter
    def fit_var(self, new_vars):
        self.set_var("fit", new_vars)

    @property
    def truth_var(self):
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

    @property
    def prior_residuals(self):
        var_id = read_var_table_as_id(self.dismod_file)
        return read_prior_residuals(self.dismod_file, var_id)

    @property
    def data_residuals(self):
        return read_data_residuals(self.dismod_file)

    @property
    def samples(self):
        var_id = read_var_table_as_id(self.dismod_file)
        return read_samples(self.dismod_file, var_id)

    def read_simulation_model_and_data(self, model, data, index):
        var_id = read_var_table_as_id(self.dismod_file)
        sim_model = read_simulation_model(self.dismod_file, model, var_id, index)
        sim_data = read_simulation_data(self.dismod_file, data, index)
        return sim_model, sim_data

    def refresh(self, list_of_tables):
        self.dismod_file.refresh(list_of_tables)

    def flush(self):
        self.dismod_file.flush()

    def close(self):
        self.flush()
        if self.dismod_file.engine is not None:
            self.dismod_file.engine.dispose()
            self.dismod_file.engine = None

    @property
    def log(self):
        self.dismod_file.refresh(["log"])
        return self.dismod_file.log

    @property
    def predict(self):
        avgint = read_avgint(self.dismod_file)
        raw = self.dismod_file.predict.merge(avgint, on="avgint_id", how="left")
        normalized = raw.drop(columns=["avgint_id", "predict_id"]).rename(columns={"avg_integrand": "mean"})
        not_predicted = avgint[~avgint.avgint_id.isin(raw.avgint_id)].drop(columns=["avgint_id"])
        return normalized, not_predicted

    @property
    def locations(self):
        raise NotImplementedError("Cannot get locations")

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
        self.location_func = location_to_node_func

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

    @contextmanager
    def close_db_while_running(self):
        if self.dismod_file.engine is not None:
            self.dismod_file.engine.dispose()
            self.dismod_file.engine = None
        try:
            yield
        finally:
            self.dismod_file.engine = get_engine(self._filename)

    def make_new_dismod_file(self, locations):
        if self._filename.exists():
            MATHLOG.info(f"{self._filename} exists so overwriting it.")
            self._filename.unlink()

        self.dismod_file = DismodFile()
        self.dismod_file.engine = get_engine(self._filename)

        self._covariate_rename = dict()
        if locations is not None:
            self.locations = locations
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
        for create_name in ["nslist", "nslist_pair", "mulcov", "smooth_grid", "smooth", "data", "avgint"]:
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
