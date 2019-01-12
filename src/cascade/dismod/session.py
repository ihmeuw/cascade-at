from contextlib import contextmanager
from math import nan
from pathlib import Path
from subprocess import run, PIPE

import numpy as np
import pandas as pd

from cascade.core import getLoggers
from cascade.dismod.db.wrapper import DismodFile, _get_engine
from cascade.dismod.model_reader import read_var_table_as_id, read_vars
from cascade.dismod.model_writer import ModelWriter

CODELOG, MATHLOG = getLoggers(__name__)


class DismodSession:
    def __init__(self, locations, parent_location, filename):
        """

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
        self.dismod_file.engine = _get_engine(self._filename)
        self.parent_location = parent_location

        self._create_node_table(locations)
        self._create_options_table()
        self._covariates = dict()  # From covariate name to the x_<number> name.
        for create_name in ["data", "avgint"]:
            setattr(self.dismod_file, create_name, self.dismod_file.empty_table(create_name))

    def fit(self, model, data):
        self.write(model)

        with self._close_db_while_running():
            completed_process = run(["dmdismod", str(self._filename), "init"], stdout=PIPE, stderr=PIPE)
            if completed_process.returncode != 0:
                print(completed_process.stdout, completed_process.stderr)
            assert completed_process.returncode == 0

        scale_vars = self.get_vars("scale")
        return scale_vars

    def get_vars(self, name):
        var_id = read_var_table_as_id(self.dismod_file)
        return read_vars(self.dismod_file, var_id, name)

    def set_option(self, name, value):
        rate_row = int(self.dismod_file.option[self.dismod_file.option.option_name == name].option_id)
        self.dismod_file.option.loc[rate_row, "option_value"] = value

    def set_covariates(self, rename_dict):
        """Both the data and avgints need to have extra columns for covariates.
        Dismod-AT wants these defined, and at least an empty data and avgint
        table, before it will write the model. This step updates the list
        of covariates in the database schema before creating empty tables
        if necessary."""
        if set(rename_dict.values()) == set(self._covariates.values()):
            self._covariates = rename_dict
            return
        else:
            # Only rewrite schema if the x_<integer> list has changed.
            self._covariates = rename_dict
        covariate_columns = list(sorted(self._covariates.values()))
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
    def _close_db_while_running(self):
        self.dismod_file.engine.dispose()
        try:
            yield
        finally:
            self.dismod_file.engine = _get_engine(self._filename)

    def _create_options_table(self):
        # Options in grey were rejected by Dismod-AT despite being in docs.
        # https://bradbell.github.io/dismod_at/doc/option_table.htm
        option = pd.DataFrame([
            dict(option_name="parent_node_id", option_value=str(self.location_func(self.parent_location))),
            # dict(option_name="meas_std_effect", option_value="add_std_scale_all"),
            dict(option_name="zero_sum_random", option_value=nan),
            dict(option_name="data_extra_columns", option_value=nan),
            dict(option_name="avgint_extra_columns", option_value=nan),
            dict(option_name="warn_on_stderr", option_value="true"),
            dict(option_name="ode_step_size", option_value="5.0"),
            # dict(option_name="age_avg_split", option_value=nan),
            dict(option_name="random_seed", option_value="0"),
            dict(option_name="rate_case", option_value="iota_pos_rho_zero"),
            # dict(option_name="derivative_test", option_value="none"),
            # dict(option_name="max_num_iter", option_value="100"),
            # dict(option_name="print_level", option_value=0),
            # dict(option_name="accept_after_max_steps", option_value="5"),
            # dict(option_name="tolerance", option_value="1e-8"),
            dict(option_name="quasi_fixed", option_value="true"),
            dict(option_name="bound_frac_fixed", option_value="1e-2"),
            dict(option_name="limited_memory_max_history_fixed", option_value="30"),
            dict(option_name="bound_random", option_value=nan),
        ], columns=["option_name", "option_value"])
        self.dismod_file.option = option.assign(option_id=option.index)

    def _create_node_table(self, locations):
        columns = dict(
            node_name=locations.name,
            parent=locations.parent,
        )
        # This adds c_location_id, if it's there.
        for add_column in [c for c in locations.columns if c.startswith("c_")]:
            columns[add_column] = locations[add_column]
        table = pd.DataFrame(columns)
        table["node_id"] = table.index

        def location_to_node_func(location_id):
            if np.isnan(location_id):
                return np.nan
            return np.where(table.c_location_id == location_id)[0][0]

        table["parent"] = table.parent.apply(location_to_node_func)
        self.dismod_file.node = table
        self.location_func = location_to_node_func

    def write(self, model):
        writer = ModelWriter(self)
        model.write(writer)
        writer.close()
        self.flush()

    def flush(self):
        self.dismod_file.flush()
