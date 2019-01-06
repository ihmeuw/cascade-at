"""
Writes a Model to a Dismod File.
"""
from math import nan
from numbers import Real

import numpy as np
import pandas as pd

from cascade.core.log import getLoggers
from cascade.dismod.db.metadata import DensityEnum
from cascade.dismod.db.wrapper import DismodFile
from cascade.dismod.serialize import (
    enum_to_dataframe, default_integrand_names, make_log_table, simplest_weight
)

CODELOG, MATHLOG = getLoggers(__name__)


class DismodSession:
    def __init__(self, locations):
        """

        Args:
            locations (pd.DataFrame): Initialize here because data refers to this.
        """
        self.dismod_file = DismodFile()
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


class ModelWriter:
    """This layer over the Dismod File is responsible for translation between
    absolute numbers in the Model and SQL-indexed numbers.

    This has an API that the client has to follow. Functions need to be called
    in order.

    For naming in the DismodFile, this chooses names based on the random
    fields.
    """

    def __init__(self, session):
        self._session = session
        self._dismod_file = session.dismod_file
        self._ages = np.empty((0,), dtype=np.float)
        self._times = np.empty((0,), dtype=np.float)
        self._rate_rows = list()  # List of dictionaries for rates.
        self._mulcov_rows = list()  # List of dictionaries for covariate multipliers.
        self._rate_id = dict()  # The rate ids with the primary rates.
        self._nslist = dict()  # rate to integer
        self._nslist_pair = dict()  # From nslist rate to list of (node, smooth)
        self._covariate_id = lambda x: nan
        self._flushed = False
        self._children = None

    def basic_db_setup(self):
        """These things are true for all databases."""
        # Density table does not depend on model.
        density_enum = enum_to_dataframe(DensityEnum)
        densities = pd.DataFrame({"density_name": density_enum["name"]})
        self._dismod_file.density = densities

        # Standard integrand naming scheme.
        all_integrands = default_integrand_names()
        self._dismod_file.integrand = all_integrands
        # Fill in the min_meas_cv later if required. Ensure integrand kinds have
        # known IDs early.
        self._dismod_file.integrand["minimum_meas_cv"] = nan

        # Defaults, empty, b/c Brad makes them empty even if there are none.
        for create_name in ["nslist", "nslist_pair", "mulcov", "smooth_grid", "smooth"]:
            setattr(self._dismod_file, create_name, self._dismod_file.empty_table(create_name))
        self._dismod_file.log = make_log_table()

    def start_model(self, nonzero_rates, children):
        """To start the model, tell me enough to know the sizes almost all of
        the tables by telling me the rate count and child random effect count."""
        self._rate_id = dict((y, x) for (x, y) in enumerate(sorted(nonzero_rates)))
        self._children = children

    def write_ages_and_times(self, ages, times):
        """
        Collects all of the ages and times because they are ordered
        and indexed by Dismod's file format. Can be called multiple times.
        """
        self._ages = np.append(self._ages, ages)
        self._times = np.append(self._times, times)

    def write_covariate(self, covariates):
        self._dismod_file.covariate = self._dismod_file.empty_table("covariate")
        CODELOG.debug(f"Covariate types {self._dismod_file.covariate.dtypes}")
        self._dismod_file.covariate, cov_col_id_func, covariate_renames = _make_covariate_table(covariates)
        self._covariate_id = cov_col_id_func

    def write_rate(self, rate_name, random_field):
        """A rate needs a smooth, which has priors and ages/times."""
        self._flush_ages_times_locations()
        smooth_id = self.add_random_field(rate_name, random_field)
        self._rate_rows.append(
            {
                "rate_id": self._rate_id[rate_name],
                "rate_name": rate_name,
                "parent_smooth_id": smooth_id,
                "child_smooth_id": nan,
                "child_nslist_id": nan,
            }
        )

    def write_random_effect(self, rate_name, child_location, random_field):
        self._flush_ages_times_locations()
        if child_location is None:
            grid_name = f"{rate_name}_re"
        else:
            grid_name = f"{rate_name}_re_{child_location}"
        smooth_id = self.add_random_field(grid_name, random_field)
        if child_location is None:
            self._rate_rows.append(
                {
                    "rate_id": len(self._rate_rows),
                    "rate_name": rate_name,
                    "parent_smooth_id": nan,
                    "child_smooth_id": smooth_id,
                    "child_nslist_id": nan,
                }
            )
        else:
            node_id = self._session.location_func(child_location)
            if rate_name in self._nslist:
                self._nslist_pair[rate_name].append((node_id, smooth_id))
            else:
                self._nslist[rate_name] = len(self._nslist)
                self._nslist_pair[rate_name] = [(node_id, smooth_id)]

    def write_mulcov(self, kind, covariate, rate_or_integrand, random_field):
        self._flush_ages_times_locations()
        print(f"write_mulcov {kind} {covariate} {rate_or_integrand}")
        grid_name = f"{kind}_{rate_or_integrand}_{covariate}"
        smooth_id = self.add_random_field(grid_name, random_field)
        if kind == "alpha":
            self._mulcov_rows.append(
                {
                    "mulcov_id": len(self._mulcov_rows),
                    "mulcov_type": "rate_value",
                    "rate_id": self._rate_id[rate_or_integrand],
                    "integrand_id": nan,
                    "covariate_id": nan,
                    "smooth_id": smooth_id,
                }
            )

    def write_weight(self, name, field_draw):
        # FIXME: The weight implementation has to create its own grid. Much simpler.
        self._dismod_file.weight, self._dismod_file.weight_grid = simplest_weight()

    def close(self):
        # Write nslists.
        pass

    def add_random_field(self, grid_name, random_field):
        """Save a new Random Field."""
        complete_table = self._add_field_priors(grid_name, random_field.priors.copy())
        smooth_id = self._add_field_smooth(grid_name, complete_table)
        self._add_field_grid(complete_table, smooth_id)
        return smooth_id

    def _add_field_grid(self, complete_table, smooth_id):
        long_table = complete_table.loc[complete_table.age_id.notna()][["age_id", "time_id", "prior_id", "kind"]]
        grid_table = long_table[["age_id", "time_id"]]
        for kind in ["value", "dage", "dtime"]:
            grid_values = long_table.loc[long_table.kind == kind] \
                .drop("kind", axis="columns") \
                .rename(columns={"prior_id": f"{kind}_prior_id"})
            grid_table = grid_table.merge(grid_values, on=["age_id", "time_id"])
        grid_table = grid_table.sort_values(["age_id", "time_id"], axis=0).reindex()
        grid_table = grid_table.assign(const_value=nan, smooth_id=smooth_id)
        if self._dismod_file.smooth_grid.empty:
            self._dismod_file.smooth_grid = grid_table.assign(smooth_grid_id=grid_table.index)
        else:
            grid_table = grid_table.assign(smooth_grid_id=grid_table.index + len(self._dismod_file.smooth_grid))
            self._dismod_file.smooth_grid = self._dismod_file.smooth_grid.append(grid_table, ignore_index=True)

    def _add_field_smooth(self, grid_name, prior_table):
        smooth_row = dict(smooth_name=grid_name)
        smooth_row["n_age"] = len(prior_table.age_id.unique())
        smooth_row["n_time"] = len(prior_table.time_id.unique())
        for prior_kind in ["value", "dage", "dtime"]:
            smooth_row[f"mulstd_{prior_kind}_prior_id"] = prior_table.loc[
                prior_table.age.isna() & (prior_table.kind == prior_kind)
                ]
        if self._dismod_file.smooth.empty:
            smooth_row["smooth_id"] = 0
            self._dismod_file.smooth = self._dismod_file.empty_table("smooth").append(smooth_row, ignore_index=True)
        else:
            smooth_row["smooth_id"] = len(self._dismod_file.smooth)
            self._dismod_file.smooth = self._dismod_file.smooth.append(smooth_row, ignore_index=True)
        smooth_id = smooth_row["smooth_id"]
        return smooth_id

    def _add_field_priors(self, grid_name, complete_table):
        # Create new prior IDs that don't overlap.
        complete_table = complete_table.assign(prior_id=complete_table.index + len(self._dismod_file.prior))
        # Unique, informative names for the priors require care.
        null_names = complete_table.prior_name.isnull()
        complete_table.loc[~null_names, "prior_name"] = (
                complete_table.loc[~null_names, "prior_name"] + "    " +
                complete_table.loc[~null_names, "prior_id"].astype(str)
        )
        complete_table.loc[null_names, "prior_name"] = complete_table.loc[
            null_names, "prior_id"].apply(
            lambda pid: f"{grid_name}_{pid}"
        )
        # Assign age_id and time_id for age and time.
        complete_table = pd.merge_asof(complete_table.sort_values("age"), self._dismod_file.age, on="age")
        complete_table = pd.merge_asof(complete_table.sort_values("time"), self._dismod_file.time, on="time")
        # Make sure the index still matches the order in the priors list
        priors_columns = [
            "prior_id", "prior_name", "lower", "upper", "mean", "std", "eta", "nu", "density_id"
        ]
        prior_table = complete_table.sort_values(by="prior_id").reset_index(drop=True)[priors_columns]
        if self._dismod_file.prior.empty:
            self._dismod_file.prior = self._dismod_file.prior.append(prior_table)
        else:
            self._dismod_file.prior = prior_table
        return complete_table

    def _flush_ages_times_locations(self):
        if self._flushed:
            return
        unique_ages = self._ages[np.unique(self._ages.round(decimals=14), return_index=True)[1]]
        unique_ages.sort()
        self._dismod_file.age = pd.DataFrame(dict(age_id=range(len(unique_ages)), age=unique_ages))
        unique_times = self._times[np.unique(self._times.round(decimals=14), return_index=True)[1]]
        unique_times.sort()
        self._dismod_file.time = pd.DataFrame(dict(time_id=range(len(unique_times)), time=unique_times))
        self._flushed = True


def _make_covariate_table(covariates):
    null_references = list()
    for check_ref_col in covariates:
        if not isinstance(check_ref_col.reference, Real):
            null_references.append(check_ref_col.name)
    if null_references:
        raise RuntimeError(f"Covariate columns without reference values {null_references}")

    # Dismod requires us to rename covariates from names like sex, and "one"
    # to x_0, x_1,... They must be "x_<digit>".
    renames = dict()
    for cov_idx, covariate in enumerate(covariates):
        renames[covariate.name] = f"x_{cov_idx}"

    covariate_columns = pd.DataFrame(
        {
            "covariate_id": np.arange(len(covariates)),
            "covariate_name": [renames[col.name] for col in covariates],
            "reference": np.array([col.reference for col in covariates], dtype=np.float),
            "max_difference": np.array([col.max_difference for col in covariates], dtype=np.float),
        }
    )

    def cov_col_id_func(query_column):
        """From the original covariate name to the index in SQL file."""
        return [search.name for search in covariates].index(query_column)

    return covariate_columns, cov_col_id_func, renames.get
