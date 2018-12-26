"""
Writes a Model to a Dismod File.
"""
from math import nan
from numbers import Real

import numpy as np
import pandas as pd

from cascade.dismod.serialize import (
    enum_to_dataframe, default_integrand_names, make_log_table, _prior_row, make_smooth_grid_table
)
from cascade.dismod.db.metadata import DensityEnum
from cascade.dismod.db.wrapper import DismodFile
from cascade.core.log import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


class ModelWriter:
    """This layer over the Dismod File is responsible for translation between
    absolute numbers in the Model and SQL-indexed numbers.

    This has an API that the client has to follow. Functions need to be called
    in order.

    For naming in the DismodFile, this chooses names based on the random
    fields.
    """

    def __init__(self):
        self._dismod_file = DismodFile()
        self._ages = set()
        self._times = set()
        self._rate_rows = list()  # List of dictionaries for rates.
        self._mulcov_rows = list()  # List of dictionaries for covariate multipliers.
        self._rate_id = lambda x: nan
        self._covariate_id = lambda x: nan

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
        pass

    def write_ages_and_times(self, ages, times):
        """
        Collects all of the ages and times because they are ordered
        and indexed by Dismod's file format. Can be called multiple times.
        """
        self._ages |= set(ages)
        self._times |= set(times)

    def write_covariate(self, covariates):
        self._dismod_file.covariate = self._dismod_file.empty_table("covariate")
        CODELOG.debug(f"Covariate types {self._dismod_file.covariate.dtypes}")
        self._dismod_file.covariate, cov_col_id_func, covariate_renames = _make_covariate_table(covariates)
        self._covariate_id = cov_col_id_func

    def write_locations(self, locations):
        """Skip until other PR is merged in for locations."""
        pass

    def write_rate(self, rate_name, random_field):
        """A rate needs a smooth, which has priors and ages/times."""
        self._flush_ages_times_locations()
        smooth_id = self.add_prior_grid(rate_name, random_field.prior_grid)
        self._rate_rows.append(
            {
                "rate_id": len(self._rate_rows),
                "rate_name": rate_name,
                "parent_smooth_id": smooth_id,
                "child_smooth_id": nan,
                "child_nslist_id": nan,
            }
        )

    def write_random_effect(self, rate_location, random_field):
        self._flush_ages_times_locations()
        rate_name, child_location = rate_location
        grid_name = f"{rate_name}_re_{child_location}"
        smooth_id = self.add_prior_grid(grid_name, random_field.prior_grid)
        self._rate_rows.append(
            {
                "rate_id": len(self._rate_rows),
                "rate_name": rate_name,
                "parent_smooth_id": nan,
                "child_smooth_id": smooth_id,
                "child_nslist_id": nan,
            }
        )

    def write_mulcov(self, kind, cov_other, random_field):
        self._flush_ages_times_locations()
        grid_name = f"{kind}_{cov_other[1]}_{cov_other[0]}"
        smooth_id = self.add_prior_grid(grid_name, random_field.prior_grid)
        if kind == "alpha":
            self._mulcov_rows.append(
                {
                    "mulcov_id": len(self._mulcov_rows),
                    "mulcov_type": "rate_value",
                    "rate_id": self._rate_id(cov_other[1]),
                    "integrand_id": nan,
                    "covariate_id": self._covariate_id(cov_other[0]),
                    "smooth_id": smooth_id,
                }
            )

    def add_prior_grid(self, grid_name, random_field_priors):
        """Save a new PriorGrid"""
        # 1. Start with the priors themselves.
        all_prior_objects = set()
        for prior_grid in random_field_priors.values():
            all_prior_objects.update(prior_grid.priors)
        sorted_priors = sorted(all_prior_objects)
        prior_table = pd.DataFrame(
            [_prior_row(p) for p in sorted_priors],
            columns=["prior_name", "density_name", "lower", "upper", "mean",
                     "std", "eta", "nu"],
        )
        if self._dismod_file.prior:
            prior_start_idx = len(self._dismod_file.prior)
        else:
            prior_start_idx = 0
        prior_table["prior_id"] = prior_table.index + prior_start_idx
        null_names = prior_table.prior_name.isnull()
        prior_table.loc[~null_names, "prior_name"] = (
            prior_table.loc[~null_names, "prior_name"] + "    " +
            prior_table.loc[~null_names, "prior_id"].astype(str)
        )
        prior_table.loc[null_names, "prior_name"] = prior_table.loc[
            null_names, "prior_id"].apply(
            lambda pid: f"{grid_name}_{pid}"
        )

        prior_table["prior_id"] = prior_table.index
        prior_table = pd.merge(prior_table, self._dismod_file.density, on="density_name")
        # Make sure the index still matches the order in the priors list
        prior_table = prior_table.sort_values(by="prior_id").reset_index(
            drop=True)
        if self._dismod_file.prior:
            self._dismod_file.prior = self._dismod_file.prior.append(prior_table)
        else:
            self._dismod_file.prior = prior_table

        # This function is only relevant for this random field.
        def prior_id_func(prior):
            return sorted_priors.index(prior) + prior_start_idx

        # 2. Then make the smooth_grid with the priors.
        smooth = "smooth representation"
        grid_table = make_smooth_grid_table(smooth, prior_id_func)
        grid_table["smooth_id"] = len(self._dismod_file.smooth)
        grid_table = pd.merge_asof(
            grid_table.sort_values("age"), self._dismod_file.age, on="age") \
            .drop("age", "columns")
        grid_table = pd.merge_asof(
            grid_table.sort_values("time"), self._dismod_file.time, on="time") \
            .drop("time", "columns")
        grid_table = grid_table.sort_values(["time_id", "age_id"])

        # 3. Then add the smooth grid entry.
        smooth_id = 0
        return smooth_id

    def _flush_ages_times_locations(self):
        if self._flushed:
            return
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
