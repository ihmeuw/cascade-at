"""
Writes a Model to a Dismod File.
"""
from math import nan
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.core.log import getLoggers
from cascade.dismod.db.metadata import DensityEnum
from cascade.dismod.db.wrapper import DismodFile, _get_engine
from cascade.dismod.serialize import (
    enum_to_dataframe, default_integrand_names, make_log_table, simplest_weight
)

CODELOG, MATHLOG = getLoggers(__name__)


class DismodSession:
    def __init__(self, locations, filename):
        """

        Args:
            locations (pd.DataFrame): Initialize here because data refers to this.
        """
        self.dismod_file = DismodFile()
        self._filename = Path(filename)
        self.dismod_file.engine = _get_engine(self._filename)
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
        self._nslist_pair_rows = list()  # list of nslist id, node, and smooth
        self._covariate_id_func = lambda x: nan
        self._integrand_id_func = lambda x: nan
        self._flushed = False
        self._children = None
        self._mulcov_name_to_type = dict(alpha="rate_value", beta="meas_value", gamma="meas_std")

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

        self._dismod_file.rate = pd.DataFrame(dict(
            rate_id=list(range(5)),  # Must be 5, in this order.
            rate_name=["pini", "iota", "rho", "chi", "omega"],
            parent_smooth_id=nan,
            child_smooth_id=nan,
            child_nslist_id=nan,
        ))

        # Defaults, empty, b/c Brad makes them empty even if there are none.
        for create_name in ["nslist", "nslist_pair", "mulcov", "smooth_grid", "smooth"]:
            setattr(self._dismod_file, create_name, self._dismod_file.empty_table(create_name))
        self._dismod_file.log = make_log_table()

    def start_model(self, nonzero_rates, children):
        """To start the model, tell me enough to know the sizes almost all of
        the tables by telling me the rate count and child random effect count."""
        self._children = children
        self.basic_db_setup()

    def write_ages_and_times(self, ages, times):
        """
        Collects all of the ages and times because they are ordered
        and indexed by Dismod's file format. Can be called multiple times.
        """
        self._ages = np.append(self._ages, ages)
        self._times = np.append(self._times, times)

    def write_covariate(self, covariates):
        self._dismod_file.covariate = self._dismod_file.empty_table("covariate")
        CODELOG.debug(f"covariates {', '.join(c.name for c in covariates)}")
        self._dismod_file.covariate, cov_col_id_func, covariate_renames = _make_covariate_table(covariates)
        self._covariate_id_func = cov_col_id_func

    def write_rate(self, rate_name, random_field):
        """A rate needs a smooth, which has priors and ages/times."""
        self._flush_ages_times_locations()
        smooth_id = self.add_random_field(rate_name, random_field)
        self._dismod_file.rate.loc[self._rate_id_func(rate_name), "parent_smooth_id"] = smooth_id

    def write_random_effect(self, rate_name, child_location, random_field):
        self._flush_ages_times_locations()
        if child_location is None:
            grid_name = f"{rate_name}_re"
        else:
            grid_name = f"{rate_name}_re_{child_location}"
        smooth_id = self.add_random_field(grid_name, random_field)
        rate_id = self._rate_id_func(rate_name)
        CODELOG.debug(f"random effect {rate_name} {child_location} {smooth_id}")
        if child_location is None:
            self._dismod_file.rate.loc[rate_id, "child_smooth_id"] = smooth_id
        else:
            node_id = self._session.location_func(child_location)
            if rate_name not in self._nslist:
                ns_id = len(self._nslist)
                self._nslist[rate_name] = ns_id
            else:
                ns_id = self._nslist[rate_name]
            self._dismod_file.rate.loc[rate_id, "child_nslist_id"] = ns_id
            self._nslist_pair_rows.append(dict(
                nslist_pair_id=len(self._nslist_pair_rows),
                nslist_id=ns_id,
                node_id=node_id,
                smooth_id=smooth_id,
            ))

    def write_mulcov(self, kind, covariate, rate_or_integrand, random_field):
        self._flush_ages_times_locations()
        CODELOG.debug(f"write_mulcov {kind} {covariate} {rate_or_integrand}")
        grid_name = f"{kind}_{rate_or_integrand}_{covariate}"
        row = {
            "mulcov_id": len(self._mulcov_rows),
            "mulcov_type": self._mulcov_name_to_type[kind],
            "rate_id": nan,
            "integrand_id": nan,
            "covariate_id": self._covariate_id_func(covariate),
            "smooth_id": self.add_random_field(grid_name, random_field),
        }
        if kind == "alpha":
            row.update(dict(rate_id=self._rate_id_func(rate_or_integrand)))
        elif kind in ("beta", "gamma"):
            row.update(dict(integrand_id=self._integrand_id_func(rate_or_integrand)))
        else:
            raise RuntimeError(f"Unknown mulcov type {kind}")
        self._mulcov_rows.append(row)

    def write_weight(self, name, field_draw):
        # FIXME: The weight implementation has to create its own grid. Much simpler.
        self._dismod_file.weight, self._dismod_file.weight_grid = simplest_weight()

    def close(self):
        self._dismod_file.mulcov = pd.DataFrame(self._mulcov_rows)
        self._dismod_file.nslist = pd.DataFrame.from_records(
            data=list(self._nslist.items()),
            columns=["nslist_name", "nslist_id"]
        )
        self._dismod_file.nslist_pair = pd.DataFrame(
            self._nslist_pair_rows,
            columns=["nslist_pair_id", "nslist_id", "node_id", "smooth_id"]
        )

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
            smooth_row[f"mulstd_{prior_kind}_prior_id"] = int(prior_table.loc[
                prior_table.age.isna() & (prior_table.kind == prior_kind)
                ].prior_id)
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
        complete_table = self._fix_ages_times(complete_table)
        # Make sure the index still matches the order in the priors list
        priors_columns = [
            "prior_id", "prior_name", "lower", "upper", "mean", "std", "eta", "nu", "density_id"
        ]
        prior_table = complete_table.sort_values(by="prior_id").reset_index(drop=True)[priors_columns]
        if not self._dismod_file.prior.empty:
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

    def _integrand_id_func(self, name):
        return int(self._dismod_file.integrand.query("integrand_name==@name").integrand_id)

    def _rate_id_func(self, name):
        return int(self._dismod_file.rate.query("rate_name==@name").rate_id)

    def _fix_ages_times(self, df):
        """Given a Pandas df with age and time columns, assign age_id and time_id
        columns using a nearest match. Keep the same order. Allow nan ages
        or times."""
        assert "age" in df.columns
        assert "time" in df.columns
        df = df.assign(save_idx=df.index)
        for dat in ["age", "time"]:
            col_id = f"{dat}_id"
            sort_by = df.sort_values(dat)
            in_grid = sort_by[dat].notna()
            at_table = getattr(self._dismod_file, dat)
            aged = pd.merge_asof(sort_by[in_grid], at_table, on=dat, direction="nearest")
            df = df.merge(aged[["save_idx", col_id]], on="save_idx", how="left")
        assert "age_id" in df.columns
        assert "time_id" in df.columns
        return df.drop("save_idx", axis=1)


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
