"""
Writes a Model to a Dismod File.
"""
from math import nan, inf

import numpy as np
import pandas as pd

from cascade.core import getLoggers
from cascade.dismod.constants import DensityEnum, WeightEnum, MulCovEnum

CODELOG, MATHLOG = getLoggers(__name__)


class ModelWriter:
    """This layer over the Dismod File is responsible for translation between
    absolute numbers in the Model and SQL-indexed numbers.

    This has an API that the client has to follow. Functions need to be called
    in order. Then the writer has to be thrown away. It's not for use
    beyond a single function that writes a fresh model.

    For naming in the DismodFile, this chooses names based on the random
    fields.

    Assumptions this ModelWriter makes, on top of what Dismod-AT requires:

     * Locations have a ``c_location_id`` member that ``maps node_id``
       to ``location_id``.
     * Rates and integrands are always in the same order.
    """

    def __init__(self, object_wrapper, dismod_file):
        """
        Args:
            object_wrapper (Session): The Dismod-AT Session into which this writes.
        """
        self._object_wrapper = object_wrapper
        self._dismod_file = dismod_file
        self._ages = np.empty((0,), dtype=np.float)
        self._times = np.empty((0,), dtype=np.float)
        self._rate_rows = list()  # List of dictionaries for rates.
        self._mulcov_rows = list()  # List of dictionaries for covariate multipliers.
        self._rate_id = dict()  # The rate ids with the primary rates.
        self._nslist = dict()  # rate to integer
        self._nslist_pair_rows = list()  # list of nslist id, node, and smooth
        self._flushed = False
        self._children = None
        self._clear_previous_model()

    def _clear_previous_model(self):
        """We could rewrite parts of tables that have changed. This is the
        clear-cutting alternative."""
        cleared_tables = list()
        # Other tables are owned by the session.
        owned_tables = ["age", "covariate", "mulcov", "nslist", "nslist_pair",
                        "prior", "smooth_grid", "smooth", "time", "weight",
                        "weight_grid"]
        for create_name in owned_tables:
            if not getattr(self._dismod_file, create_name).empty:
                cleared_tables.append(create_name)
                setattr(self._dismod_file, create_name, self._dismod_file.empty_table(create_name))

        # Handle rate separately.
        if not self._dismod_file.rate.empty:
            cleared_tables.append("rate")
            # Then the rate has all five entries.
            self._dismod_file.rate.loc[:, "parent_smooth_id"] = nan
            self._dismod_file.rate.loc[:, "child_smooth_id"] = nan
            self._dismod_file.rate.loc[:, "child_nslist_id"] = nan
        if cleared_tables:
            CODELOG.debug(f"Writing model cleared tables from previous model: {', '.join(cleared_tables)}")

    def start_model(self, nonzero_rates, children):
        """To start the model, tell me enough to know the sizes almost all of
        the tables by telling me the rate count and child random effect count."""
        self._children = children
        iota_case = "pos" if "iota" in nonzero_rates else "zero"
        rho_case = "pos" if "rho" in nonzero_rates else "zero"
        self._object_wrapper.set_option(rate_case=f"iota_{iota_case}_rho_{rho_case}")

    def write_ages_and_times(self, ages, times):
        """
        Collects all of the ages and times because they are ordered
        and indexed by Dismod's file format. Can be called multiple times.
        """
        self._ages = np.append(self._ages, ages)
        self._times = np.append(self._times, times)

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
            locs = self._object_wrapper.locations
            node_id = locs[locs.location_id == child_location].node_id.iloc[0]
            if rate_name not in self._nslist:
                ns_id = len(self._nslist)
                self._nslist[rate_name] = ns_id
            else:
                ns_id = self._nslist[rate_name]
            self._dismod_file.rate.loc[rate_id, "child_nslist_id"] = ns_id
            self._nslist_pair_rows.append(dict(
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
            "mulcov_type": MulCovEnum[kind].value,
            "rate_id": nan,
            "integrand_id": nan,
            "covariate_id": self._object_wrapper.covariate_to_index[covariate],
            "smooth_id": self.add_random_field(grid_name, random_field),
        }
        if kind == "alpha":
            row.update(dict(rate_id=self._rate_id_func(rate_or_integrand)))
        elif kind in ("beta", "gamma"):
            row.update(dict(integrand_id=self._integrand_id_func(rate_or_integrand)))
        else:
            raise RuntimeError(f"Unknown mulcov type {kind}")
        self._mulcov_rows.append(row)

    def write_weights(self, weights):
        """Always write 4 weights in same order."""
        self._flush_ages_times_locations()
        names = [w.name for w in WeightEnum]
        weight_table = pd.DataFrame(dict(
            weight_id=[w.value for w in WeightEnum],
            weight_name=names,
            n_age=[len(weights[name].ages) for name in names],
            n_time=[len(weights[name].times) for name in names],
        ))
        grids = list()
        for w in WeightEnum:
            one_grid = weights[w.name].grid[["age", "time", "mean"]].rename(columns={"mean": "weight"})
            grids.append(one_grid.assign(weight_id=w.value))
        un_aged = pd.concat(grids).reset_index(drop=True)
        total = self._fix_ages_times(un_aged).drop(columns=["age", "time"])
        total.reset_index(drop=True)
        total["weight_grid_id"] = total.index
        self._dismod_file.weight = weight_table
        self._dismod_file.weight_grid = total

    def close(self):
        if self._mulcov_rows:
            self._dismod_file.mulcov = pd.DataFrame(self._mulcov_rows)
        else:
            self._dismod_file.mulcov = self._dismod_file.empty_table("mulcov")
        self._dismod_file.nslist = pd.DataFrame.from_records(
            data=list(self._nslist.items()),
            columns=["nslist_name", "nslist_id"]
        )
        # Can sort this here and generate its index because nothing uses
        # the nslist_pair_id as a key. The sort ensures a sensible order.
        self._dismod_file.nslist_pair = pd.DataFrame(
            self._nslist_pair_rows,
            columns=["nslist_id", "node_id", "smooth_id"]
        ).sort_values(["nslist_id", "node_id"])

    def add_random_field(self, grid_name, random_field):
        """Save a new Random Field. Creates the ``smooth``, ``smooth_grid``,
        and ``priors`` tables."""
        # The smooth_grid table points to the priors and the smooth, itself,
        # so write it last.
        complete_table = self._add_field_priors(grid_name, random_field.priors)
        age_cnt, time_cnt = (len(random_field.ages), len(random_field.times))
        assert len(complete_table) == (age_cnt * time_cnt + 1) * 3
        smooth_id = self._add_field_smooth(grid_name, complete_table, (age_cnt, time_cnt))
        self._add_field_grid(complete_table, smooth_id)
        return smooth_id

    def _add_field_grid(self, complete_table, smooth_id):
        """Each age-time entry in the smooth_grid table, including the mulstds."""
        long_table = complete_table.loc[complete_table.age_id.notna()][["age_id", "time_id", "prior_id", "kind"]]
        grid_table = long_table[["age_id", "time_id"]].sort_values(["age_id", "time_id"]).drop_duplicates()
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

    def _add_field_smooth(self, grid_name, prior_table, age_time_cnt):
        """Ths one row in the smooth grid table."""
        smooth_row = dict(smooth_name=grid_name)
        smooth_row["n_age"] = age_time_cnt[0]
        smooth_row["n_time"] = age_time_cnt[1]
        for prior_kind in ["value", "dage", "dtime"]:
            mulstd_row = prior_table.loc[prior_table.age.isna() & (prior_table.kind == prior_kind)]
            if all(mulstd_row.assigned):
                smooth_row[f"mulstd_{prior_kind}_prior_id"] = int(mulstd_row.prior_id)
        if self._dismod_file.smooth.empty:
            smooth_row["smooth_id"] = 0
            self._dismod_file.smooth = self._dismod_file.empty_table("smooth").append(smooth_row, ignore_index=True)
        else:
            smooth_row["smooth_id"] = len(self._dismod_file.smooth)
            self._dismod_file.smooth = self._dismod_file.smooth.append(smooth_row, ignore_index=True)
        smooth_id = smooth_row["smooth_id"]
        return smooth_id

    def _add_field_priors(self, grid_name, complete_table):
        """These are all entries in the priors table for this smooth grid."""
        # The assigned column will tell us whether mulstds were assigned.
        complete_table = complete_table.assign(assigned=complete_table.density.notna())
        complete_table.loc[complete_table.density.isnull(), ["density", "mean", "lower", "upper"]] = \
            ["uniform", 0, -inf, inf]
        complete_table = complete_table.assign(density_id=complete_table.density.apply(lambda x: DensityEnum[x].value))
        # Create new prior IDs that don't overlap.
        complete_table = complete_table.assign(prior_id=complete_table.index + len(self._dismod_file.prior))
        complete_table = complete_table.rename(columns={"name": "prior_name"})
        # Unique, informative names for the priors require care.
        null_names = complete_table.prior_name.isnull()
        complete_table.loc[~null_names, "prior_name"] = (
            complete_table.loc[~null_names, "prior_name"].astype(str) + "    " +
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
        # Remove columns before saving, but keep extra columns in complete_table for
        # further construction of grids.
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
        # Dismod-AT doesn't like min and max ages nearly equal.
        if unique_ages[-1] - unique_ages[0] < 1:
            unique_ages = np.append(unique_ages, unique_ages[-1] + 1)
        self._dismod_file.age = pd.DataFrame(dict(age_id=range(len(unique_ages)), age=unique_ages))
        unique_times = self._times[np.unique(self._times.round(decimals=14), return_index=True)[1]]
        unique_times.sort()
        if unique_times[-1] - unique_times[0] < 1:
            unique_times = np.append(unique_times, unique_times[-1] + 1)
        self._dismod_file.time = pd.DataFrame(dict(time_id=range(len(unique_times)), time=unique_times))
        self._flushed = True

    def _integrand_id_func(self, name):
        return int(self._dismod_file.integrand.query("integrand_name==@name").integrand_id.values[0])

    def _rate_id_func(self, name):
        if not isinstance(name, str):
            raise TypeError(f"To get the rate ID, use one of the rate names, not {name}.")
        rate_record = self._dismod_file.rate.query("rate_name==@name")
        if len(rate_record) == 0:
            raise RuntimeError(
                f"The rate {name} was not found in db file rate table, and all rates should be in there. "
                f"The whole rate table is:\n{self._dismod_file.rate}"
            )
        elif len(rate_record) > 1:
            raise RuntimeError(
                f"There are multiple rows for rate {name} in the db file rate table, which should not be: "
                f"{rate_record}"
            )
        return int(rate_record.rate_id)

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

    def write_covariate(self, covariates):
        self._dismod_file.covariate = self._dismod_file.empty_table("covariate")
        reorder = list()
        lookup = {search.name: search for search in covariates}
        for special in ["sex", "one"]:
            if special in lookup:
                reorder.append(lookup[special])
                del lookup[special]
        for remaining in sorted(lookup.keys()):
            reorder.append(lookup[remaining])
        CODELOG.debug(f"covariates {', '.join(c.name for c in reorder)}")
        self._object_wrapper.covariates = reorder
