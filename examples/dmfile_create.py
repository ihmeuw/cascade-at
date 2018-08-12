"""
Converts the internal representation to a Dismod File.
"""
import logging
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd

from cascade.dismod.db.metadata import IntegrandEnum, DensityEnum
from cascade.dismod.db.wrapper import _get_engine, DismodFile


LOGGER = logging.getLogger(__name__)


def enum_to_dataframe(enum_name):
    """Given an enum, return a dataframe with two columns, name and value."""
    return pd.DataFrame.from_records(
        np.array(
            [(measure, enum_value.value) for (measure, enum_value) in enum_name.__members__.items()],
            dtype=np.dtype([('name', object), ('value', np.int)])
        )
    )


def default_integrand_names():
    # Converting an Enum to a DataFrame with specific parameters
    integrands = enum_to_dataframe(IntegrandEnum)
    df = pd.DataFrame({"integrand_name": integrands["name"]})
    df["minimum_meas_cv"] = 0.0
    return df


def simplest_weight():
    """Defines one weight for everything by defining it on one age-time point."""
    weight = pd.DataFrame({
        "weight_name": ["constant"],
        "n_age": [1],
        "n_time": [1],
    })
    weight_grid = pd.DataFrame({
        "weight_id": [0],
        "age_id": [0],
        "time_id": [0],
        "weight": [1.0],
    })
    return weight, weight_grid


def observations_to_data(dismodel, observations_df, hold_out=0):
    """Turn an internal format into a Dismod format."""
    # Don't make the data_name here because could convert multiple observations.
    return pd.DataFrame({
        "integrand_id": observations_df["measure"].apply(lambda x: IntegrandEnum[x].value),
        # The node_id is the location_id.
        "node_id": observations_df["location_id"],
        # Density is an Enum at this point.
        "density_id": observations_df["density"].apply(lambda x: x.value),
        # Translate weight from string
        "weight_id": 0,
        "age_lower": observations_df["age_start"],
        "age_upper": observations_df["age_end"],
        "time_lower": observations_df["year_start"].astype(np.float),
        "time_upper": observations_df["year_end"],
        "meas_value": observations_df["mean"],
        "meas_std": observations_df["standard_error"],
        "eta": np.NaN,
        "nu": np.NaN,
        "hold_out": hold_out,
    })


def age_time_from_grids(smoothers):
    results = list()
    for column, name in [("age", "age"), ("year", "time")]:
        ages = set()
        for grid_df in smoothers:
            ages.update(set(grid_df[column].unique()))
        age_list = list(ages)
        age_list.sort()
        as_floats = np.array(age_list, dtype=np.float)
        results.append(pd.DataFrame({name: as_floats}))

    age_df, time_df = results
    return age_df, time_df


def convert_smoothers(smoothers, age_df, time_df, prior_df):
    LOGGER.debug(f"age_df {age_df.dtypes}")
    LOGGER.debug(f"time_df {time_df.dtypes}")
    LOGGER.debug(f"prior_df {prior_df.dtypes}")
    sm_name = list()
    sm_age = list()
    sm_time = list()

    smooth_grid = list()

    smooth_idx = 0
    for name, smoothing in smoothers.items():
        # The name is a RateName enum value. Hence name.name to get the string.
        LOGGER.debug(f"{name} f{smoothing.dtypes}")
        sm_name.append(name)
        sm_age.append(len(smoothing["age"].unique()))
        sm_time.append(len(smoothing["year"].unique()))

        smoothing = smoothing.sort_values(by="age")
        with_age = pd.merge_asof(smoothing, age_df, left_on="age", right_on="age")
        # Time comes in as year from the model.
        with_age = with_age.sort_values(by="year")
        with_age_time = pd.merge_asof(with_age, time_df, left_on="year", right_on="time")
        # Use left outer join because priors can be none and should be kept.
        with_val = with_age_time.merge(
            prior_df, how="left", left_on="value_prior", right_on="prior_name")
        with_val = with_val.rename(index=str, columns={"prior_id": "value_prior_id"})
        with_dage = with_val.merge(
            prior_df, left_on="age_difference_prior", right_on="prior_name", how="left")
        with_dage = with_dage.rename(index=str, columns={"prior_id": "dage_prior_id"})
        with_dtime = with_dage.merge(
            prior_df, left_on="time_difference_prior", right_on="prior_name", how="left")
        with_ids = with_dtime.rename(index=str, columns={"prior_id": "dtime_prior_id"})

        smooth_grid.append(pd.DataFrame({
            "smooth_id": smooth_idx,
            "age_id": with_ids["age_id"],
            "time_id": with_ids["time_id"],
            "value_prior_id": with_ids["value_prior_id"],
            "dage_prior_id": with_ids["dage_prior_id"],
            "dtime_prior_id": with_ids["dtime_prior_id"],
            "const_value": with_ids["const_value"],
        }))

        smooth_idx += 1
    LOGGER.debug(f"Added {len(smooth_grid)} smoothers for {list(smoothers.keys())}.")

    smooth_df = pd.DataFrame({
        "smooth_name": sm_name,
        "n_age": sm_age,
        "n_time": sm_time,
        "mulstd_value_prior_id": np.NaN,
        "mulstd_dage_prior_id": np.NaN,
        "mulstd_dtime_prior_id": np.NaN,
    })
    grids_together = pd.concat(smooth_grid, ignore_index=True)
    return smooth_df, grids_together


def write_to_file(config, model):
    avgint_columns = dict()
    data_columns = dict()
    bundle_dismod_db = Path("fit_no.db")
    bundle_file_engine = _get_engine(bundle_dismod_db)
    bundle_fit = DismodFile(bundle_file_engine, avgint_columns, data_columns)

    # Standard Density table.
    density_enum = enum_to_dataframe(DensityEnum)
    densities = pd.DataFrame({"density_name": density_enum["name"]})
    bundle_fit.density = densities

    # Standard integrand naming scheme.
    all_integrands = default_integrand_names()
    bundle_fit.integrand = all_integrands

    # No covariates.
    bundle_fit.covariate = pd.DataFrame(
        columns=["covariate_name", "reference", "max_difference"])

    # Defaults, empty, b/c Brad makes them empty.
    bundle_fit.nslist = pd.DataFrame(columns=["nslist_name"])
    bundle_fit.mulcov = pd.DataFrame(columns=["mulcov_type", "integrand_id", "covariate_id", "smooth_id"])

    # Assume we have one location, so no parents.
    # If we had a hierarchy, that would be used to determine parents.
    unique_locations = model.observations["location_id"].unique()
    assert len(unique_locations) == 1
    node_table = pd.DataFrame({
        "node_name": unique_locations.astype(int).astype(str),
        "parent": np.array([np.NaN]),
    },
    index=unique_locations.astype(int)
    )
    bundle_fit.node = node_table

    # Ages and times are used by Weight grids and smooth grids,
    # so pull all ages and times from those two objects in the
    # internal model. Skip weight grid here b/c assuming use constant.
    bundle_fit.age, bundle_fit.time = age_time_from_grids(model.smoothers.values())

    # These are used to get the age_id into another dataframe.
    # Use pd.merge_asof to do this.
    age_df = bundle_fit.age.copy()
    age_df["age_id"] = age_df.index
    time_df = bundle_fit.time.copy()
    time_df["time_id"] = time_df.index

    bundle_fit.weight, bundle_fit.weight_grid = simplest_weight()

    bundle_fit.prior = model.priors
    prior_df = bundle_fit.prior.copy()
    prior_df["prior_id"] = prior_df.index

    # The avgint needs to be translated.
    bundle_fit.avgint = pd.DataFrame({
        "integrand_id": model.outputs.integrand.apply(lambda x: x.value),
        # We made the location_id the index of the node_id.
        "node_id": model.outputs.location_id,
        # Assuming using the first set of weights, which is constant.
        "weight_id": 0,
        "age_lower": model.outputs.age_start,
        "age_upper": model.outputs.age_end,
        "time_lower": model.outputs.year_start,
        "time_upper": model.outputs.year_end,
    })

    observations = observations_to_data(bundle_fit, model.observations)
    constraints = observations_to_data(bundle_fit, model.constraints, hold_out=1)
    total_data = pd.concat([observations, constraints], ignore_index=True)
    # Why a unique string name?
    total_data["data_name"] = total_data.index.astype(str)
    bundle_fit.data = total_data

    bundle_fit.smooth, bundle_fit.smooth_grid = convert_smoothers(
        model.smoothers, age_df, time_df, prior_df)

    flush_begin = timer()
    bundle_fit.flush()
    LOGGER.debug(f"Flush db {timer() - flush_begin}")