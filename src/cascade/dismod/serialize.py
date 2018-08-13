"""
Converts the internal representation to a Dismod File.
"""
import logging
from pathlib import Path
import time
from timeit import default_timer as timer

import numpy as np
import pandas as pd

from cascade.dismod.db.metadata import IntegrandEnum, DensityEnum, RateName
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


def observations_to_data(observations_df, hold_out=0):
    """Turn an internal format into a Dismod format."""
    # Don't make the data_name here because could convert multiple observations.
    return pd.DataFrame({
        "integrand_id": observations_df["measure"].apply(lambda x: IntegrandEnum[x].value),
        # Assumes one location_id.
        "node_id": 0,
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


def age_time_from_grids(smoothers, total_data):
    """
    The ages and times must correspond exactly to smoother ages and times
    but they must also include the minimum and maximum of all ages and times
    used.

    Args:
        smoothers: List of smoothers.
        total_data (pd.DataFrame): In the Dismod format, so uses time not year.

    Returns:
        (pd.DataFrame, pd.DataFrame): The age and time dataframes, ready to write.
    """
    from_data = dict(
        age=np.hstack([total_data["age_lower"].values, total_data["age_upper"].values]),
        time=np.hstack([total_data["time_lower"].values, total_data["time_upper"].values])
    )

    results = list()
    for column, name in [("age", "age"), ("year", "time")]:
        ages = set()
        for grid_df in smoothers:
            ages.update(set(grid_df[column].unique()))
        age_list = list(ages)

        # The min and max of integrands should be included in age list.
        # This is a separate purpose of the age list, to define min and max
        # integration values.
        age_list.append(from_data[name].min())
        age_list.append(from_data[name].max())

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


def write_to_file(model_context, filename):
    """
    This is a one-way translation from a model context to a new Dismod file.
    It assumes a lot. One location, no covariates, and more.

    Args:
        model_context (ModelContext): The one big object.

    """
    model = model_context.input_data
    avgint_columns = dict()
    data_columns = dict()
    bundle_dismod_db = Path(filename)
    bundle_file_engine = _get_engine(bundle_dismod_db)
    bundle_fit = DismodFile(bundle_file_engine, avgint_columns, data_columns)

    # Standard Density table.
    density_enum = enum_to_dataframe(DensityEnum)
    densities = pd.DataFrame({"density_name": density_enum["name"]})
    bundle_fit.density = densities

    # Standard integrand naming scheme.
    all_integrands = default_integrand_names()
    bundle_fit.integrand = all_integrands

    bundle_fit.covariate = bundle_fit.empty_table("covariate")
    LOGGER.debug(f"Covariate types {bundle_fit.covariate.dtypes}")

    # Defaults, empty, b/c Brad makes them empty.
    bundle_fit.nslist = bundle_fit.empty_table("nslist")
    bundle_fit.nslist_pair = bundle_fit.empty_table("nslist_pair")
    bundle_fit.mulcov = bundle_fit.empty_table("mulcov")

    bundle_fit.log = pd.DataFrame({
        "message_type": ["command"],
        "table_name": np.array([None], dtype=np.object),
        "row_id": np.NaN,
        "unix_time": int(round(time.time())),
        "message": ["fit_no_covariates.py"],
    })

    # Assume we have one location, so no parents.
    # If we had a hierarchy, that would be used to determine parents.
    unique_locations = model.observations["location_id"].unique()
    assert len(unique_locations) == 1
    node_table = pd.DataFrame({
        "node_name": unique_locations.astype(int).astype(str),
        "parent": np.array([np.NaN]),
    })
    bundle_fit.node = node_table

    non_zero_rates = list(model.smoothers.keys())
    if 'iota' in non_zero_rates:
        if 'rho' in non_zero_rates:
            value = 'iota_pos_rho_pos'
        else:
            value = 'iota_pos_rho_zero'
    else:
        if 'rho' in non_zero_rates:
            value = 'iota_zero_rho_pos'
        else:
            value = 'iota_zero_rho_zero'
    bundle_fit.option = pd.DataFrame({
        "option_name": ["parent_node_name", "rate_case"],
        "option_value": [node_table.iloc[0]["node_name"], value],
    })

    observations = observations_to_data(model.observations)
    constraints = observations_to_data(model.constraints, hold_out=1)
    total_data = pd.concat([observations, constraints], ignore_index=True)
    # Why a unique string name?
    total_data["data_name"] = total_data.index.astype(str)
    bundle_fit.data = total_data

    # Include all data in the data_subset.
    bundle_fit.data_subset = pd.DataFrame({
        "data_id": np.arange(len(total_data)),
    })

    # Ages and times are used by Weight grids and smooth grids,
    # so pull all ages and times from those two objects in the
    # internal model. Skip weight grid here b/c assuming use constant.
    bundle_fit.age, bundle_fit.time = age_time_from_grids(
        model.smoothers.values(), total_data)

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
        # Assumes one location_id.
        "node_id": 0,
        # Assuming using the first set of weights, which is constant.
        "weight_id": 0,
        "age_lower": model.outputs.age_start,
        "age_upper": model.outputs.age_end,
        "time_lower": model.outputs.year_start,
        "time_upper": model.outputs.year_end,
    })

    bundle_fit.smooth, bundle_fit.smooth_grid = convert_smoothers(
        model.smoothers, age_df, time_df, prior_df)
    smooth_df = bundle_fit.smooth.copy()
    smooth_df["smooth_id"] = smooth_df.index

    rate_enum = enum_to_dataframe(RateName)
    with_smooth = rate_enum.merge(smooth_df, left_on="name", right_on="smooth_name", how="outer")
    bundle_fit.rate = pd.DataFrame({
        "rate_name": with_smooth["name"],
        "parent_smooth_id": with_smooth["smooth_id"],
        "child_smooth_id": np.NaN,
        "child_nslist_id": np.NaN,
    })

    flush_begin = timer()
    bundle_fit.flush()
    LOGGER.debug(f"Flush db {timer() - flush_begin}")


def read_predict(db_path):
    avgint_columns = dict()
    data_columns = dict()
    bundle_dismod_db = Path(db_path)
    bundle_file_engine = _get_engine(bundle_dismod_db)
    bundle_fit = DismodFile(bundle_file_engine, avgint_columns,
                            data_columns)

    desired_outputs = bundle_fit.avgint

    # Use the integrand table to convert the integrand_id into an integrand_name.
    integrand_names = bundle_fit.integrand
    desired_outputs = desired_outputs.merge(integrand_names, left_on="integrand_id",
                                            right_on=integrand_names.index)

    # Associate the result with the desired integrand.
    prediction = bundle_fit.predict.merge(
        desired_outputs, left_on="avgint_id", right_on=desired_outputs.index)
    return prediction


def read_prevalence(prediction):
    return prediction[prediction["integrand_name"] == "prevalence"][
        ["avg_integrand", "time_lower", "age_lower"]]
