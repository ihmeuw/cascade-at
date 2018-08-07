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


def observations_to_data(dismodel, observations_df):
    """Turn an internal format into a Dismod format."""
    measure_to_integrand = dict(
        incidence=IntegrandEnum.Sincidence.value,
        mtexcess=IntegrandEnum.mtexcess.value,
    )
    return pd.DataFrame({
        "measure": observations_df["measure"].apply(measure_to_integrand.get),
        # Translate node id from location_id
        "node_id": observations_df["location_id"],
        # Translate density from string
        "density_id": observations_df["density"],
        # Translate weight from string
        "weight_id": observations_df["weight"],
        "age_lower": observations_df["age_start"],
        "age_upper": observations_df["age_end"],
        "time_lower": observations_df["year_start"].astype(np.float),
        "time_upper": observations_df["year_end"],
        "meas_value": observations_df["mean"],
        "meas_std": observations_df["standard_error"],
        "hold_out": 0,
    })


def convert_smoothers(smoother):
    pass


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
    bundle_fit.integrands = all_integrands

    # Defaults, empty, b/c Brad makes them empty.
    bundle_fit.nslist = pd.DataFrame()
    bundle_fit.mulcov = pd.DataFrame()

    # Assume we have one location, so no parents.
    # If we had a hierarchy, that would be used to determine parents.
    unique_locations = model.observations["location_id"].unique()
    assert len(unique_locations) == 1
    node_table = pd.DataFrame({
        "node_name": unique_locations.astype(int).astype(str),
        "parent": None,
    },
    index=unique_locations.astype(int)
    )
    bundle_fit.node = node_table

    # Ages and times are used by Weight grids and smooth grids,
    # so pull all ages and times from those two objects in the
    # internal model. Skip weight grid here b/c assuming use constant.

    bundle_fit.age = model.age
    bundle_fit.time = model.time

    # These are helpers to convert from ages and times to age and time indexes.
    # pd.merge_asof will do an approximate merge.
    age_idx = pd.DataFrame(model.age)
    age_idx["index0"] = age_idx.index

    time_idx = pd.DataFrame(model.time)
    time_idx["index0"] = time_idx.index

    bundle_fit.node = node_table
    bundle_fit.weight, bundle_fit.weight_grid = simplest_weight()

    bundle_fit.prior = model.priors

    # The avgint needs to be translated.
    bundle_fit.avgint = pd.DataFrame({
        "integrand_id": model.outputs.integrand.apply(lambda x: x.value),
        "node_id": model.outputs.location_id,
        # Assuming using the first set of weightw, which is constant.
        "weight_id": 0,
        "age_lower": model.outputs.age_start,
        "age_upper": model.outputs.age_end,
        "time_lower": model.outputs.year_start,
        "time_upper": model.outputs.year_end,
    })

    observations = observations_to_data(bundle_fit, model.observations)
    constraints = model.constraints
    bundle_fit.data = pd.concat([observations, constraints], ignore_index=True)

    bundle_fit.smooth, bundle_fit.smooth_grid = convert_smoothers(model.smoothers)

    bundle_fit.covariate = pd.DataFrame({
        "covariate_name": np.array(0, object),
        "reference": np.array(0, np.float),
        "max_difference": np.array(0, np.float),
    })

    flush_begin = timer()
    bundle_fit.flush()
    LOGGER.debug(f"Flush db {timer() - flush_begin}")