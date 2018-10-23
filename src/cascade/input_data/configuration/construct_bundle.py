import numpy as np
import pandas as pd

from cascade.input_data import InputDataError
from cascade.input_data.configuration.construct_study import \
    get_bundle_study_covariates
from cascade.input_data.configuration.id_map import make_integrand_map
from cascade.input_data.db.bundle import _get_bundle_id, _get_bundle_data
from cascade.dismod.db.metadata import DensityEnum
from cascade.core import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


def _normalize_measures(data):
    """Transform measure_ids into canonical measure names, for instance,
    GBD measure 38 for birth prevalence becomes pini.
    """
    data = data.copy()
    gbd_measure_id_to_integrand = make_integrand_map()
    if any(data.measure_id == 6):
        MATHLOG.warn(f"Found incidence, measure_id=6, in data. Should be Tincidence or Sincidence.")
    try:
        data["measure"] = data.measure_id.apply(lambda k: gbd_measure_id_to_integrand[k].name)
    except KeyError as ke:
        raise RuntimeError(
            f"The bundle data uses measure {str(ke)} which doesn't map "
            f"to an integrand. The map is {gbd_measure_id_to_integrand}."
        )
    return data


def _normalize_sex(data):
    """Transform sex_ids from 1, 2, 3 to male, female, both.
    """
    data = data.copy()
    try:
        data["sex"] = data.sex_id.apply({1: "Male", 2: "Female", 3: "Both", 4: "Unspecified"}.get)
    except KeyError as ke:
        raise InputDataError(f"Unrecognized sex id") from ke
    return data


def _normalize_bundle_data(data):
    """Normalize bundle columns, strip extra columns and index on `seq`
    """
    data = _normalize_measures(data)
    data = _normalize_sex(data)

    data = data.set_index("seq")

    cols = ["measure", "mean", "sex", "sex_id", "standard_error", "age_start", "age_end", "year_start", "year_end", "location_id"]

    return data[cols]


def bundle_to_observations(config, bundle_df):
    """
    Convert bundle into an internal format. It removes the sex column and changes
    location to node. It also adjusts for the demographic specification.
    """
    if "location_id" in bundle_df.columns:
        location_id = bundle_df["location_id"]
    else:
        location_id = np.full(len(bundle_df), config.location_id, dtype=np.int)

    # assume using demographic notation because this bundle uses it.
    demographic_interval_specification = 0
    MATHLOG.info(f"Does this bundle assume demographic notation? {demographic_interval_specification}. "
                 f"A 1 means that 1 year is added to both end ages and end times. A 0 means nothing is added.")

    weight_method = "constant"
    MATHLOG.info(f"The set of weights for this bundle is {weight_method}.")
    # Stick with year_start instead of time_start because that's what's in the
    # bundle, so it's probably what modelers use. Would be nice to pair
    # start with finish or begin with end.
    return pd.DataFrame(
        {
            "measure": bundle_df["measure"],
            "node_id": location_id,
            "density": DensityEnum.gaussian,
            "weight": weight_method,
            "age_start": bundle_df["age_start"],
            "age_end": bundle_df["age_end"] + demographic_interval_specification,
            # The years should be floats in the bundle.
            "year_start": bundle_df["year_start"].astype(np.float),
            "year_end": bundle_df["year_end"].astype(np.float) + demographic_interval_specification,
            "mean": bundle_df["mean"],
            "standard_error": bundle_df["standard_error"],
        }
    )


def bundle_with_study_covariates(execution_context, bundle_id=None, tier=3):
    """Get bundle data with associated study covariate labels.

    Args:
        execution_context (ExecutionContext): The context within which to make the query
        bundle_id (int): Bundle to load. Defaults to the bundle associated with the context
        tier (int): Tier to load data from. Defaults to 3 (frozen data) but will also accept 2 (scratch space)

    Returns:
        A tuple of (bundle data, study covariate labels) where the bundle data is a pd.DataFrame and the labels are a
        pd.DataFrame with an index aligned with bundle data and a column without ``x_`` for each study covariate.
    """
    if bundle_id is None:
        bundle_id = _get_bundle_id(execution_context)

    bundle = _get_bundle_data(execution_context, bundle_id, tier=tier)
    bundle = _normalize_bundle_data(bundle)

    normalized_covariate = get_bundle_study_covariates(bundle.index, bundle_id, execution_context, tier)
    return (bundle, normalized_covariate)