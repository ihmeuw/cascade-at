from collections import defaultdict

import numpy as np
import pandas as pd
from numpy import nan

from cascade.core import getLoggers
from cascade.input_data import InputDataError
from cascade.input_data.configuration.id_map import make_integrand_map
from cascade.input_data.db.bundle import _get_bundle_id, _get_bundle_data
from cascade.stats.estimation import bounds_to_stdev

CODELOG, MATHLOG = getLoggers(__name__)


def _normalize_measures(data):
    """Transform measure_ids into canonical measure names, for instance,
    GBD measure 38 for birth prevalence becomes pini.
    """
    data = data.copy()
    gbd_measure_id_to_integrand = make_integrand_map()
    if any(data.measure_id == 6):
        MATHLOG.warn(f"Found incidence, measure_id=6, in data. Should be Tincidence or Sincidence.")

    if any(data.measure_id == 17):
        MATHLOG.info(
            f"Found case fatality rate, measure_id=17, in data. Ignoring it because it does not "
            f"map to a Dismod-AT integrand and cannot be used by the model."
        )
        data = data[data.measure_id != 17]

    try:
        data["measure"] = data.measure_id.apply(lambda k: gbd_measure_id_to_integrand[k].name)
    except KeyError as ke:
        raise RuntimeError(
            f"The bundle data uses measure {str(ke)} which doesn't map "
            f"to an integrand. The map is {gbd_measure_id_to_integrand}."
        )
    return data


def strip_bundle_exclusions(bundle, ev_settings):
    """Remove measures from bundle as requested in EpiViz-AT settings.

    Args:
        bundle (pd.DataFrame): This bundle has the ``measure`` column with
            each measure as a string name.
        ev_settings: From the settings form, for
            ``model.exclude_data_for_param``.

    Returns:
        A bundle with the same or fewer rows and the same number of columns.
    """
    if not ev_settings.model.is_field_unset("exclude_data_for_param"):
        integrand_map = make_integrand_map()
        measures_to_exclude = [integrand_map[m].name
                               for m in ev_settings.model.exclude_data_for_param
                               if m in integrand_map]
    else:
        measures_to_exclude = list()
    if ev_settings.policies.exclude_relative_risk:
        measures_to_exclude.append("relrisk")
    # else don't add relrisk to excluded measures

    if measures_to_exclude:
        mask = bundle.measure.isin(measures_to_exclude)
        if mask.sum() > 0:
            bundle.loc[mask, "hold_out"] = 1
            MATHLOG.info(
                f"Filtering {mask.sum()} rows of of data where the measure has been excluded. "
                f"Measures marked for exclusion: {measures_to_exclude}. "
                f"{len(bundle)} rows remaining."
            )
    return bundle


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
    """Normalize bundle columns, strip extra columns and index on `seq`.
    Change measures to string names. Add sex as string names.
    Assign ``hold_out`` column.
    """
    data = _normalize_measures(data)
    data = data.assign(hold_out=0)

    cols = ["seq", "measure", "mean", "sex_id", "hold_out",
            "age_start", "age_end", "year_start", "year_end", "location_id",
            "lower", "upper"]

    return data[cols].rename(columns={"age_start": "age_lower", "age_end": "age_upper",
                                      "year_start": "time_lower", "year_end": "time_upper"})


def bundle_to_observations(bundle_df, parent_location_id, data_eta, density, nu):
    """
    Convert bundle into an internal format. It removes the sex column and changes
    location to node. It also adjusts for the demographic specification.

    Args:
        bundle_df (pd.DataFrame): Measurement data.
        parent_location_id (int): Parent location

        data_eta (Dict[str,float]): Default value for eta parameter on distributions as
            a dictionary from measure name to float.
        density (Dict[str,str]): Default values for density parameter on distributions as
            a dictionary from measure name to string.
        nu (Dict[str,float]): The parameter for students-t distributions.

    Returns:
        pd.DataFrame: Includes ``sex_id`` and which indicates
            that these particular observations are from the bundle as
            opposed to ones we add separately. It also keeps the `seq` column
            which aligns bundle data with covariates.
    """
    if "location_id" in bundle_df.columns:
        location_id = bundle_df["location_id"]
    else:
        location_id = np.full(len(bundle_df), parent_location_id, dtype=np.int)
    data_eta = data_eta if data_eta else defaultdict(lambda: nan)
    density = density if density else defaultdict(lambda: "gaussian")

    # assume using demographic notation because this bundle uses it.
    demographic_interval_specification = 0
    MATHLOG.info(f"Does this bundle assume demographic notation? {demographic_interval_specification}. "
                 f"A 1 means that 1 year is added to both end ages and end times. A 0 means nothing is added.")

    weight_method = "constant"
    MATHLOG.info(f"The set of weights for this bundle is {weight_method}.")
    return pd.DataFrame(
        {
            "integrand": bundle_df["measure"],
            "location": location_id,
            # Using getitem instead of get because defaultdict.get returns None
            # when the item is missing.
            "density": bundle_df["measure"].apply(density.__getitem__),
            "eta": bundle_df["measure"].apply(data_eta.__getitem__),
            "nu": bundle_df["measure"].apply(nu.__getitem__),
            "age_lower": bundle_df["age_lower"],
            "age_upper": bundle_df["age_upper"] + demographic_interval_specification,
            # The years should be floats in the bundle.
            "time_lower": bundle_df["time_lower"].astype(np.float),
            "time_upper": bundle_df["time_upper"].astype(np.float) + demographic_interval_specification,
            "mean": bundle_df["mean"],
            "std": bounds_to_stdev(bundle_df.lower, bundle_df.upper),
            "sex_id": bundle_df["sex_id"],
            "name": bundle_df["seq"].astype(str),
            "seq": bundle_df["seq"],  # Keep this until study covariates are added.
            "hold_out": bundle_df["hold_out"],
        }
    )


def normalized_bundle_from_database(execution_context, model_version_id, bundle_id=None, tier=3):
    """Get bundle data with associated study covariate labels.

    Args:
        execution_context (ExecutionContext): The context within which to make the query
        bundle_id (int): Bundle to load. Defaults to the bundle associated with the context
        tier (int): Tier to load data from. Defaults to 3 (frozen data) but will also accept 2 (scratch space)

    Returns:
        bundle data, where the bundle data is a pd.DataFrame.
    """
    if bundle_id is None:
        bundle_id = _get_bundle_id(execution_context, model_version_id)

    bundle = _get_bundle_data(execution_context, model_version_id, bundle_id, tier=tier)
    bundle = _normalize_bundle_data(bundle)

    return bundle


def normalized_bundle_from_disk(path):
    """Load a bundle off disk. It is assumed to be in the same format that Dismod-ODE
    used and we do a bit of adjusting to get it into the same format as our normalized
    from database bundles.
    """
    bundle = dataframe_from_disk(path)
    bundle = bundle.rename(columns={"measure": "measure_id"})
    return _normalize_measures(bundle)


def dataframe_from_disk(path):
    """ Load the file at `path` as a pandas dataframe.
    """
    if any(path.endswith(extension) for extension in [".hdf", ".h5", ".hdf5", ".he5"]):
        return pd.read_hdf(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unknown file format for bundle: {path}")
