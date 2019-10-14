from collections import defaultdict

import numpy as np
import pandas as pd
from numpy import nan

from cascade.core import getLoggers
from cascade.input_data import InputDataError
from cascade.input_data.configuration.id_map import make_integrand_map
from cascade.input_data.db.crosswalk_version import _get_crosswalk_version_id, _get_crosswalk_version
from cascade.stats.estimation import bounds_to_stdev

CODELOG, MATHLOG = getLoggers(__name__)


def _normalize_measures(data):
    """Transform measure_ids into canonical measure names, for instance,
    GBD measure 38 for birth prevalence becomes pini.
    """
    data = data.copy()
    gbd_measure_id_to_integrand = make_integrand_map()
    if any(data.measure_id == 6):
        MATHLOG.warning(f"Found incidence, measure_id=6, in data. Should be Tincidence or Sincidence.")

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
            f"The crosswalk version data uses measure {str(ke)} which doesn't map "
            f"to an integrand. The map is {gbd_measure_id_to_integrand}."
        )
    return data


def strip_crosswalk_version_exclusions(crosswalk_version, ev_settings):
    """Remove measures from crosswalk version as requested in EpiViz-AT settings.

    Args:
        crosswalk_version (pd.DataFrame): This crosswalk version has the ``measure`` column with
            each measure as a string name.
        ev_settings: From the settings form, for
            ``model.exclude_data_for_param``.

    Returns:
        A crosswalk version with the same or fewer rows and the same number of columns.
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
        mask = crosswalk_version.measure.isin(measures_to_exclude)
        if mask.sum() > 0:
            crosswalk_version.loc[mask, "hold_out"] = 1
            MATHLOG.info(
                f"Filtering {mask.sum()} rows of of data where the measure has been excluded. "
                f"Measures marked for exclusion: {measures_to_exclude}. "
                f"{len(crosswalk_version)} rows remaining."
            )
    return crosswalk_version


def _normalize_sex(data):
    """Transform sex_ids from 1, 2, 3 to male, female, both.
    """
    data = data.copy()
    try:
        data["sex"] = data.sex_id.apply({1: "Male", 2: "Female", 3: "Both", 4: "Unspecified"}.get)
    except KeyError as ke:
        raise InputDataError(f"Unrecognized sex id") from ke
    return data


def _normalize_crosswalk_version(data):
    """Normalize crosswalk version columns, strip extra columns and index on `seq`.
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


def crosswalk_version_to_observations(crosswalk_version, parent_location_id, data_eta, density, nu):
    """
    Convert crosswalk version into an internal format. It removes the sex column and changes
    location to node. It also adjusts for the demographic specification.

    Args:
        crosswalk_version (pd.DataFrame): Measurement data.
        parent_location_id (int): Parent location

        data_eta (Dict[str,float]): Default value for eta parameter on distributions as
            a dictionary from measure name to float.
        density (Dict[str,str]): Default values for density parameter on distributions as
            a dictionary from measure name to string.
        nu (Dict[str,float]): The parameter for students-t distributions.

    Returns:
        pd.DataFrame: Includes ``sex_id`` and which indicates
            that these particular observations are from the crosswalk version as
            opposed to ones we add separately. It also keeps the `seq` column
            which aligns crosswalk version data with covariates.
    """
    if "location_id" in crosswalk_version.columns:
        location_id = crosswalk_version["location_id"]
    else:
        location_id = np.full(len(crosswalk_version), parent_location_id, dtype=np.int)
    data_eta = data_eta if data_eta else defaultdict(lambda: nan)
    density = density if density else defaultdict(lambda: "gaussian")

    # assume using demographic notation because this crosswalk version uses it.
    demographic_interval_specification = 0
    MATHLOG.info(f"Does this crosswalk version assume demographic notation? {demographic_interval_specification}. "
                 f"A 1 means that 1 year is added to both end ages and end times. A 0 means nothing is added.")

    weight_method = "constant"
    MATHLOG.info(f"The set of weights for this crosswalk version is {weight_method}.")
    return pd.DataFrame(
        {
            "integrand": crosswalk_version["measure"],
            "location": location_id,
            # Using getitem instead of get because defaultdict.get returns None
            # when the item is missing.
            "density": crosswalk_version["measure"].apply(density.__getitem__),
            "eta": crosswalk_version["measure"].apply(data_eta.__getitem__),
            "nu": crosswalk_version["measure"].apply(nu.__getitem__),
            "age_lower": crosswalk_version["age_lower"],
            "age_upper": crosswalk_version["age_upper"] + demographic_interval_specification,
            # The years should be floats in the crosswalk version.
            "time_lower": crosswalk_version["time_lower"].astype(np.float),
            "time_upper": crosswalk_version["time_upper"].astype(np.float) + demographic_interval_specification,
            "mean": crosswalk_version["mean"],
            "std": bounds_to_stdev(crosswalk_version.lower, crosswalk_version.upper),
            "sex_id": crosswalk_version["sex_id"],
            "name": crosswalk_version["seq"].astype(str),
            "seq": crosswalk_version["seq"],  # Keep this until study covariates are added.
            "hold_out": crosswalk_version["hold_out"],
        }
    )


def normalized_crosswalk_version_from_database(execution_context, model_version_id, crosswalk_version_id=None):
    """Get crosswalk version with associated study covariate labels.

    Args:
        execution_context (ExecutionContext): The context within which to make the query
        model_version_id (int): The model version ID to get the crosswalk version ID from
        crosswalk_version_id (int): Crosswalk version to load.
            Defaults to the crosswalk version associated with the context

    Returns:
        crosswalk data, where the crosswalk data is a pd.DataFrame.
    """
    if crosswalk_version_id is None:
        crosswalk_version_id = _get_crosswalk_version_id(execution_context, model_version_id)

    crosswalk_version = _get_crosswalk_version(model_version_id, crosswalk_version_id)
    crosswalk_version = _normalize_crosswalk_version(crosswalk_version)

    return crosswalk_version
