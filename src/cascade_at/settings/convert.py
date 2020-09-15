from collections import defaultdict
from copy import deepcopy
from typing import List, Dict

import numpy as np

from cascade_at.dismod.integrand_mappings import INTEGRAND_MAP
from cascade_at.settings.settings_config import SettingsConfig

CASCADE_LEVEL_ID = ['most_detailed']


def midpoint_list_from_settings(settings: SettingsConfig) -> List[str]:
    """
    Takes the settings configuration for which integrands to midpoint
    which comes in as measure ID and translates that to integrand enums.

    Parameters
    ----------
    settings
        The settings configuration to convert from
    """
    if not settings.model.is_field_unset("midpoint_approximation"):
        measures_midpoint = [
            INTEGRAND_MAP[m].name
            for m in settings.model.midpoint_approximation
            if m in INTEGRAND_MAP]
    else:
        measures_midpoint = list()
    return measures_midpoint


def measures_to_exclude_from_settings(settings: SettingsConfig) -> List[str]:
    """
    Gets the measures to exclude from the data from the model
    settings configuration.

    Parameters
    ----------
    settings
        The settings configuration to convert from
    """
    if not settings.model.is_field_unset("exclude_data_for_param"):
        measures_to_exclude = [
            INTEGRAND_MAP[m].name
            for m in settings.model.exclude_data_for_param
            if m in INTEGRAND_MAP]
    else:
        measures_to_exclude = list()
    if settings.policies.exclude_relative_risk:
        measures_to_exclude.append("relrisk")
    return measures_to_exclude


def data_eta_from_settings(settings: SettingsConfig, default: float = np.nan) -> Dict[str, float]:
    """
    Gets the data eta from the settings Configuration.
    The default data eta is np.nan.

    Parameters
    ----------
    settings
        The settings configuration to convert from
    default
        The default eta to use
    """
    data_eta = defaultdict(lambda: default)
    if not settings.eta.is_field_unset("data") and settings.eta.data:
        data_eta = defaultdict(lambda: float(settings.eta.data))
    for set_eta in settings.data_eta_by_integrand:
        data_eta[INTEGRAND_MAP[set_eta.integrand_measure_id].name] = float(set_eta.value)
    return data_eta


def density_from_settings(settings: SettingsConfig, default: str = "gaussian") -> Dict[str, str]:
    """
    Gets the density from the settings Configuration.
    The default density is "gaussian".

    Parameters
    ----------
    settings
        The settings configuration to convert from
    default
        The default data density to use
    """
    density = defaultdict(lambda: default)
    if not settings.model.is_field_unset("data_density") and settings.model.data_density:
        density = defaultdict(lambda: settings.model.data_density)
    for set_density in settings.data_density_by_integrand:
        density[INTEGRAND_MAP[set_density.integrand_measure_id].name] = set_density.value
    return density


def data_cv_from_settings(settings: SettingsConfig, default: float = 0.0) -> Dict[str, float]:
    """ Gets the data min coefficient of variation from the settings Configuration

    Parameters
    ----------
    settings
        The settings configuration to convert from
    default
        The default data coefficient of variation
    """
    data_cv = defaultdict(lambda: default)
    if not settings.model.is_field_unset("minimum_meas_cv") and settings.model.minimum_meas_cv:
        data_cv = defaultdict(
            lambda: float(settings.model.minimum_meas_cv))
    for set_data_cv in settings.data_cv_by_integrand:
        data_cv[INTEGRAND_MAP[
            set_data_cv.integrand_measure_id].name] = float(
                set_data_cv.value)
    return data_cv


def min_cv_from_settings(settings: SettingsConfig, default: float = 0.0) -> defaultdict:
    """
    Gets the minimum coefficient of variation by rate and level
    of the cascade from settings. First key is cascade level, second is rate

    Parameters
    ----------
    settings
        The settings configuration from which to pull
    default
        The default min CV to use when not specified
    """
    # This is a hack to over-ride the default value while the visualization
    # team is fixing the bug in the cascade level drop-down menu.
    if not settings.is_field_unset("min_cv") and settings.min_cv:
        cascade_levels = [cv.cascade_level_id for cv in settings.min_cv]
        values = [cv.value for cv in settings.min_cv]
        if "most_detailed" in cascade_levels:
            default = values[cascade_levels.index("most_detailed")]
    inner = defaultdict(lambda: default)
    outer = defaultdict(lambda: deepcopy(inner))

    if not settings.is_field_unset("min_cv") and settings.min_cv:
        for cv in settings.min_cv:
            # The following lambda function is the only way to get the defaultdict
            # creation to work within a loop. It does *not* work if you only do
            # defaultdict(lambda: cv.value). As you go through the iterations, it just
            # pulls in the current value for cv because lambda functions can't go back in time.
            outer.update({cv.cascade_level_id: defaultdict(lambda val=cv.value: val)})
    if not settings.is_field_unset("min_cv_by_rate") and settings.min_cv_by_rate:
        for rate_cv in settings.min_cv_by_rate:
            outer[rate_cv.cascade_level_id].update({rate_cv.rate_measure_id: rate_cv.value})
    return outer


def nu_from_settings(settings: SettingsConfig, default: float = np.nan) -> Dict[str, float]:
    """
    Gets nu from the settings Configuration.
    The default nu is np.nan.

    Parameters
    ----------
    settings
        The settings configuration from which to pull
    default
        The default nu to use when not specified in the settings
    """
    nu = defaultdict(lambda: default)
    nu["students"] = settings.students_dof.data
    nu["log_students"] = settings.log_students_dof.data
    return nu
