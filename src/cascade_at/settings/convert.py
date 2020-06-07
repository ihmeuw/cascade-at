from collections import defaultdict
import numpy as np

from cascade_at.settings.settings_config import SettingsConfig
from cascade_at.dismod.integrand_mappings import INTEGRAND_MAP


def measures_to_exclude_from_settings(settings: SettingsConfig):
    """
    Gets the measures to exclude from the data from the model
    settings configuration.
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


def data_eta_from_settings(settings: SettingsConfig, default: float = np.nan):
    """
    Gets the data eta from the settings Configuration.
    The default data eta is np.nan.
    """
    data_eta = defaultdict(lambda: default)
    if not settings.eta.is_field_unset("data") and settings.eta.data:
        data_eta = defaultdict(lambda: float(settings.eta.data))
    for set_eta in settings.data_eta_by_integrand:
        data_eta[INTEGRAND_MAP[set_eta.integrand_measure_id].name] = float(set_eta.value)
    return data_eta


def density_from_settings(settings: SettingsConfig, default: str = "gaussian"):
    """
    Gets the density from the settings Configuration.
    The default density is "gaussian".
    """
    density = defaultdict(lambda: default)
    if not settings.model.is_field_unset("data_density") and settings.model.data_density:
        density = defaultdict(lambda: settings.model.data_density)
    for set_density in settings.data_density_by_integrand:
        density[INTEGRAND_MAP[set_density.integrand_measure_id].name] = set_density.value
    return density


def data_cv_from_settings(settings: SettingsConfig, default: float = 0.0):
    """
    Gets the data min coefficient of variation from the settings Configuration

    Args:
        settings: (cascade.settings.configuration.Configuration)
        default: (float) default data cv

    Returns:
        dictionary of data cv's from settings
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


def nu_from_settings(settings: SettingsConfig, default: float = np.nan):
    """
    Gets nu from the settings Configuration.
    The default nu is np.nan.
    """
    nu = defaultdict(lambda: default)
    nu["students"] = settings.students_dof.data
    nu["log_students"] = settings.log_students_dof.data
    return nu
