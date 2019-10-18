import pandas as pd

from cascade.core import getLoggers
from cascade.input_data.configuration.construct_country import (
    convert_gbd_ids_to_dismod_values
)
from cascade.input_data.db.csmr import get_csmr_data
from cascade.input_data.db.locations import location_hierarchy, get_descendants
from cascade.input_data.db.mortality import get_frozen_cause_specific_mortality_data
from cascade.runner.application_config import application_config

CODELOG, MATHLOG = getLoggers(__name__)


def get_raw_csmr(execution_context, data_access,
                 included_locations, age_spans):
    """Gets CSMR that has age_lower, age_upper, but no further processing."""
    assert isinstance(age_spans, pd.DataFrame)
    parameters = application_config()["NonModel"]
    if data_access.tier == 3:
        CODELOG.debug(f"Getting CSMR from tier 3")
        raw_csmr = get_frozen_cause_specific_mortality_data(
            execution_context, data_access.model_version_id)
    else:
        small_number_locations = parameters.getint("small-location-count")
        if included_locations is not None and len(included_locations) < small_number_locations:
            raw_csmr = get_csmr_data(
                execution_context,
                included_locations,
                data_access.add_csmr_cause,
                data_access.cod_version,
                data_access.gbd_round_id,
                data_access.decomp_step,
            )
            assert set(raw_csmr.location_id.unique()) == set(included_locations), \
                "The get_raw_csmr csmr query failed to return all of the requested locations."
        else:
            CODELOG.debug(
                f"CSMR retrieval for location_set_id {data_access.location_set_id}.")
            raw_csmr = get_csmr_data(
                execution_context,
                data_access.location_set_id,
                data_access.add_csmr_cause,
                data_access.cod_version,
                data_access.gbd_round_id,
                data_access.decomp_step,
            )
    return convert_gbd_ids_to_dismod_values(raw_csmr, age_spans)


def normalize_csmr(raw_csmr, sex_id):
    assert isinstance(sex_id, list)
    csmr = raw_csmr.assign(measure="mtspecific")
    csmr = csmr.query(f"sex_id in @sex_id")
    MATHLOG.debug(f"Creating a set of {csmr.shape[0]} mtspecific observations from IHME CSMR database.")
    return csmr.assign(hold_out=0)


def location_and_children_from_settings(data_access, parent_id):
    """Given the data_access member of a local_settings, return location and its
    children."""
    not_just_children_but_all_descendants = True
    locations = location_hierarchy(
        data_access.gbd_round_id, location_set_version_id=data_access.location_set_version_id)
    location_and_children = get_descendants(
        locations, parent_id, children_only=not_just_children_but_all_descendants, include_parent=True)
    return location_and_children
