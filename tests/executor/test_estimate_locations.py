from numpy.random import RandomState

from cascade.executor.covariate_description import create_covariate_specifications
from cascade.executor.create_settings import create_local_settings
from cascade.executor.estimate_location import (
    retrieve_data, modify_input_data, construct_model, set_priors_from_parent_draws
)
from cascade.input_data.db.locations import location_hierarchy
from cascade.testing_utilities import make_execution_context


def test_retrieve_data(ihme):
    ec = make_execution_context()
    rng = RandomState(2425397)
    locs = location_hierarchy(5, 429)
    local_settings, locations = create_local_settings(rng, locations=locs)
    covariate_multipliers, covariate_data_spec = create_covariate_specifications(
        local_settings.settings.country_covariate, local_settings.settings.study_covariate
    )
    input_data = retrieve_data(ec, local_settings, covariate_data_spec)
    modified_data = modify_input_data(input_data, local_settings, covariate_data_spec)
    model = construct_model(modified_data, local_settings, covariate_multipliers)
    set_priors_from_parent_draws(model, input_data.draws)
