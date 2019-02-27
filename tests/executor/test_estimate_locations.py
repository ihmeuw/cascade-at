import pytest
from numpy.random import RandomState

from cascade.executor.covariate_description import create_covariate_specifications
from cascade.executor.create_settings import create_local_settings
from cascade.executor.estimate_location import (
    modify_input_data, construct_model, set_priors_from_parent_draws
)
from cascade.input_data.db.locations import location_hierarchy
from cascade.testing_utilities import make_execution_context
from cascade.testing_utilities.fake_data import retrieve_fake_data


@pytest.mark.parametrize("draw", list(range(10)))
def test_retrieve_data(ihme, draw):
    ec = make_execution_context()
    rng = RandomState(524287 + 131071 * draw)
    locs = location_hierarchy(5, 429)
    local_settings, locations = create_local_settings(rng, locations=locs)
    covariate_multipliers, covariate_data_spec = create_covariate_specifications(
        local_settings.settings.country_covariate, local_settings.settings.study_covariate
    )
    input_data = retrieve_fake_data(ec, local_settings, covariate_data_spec)
    modified_data = modify_input_data(input_data, local_settings, covariate_data_spec)
    model = construct_model(modified_data, local_settings, covariate_multipliers)
    set_priors_from_parent_draws(model, input_data.draws)
