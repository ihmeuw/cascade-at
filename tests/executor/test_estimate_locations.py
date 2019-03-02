from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.random import RandomState

from cascade.executor.covariate_description import create_covariate_specifications
from cascade.executor.create_settings import create_local_settings
from cascade.executor.estimate_location import (
    modify_input_data, construct_model, set_priors_from_parent_draws, estimate_location
)
from cascade.input_data.db.locations import location_hierarchy
from cascade.testing_utilities import make_execution_context
from cascade.testing_utilities.fake_data import retrieve_fake_data
from cascade.executor.dismodel_main import generate_plan, parse_arguments


@pytest.mark.parametrize("meid,mvid", [
    (None, 267800),
])
def test_with_known_id(ihme, meid, mvid):
    """This runs the equivalent of dismodel_main.main"""
    ec = make_execution_context()
    # no-upload keeps this from going to the databases when it's done.
    args = ["z.db", "--no-upload", "--db-only"]
    if mvid:
        args += ["--mvid", str(mvid)]
    elif meid:
        args += ["--meid", str(meid)]
    else:
        assert meid or mvid
    plan = generate_plan(ec, parse_arguments(args))
    for task_id in plan.cascade_jobs:
        job, this_location_work = plan.cascade_job(task_id)
        if job == "estimate_location":
            estimate_location(ec, this_location_work)
            break  # Do one, not the whole tree.
        # else is a bundle setup.


@pytest.mark.parametrize("draw", list(range(10)))
def test_retrieve_data(ihme, draw):
    ec = make_execution_context()
    rng = RandomState(524287 + 131071 * draw)
    locs = location_hierarchy(5, 429)
    local_settings, locations = create_local_settings(rng, locations=locs)
    covariate_multipliers, covariate_data_spec = create_covariate_specifications(
        local_settings.settings.country_covariate, local_settings.settings.study_covariate
    )
    # Here, we create a fake input data so that there is a bundle and study covariates.
    input_data = retrieve_fake_data(ec, local_settings, covariate_data_spec)
    modified_data = modify_input_data(input_data, local_settings, covariate_data_spec)
    model = construct_model(modified_data, local_settings, covariate_multipliers, covariate_data_spec)
    set_priors_from_parent_draws(model, input_data.draws)


def print_types(data):
    """Walks through data types and prints them out.

    >>> input_data = retrieve_data(ec, local_settings, covariate_data_spec)
    >>> print_types(input_data)

    """
    with Path(f"dtypes{np.random.randint(0, 999)}.txt").open("w") as outfile:
        for member_name in dir(data):
            member = getattr(data, member_name)
            if isinstance(member, pd.DataFrame):
                dtypes = member.dtypes
                as_data = list(zip(dtypes.index.tolist(), dtypes.tolist()))
                print(f"data_type {member_name}: {as_data}", file=outfile)
