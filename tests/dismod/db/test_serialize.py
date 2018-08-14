import pytest

import pandas as pd
import numpy as np

from cascade.core.context import ModelContext
from cascade.model.grids import PriorGrid, AgeTimeGrid
from cascade.model.rates import Smooth
from cascade.model.priors import GaussianPrior
from cascade.dismod.db.serialize import (
    dismodfile_from_model_context,
    collect_ages_or_times,
    collect_priors,
    make_age_table,
    make_time_table,
    make_prior_table,
    make_smooth_and_smooth_grid_tables,
)
from cascade.dismod.db.wrapper import DismodFile, _get_engine


@pytest.fixture
def base_context():
    context = ModelContext()

    grid = AgeTimeGrid.uniform(age_start=0, age_end=120, age_step=1, time_start=1990, time_end=2018, time_step=5)

    d_time = PriorGrid(grid)
    d_time[:, :].prior = GaussianPrior(0, 0.1)
    d_age = PriorGrid(grid)
    d_age[:, :].prior = GaussianPrior(0, 0.1)
    value = PriorGrid(grid)
    value[:, :].prior = GaussianPrior(0, 0.1)

    smooth = Smooth()
    smooth.d_time_priors = d_time
    smooth.d_age_priors = d_age
    smooth.value_priors = value

    context.rates.iota.parent_smooth = smooth

    return context


def test_development_target(base_context):
    dm = dismodfile_from_model_context(base_context)
    e = _get_engine(None)
    dm.engine = e
    dm.flush()
    dm2 = DismodFile(e, {}, {})
    print(dm2.smooth_grid)


def test_collect_priors(base_context):
    priors = collect_priors(base_context)
    assert priors == {GaussianPrior(0, 0.1)}


def test_collect_ages_or_times__ages(base_context):
    ages = collect_ages_or_times(base_context, "ages")
    assert set(ages) == set(range(0, 120, 1))


def test_collect_ages_or_times__times(base_context):
    ages = collect_ages_or_times(base_context, "times")
    assert set(ages) == set(range(1990, 2018, 5))


def test_make_age_table(base_context):
    df = make_age_table(base_context)

    assert df.age.equals(pd.Series(range(0, 120, 1)))


def test_make_time_table(base_context):
    df = make_time_table(base_context)

    assert df.time.equals(pd.Series(range(1990, 2018, 5)))


def test_make_prior_table(base_context):
    dm = DismodFile(None, {}, {})
    dm.make_densities()

    prior_table, prior_objects = make_prior_table(base_context, dm.density)

    assert len(prior_table) == len(prior_objects)

    prior_table = prior_table.merge(dm.density, on="density_id")
    prior_table = prior_table.rename(columns={"density_name": "density"}).drop(["density_id", "prior_id"], 1)

    def p_to_r(p):
        d = dict(
            prior_name=None, density=None, upper=np.nan, lower=np.nan, mean=np.nan, std=np.nan, eta=np.nan, nu=np.nan
        )
        d.update(p.parameters())
        return d

    for row, obj in zip(prior_table.iterrows(), prior_objects):
        r_dict = dict(row[1])
        o_dict = p_to_r(obj)

        if o_dict["prior_name"] is None:
            # In this case the code made up a name for the prior and we don't care what it is
            r_dict["prior_name"] = None

        assert set(r_dict.keys()) == set(o_dict.keys())
        for k, v in r_dict.items():
            assert r_dict[k] == o_dict[k] or (np.isnan(r_dict[k]) and np.isnan(o_dict[k]))


def test_make_smooth_and_smooth_grid_tables(base_context):
    dm = DismodFile(None, {}, {})
    dm.make_densities()

    age_table = make_age_table(base_context)
    time_table = make_time_table(base_context)

    prior_table, prior_objects = make_prior_table(base_context, dm.density)

    smooth_table, smooth_grid_table = make_smooth_and_smooth_grid_tables(
        base_context, age_table, time_table, prior_objects
    )

    assert len(smooth_table) == 1

    assert len(smooth_grid_table) == len(age_table) * len(time_table)
