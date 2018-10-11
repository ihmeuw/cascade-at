import pytest

import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal

from cascade.core.context import ModelContext
from cascade.model.grids import PriorGrid, AgeTimeGrid
from cascade.model.rates import Smooth
from cascade.model.priors import Gaussian, Uniform
from cascade.testing_utilities import make_execution_context
from cascade.model.covariates import Covariate, CovariateMultiplier
from cascade.dismod.serialize import (
    model_to_dismod_file,
    collect_ages_or_times,
    collect_priors,
    make_age_table,
    make_time_table,
    integrand_to_id,
    make_prior_table,
    make_smooth_and_smooth_grid_tables,
    make_node_table,
    make_data_table,
    make_rate_table,
    make_covariate_table,
)
from cascade.dismod.db.wrapper import DismodFile, _get_engine
from cascade.dismod.db.metadata import DensityEnum


class LocationNode:
    def __init__(self, name, location_id, children):
        self.info = {"location_name_short": name}
        self.id = location_id
        self._children = children
        self.root = None

    def level_n_descendants(self, n):
        assert n == 1, "Levels other than 1 not supported for this mock"
        return self._children


@pytest.fixture
def mock_get_location_hierarchy_from_gbd(mocker):
    get_location_hierarchy_from_gbd = mocker.patch("cascade.dismod.serialize.get_location_hierarchy_from_gbd")
    the_uk = LocationNode("United Kingdom", 1004, [])
    europe = LocationNode("Europe", 10, [the_uk])
    earth = LocationNode("Earth", 42, [europe])
    the_uk.root = earth
    europe.root = earth
    earth.root = earth
    get_location_hierarchy_from_gbd.return_value = earth


def make_data(integrands):
    ages = np.arange(0, 121, 5, dtype=float)
    times = np.arange(1980, 2016, 5, dtype=float)
    df = pd.MultiIndex.from_product([ages, times, integrands], names=["age_start", "year_start", "measure"])
    df = pd.DataFrame(index=df).reset_index()
    df["age_end"] = df.age_start + 5
    df["year_end"] = df.year_start + 5
    df["node_id"] = 1
    df["sex"] = "Both"
    df["density"] = DensityEnum.gaussian
    df["weight"] = "constant"

    df["mean"] = 0
    df["standard_error"] = 0.001

    return df


@pytest.fixture
def observations():
    return make_data(["Sincidence", "mtexcess"])


@pytest.fixture
def constraints():
    return make_data(["mtother"])


@pytest.fixture
def base_context(observations, constraints):
    context = ModelContext()
    context.parameters.rate_case = "iota_pos_rho_zero"
    context.parameters.minimum_meas_cv = 0

    context.input_data.observations = observations
    context.input_data.constraints = constraints

    grid = AgeTimeGrid.uniform(age_start=0, age_end=100, age_step=1, time_start=1990, time_end=2018, time_step=5)

    d_time = PriorGrid(grid)
    d_time[:, :].prior = Gaussian(0, 0.1, eta=1)
    d_age = PriorGrid(grid)
    d_age[:, :].prior = Gaussian(0, 0.1, name="TestPrior")
    value = PriorGrid(grid)
    value[:, :].prior = Gaussian(0, 0.1)

    smooth = Smooth(name="iota_smooth")
    smooth.d_time_priors = d_time
    smooth.d_age_priors = d_age
    smooth.value_priors = value
    context.rates.iota.parent_smooth = smooth

    smooth = Smooth()
    d_time = PriorGrid(grid)
    d_time[:, :].prior = Gaussian(1, 0.1)
    d_time.hyper_prior = Uniform(0, 1, 0.5)
    smooth.d_time_priors = d_time
    context.rates.pini.parent_smooth = smooth

    context.average_integrand_cases = pd.DataFrame(
        [],
        columns=[
            "integrand_name",
            "age_lower",
            "age_upper",
            "time_lower",
            "time_upper",
            "weight_id",
            "node_id",
            "x_sex",
        ],
    )
    return context


def test_development_target(base_context, mock_get_location_hierarchy_from_gbd):
    ec = make_execution_context(location_id=180)
    dm = model_to_dismod_file(base_context, ec)
    e = _get_engine(None)
    dm.engine = e
    dm.flush()
    dm2 = DismodFile(e)

    age_table = make_age_table(base_context)
    time_table = make_time_table(base_context)

    prior_table, prior_objects = make_prior_table(base_context, dm.density)

    smooth_table, smooth_grid_table, smooth_id_func = make_smooth_and_smooth_grid_tables(
        base_context, age_table, time_table, prior_objects
    )

    def _compare_tables(a, b):
        assert_frame_equal(a.sort_index("columns"), b.sort_index("columns"), check_names=False)

    _compare_tables(dm2.smooth_grid, smooth_grid_table)
    _compare_tables(dm2.age, age_table)
    _compare_tables(dm2.time, time_table)
    _compare_tables(dm2.prior, prior_table)


def test_collect_priors(base_context):
    priors = collect_priors(base_context)
    assert priors == {
        Gaussian(0, 0.1, name="TestPrior"),
        Uniform(0, 1, 0.5),
        Gaussian(1, 0.1),
        Gaussian(0, 0.1),
        Gaussian(0, 0.1, eta=1),
    }


def test_collect_ages_or_times__ages(base_context):
    ages = collect_ages_or_times(base_context, "ages")
    assert set(ages) == set(range(0, 100, 1)) | {125}


def test_collect_ages_or_times__no_data(base_context):
    base_context.input_data.observations = None
    base_context.input_data.constraints = None
    ages = collect_ages_or_times(base_context, "ages")
    assert set(ages) == set(range(0, 100, 1))


def test_collect_ages_or_times__times(base_context):
    times = collect_ages_or_times(base_context, "times")
    true_times = set(range(1990, 2016, 5)) | {1980, 2020}
    assert set(times) == true_times


def test_make_age_table(base_context):
    df = make_age_table(base_context)

    assert df.age.equals(pd.Series(list(range(0, 100, 1)) + [125], dtype=float))


def test_make_time_table(base_context):
    df = make_time_table(base_context)

    assert df.time.equals(pd.Series([1980] + list(range(1990, 2016, 5)) + [2020], dtype=float))


def test_make_prior_table(base_context):
    dm = DismodFile(None)
    dm.make_densities()

    prior_table, prior_id_func = make_prior_table(base_context, dm.density)
    prior_objects = collect_priors(base_context)

    assert len(prior_table) == len(prior_objects)

    raw_prior_table = prior_table.merge(dm.density, on="density_id")
    prior_table = raw_prior_table.rename(columns={"density_name": "density"}).drop(["density_id", "prior_id"], 1)

    def p_to_r(p):
        d = dict(
            prior_name=None, density=None, upper=np.nan, lower=np.nan, mean=np.nan, std=np.nan, eta=np.nan, nu=np.nan
        )
        d.update(p.parameters())
        return d

    for obj in prior_objects:
        r_dict = dict(prior_table.loc[prior_id_func(obj)])
        o_dict = p_to_r(obj)

        if o_dict["prior_name"] is None:
            # In this case the code made up a name for the prior and we don't care what it is
            r_dict["prior_name"] = None

        assert set(r_dict.keys()) == set(o_dict.keys())
        for k, v in r_dict.items():
            if k == "eta" and o_dict[k] is None:
                assert np.isnan(r_dict[k])
            else:
                assert r_dict[k] == o_dict[k] or (np.isnan(r_dict[k]) and np.isnan(o_dict[k]))


def test_make_smooth_and_smooth_grid_tables(base_context):
    dm = DismodFile(None)
    dm.make_densities()

    age_table = make_age_table(base_context)
    time_table = make_time_table(base_context)

    prior_table, prior_objects = make_prior_table(base_context, dm.density)

    smooth_table, smooth_grid_table, smooth_id_func = make_smooth_and_smooth_grid_tables(
        base_context, age_table, time_table, prior_objects
    )

    assert len(smooth_table) == 2
    assert "iota_smooth" in smooth_table.smooth_name.values

    assert set(smooth_table.index) == set(smooth_grid_table.smooth_id)


def test_make_node_table(base_context, mock_get_location_hierarchy_from_gbd):
    node_table, _ = make_node_table(base_context)

    expected = pd.DataFrame(
        [["Earth", 0, np.nan, 42], ["Europe", 1, 0, 10], ["United Kingdom", 2, 1, 1004]],
        columns=["node_name", "node_id", "parent", "c_location_id"],
    )
    assert_frame_equal(node_table, expected, check_like=True)


def test_make_data_table(base_context, mock_get_location_hierarchy_from_gbd):
    node_table, _ = make_node_table(base_context)
    data_table = make_data_table(base_context, node_table)

    assert len(data_table) == len(base_context.input_data.observations) + len(base_context.input_data.constraints)
    assert len(data_table.query("hold_out==1")) == len(base_context.input_data.constraints)


def test_make_rate_table(base_context):
    dm = DismodFile(None)
    dm.make_densities()
    age_table = make_age_table(base_context)
    time_table = make_time_table(base_context)
    prior_table, prior_objects = make_prior_table(base_context, dm.density)
    smooth_table, smooth_grid_table, smooth_id_func = make_smooth_and_smooth_grid_tables(
        base_context, age_table, time_table, prior_objects
    )

    rate_table, rate_to_id = make_rate_table(base_context, smooth_id_func)

    assert rate_table.rate_name.tolist() == ["pini", "iota", "rho", "chi", "omega"]

    for rate in base_context.rates:
        df = rate_table.loc[rate_table["rate_name"] == rate.name]
        if not df.empty:
            rate_id = float(df.rate_id)
            assert rate_to_id(rate) == rate_id


def test_make_covariate_table(base_context):
    at_grid = AgeTimeGrid.uniform(age_start=0, age_end=120, age_step=5, time_start=1990, time_end=2018, time_step=1)
    value_priors = PriorGrid(at_grid)
    value_priors[:, :].prior = Gaussian(0, 0.8)
    at_priors = PriorGrid(at_grid)
    at_priors[:, :].prior = Gaussian(0, 0.15)

    income = Covariate("income")
    income.reference = 1000
    income.max_difference = None
    income_time_tight = CovariateMultiplier(income, Smooth(value_priors, at_priors, at_priors))

    wash = Covariate("wash")
    wash.reference = 0
    wash.max_difference = None
    wash_cov = CovariateMultiplier(wash, Smooth(value_priors, at_priors, at_priors))

    sex_id = Covariate("sex")
    sex_id.reference = -0.5
    sex_id.max_difference = 0.5
    # A sex covariate is often also used as a study covariate.

    base_context.input_data.covariates.extend([income, wash, sex_id])

    # There isn't much to test about the lists of covariate multipliers.
    # They are lists and would permit, for instance, adding the same one twice.
    base_context.rates.iota.covariate_multipliers.append(income_time_tight)
    base_context.integrand_covariate_multipliers["remission"].value_covariate_multipliers.append(income_time_tight)
    base_context.integrand_covariate_multipliers["prevalence"].std_covariate_multipliers.append(income_time_tight)

    for rate_adj in base_context.rates:
        rate_adj.covariate_multipliers.append(wash_cov)

    dm = DismodFile(None)
    dm.make_densities()
    age_table = make_age_table(base_context)
    time_table = make_time_table(base_context)
    prior_table, prior_objects = make_prior_table(base_context, dm.density)
    smooth_table, smooth_grid_table, smooth_id_func = make_smooth_and_smooth_grid_tables(
        base_context, age_table, time_table, prior_objects
    )

    rate_table, rate_to_id = make_rate_table(base_context, smooth_id_func)
    columns_df, mulcov_df, columns_func = make_covariate_table(
        base_context, smooth_id_func, rate_to_id, integrand_to_id
    )
