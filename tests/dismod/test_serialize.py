import pytest

import networkx as nx
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
    make_covariate_table,
    make_prior_table,
    make_smooth_and_smooth_grid_tables,
    make_node_table,
    make_data_table,
    make_rate_and_nslist_tables,
    make_mulcov_table,
    _infer_rate_case,
)
from cascade.dismod.db.wrapper import DismodFile, get_engine
from cascade.dismod.constants import DensityEnum


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
    location_hierarchy = mocker.patch("cascade.dismod.serialize.location_hierarchy")
    g = nx.DiGraph()
    g.add_nodes_from([
        (1004, dict(location_name="United Kingdom")),
        (10, dict(location_name="Europe")),
        (42, dict(location_name="Earth")),
    ])
    g.add_edges_from([(42, 10), (10, 1004)])
    location_hierarchy.return_value = g


def make_data(integrands):
    ages = np.arange(0, 121, 5, dtype=float)
    times = np.arange(1980, 2016, 5, dtype=float)
    df = pd.MultiIndex.from_product([ages, times, integrands], names=["age_lower", "time_lower", "measure"])
    df = pd.DataFrame(index=df).reset_index()
    df["age_upper"] = df.age_lower + 5
    df["time_upper"] = df.time_lower + 5
    df["node_id"] = 10
    df["sex"] = "Both"
    df["density"] = DensityEnum.gaussian
    df["weight"] = "constant"

    df["mean"] = 0
    df["standard_error"] = 0.001
    df["hold_out"] = 0

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
    context.parameters.parent_location_id = 42
    context.parameters.ode_step_size = 5
    context.parameters.additional_ode_steps = [0.019, 0.25, 0.5]
    context.parameters.minimum_meas_cv = 0

    context.input_data.observations = observations

    grid = AgeTimeGrid.uniform(age_lower=0, age_upper=100, age_step=1, time_lower=1990, time_upper=2018, time_step=5)

    d_time = PriorGrid(grid)
    d_time[:, :].prior = Gaussian(0, 0.1, eta=1)
    d_age = PriorGrid(grid)
    d_age[:, :].prior = Gaussian(0, 0.1, name="TestPrior")
    value = PriorGrid(grid)
    value[:, :].prior = Gaussian(0.1, 0.1, lower=0.01)

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
    ec = make_execution_context(parent_location_id=180)
    dm = model_to_dismod_file(base_context, ec)
    e = get_engine(None)
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
        Gaussian(0.1, 0.1, lower=0.01),
        Gaussian(0, 0.1, eta=1),
    }


def test_collect_ages_or_times__ages(base_context):
    ages = collect_ages_or_times(base_context, "ages")
    assert set(ages) == set(range(0, 100, 1)) | {125}


def test_collect_ages_or_times__no_data(base_context):
    base_context.input_data.observations = None
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
    density = pd.DataFrame({"density_name": [x.name for x in DensityEnum]})
    dm.density = density.assign(density_id=density.index)

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
    density = pd.DataFrame({"density_name": [x.name for x in DensityEnum]})
    dm.density = density.assign(density_id=density.index)

    age_table = make_age_table(base_context)
    time_table = make_time_table(base_context)

    prior_table, prior_objects = make_prior_table(base_context, dm.density)

    smooth_table, smooth_grid_table, smooth_id_func = make_smooth_and_smooth_grid_tables(
        base_context, age_table, time_table, prior_objects
    )

    assert len(smooth_table) == 2
    assert "iota_smooth    1" in smooth_table.smooth_name.values

    assert set(smooth_table.index) == set(smooth_grid_table.smooth_id)


def test_make_node_table(base_context, mock_get_location_hierarchy_from_gbd):
    node_table, _ = make_node_table(base_context)

    expected = pd.DataFrame(
        [["Earth", 0, np.nan, 42], ["Europe", 1, 0, 10], ["United Kingdom", 2, 1, 1004]],
        columns=["node_name", "node_id", "parent", "c_location_id"],
    )
    assert_frame_equal(node_table, expected, check_like=True)


def test_live_locations_node_table(ihme):
    """Ensure Global has no parent."""
    ec = make_execution_context(gbd_round_id=5)
    node_table, location_to_node_func = make_node_table(ec)
    first_row = node_table.iloc[0]
    assert np.isnan(first_row.parent)
    assert first_row.c_location_id == 1
    assert first_row.node_id == 0


def test_make_data_table(base_context, mock_get_location_hierarchy_from_gbd):
    node_table, _ = make_node_table(base_context)
    renames = dict(x_sex="x_0")
    data_table = make_data_table(base_context, node_table, renames)

    assert len(data_table) == len(base_context.input_data.observations)
    assert all(data_table.node_id == 1)


def test_make_rate_and_nslist_tables(base_context):
    dm = DismodFile(None)
    density = pd.DataFrame({"density_name": [x.name for x in DensityEnum]})
    dm.density = density.assign(density_id=density.index)
    age_table = make_age_table(base_context)
    time_table = make_time_table(base_context)
    prior_table, prior_objects = make_prior_table(base_context, dm.density)
    smooth_table, smooth_grid_table, smooth_id_func = make_smooth_and_smooth_grid_tables(
        base_context, age_table, time_table, prior_objects
    )

    rate_table, rate_to_id, nslist_table, nspairs_table = make_rate_and_nslist_tables(
        base_context, smooth_id_func, lambda location_id: location_id
    )

    assert len(nslist_table) == 0
    assert len(nspairs_table) == 0

    assert rate_table.rate_name.tolist() == ["pini", "iota", "rho", "chi", "omega"]

    for rate in base_context.rates:
        df = rate_table.loc[rate_table["rate_name"] == rate.name]
        if not df.empty:
            rate_id = float(df.rate_id)
            assert rate_to_id(rate) == rate_id


def test_make_covariate_table(base_context):
    at_grid = AgeTimeGrid.uniform(age_lower=0, age_upper=120, age_step=5, time_lower=1990, time_upper=2018, time_step=1)
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
    density = pd.DataFrame({"density_name": [x.name for x in DensityEnum]})
    dm.density = density.assign(density_id=density.index)

    density = pd.DataFrame({"density_name": [x.name for x in DensityEnum]})
    dm.density = density.assign(density_id=density.index)

    age_table = make_age_table(base_context)
    time_table = make_time_table(base_context)
    prior_table, prior_objects = make_prior_table(base_context, dm.density)
    smooth_table, smooth_grid_table, smooth_id_func = make_smooth_and_smooth_grid_tables(
        base_context, age_table, time_table, prior_objects
    )
    covariate_columns, cov_id_func, covariate_renames = make_covariate_table(base_context)
    rate_table, rate_to_id, _, _ = make_rate_and_nslist_tables(
        base_context, smooth_id_func, lambda location_id: location_id
    )
    make_mulcov_table(base_context, smooth_id_func, rate_to_id, integrand_to_id, cov_id_func)


def test_infer_rate_case(base_context):
    case = _infer_rate_case(base_context)
    assert case == "iota_pos_rho_zero"
