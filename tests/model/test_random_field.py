import networkx as nx

from cascade.model.priors import Gaussian, Uniform
from cascade.model.grids import AgeTimeGrid, PriorGrid
from cascade.model.random_field import Model, RandomField


def test_make_model():
    nonzero_rates = ["iota", "chi", "omega"]
    locations = nx.DiGraph()
    locations.add_edges_from([(1, 2), (1, 3), (1, 4)])
    parent_location = 1

    m = Model(nonzero_rates, locations, parent_location)

    covariate_age_time = AgeTimeGrid([40], [2000])
    value = PriorGrid(covariate_age_time)
    value[:, :].prior = Gaussian(0, 0.1)

    m.alpha[("iota", "traffic")] = RandomField(covariate_age_time, dict(value=value))

    dense_age_time = AgeTimeGrid.uniform(
        age_lower=0, age_upper=120, age_step=5, time_lower=1990, time_upper=2015, time_step=5)
    rate_value_priors = PriorGrid(dense_age_time)
    rate_value_priors[:, :].prior = Uniform(0, 0.1, 0, 1)
    rate_dage_priors = PriorGrid(dense_age_time)
    rate_dage_priors[:, :].prior = Gaussian(0, 0.1)
    rate_dtime_priors = PriorGrid(dense_age_time)
    rate_dtime_priors[:, :].prior = Gaussian(0, 0.1)
    rate_priors = dict(value=rate_value_priors, dage=rate_dage_priors, dtime=rate_dtime_priors)

    m.rate["omega"] = RandomField(dense_age_time, rate_priors)
    m.rate["iota"] = RandomField(dense_age_time, rate_priors)
    m.rate["chi"] = RandomField(dense_age_time, rate_priors)
