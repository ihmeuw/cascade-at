import pytest

import numpy as np
import pandas as pd

from scipy.stats import norm

from cascade.saver.save_prediction import uncertainty_from_prediction_draws, _predicted_to_uploadable_format


def test_uncertainty_from_prediction_draws():
    rng = np.random.RandomState(4398221)

    num_draws = 100
    locations = [1, 2]
    age_lower, age_upper = zip(*[(0, 0.5), (50, 60)])
    sexes = [-0.5]
    time_lower, time_upper = zip(*[(y, y) for y in [2000, 2020]])
    integrands = ["prevalence", "Sincidence"]

    df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [locations, age_lower, age_upper, time_lower, time_upper, integrands, sexes],
            names=["location", "age_lower", "age_upper", "time_lower", "time_upper", "integrand", "s_sex"],
        )
    ).reset_index()

    true_mean = np.linspace(0.0001, 2.0, len(df))
    rng.shuffle(true_mean)
    true_std = np.linspace(0.00001, 0.001, len(df))
    rng.shuffle(true_std)
    df["true_mean"] = true_mean
    df["true_std"] = true_std

    draws = []
    for sample_index in range(num_draws):
        new_df = df.copy()
        new_df["sample_index"] = sample_index
        new_df["mean"] = new_df[["true_mean", "true_std"]].apply(
            lambda r: rng.normal(r["true_mean"], r["true_std"]), axis="columns"
        )
        draws.append(new_df)

    computed_fit = pd.DataFrame()
    with_uncertainty = uncertainty_from_prediction_draws(
        computed_fit,
        [
            draw[
                [
                    "location",
                    "age_lower",
                    "age_upper",
                    "time_lower",
                    "time_upper",
                    "integrand",
                    "s_sex",
                    "sample_index",
                    "mean",
                ]
            ]
            for draw in draws
        ]
    )

    with_uncertainty.set_index(
        ["location", "integrand", "age_lower", "age_upper", "time_lower", "time_upper", "s_sex"]
    ).sort_index()
    source = pd.concat(draws).drop_duplicates(
        ["location", "integrand", "age_lower", "age_upper", "time_lower", "time_upper", "s_sex"]
    )
    source = source.set_index(
        ["location", "integrand", "age_lower", "age_upper", "time_lower", "time_upper", "s_sex"]
    ).sort_index()
    source["lower"] = source[["true_mean", "true_std"]].apply(
        lambda r: norm(r["true_mean"], r["true_std"]).ppf(0.025), axis="columns"
    )
    source["upper"] = source[["true_mean", "true_std"]].apply(
        lambda r: norm(r["true_mean"], r["true_std"]).ppf(0.975), axis="columns"
    )

    assert np.allclose(with_uncertainty["lower"], source["lower"], rtol=0.04)
    assert np.allclose(with_uncertainty["upper"], source["upper"], rtol=0.1)
    assert np.allclose(with_uncertainty["mean"], source["true_mean"], atol=0.003)


@pytest.mark.parametrize(
    "s_sex,sex_id",
    [
        ([-0.5, -0.5, -0.5], [2, 2, 2]),
        ([0, 0, 0], [3, 3, 3]),
        ([0.5, 0.5, 0.5], [1, 1, 1]),
        ([-0.5, 0, 0.5], [2, 3, 1]),
    ],
)
def test_predicted_to_uploadable_format(s_sex, sex_id, mocker):
    fake_age_to_group = mocker.patch("cascade.saver.save_prediction.age_ranges_to_groups")
    fake_age_to_group.side_effect = lambda _, df: df

    predicted = pd.DataFrame(
        {
            "mean": [1, 1, 1],
            "location": [1, 2, 3],
            "integrand": ["prevalence", "Sincidence", "remission"],
            "age_lower": [0, 1, 2],
            "age_upper": [1, 2, 3],
            "time_lower": [1990, 2000, 2002],
            "time_upper": [1990, 2000, 2002],
            "s_sex": s_sex,
        }
    )

    new_predicted = _predicted_to_uploadable_format(None, predicted)

    assert fake_age_to_group.called
    assert new_predicted.location_id.tolist() == predicted.location.tolist()
    assert new_predicted.sex_id.tolist() == sex_id
    assert new_predicted.year_id.tolist() == predicted.time_lower.tolist()
    assert new_predicted.measure_id.tolist() == [5, 41, 7]
