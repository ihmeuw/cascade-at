import pytest

import pandas as pd

from cascade.core.input_data import InputData

from cascade.executor.epiviz_runner import compute_age_steps, limit_omega_to_observed_times


@pytest.mark.parametrize("min_step", [(1,), (5,), (0.5,), (0.1,), (10,)])
def test_age_steps(min_step):
    steps = compute_age_steps(min_step[0])
    assert steps[0] <= 7 / 365
    assert 3.5 / 365 <= steps[0]
    assert steps[-1] < 0.75 * min_step[0]
    assert steps[-1] > 0.4 * min_step[0]


@pytest.mark.parametrize("min_step", [(6 / 365,), (1 / 365,), (0.000001,)])
def test_small_small_step(min_step):
    steps = compute_age_steps(min_step[0])
    assert len(steps) == 0


def test_omega_constraint():
    single_time = (
        pd.DataFrame(dict(time_lower=[2000, 2000, 2000], time_upper=[2000, 2000, 2000], values=[1, 2, 3])),
        {
            (1994, 1995),
            (1995, 1996),
            (1996, 1997),
            (1997, 1998),
            (1998, 1999),
            (1999, 2000),
            (2000, 2001),
            (2001, 2002),
            (2002, 2003),
            (2003, 2004),
            (2004, 2005),
            (2005, 2006),
        },
    )
    point_times = (
        pd.DataFrame(dict(time_lower=[1990, 2000, 2010], time_upper=[1990, 2000, 2010], values=[1, 2, 3])),
        {
            (1992, 1993),
            (1993, 1994),
            (1994, 1995),
            (1995, 1996),
            (1996, 1997),
            (1997, 1998),
            (1998, 1999),
            (1999, 2000),
            (2000, 2001),
            (2001, 2002),
            (2002, 2003),
            (2003, 2004),
            (2004, 2005),
            (2005, 2006),
            (2006, 2007),
            (2007, 2008),
            (2008, 2009),
            (2009, 2010),
            (2010, 2011),
            (2011, 2012),
            (2012, 2013),
        },
    )
    interval_times = (
        pd.DataFrame(dict(time_lower=[1990, 2000, 2010], time_upper=[1991, 2001, 2011], values=[1, 2, 3])),
        {
            (1992, 1993),
            (1993, 1994),
            (1994, 1995),
            (1995, 1996),
            (1996, 1997),
            (1997, 1998),
            (1998, 1999),
            (1999, 2000),
            (2000, 2001),
            (2001, 2002),
            (2002, 2003),
            (2003, 2004),
            (2004, 2005),
            (2005, 2006),
            (2006, 2007),
            (2007, 2008),
            (2008, 2009),
            (2009, 2010),
            (2010, 2011),
            (2011, 2012),
            (2012, 2013),
        },
    )

    intervals = [
        (1992, 1993),
        (1993, 1994),
        (1994, 1995),
        (1995, 1996),
        (1996, 1997),
        (1997, 1998),
        (1998, 1999),
        (1999, 2000),
        (2000, 2001),
        (2001, 2002),
        (2002, 2003),
        (2003, 2004),
        (2004, 2005),
        (2005, 2006),
        (2006, 2007),
        (2007, 2008),
        (2008, 2009),
        (2009, 2010),
        (2010, 2011),
        (2011, 2012),
        (2012, 2013),
    ]
    intervals = pd.DataFrame(
        [dict(time_lower=time_lower, time_upper=time_upper) for time_lower, time_upper in intervals]
    )

    for test_data, expected_times in [single_time, point_times, interval_times]:
        input_data = InputData()
        input_data.observations = test_data
        assert expected_times == {
            (r["time_lower"], r["time_upper"])
            for _, r in limit_omega_to_observed_times(input_data, intervals).iterrows()
        }
