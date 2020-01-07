import pandas as pd
import numpy as np

from cascade_at.inputs.base_input import BaseInput


def test_convert_to_age_lower_upper(ihme):
    df = pd.DataFrame({
        'age_group_id': [10, 12, 14]
    })
    c_df = BaseInput().convert_to_age_lower_upper(df=df)
    assert len(c_df) == 3
    assert np.isfinite(c_df.age_lower.values).all()
    assert np.isfinite(c_df.age_upper.values).all()


def test_demographic_notation():
    df = pd.DataFrame({
        'year_lower': [2000, 2000, 2001, 2001],
        'year_upper': [2000, 2001, 2001, 2002],
    })
    c_df = BaseInput.get_out_of_demographic_notation(df=df, columns=['year'])
    assert (c_df == pd.DataFrame({
        'year_lower': [2000, 2000, 2001, 2001],
        'year_upper': [2001, 2001, 2002, 2002]
    })).all().all()
