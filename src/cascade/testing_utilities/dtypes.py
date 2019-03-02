from pathlib import Path

import numpy as np
import pandas as pd


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
