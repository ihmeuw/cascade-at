from cascade_at.model.grid_alchemy import Alchemy
from cascade_at.model.utilities.grid_helpers import integrand_grids, expand_grid
from cascade_at.dismod.api.fill_extract_helpers.utils import vec_to_midpoint


def get_prior_avgint_dict(settings, integrands, sexes, locations, midpoint=True):
    """
    Get a data frame to use with the AvgInt

    Args:
        settings: (cascade_at.settings.settings_configuration.SettingsConfiguration)
        integrands: (list of str)
        sexes: (list of int)
        locations: (list of int)
        midpoint: (bool)

    Returns: (pd.DataFrame)

    """
    prior_dfs = dict()
    alchemy = Alchemy(settings)
    grids = integrand_grids(alchemy=alchemy, integrands=integrands)
    for k, v in grids.items():
        if midpoint:
            time_lower = vec_to_midpoint(v['time'])
            time_upper = time_lower
            age_lower = vec_to_midpoint(v['age'])
            age_upper = age_lower
        else:
            time_lower = v['time'][1:]
            time_upper = v['time'][:-1]
            age_lower = v['age'][1:]
            age_upper = v['age'][:-1]

        time_dict = dict(zip(time_lower, time_upper))
        age_dict = dict(zip(age_lower, age_upper))

        prior_dfs[k] = expand_grid({
            'ages': age_lower,
            'times': time_lower,
            'location_id': locations,
            'sex_id': sexes
        })
        prior_dfs[k]['time_upper'] = prior_dfs[k]['time_lower'].map(time_dict)
        prior_dfs[k]['age_upper'] = prior_dfs[k]['age_lower'].map(age_dict)

    return prior_dfs
