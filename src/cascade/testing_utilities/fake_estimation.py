from copy import deepcopy

from cascade.core import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


def fake_compute_location(execution_context, local_settings, input_data, model):
    """This doesn't run fit but returns a fit result and draws that
    have the correct shape and values that won't violate assumptions."""
    fit_var = model.var_from_mean()
    draws = [deepcopy(fit_var) for _ in range(local_settings.number_of_fixed_effect_samples)]
    return fit_var, draws
