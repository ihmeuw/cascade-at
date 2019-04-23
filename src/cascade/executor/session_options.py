from cascade.dismod.constants import IntegrandEnum


def make_options(ev_settings, model_options):
    """This sets Dismod-AT options from the EpiViz-AT settings and model.
    https://bradbell.github.io/dismod_at/doc/option_table.htm

    This should get or report missing every last one.

    Args:
        ev_settings: The Configuration of the EpiViz settings.
    """
    options = dict()
    # parent location id will be set for us
    # parent node name is set for us
    options["meas_noise_effect"] = ev_settings.policies.meas_std_effect
    if not ev_settings.model.is_field_unset("zero_sum_random"):
        options["zero_sum_random"] = " ".join(ev_settings.model.zero_sum_random)
    # data extra columns is set for us.
    # avgint extra columns is set for us.
    options["warn_on_stderr"] = True
    options["ode_step_size"] = ev_settings.model.ode_step_size
    if not ev_settings.model.is_field_unset("additional_ode_steps"):
        options["age_avg_split"] = " ".join(str(ea) for ea in ev_settings.model.additional_ode_steps)
    if not ev_settings.model.is_field_unset("random_seed"):
        options["random_seed"] = ev_settings.model.random_seed
    # rate case is set for us.

    for kind in ["fixed", "random"]:
        for opt in ["derivative_test", "max_num_iter", "print_level", "accept_after_max_steps", "tolerance"]:
            if hasattr(ev_settings, opt):
                dtest = getattr(ev_settings, opt)
                # Check for None b/c would be a mistake to set tolerance_random
                # to None.
                if not dtest.is_field_unset(kind):
                    options[f"{opt}_{kind}"] = getattr(dtest, kind)

    if not ev_settings.model.is_field_unset("quasi_fixed"):
        options["quasi_fixed"] = ev_settings.model.quasi_fixed == 1
        options["bound_frac_fixed"] = ev_settings.model.bound_frac_fixed
    # limited_memory_max_history_fixed is not in the Form.
    if not ev_settings.policies.is_field_unset("limited_memory_max_history_fixed"):
        options["limited_memory_max_history_fixed"] = ev_settings.policies.limited_memory_max_history_fixed
    options["bound_random"] = model_options.bound_random

    return options


def make_minimum_meas_cv(ev_settings):
    if not ev_settings.model.is_field_unset("minimum_meas_cv"):
        return {integrand.name: ev_settings.model.minimum_meas_cv for integrand in IntegrandEnum}
    else:
        return {}
