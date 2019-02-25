
def make_options(ev_settings):
    """This sets Dismod-AT options from the EpiViz-AT settings and model.
    https://bradbell.github.io/dismod_at/doc/option_table.htm

    This should get or report missing every last one.

    Args:
        ev_settings: The Configuration of the EpiViz settings.
    """
    options = dict()
    # parent location id will be set for us
    # parent node name is set for us
    options["meas_std_effect"] = ev_settings.policies.meas_std_effect
    if hasattr(ev_settings.model, "zero_sum_random") and ev_settings.model.zero_sum_random:
        options["zero_sum_random"] = " ".join(ev_settings.model.zero_sum_random)
    # data extra columns is set for us.
    # avgint extra columns is set for us.
    options["warn_on_stderr"] = True
    if hasattr(ev_settings, "minimum_meas_cv") and ev_settings.minimum_meas_cv:
        options["minimum_meas_cv"] = ev_settings.minimum_meas_cv

    options["ode_step_size"] = ev_settings.model.ode_step_size
    if ev_settings.model.additional_ode_steps:
        options["age_avg_split"] = " ".join(str(ea) for ea in ev_settings.model.additional_ode_steps)
    if hasattr(ev_settings.model, "random_seed") and ev_settings.model.random_seed:
        options["random_seed"] = ev_settings.model.random_seed
    # rate case is set for us.

    for kind in ["fixed", "random"]:
        for opt in ["derivative_test", "max_num_iter", "print_level", "accept_after_max_steps", "tolerance"]:
            if hasattr(ev_settings, opt):
                dtest = getattr(ev_settings, opt)
                # Check for None b/c would be a mistake to set tolerance_random
                # to None.
                if hasattr(dtest, kind) and getattr(dtest, kind) is not None:
                    options[f"{opt}_{kind}"] = getattr(dtest, kind)

    if hasattr(ev_settings.model, "quasi_fixed"):
        options["quasi_fixed"] = ev_settings.model.quasi_fixed == 1
    # bound_frac_fixed is not in the Form.
    if hasattr(ev_settings.model, "bound_frac_fixed"):
        options["bound_frac_fixed"] = ev_settings.model.bound_frac_fixed
    # limited_memory_max_history_fixed is not in the Form.
    if hasattr(ev_settings.policies, "limited_memory_max_history_fixed"):
        options["limited_memory_max_history_fixed"] = ev_settings.policies.limited_memory_max_history_fixed
    # bound_frac_fixed is not in the Form.
    if hasattr(ev_settings.model, "bound_random"):
        options["bound_random"] = ev_settings.model.bound_random

    return options
