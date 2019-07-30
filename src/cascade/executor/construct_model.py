from collections import defaultdict

import numpy as np
from numpy import inf

from cascade.core.log import getLoggers
from cascade.model import (
    Model, Var, SmoothGrid, Covariate, Constant
)

CODELOG, MATHLOG = getLoggers(__name__)


def rectangular_data_to_var(gridded_data):
    """Using this very regular data, where every age and time is present,
    construct an initial guess as a Var object. Very regular means that there
    is a complete set of ages-cross-times."""
    try:
        initial_ages = np.sort(
            np.unique(0.5 * (gridded_data.age_lower + gridded_data.age_upper))
        )
        initial_times = np.sort(
            np.unique(0.5 * (gridded_data.time_lower + gridded_data.time_upper))
        )
    except AttributeError:
        CODELOG.error(f"Data to make a var has columns {gridded_data.columns}")
        raise RuntimeError(
            f"Wrong columns in rectangular_data_to_var {gridded_data.columns}")

    guess = Var(ages=initial_ages, times=initial_times)
    for age, time in guess.age_time():
        found = gridded_data.query(
            "(age_lower <= @age) & (@age <= age_upper) & "
            "(time_lower <= @time) & (@time <= time_upper)"
        )
        assert len(found) == 1, f"found {found}"
        guess[age, time] = float(found.iloc[0]["mean"])
    return guess


def const_value(value):

    def at_function(age, time):
        return value

    return at_function


def construct_model(data, local_settings, covariate_multipliers, covariate_data_spec):
    """Makes a Cascade model from EpiViz-AT settings and data.

    Args:
        data: An object with both ``age_specific_death_rate`` and ``locations``.
        local_settings: A settings object from ``cascade_plan``.
        covariate_multipliers (List[EpiVizCovariateMultiplier]): descriptions of
            covariate multipliers.
        covariate_data_spec (List[EpiVizCovariate]): the covariates themselves.
            Some covariates aren't used by covariate multipliers but are
            included to calculate hold outs.

    Returns:
        cascade.model.Model: The model to fit.
    """
    ev_settings = local_settings.settings
    parent_location_id = local_settings.parent_location_id
    default_age_time = dict()
    default_age_time["age"] = np.linspace(0, 100, 21)
    default_age_time["time"] = np.linspace(1990, 2015, 6)
    for kind in ["age", "time"]:
        default_grid = getattr(ev_settings.model, f"default_{kind}_grid")
        if default_grid is not None:
            default_age_time[kind] = np.sort(np.array(default_grid, dtype=np.float))

    # Use this age and time when a smooth grid doesn't depend on age and time.
    single_age = default_age_time["age"][:1]
    single_time = [default_age_time["time"][len(default_age_time["time"]) // 2]]
    single_age_time = (single_age, single_time)

    nonzero_rates = [smooth.rate for smooth in ev_settings.rate]

    children = list(data.locations.successors(parent_location_id))
    model = Model(
        nonzero_rates=nonzero_rates,
        parent_location=parent_location_id,
        child_location=children,
        covariates=covariates_list(covariate_data_spec),
        weights=None,
    )

    construct_model_rates(default_age_time, single_age_time, ev_settings, model)
    # No random effects if there is only one child.
    if children and len(children) > 1:
        construct_model_random_effects(default_age_time, single_age_time, ev_settings, model)
    construct_model_covariates(default_age_time, single_age_time, covariate_multipliers, model)
    asdr = data.age_specific_death_rate
    if ev_settings.model.constrain_omega:
        constrain_omega(
            default_age_time, asdr, ev_settings, model, parent_location_id, children
        )

    return model


def construct_model_rates(default_age_time, single_age_time, ev_settings, model):
    for smooth in ev_settings.rate:
        rate_grid = smooth_grid_from_smoothing_form(default_age_time, single_age_time, smooth)
        model.rate[smooth.rate] = rate_grid


def constrain_omega(default_age_time, asdr, ev_settings, model, parent_location_id, children):
    r"""Set parent rate to fixed value and define children as fixed random effects.
    Constrains parent omega to age-specific death rate and constrains child
    rates, if they exist, to

    .. math::

        u_j = \log(r_c / r_p)

    where :math:`r_p` is the parent rate and :math:`r_c` is the child rate.

    Args:
        default_age_time (Dict[str,ndarray]): Age and time grids chosen by user.
        asdr: Age-specific death rate
        ev_settings: The Form.py settings object.
        model (Model): Writes to rate and random_effect parts of the model.
        parent_location_id (int): parent location
        children (List[int]): Child location ids.
    """
    CODELOG.debug(
        f"Constrain omega, asdr columns {asdr.columns} "
        f"parent loc {parent_location_id} child locs {children}.")
    sexes_present = set(asdr.sex_id.unique())
    if len(sexes_present) > 1:
        if sexes_present == {1, 2}:
            # XXX use population weighting
            axes = ["time_lower", "time_upper", "age_lower", "age_upper", "location"]
            gridded_data = asdr[axes + ["mean"]]
            single_sex_asdr = gridded_data.groupby(axes).mean().reset_index()
        elif sexes_present == {1, 2, 3}:
            single_sex_asdr = asdr[asdr.sex_id == 3]
        elif 3 in sexes_present:
            keep = (sexes_present - {3}).pop()
            single_sex_asdr = asdr[asdr.sex_id == keep]
        else:
            raise AssertionError(f"ASDR had sexes {sexes_present}.")
    else:
        single_sex_asdr = asdr

    parent_asdr = single_sex_asdr[single_sex_asdr.location == parent_location_id]
    if len(parent_asdr) == 0:
        raise RuntimeError(
            f"Age-specific death rate has no values for this location "
            f"({parent_location_id}). It has locations "
            f"{', '.join(str(x) for x in single_sex_asdr.location.unique())}."
        )

    omega = rectangular_data_to_var(parent_asdr)
    model.rate["omega"] = constraint_from_rectangular_data(omega, default_age_time)
    asdr_locations = set(single_sex_asdr.location.unique().tolist())
    children_without_asdr = set(children) - set(asdr_locations)
    if children_without_asdr:
        MATHLOG.warning(f"Children of {parent_location_id} missing ASDR {children_without_asdr} "
                        f"out of {len(children)} children so not including child omega constraints")
        return

    for child in children:
        child_asdr = single_sex_asdr[single_sex_asdr.location == child]
        assert len(child_asdr) > 0
        child_rate = rectangular_data_to_var(child_asdr)

        def child_effect(age, time):
            return np.log(child_rate(age, time) / omega(age, time))

        model.random_effect[("omega", child)] = constraint_from_rectangular_data(child_effect, default_age_time)


def constraint_from_rectangular_data(rate_var, default_age_time):
    """Takes data on a complete set of ages and times, makes a constraint grid.

    Args:
        rate_var: A function of age and time to represent a rate.
    """
    omega_grid = SmoothGrid(ages=default_age_time["age"], times=default_age_time["time"])
    for age, time in omega_grid.age_time():
        omega_grid.value[age, time] = Constant(rate_var(age, time))
    return omega_grid


def smooth_grid_from_smoothing_form(default_age_time, single_age_time, smooth):
    """
    Create a new SmoothGrid from the settings in EpiViz-AT at the Smoothing
    level.

    Args:
        default_age_time (List[ages, times]): Two members, the ages and the time.
        single_age_time (List[float]): Two members, an age and a time.
        smooth (cascade.input_data.configuration.form.Smoothing): The form element.
    Returns:
        SmoothGrid: A new smooth grid.
    """
    ages, times = construct_grid_ages_times(default_age_time, single_age_time, smooth)
    rate_grid = SmoothGrid(ages=ages, times=times)
    for kind in ["value", "dage", "dtime"]:
        if not smooth.is_field_unset("default") and not smooth.default.is_field_unset(kind):
            getattr(rate_grid, kind)[:, :] = getattr(smooth.default, kind).prior_object
        if not smooth.is_field_unset("mulstd") and not smooth.mulstd.is_field_unset(kind):
            getattr(rate_grid, kind).mulstd_prior = getattr(smooth.mulstd, kind).prior_object
    if not smooth.is_field_unset("detail"):
        for smoothing_prior in smooth.detail:
            for a, t in matching_knots(rate_grid, smoothing_prior):
                getattr(rate_grid, smoothing_prior.prior_type)[a, t] = smoothing_prior.prior_object
    return rate_grid


def matching_knots(rate_grid, smoothing_prior):
    """
    Get lower and upper out of the smoothing prior. This uses the age, time,
    and "born" lower and upper bounds to return
    ages and times in the grid that are within those bounds.
    The goal is to apply a prior selectively to those knots.

    Args:
        smoothing_prior (cascade.input_data.configuration.form.SmoothingPrior):
            A single smoothing prior.

    Returns:
        Iterator over (a, t) that match. Can be nothing.
    """
    extents = dict()
    for extent in ["age", "time", "born"]:
        extents[extent] = np.zeros(2, dtype=np.float)
        for side_idx, side, default_extent in [(0, "lower", -inf), (1, "upper", inf)]:
            name = f"{extent}_{side}"
            if smoothing_prior.is_field_unset(name):
                extents[extent][side_idx] = default_extent
            else:
                extents[extent][side_idx] = getattr(smoothing_prior, name)
    # meshgrid generates every combination of age and time as two numpy arrays.
    ages, times = np.meshgrid(rate_grid.ages, rate_grid.times)
    assert ages.shape == (len(rate_grid.times), len(rate_grid.ages))
    assert times.shape == (len(rate_grid.times), len(rate_grid.ages))
    in_age = (ages >= extents["age"][0]) & (ages <= extents["age"][1])
    in_time = (times >= extents["time"][0]) & (times <= extents["time"][1])
    in_born = (ages <= times - extents["born"][0]) & (ages >= times - extents["born"][1])
    cover = in_age & in_time & in_born
    if not np.any(cover):
        MATHLOG.info(f"No ages and times match prior with extents {extents}.")
    yield from zip(ages[cover], times[cover])


def construct_grid_ages_times(default_age_time, single_age_time, smooth):
    if not smooth.is_field_unset("age_time_specific") and smooth.age_time_specific == 0:
        return single_age_time

    ages = smooth.age_grid
    if ages is None:
        # hasattr because this may be a Smoothing form or a Covariate form.
        if hasattr(smooth, "rate") and smooth.rate == "pini":
            ages = np.array([0], dtype=np.float)
        else:
            ages = default_age_time["age"]
    else:
        ages = np.sort(np.array(ages, dtype=np.float))
    times = smooth.time_grid
    if times is None:
        times = default_age_time["time"]
    else:
        times = np.sort(np.array(times, dtype=np.float))
    return ages, times


def construct_model_random_effects(default_age_time, single_age_time, ev_settings, model):
    """The settings may have random effects for many locations which aren't children
    of the current parent location. Only those random effects that apply to the children
    explicitly or to all locations (specified as location=None) are included."""
    if not ev_settings.random_effect:
        return

    random_effect_by_rate = defaultdict(list)
    for smooth in ev_settings.random_effect:
        re_grid = smooth_grid_from_smoothing_form(default_age_time, single_age_time, smooth)

        if not smooth.is_field_unset("location") and smooth.location in model.child_location:
            location = smooth.location
        else:
            # One smooth for all children when there isn't a child location.
            location = None
        model.random_effect[(smooth.rate, location)] = re_grid
        random_effect_by_rate[smooth.rate].append(location)

    for rate_to_check, locations in random_effect_by_rate.items():
        if locations != [None] \
                and len(locations) != len(model.child_location) \
                and set(locations) != set(model.child_location):
            message = (f"Random effect for {rate_to_check} doesn't have "
                       f"entries for all child locations, only {locations} "
                       f"instead of {model.child_location}.")
            MATHLOG.error(message)
            raise RuntimeError(message)


def construct_model_covariates(default_age_time, single_age_time, covariate_multipliers, model):
    """The covariate multipliers are of all types: alpha, beta, and gamma. This adds
    their priors to the Model.

    Args:
        default_age_time (Tuple[ndarray,ndarray]): ages and times
        single_age_time (float, float): The single age and time to use if it's
            a point value.
        covariate_multipliers (List[EpiVizCovariateMultiplier): A list of specifications
            for covariate multipliers. This assumes data has already been read,
            because that data determines names for the multipliers.
    """
    for mulcov in covariate_multipliers:
        grid = smooth_grid_from_smoothing_form(default_age_time, single_age_time, mulcov.grid_spec)
        model[mulcov.group][mulcov.key] = grid


def covariates_list(covariate_data_spec):
    covariate_list = list()
    for c in covariate_data_spec:
        CODELOG.debug(f"Adding covariate reference {c.name}.reference={c.reference}")
        covariate_list.append(Covariate(c.name, c.reference, c.max_difference))
    return covariate_list
