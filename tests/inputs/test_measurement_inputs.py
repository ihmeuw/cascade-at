import pytest
import numpy as np
from copy import deepcopy
from random import choice, sample, randint

from cascade_at.settings.base_case import BASE_CASE
from cascade_at.settings.settings import load_settings
from cascade_at.inputs.measurement_inputs import (
    MeasurementInputs, MeasurementInputsFromSettings)
from cascade_at.inputs.locations import LocationDAG


@pytest.mark.parametrize("column,values", [
    ("age_lower", {0: 0.0, 1: 0.0, 2: 0.0}),
    ("age_upper", {0: 0.01917808, 1: 0.01917808, 2: 0.01917808}),
    ("hold_out", {0: 0.0, 1: 1.0, 2: 0.0}),
    ("location_id", {0: 70.0, 1: 70.0, 2: 70.0}),
    ("meas_std", {0: 3e-06, 1: 0.010204269138493082, 2: 1.020426913849308e-06}),
    ("meas_value", {0: 4e-05, 1: 0.17, 2: 5e-06}),
    ("measure", {0: 'Tincidence', 1: 'mtall', 2: 'mtspecific'}),
    ("name", {0: '342686', 1: np.nan, 2: np.nan}),
    ("sex_id", {0: 2.0, 1: 2.0, 2: 2.0}),
    ("time_lower", {0: 1990.0, 1: 1990.5, 2: 1990.5}),
    ("time_upper", {0: 1991.0, 1: 1990.5, 2: 1990.5}),
    ("density", {0: 'log_gaussian', 1: 'log_gaussian', 2: 'log_gaussian'}),
    ("eta", {0: 1e-05, 1: 1e-05, 2: 1e-05}),
    ("c_diabetes_fpg", {0: 0.96, 1: 0.96, 2: 0.96}),
    ("s_sex", {0: -0.5, 1: -0.5, 2: -0.5}),
    ("s_one", {0: 1.0, 1: 1.0, 2: 1.0})
])
def test_dismod_data(dismod_data, column, values):
    assert dismod_data.to_dict()[column] == values


def test_pickle(mi, context):
    settings = BASE_CASE
    context.write_inputs(inputs=mi, settings=settings)
    p_inputs, p_alchemy, p_settings = context.read_inputs()
    assert len(p_inputs.dismod_data) == len(mi.dismod_data)


def test_data_cv_from_settings():
    settings = BASE_CASE.copy()
    s = load_settings(settings)
    cv = MeasurementInputs.data_cv_from_settings(settings=s)
    for k, v in cv.items():
        assert v == 0.1


def test_data_cv_from_settings_by_integrand():
    settings = BASE_CASE.copy()
    settings.update({
        "data_cv_by_integrand": [{
            "integrand_measure_id": 5,
            "value": 0.5
        }]
    })
    s = load_settings(settings)
    cv = MeasurementInputs.data_cv_from_settings(settings=s)
    for k, v in cv.items():
        if k == 'prevalence':
            assert v == 0.5
        else:
            assert v == 0.1


# Commenting here to promote discussion.  These tests are a little silly,
# since I've basically recreated the logic implemented in the
# measurement_inputs module, meaning that if a bug is introduced into the
# LocationDAG class we won't necessarily catch it. I could have used
# hard-coded test locations and number of descendants to test the underlying
# logic but that would open up the test to failure if the
# location_set_version_id cited by the BASE_CASE is changed. We could import
# the hierarchies.dbtrees module or hit the database as an independent check.
# This test at least ensures that drill location start and drill location end
# are being correctly passed to the MeasurementInputs class.

def test_location_drill_start_only(ihme):
    these_settings = deepcopy(BASE_CASE)

    model_settings = these_settings["model"]

    tree = LocationDAG(these_settings['location_set_version_id'],
                       these_settings['gbd_round_id'])
    region_ids = tree.parent_children(1)
    test_loc = choice(region_ids)
    num_descendants = len(tree.descendants(test_loc))
    num_mr_locs = len(tree.parent_children(test_loc))

    model_settings.pop("drill_location_end")
    model_settings['drill_location_start'] = test_loc
    these_settings["model"] = model_settings
    s = load_settings(these_settings)
    mi = MeasurementInputsFromSettings(settings=s)

    # with drill_location_end unset, demographics.location_id should
    # be set to all descendants of the test loc, plus the test loc itself
    assert len(mi.demographics.location_id) == num_descendants + 1
    assert len(mi.demographics.mortality_rate_location_id) == num_mr_locs
    these_settings


def test_location_drill_start_end(ihme):
    these_settings = deepcopy(BASE_CASE)

    model_settings = these_settings["model"]

    tree = LocationDAG(these_settings['location_set_version_id'],
                       these_settings['gbd_round_id'])
    region_ids = tree.parent_children(1)
    parent_test_loc = choice(region_ids)
    test_children = list(tree.parent_children(parent_test_loc))
    num_test_children = randint(2, len(test_children))

    children_test_locs = sample(test_children, num_test_children)
    num_descendants = 0
    for child in children_test_locs:
        num_descendants += len(tree.descendants(child))

    model_settings['drill_location_end'] = children_test_locs
    model_settings['drill_location_start'] = parent_test_loc
    these_settings['model'] = model_settings
    s = load_settings(these_settings)
    mi = MeasurementInputsFromSettings(settings=s)

    # demographics.location_id shoul be set to all descendants of each
    # location in drill_location_end, plus drill_location_end locations
    # themselves, plus the drill_location_start location
    assert len(mi.demographics.location_id) == (
        num_descendants + len(children_test_locs) + 1)
    assert len(mi.demographics.mortality_rate_location_id) == (
        len(children_test_locs) + 1)


def test_no_drill(ihme):
    these_settings = deepcopy(BASE_CASE)

    model_settings = these_settings["model"]

    tree = LocationDAG(these_settings['location_set_version_id'],
                       these_settings['gbd_round_id'])
    num_descendants = len(tree.descendants(1))

    model_settings.pop('drill_location_end')
    model_settings.pop('drill_location_start')

    these_settings['model'] = model_settings
    s = load_settings(these_settings)
    mi = MeasurementInputsFromSettings(settings=s)

    # since we haven't set either drill_location_start or
    # drill_location_end, demographics.location_id should be set
    # to the entire hierarchy
    assert len(mi.demographics.location_id) == num_descendants + 1
    assert len(mi.demographics.mortality_rate_location_id) == num_descendants + 1
