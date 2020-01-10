from cascade_at.cascade.cascade_operations import ConfigureInputs, FitBoth, FormatAndUpload, CleanUp
from cascade_at.cascade.cascade_operations import CASCADE_OPERATIONS


def test_cascade_dict():
    assert type(CASCADE_OPERATIONS) == dict
    assert CASCADE_OPERATIONS['configure_inputs'] == ConfigureInputs
    assert CASCADE_OPERATIONS['fit_both'] == FitBoth
    assert CASCADE_OPERATIONS['format_upload'] == FormatAndUpload


def test_configure_inputs():
    obj = ConfigureInputs(
        model_version_id=0
    )
    assert obj.command == (
        f'configure_inputs -model-version-id 0 '
        f'--make --configure'
    )


def test_configure_inputs_with_drill():
    obj = ConfigureInputs(
        model_version_id=0
    )
    assert obj.command == (
        f'configure_inputs -model-version-id 0 '
        f'--make --configure'
    )


def test_fit_both():
    obj = FitBoth(
        model_version_id=0,
        parent_location_id=1,
        sex_id=1
    )
    assert obj.command == (
        f'dismod_db '
        f'-model-version-id 0 '
        f'-parent-location-id 1 '
        f'-sex-id 1 '
        f'--commands init fit-fixed set-start_var-fit_var set-scale_var-fit_var fit-both predict-fit_var '
    )


def test_format_upload():
    obj = FormatAndUpload(
        model_version_id=0,
        parent_location_id=1,
        sex_id=1
    )
    assert obj.command == (
        f'format_upload '
        f'-model-version-id 0 '
        f'-parent-location-id 1 '
        f'-sex-id 1'
    )


def test_cleanup():
    obj = CleanUp(
        model_version_id=0
    )
    assert obj.command == (
        f'cleanup '
        f'-model-version-id 0'
    )
