from cascade_at.cascade.cascade_operations import (
    ConfigureInputs, FitFixed, FitBoth,
    FormatAndUpload, CleanUp, SampleSimulate
)


def test_configure_inputs():
    obj = ConfigureInputs(
        model_version_id=0
    )
    assert obj.command == (
        f'configure_inputs --model-version-id 0 '
        f'--make --configure'
    )


def test_fit_fixed():
    obj = FitFixed(
        model_version_id=0,
        parent_location_id=1,
        sex_id=1,
        fill=True
    )
    assert obj.command == (
        f'dismod_db '
        f'--model-version-id 0 '
        f'--parent-location-id 1 '
        f'--sex-id 1 '
        f'--fill '
        f'--dm-commands init fit-fixed predict-fit_var'
    )


def test_fit_both():
    obj = FitBoth(
        model_version_id=0,
        parent_location_id=1,
        sex_id=1,
        fill=True
    )
    assert obj.command == (
        f'dismod_db '
        f'--model-version-id 0 '
        f'--parent-location-id 1 '
        f'--sex-id 1 '
        f'--fill '
        f'--dm-commands init fit-fixed set-start_var-fit_var set-scale_var-fit_var fit-both predict-fit_var'
    )


def test_sample_simulate():
    obj = SampleSimulate(
        model_version_id=0,
        parent_location_id=1,
        sex_id=1,
        n_sim=5,
        n_pool=1,
        fit_type='both'
    )
    assert obj.command == (
        f'sample_simulate '
        f'--model-version-id 0 '
        f'--parent-location-id 1 '
        f'--sex-id 1 '
        f'--n-sim 5 '
        f'--n-pool 1 '
        f'--fit-type both'
    )


def test_format_upload():
    obj = FormatAndUpload(
        model_version_id=0,
        parent_location_id=1,
        sex_id=1
    )
    assert obj.command == (
        f'format_upload '
        f'--model-version-id 0 '
        f'--parent-location-id 1 '
        f'--sex-id 1'
    )


def test_cleanup():
    obj = CleanUp(
        model_version_id=0
    )
    assert obj.command == (
        f'cleanup '
        f'--model-version-id 0'
    )
