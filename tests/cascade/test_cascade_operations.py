from cascade_at.cascade.cascade_operations import (
    ConfigureInputs, Fit,
    Upload, CleanUp, Sample, Predict
)


def test_configure_inputs():
    obj = ConfigureInputs(
        model_version_id=0
    )
    assert obj.command == (
        f'configure_inputs --model-version-id 0 '
        f'--make --configure'
    )
    assert obj.template_kwargs['model_version_id'] == '--model-version-id 0'
    assert obj.template_kwargs['make'] == '--make'
    assert obj.template_kwargs['configure'] == '--configure'
    assert obj.template_kwargs['log_level'] == '--log-level info'
    assert obj.template_kwargs['json_file'] == ''
    assert obj.template_kwargs['test_dir'] == ''
    assert obj.template_kwargs['midpoint'] == ''


def test_fit_fixed():
    obj = Fit(
        model_version_id=0,
        parent_location_id=1,
        sex_id=1,
        fill=True,
        both=False,
        prior_samples=False
    )
    assert obj.command == (
        f'dismod_db '
        f'--model-version-id 0 '
        f'--parent-location-id 1 '
        f'--sex-id 1 '
        f'--fill '
        f'--dm-commands init fit-fixed predict-fit_var'
    )
    assert obj.template_kwargs['prior_samples'] == ''
    assert obj.template_kwargs['dm_commands'] == '--dm-commands init fit-fixed predict-fit_var'


def test_fit_both():
    obj = Fit(
        model_version_id=0,
        parent_location_id=1,
        sex_id=1,
        fill=True,
        prior_samples=False,
        both=True
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
    obj = Sample(
        model_version_id=0,
        parent_location_id=1,
        sex_id=1,
        n_sim=5,
        n_pool=1,
        fit_type='both',
        asymptotic=False
    )
    assert obj.command == (
        f'sample '
        f'--model-version-id 0 '
        f'--parent-location-id 1 '
        f'--sex-id 1 '
        f'--n-sim 5 '
        f'--n-pool 1 '
        f'--fit-type both'
    )


def test_sample_asymptotic():
    obj = Sample(
        model_version_id=0,
        parent_location_id=1,
        sex_id=1,
        n_sim=5,
        n_pool=1,
        fit_type='both',
        asymptotic=True
    )
    assert obj.command == (
        f'sample '
        f'--model-version-id 0 '
        f'--parent-location-id 1 '
        f'--sex-id 1 '
        f'--n-sim 5 '
        f'--n-pool 1 '
        f'--fit-type both '
        f'--asymptotic'
    )


def test_predict():
    obj = Predict(
        model_version_id=0,
        parent_location_id=1,
        sex_id=1,
        prior_grid=True,
        save_fit=False,
        sample=True,
        child_locations=[1, 2],
        child_sexes=[2]
    )
    assert obj.command == (
        'predict '
        f'--model-version-id 0 '
        f'--parent-location-id 1 '
        f'--sex-id 1 '
        f'--child-locations 1 2 '
        f'--child-sexes 2 '
        f'--prior-grid '
        f'--sample'
    )
    assert obj.template_kwargs['child_locations'] == '--child-locations 1 2'


def test_format_upload():
    obj = Upload(
        model_version_id=0,
        fit=True,
        prior=True,
        final=True
    )
    assert obj.command == (
        f'upload '
        f'--model-version-id 0 '
        f'--final --fit --prior'
    )


def test_cleanup():
    obj = CleanUp(
        model_version_id=0
    )
    assert obj.command == (
        f'cleanup '
        f'--model-version-id 0'
    )
