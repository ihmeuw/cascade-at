from cascade_at.cascade.cascade_commands import Drill, TraditionalCascade
from cascade_at.cascade.cascade_operations import CASCADE_OPERATIONS


def test_drill():
    cascade_command = Drill(
        model_version_id=0,
        drill_parent_location_id=1,
        drill_sex=1,
        n_sim=100,
        n_pool=20,
    )
    assert len(cascade_command.task_dict) == 5
    assert len(cascade_command.get_commands()) == 5
    assert isinstance(
        cascade_command.task_dict['configure_inputs --model-version-id 0 --make --configure'],
        CASCADE_OPERATIONS['configure_inputs']
    )
    assert isinstance(
        cascade_command.task_dict[
            'dismod_db --model-version-id 0 --parent-location-id 1 --sex-id 1 --fill '
            '--dm-commands ODE-init ODE-fit predict-fit_var '
            '--save-fit'
        ],
        CASCADE_OPERATIONS['dismod_db']
    )
    assert isinstance(
        cascade_command.task_dict[
            'sample --model-version-id 0 --parent-location-id 1 --sex-id 1 '
            '--n-sim 100 --n-pool 20 --fit-type both --asymptotic'
        ],
        CASCADE_OPERATIONS['sample']
    )
    assert isinstance(
        cascade_command.task_dict[
            'predict --model-version-id 0 --parent-location-id 1 --sex-id 1 '
            '--save-final'
        ],
        CASCADE_OPERATIONS['predict']
    )
    assert isinstance(
        cascade_command.task_dict['upload --model-version-id 0 --final --fit'],
        CASCADE_OPERATIONS['upload']
    )
