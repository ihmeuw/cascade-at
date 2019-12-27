from cascade_at.cascade.cascade_commands import Drill, TraditionalCascade
from cascade_at.cascade.cascade_commands import CASCADE_COMMANDS
from cascade_at.cascade.cascade_operations import CASCADE_OPERATIONS


def test_cascade_dict():
    assert type(CASCADE_COMMANDS) == dict
    assert CASCADE_COMMANDS['drill'] == Drill
    assert CASCADE_COMMANDS['cascade'] == TraditionalCascade


def test_drill():
    cascade_command = Drill(
        model_version_id=0,
        drill_parent_location_id=1,
        drill_sex=1
    )
    assert len(cascade_command.task_dict) == 3
    assert len(cascade_command.get_commands()) == 3

    assert type(
        cascade_command.task_dict['configure_inputs -model-version-id 0 --make --configure --drill 1']
    ) == CASCADE_OPERATIONS['configure_inputs']
    assert type(
        cascade_command.task_dict['dismod_db -model-version-id 0 -parent-location-id 1 -sex-id 1 '
                                  '--commands init fit-fixed fit-both predict-fit_var']
    ) == CASCADE_OPERATIONS['fit_both']
    assert type(
        cascade_command.task_dict['format_upload -model-version-id 0 -parent-location-id 1 -sex-id 1']
    ) == CASCADE_OPERATIONS['format_upload']
