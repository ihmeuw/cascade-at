from typing import List

from cascade_at.inputs.utilities.gbd_ids import SEX_NAME_TO_ID, SEX_ID_TO_NAME
from cascade_at.inputs.locations import LocationDAG
from cascade_at.cascade.cascade_operations import _CascadeOperation, Upload
from cascade_at.cascade.cascade_stacks import root_fit, branch_fit, leaf_fit


def branch_or_leaf(dag, location_id, sex, model_version_id, parent_location, parent_sex,
                   n_sim, n_pool, upstream, tasks):
    if list(dag.dag.successors(location_id)):
        branch = branch_fit(
            model_version_id=model_version_id,
            location_id=location_id, sex_id=sex,
            prior_parent=parent_location, prior_sex=sex,
            child_locations=list(dag.dag.successors(location_id)), child_sexes=[sex],
            n_sim=n_sim, n_pool=n_pool,
            upstream_commands=upstream
        )
        tasks += branch
        for location in dag.dag.successors(location_id):
            branch_or_leaf(dag=dag, location_id=location, sex=sex, model_version_id=model_version_id,
                           parent_location=location_id, parent_sex=sex,
                           n_sim=n_sim, n_pool=n_pool, upstream=branch[-1].command, tasks=tasks)
    else:
        leaf = leaf_fit(
            model_version_id=model_version_id,
            location_id=location_id,
            sex_id=sex,
            prior_parent=parent_location,
            prior_sex=parent_sex,
            n_sim=n_sim, n_pool=n_pool,
            upstream_commands=upstream
        )
        tasks += leaf


def make_cascade_dag(model_version_id: int, dag: LocationDAG,
                     location_start: int, sex_start: int, split_sex: bool,
                     n_sim: int = 100, n_pool: int = 100) -> List[_CascadeOperation]:

    tasks = []

    sexes = [sex_start]
    if SEX_ID_TO_NAME[sex_start] == 'Both':
        if split_sex:
            sexes = [
                SEX_NAME_TO_ID['Female'],
                SEX_NAME_TO_ID['Male']
            ]

    top_level = root_fit(
        model_version_id=model_version_id,
        location_id=location_start, sex_id=sex_start,
        child_locations=dag.dag.successors(location_start), child_sexes=sexes
    )
    tasks += top_level
    for sex in sexes:
        for location1 in dag.dag.successors(location_start):
            branch_or_leaf(
                dag=dag, location_id=location1, sex=sex, model_version_id=model_version_id,
                parent_location=location_start, parent_sex=sexes,
                n_sim=n_sim, n_pool=n_pool, upstream=top_level[-1].command, tasks=tasks
            )
    tasks.append(Upload(
        model_version_id=model_version_id,
        fit=True, prior=True,
        upstream_commands=tasks[-1].command
    ))
    return tasks
