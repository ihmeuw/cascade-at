from typing import List

from cascade_at.inputs.utilities.gbd_ids import SEX_NAME_TO_ID, SEX_ID_TO_NAME
from cascade_at.inputs.locations import LocationDAG
from cascade_at.cascade.cascade_operations import _CascadeOperation, Upload, MulcovStatistics
from cascade_at.cascade.cascade_stacks import root_fit, branch_fit, leaf_fit


def branch_or_leaf(dag: LocationDAG, location_id: int, sex: int, model_version_id: int,
                   parent_location: int, parent_sex: int,
                   n_sim: int, n_pool: int, upstream: List[str], tasks: List[_CascadeOperation]):
    """
    Recursive function that either creates a branch (by calling itself) or a leaf fit depending
    on whether or not it is at a terminal node. Determines if it's at a terminal node using
    the dag.successors() method from networkx. Appends tasks onto the tasks parameter.
    """
    if not dag.is_leaf(location_id=location_id):
        branch = branch_fit(
            model_version_id=model_version_id,
            location_id=location_id, sex_id=sex,
            prior_parent=parent_location, prior_sex=sex,
            child_locations=dag.children(location_id), child_sexes=[sex],
            n_sim=n_sim, n_pool=n_pool,
            upstream_commands=upstream
        )
        tasks += branch
        for location in dag.children(location_id):
            branch_or_leaf(dag=dag, location_id=location, sex=sex, model_version_id=model_version_id,
                           parent_location=location_id, parent_sex=sex,
                           n_sim=n_sim, n_pool=n_pool, upstream=[branch[-1].command], tasks=tasks)
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
                     n_sim: int = 100, n_pool: int = 100, skip_configure: bool = False,
                     json_file: str = '') -> List[_CascadeOperation]:
    """
    Make a traditional cascade dag for a model version. Relies on a location DAG and a starting
    point in the DAG for locations and sexes.

    Parameters
    ----------
    model_version_id
        Model version ID
    dag
        A location DAG that specifies the location hierarchy
    location_start
        Where to start in the location hierarchy
    sex_start
        Which sex to start with, can be most detailed or both.
    split_sex
        Whether or not to split sex into most detailed. If not, then will just stay at 'both' sex.
    n_sim
        Number of simulations to do in sample simulate
    n_pool
        Number of multiprocessing pools to create during sample simulate
    skip_configure
        Don't configure inputs. Only do this if it's already been done.

    Returns
    -------
    List of _CascadeOperation.
    """

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
        child_locations=dag.children(location_start), child_sexes=sexes,
        mulcov_stats=True,
        skip_configure=skip_configure,
        n_sim=n_sim, n_pool=n_pool,
        ode_fit_strategy=True,
        json_file=json_file,
    )
    tasks += top_level
    for sex in sexes:
        for location1 in dag.children(location_start):
            branch_or_leaf(
                dag=dag, location_id=location1, sex=sex, model_version_id=model_version_id,
                parent_location=location_start, parent_sex=sex,
                n_sim=n_sim, n_pool=n_pool, upstream=[top_level[-1].command], tasks=tasks
            )
    tasks.append(Upload(
        model_version_id=model_version_id,
        fit=True, prior=True,
        upstream_commands=[tasks[-1].command],
        executor_parameters={
            'm_mem_free': '50G'
        }
    ))
    return tasks
