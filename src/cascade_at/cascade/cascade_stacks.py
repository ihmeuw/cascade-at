"""
====================================
Cascade Operation Stacking Functions
====================================

These functions make sequences of _CascadeOperation
and the appropriate upstream dependencies. They can then be
used together to create a _CascadeCommand.
"""
from typing import List

from cascade_at.cascade.cascade_operations import _CascadeOperation
from cascade_at.cascade.cascade_operations import (
    ConfigureInputs, Fit, SampleSimulate, PredictSample,
    Upload, CleanUp
)


def single_fit(model_version_id: int,
               location_id: int, sex_id: int) -> List[_CascadeOperation]:
    """
    Create a sequence of tasks to do a single fit both model. Configures
    inputs, does a fit fixed, then fit both, then predict and uploads the result.
    Will fit the model based on the settings attached to the model version ID.

    Parameters
    ----------
    model_version_id
        The model version ID.
    location_id
        The parent location ID to run the model for.
    sex_id
        The sex ID to run the model for.

    Returns
    -------
    List of CascadeOperations.
    """
    t1 = ConfigureInputs(
        model_version_id=model_version_id
    )
    t2 = Fit(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        fill=True,
        predict=True,
        both=True,
        save_fit=True,
        upstream_commands=[t1.command]
    )
    t3 = Upload(
        model_version_id=model_version_id,
        fit=True,
        upstream_commands=[t2.command]
    )
    return [t1, t2, t3]


def leaf_fits(model_version_id: int, location_id: int, sex_id: int,
              n_sim: int = 100, n_pool: int = 1,
              upstream_commands: List[str] = None) -> List[_CascadeOperation]:
    """
    Performs a fit at a leaf node, only running a fit fixed, no children.

    Parameters
    ----------
    model_version_id
    location_id
    sex_id
    n_sim
    n_pool
    upstream_commands
    """
    t1 = Fit(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        fill=False,
        both=False,
        upstream_commands=upstream_commands
    )
    t2 = SampleSimulate(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        n_sim=n_sim,
        n_pool=n_pool,
        fit_type='fixed',
        upstream_commands=[t1.command]
    )
    t3 = PredictSample(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        target_locations=[location_id],
        target_sexes=[sex_id],
        upstream_commands=[t2.command]
    )
    return [t1.command, t2.command, t3.command]


def cascade_fits(model_version_id: int, location_id: int, sex_id: int,
                 child_locations: List[int], child_sexes: List[int],
                 n_sim: int = 100, n_pool: int = 1,
                 upstream_commands: List[str] = None) -> List[_CascadeOperation]:
    t1 = Fit(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        fill=True,
        both=True,
        predict=True,
        upstream_commands=upstream_commands
    )
    t2 = SampleSimulate(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        n_sim=n_sim,
        n_pool=n_pool,
        fit_type='both',
        upstream_commands=[t1.command]
    )
    t3 = PredictSample(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        target_locations=child_locations,
        target_sexes=child_sexes,
        upstream_commands=[t2.command]
    )
    return [t1, t2, t3]


def root_fit(model_version_id: int, location_id: int, sex_id: int,
             child_locations: List[int], child_sexes: List[int],
             n_sim: int = 100, n_pool: int = 1,
             upstream_commands: List[str] = None) -> List[_CascadeOperation]:
    """
    Create a sequence of tasks to do a top-level prior fit.
    Does a fit fixed, then fit both, then predict and uploads the result.
    Will fit the model based on the settings attached to the model version ID.

    Parameters
    ----------
    model_version_id
        The model version ID.
    location_id
        The parent location ID to run the model for.
    sex_id
        The sex ID to run the model for.
    n_sim
        The number of simulations to do to get the posterior fit.
    n_pool
        The number of pools to use to do the simulation fits.
    child_locations
        The children to fill the avgint table with
    child_sexes
        The sexes to predict for.
    upstream_commands
        List of upstream commands for this sequence.

    Returns
    -------
    List of CascadeOperations.
    """
    t1 = ConfigureInputs(
        model_version_id=model_version_id
    )
    t2 = Fit(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        fill=True,
        both=True,
        predict=True,
        upstream_commands=[t1.command]
    )
    t3 = SampleSimulate(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        n_sim=n_sim,
        n_pool=n_pool,
        fit_type='both',
        upstream_commands=[t2.command]
    )
    t4 = PredictSample(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        target_locations=child_locations,
        target_sexes=child_sexes,
        upstream_commands=[t3.command]
    )
    return [t1, t2, t3]
