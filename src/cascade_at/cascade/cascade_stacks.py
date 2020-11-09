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
    ConfigureInputs, Fit, Sample, Predict,
    Upload, CleanUp, MulcovStatistics
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
        save_prior=False,
        upstream_commands=[t1.command]
    )
    t3 = Upload(
        model_version_id=model_version_id,
        fit=True,
        upstream_commands=[t2.command]
    )
    return [t1, t2, t3]


def single_fit_with_uncertainty(model_version_id: int,
                                location_id: int, sex_id: int,
                                n_sim: int = 100, n_pool: int = 20,
                                asymptotic: bool = False, cv_priors: bool = False) -> List[_CascadeOperation]:
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
    n_sim
        The number of simulations to do, number of draws to make
    n_pool
        The number of multiprocessing pools to use in creating the draws
    asymptotic
        Use asymptotic sampling to compute posteriors and prior prediction
    cv_priors
        Use the coefficient of variation of the fit predictions for priors
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
    t3 = Sample(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        n_sim=n_sim,
        n_pool=n_pool,
        fit_type='both',
        asymptotic=asymptotic,
        cv_priors=cv_priors,
        upstream_commands=[t2.command],
        executor_parameters={
            'num_cores': n_pool
        },
    )
    t4 = Predict(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        save_final=True,
        sample=not cv_priors,
        prior_grid=asymptotic,
        upstream_commands=[t3.command]
    )
    t5 = Upload(
        model_version_id=model_version_id,
        fit=True,
        final=True,
        upstream_commands=[t4.command]
    )
    return [t1, t2, t3, t4, t5]


def root_fit(model_version_id: int, location_id: int, sex_id: int,
             child_locations: List[int], child_sexes: List[int],
             skip_configure: bool = False,
             mulcov_stats: bool = True,
             n_sim: int = 100, n_pool: int = 10,
             asymptotic: bool = False, cv_priors: bool = False) -> List[_CascadeOperation]:
    """
    Create a sequence of tasks to do a top-level prior fit.
    Does a fit fixed, then fit both, then creates posteriors
    that can be used as priors later on. Saves its fit to be uploaded.

    Parameters
    ----------
    model_version_id
        The model version ID.
    location_id
        The parent location ID to run the model for.
    sex_id
        The sex ID to run the model for.
    child_locations
        The children to fill the avgint table with
    child_sexes
        The sexes to predict for.
    skip_configure
        Don't run a task to configure the inputs. Only do this if it has already happened.
    mulcov_stats
        Compute mulcov statistics at this level
    n_sim
        The number of simulations to do, number of draws to make
    n_pool
        The number of multiprocessing pools to use in creating the draws
    asymptotic
        Use asymptotic sampling to compute posteriors and prior prediction
    cv_priors
        Use the coefficient of variation of the fit predictions for priors
    Returns
    -------
    List of CascadeOperations.
    """
    tasks = []

    if not skip_configure:
        t1 = ConfigureInputs(
            model_version_id=model_version_id
        )
        upstream = [t1.command]
        tasks.append(t1)
    else:
        upstream = None
    t2 = Fit(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        fill=True,
        both=False,
        predict=True,
        upstream_commands=upstream,
        save_fit=True,
        save_prior=True
    )
    tasks.append(t2)
    t3 = Sample(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        n_sim=n_sim,
        n_pool=n_pool,
        fit_type='fixed',
        asymptotic=asymptotic,
        cv_priors=cv_priors,
        upstream_commands=[t2.command],
        executor_parameters={
            'num_cores': n_pool
        }
    )
    tasks.append(t3)
    t4 = Predict(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        child_locations=child_locations,
        child_sexes=child_sexes,
        sample=not cv_priors,
        upstream_commands=[t3.command]
    )
    tasks.append(t4)
    if mulcov_stats:
        t5 = MulcovStatistics(
            model_version_id=model_version_id,
            locations=[location_id],
            sexes=[sex_id],
            sample=True,
            mean=True,
            std=True,
            quantile=[0.025, 0.975],
            upstream_commands=[t4.command]
        )
        tasks.append(t5)
    return tasks


def branch_fit(model_version_id: int, location_id: int, sex_id: int,
               prior_parent: int, prior_sex: int,
               child_locations: List[int], child_sexes: List[int],
               n_sim: int = 100, n_pool: int = 10,
               asymptotic: bool = False, cv_priors: bool = False,
               upstream_commands: List[str] = None) -> List[_CascadeOperation]:
    """
    Create a sequence of tasks to do a cascade fit (mid-level).
    Does a fit fixed, then fit both, predicts on the prior rate grid to create posteriors
    that can be used as priors later on. Saves its fit to be uploaded.

    Parameters
    ----------
    model_version_id
        The model version ID.
    location_id
        The parent location ID to run the model for.
    sex_id
        The sex ID to run the model for.
    prior_parent
        The location ID corresponding to a database to pull the prior from
    prior_sex
        The sex ID corresponding to a database to pull the prior from
    child_locations
        The children to fill the avgint table with
    child_sexes
        The sexes to predict for.
    asymptotic
        Use asymptotic sampling to compute posteriors and prior prediction
    cv_priors
        Use the coefficient of variation of the fit predictions for priors
    upstream_commands
        Commands that need to be run before this stack.

    Returns
    -------
    List of CascadeOperations.
    """
    t1 = Fit(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        fill=True,
        both=False,
        predict=True,
        prior_mulcov=model_version_id,
        prior_samples=n_sim>1,
        prior_parent=prior_parent,
        prior_sex=prior_sex,
        save_fit=True,
        save_prior=True,
        upstream_commands=upstream_commands
    )
    t2 = Sample(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        n_sim=n_sim,
        n_pool=n_pool,
        fit_type='fixed',
        asymptotic=asymptotic,
        cv_priors=cv_priors,
        upstream_commands=[t1.command],
        executor_parameters={
            'num_cores': n_pool
        }
    )
    t3 = Predict(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        child_locations=child_locations,
        child_sexes=child_sexes,
        sample=not cv_priors,
        prior_grid=n_sim>0,
        upstream_commands=[t2.command]
    )
    return [t1, t2, t3]


def leaf_fit(model_version_id: int, location_id: int, sex_id: int,
             prior_parent: int, prior_sex: int,
             n_sim: int = 100, n_pool: int = 1,
             asymptotic: bool = False, cv_priors: bool = False,
             upstream_commands: List[str] = None) -> List[_CascadeOperation]:
    """
    Create a sequence of tasks to do a for a leaf-node fit, no children.
    Does a fit fixed then sample simulate to create posteriors. Saves its fit to be uploaded.

    Parameters
    ----------
    model_version_id
        The model version ID.
    location_id
        The parent location ID to run the model for.
    sex_id
        The sex ID to run the model for.
    prior_parent
        The location ID corresponding to a database to pull the prior from
    prior_sex
        The sex ID corresponding to a database to pull the prior from
    n_sim
        The number of simulations to do to get the posterior fit.
    n_pool
        The number of pools to use to do the simulation fits.
    asymptotic
        Use asymptotic sampling to compute posteriors and prior prediction
    cv_priors
        Use the coefficient of variation of the fit predictions for priors
    upstream_commands
        Commands that need to be run before this stack.

    Returns
    -------
    List of CascadeOperations.
    """
    t1 = Fit(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        fill=True,
        both=False,
        prior_mulcov=model_version_id,
        prior_samples=n_sim>1,
        prior_parent=prior_parent,
        prior_sex=prior_sex,
        save_fit=False,
        save_prior=True,
        upstream_commands=upstream_commands
    )
    t2 = Sample(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        n_sim=n_sim,
        n_pool=n_pool,
        fit_type='fixed',
        asymptotic=asymptotic,
        cv_priors=cv_priors,
        upstream_commands=[t1.command],
        executor_parameters={
            'num_cores': n_pool
        }
    )
    t3 = Predict(
        model_version_id=model_version_id,
        parent_location_id=location_id,
        sex_id=sex_id,
        child_locations=[location_id],
        child_sexes=[sex_id],
        save_fit=True,
        save_final=True,
        sample=not cv_priors,
        prior_grid=n_sim>0,
        upstream_commands=[t2.command]
    )
    return [t1, t2, t3]
