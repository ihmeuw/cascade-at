import os
import logging
import asyncio
import math
from pathlib import Path
from pprint import pformat
from bdb import BdbQuit
from pkg_resources import get_distribution, DistributionNotFound
from tempfile import TemporaryDirectory
import shutil

import pandas as pd
import numpy as np

import cascade
from cascade.core.cascade_plan import CascadePlan
from cascade.core.db import dataframe_from_disk
from cascade.input_data.configuration.id_map import make_integrand_map
from cascade.dismod.db.wrapper import DismodFile, _get_engine
from cascade.stats import meas_bounds_to_stdev
from cascade.executor.argument_parser import DMArgumentParser
from cascade.input_data.db.demographics import age_groups_to_ranges
from cascade.testing_utilities import make_execution_context
from cascade.input_data.db.configuration import load_settings
from cascade.input_data.db.csmr import load_csmr_to_t3, get_csmr_data
from cascade.input_data.db.locations import get_descendents, location_id_from_location_and_level
from cascade.input_data.db.asdr import load_asdr_to_t3, get_asdr_data
from cascade.input_data.db.mortality import (
    get_frozen_cause_specific_mortality_data,
    normalize_mortality_data
)
from cascade.input_data.emr import add_emr_from_prevalence
from cascade.executor.dismod_runner import run_and_watch, async_run_and_watch, DismodATException
from cascade.input_data.configuration.construct_bundle import normalized_bundle_from_database, bundle_to_observations
from cascade.input_data.db.bundle import freeze_bundle
from cascade.dismod.serialize import model_to_dismod_file
from cascade.model.integrands import make_average_integrand_cases_from_gbd
from cascade.saver.save_model_results import save_model_results
from cascade.input_data.configuration import SettingsError
from cascade.input_data.configuration.builder import (
    initial_context_from_epiviz,
    fixed_effects_from_epiviz,
    random_effects_from_epiviz,
    build_constraint,
)

from cascade.core import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


def add_settings_to_execution_context(ec, settings):
    to_append = dict(
        modelable_entity_id=settings.model.modelable_entity_id,
        model_version_id=settings.model.model_version_id,
        model_title=settings.model.title,
        gbd_round_id=settings.gbd_round_id,
        bundle_id=settings.model.bundle_id,
        add_csmr_cause=settings.model.add_csmr_cause,
        location_id=settings.model.drill_location,
        cod_version=settings.csmr_cod_output_version_id,
    )
    for param, value in to_append.items():
        setattr(ec.parameters, param, value)

    # FIXME: We are using split sex to represent the drill start because
    # there isn't an entry for it in the GUI yet.
    ec.parameters.drill_start = location_id_from_location_and_level(
        ec, settings.model.drill_location, settings.model.split_sex
    )[0]


def add_mortality_data(model_context, execution_context, sex_id):
    """
    Gets cause-specific mortality rate and adds that data as an ``mtspecific``
    measurement by appending it to the bundle. Uses ranges for ages and years.
    This doesn't determine point-data values.
    """
    MATHLOG.debug(f"Creating a set of mtspecific observations from IHME CSMR database.")
    MATHLOG.debug("Assigning standard error from measured upper and lower.")

    if execution_context.parameters.tier == 3:
        raw_csmr = normalize_mortality_data(get_frozen_cause_specific_mortality_data(execution_context))
    else:
        raw_csmr = normalize_mortality_data(get_csmr_data(execution_context))

    csmr = meas_bounds_to_stdev(
        age_groups_to_ranges(execution_context, raw_csmr)
    )
    csmr["measure"] = "mtspecific"
    csmr = csmr.rename(columns={"location_id": "node_id"})
    csmr = csmr.query(f"sex_id == @sex_id")
    MATHLOG.debug(f"Creating a set of {csmr.shape[0]} mtspecific observations from IHME CSMR database.")
    csmr = csmr.assign(hold_out=0)

    model_context.input_data.observations = pd.concat(
        [model_context.input_data.observations, csmr], ignore_index=True, sort=True
    )

    if model_context.policies["estimate_emr_from_prevalence"]:
        MATHLOG.debug(f"estimate_emr_from_prevalence policy is selected")
        add_emr_from_prevalence(model_context, execution_context)


def add_omega_constraint(model_context, execution_context, sex_id):
    r"""
    Adds a constraint to other-cause mortality rate. Removes mtother,
    mtall, and mtspecific from observation data. Uses
    :py:func:`cascade.input_data.configuration.builder.build_constraint` to make smoothing priors.

    Child constraints are constrained random effects, so they are offsets
    from the parent omega constraint, as

    .. math::

        \omega_c(a,t) = \omega_p(a,t) e^{u_c(a,t)}

    It is the :math:`u_c` that we put into the child smoothings. We define
    these grids so that all grid points match. At each grid point,
    :math:`u_c=\log (c/p)` with c for child, p for parent.
    """
    MATHLOG.debug(f"Add omega constraint from age-standardized death rate data.")
    MATHLOG.debug("Assigning standard error from measured upper and lower.")

    if execution_context.parameters.tier == 3:
        raw_asdr = normalize_mortality_data(
            # Call this the long way so it can be monkeypatched in testing.
            cascade.input_data.db.mortality.get_frozen_age_standardized_death_rate_data(execution_context)
        )
    else:
        raw_asdr = normalize_mortality_data(get_asdr_data(execution_context))

    asdr = meas_bounds_to_stdev(
        age_groups_to_ranges(execution_context, raw_asdr)
    )
    asdr["measure"] = "mtall"
    asdr = asdr.rename(columns={"location_id": "node_id"})
    asdr = asdr.query(f"sex_id == @sex_id")
    if model_context.input_data.times:  # The times are a set so can be tested this way.
        min_time = np.min(list(model_context.input_data.times))  # noqa: F841
        max_time = np.max(list(model_context.input_data.times))  # noqa: F841
        # The % 5 is to exclude annual data points.
        asdr = asdr.query("time_lower >= @min_time and time_upper <= @max_time and time_lower % 5 == 0")

    parent_asdr = asdr[asdr.node_id == model_context.parameters.location_id]
    model_context.rates.omega.parent_smooth = build_constraint(parent_asdr)
    MATHLOG.debug(f"Add {parent_asdr.shape[0]} omega constraints from age-standardized death rate data to the parent.")

    children = get_descendents(execution_context, children_only=True)  # noqa: F841
    children_asdr = asdr.query("node_id in @children")
    # Transform the children to be the random effect for the rate.
    parent_value = parent_asdr[["age_lower", "time_lower", "mean"]].rename({"mean": "parent_mean"}, axis=1)
    parent_and_child = children_asdr.merge(parent_value, how="left", on=["age_lower", "time_lower"])
    # This could result in an Inf value, but that's a legal value, so...
    children_effects = parent_and_child.assign(
        mean=np.log(parent_and_child["mean"] / parent_and_child["parent_mean"])
    ).drop("parent_mean", axis=1)

    model_context.rates.omega.child_smoothings = [
        (node_id, build_constraint(child_asdr))
        for node_id, child_asdr in children_effects.groupby('node_id')
    ]
    MATHLOG.debug(
        f"Add {children_asdr.shape[0]} omega constraints from "
        f"age-standardized death rate data to the children."
    )

    observations = model_context.input_data.observations
    observations.loc[observations.measure == "mtall", "hold_out"] = 1
    asdr = asdr.assign(hold_out=1)
    model_context.input_data.observations = pd.concat([observations, asdr], ignore_index=True, sort=True)


def compute_age_steps(smallest_step):
    """We will add age steps to the ODE steps that are for "every 1 year" or
    "every 5 years," as given by the settings. The GBD chooses
    age step sizes of 0, 7 days, 28 days, 1 year, 5 years. These look roughly
    like a pattern of multiplying by 4, so 1 week, 4 weeks 16 weeks, 64 weeks.
    Let's play with the numbers to respect that organization.

    This means there are two magic numbers: The smallest step must be less
    than 7 days, and all steps are 2 times the size of the last step, instead
    of 4, just to be safe.
    So an ODE step size of 1 year gives (0.015625, 0.0625, 0.25).
    An ODE step size of 5 years gives [0.0049, 0.019, 0.078, 0.31, 1.25].

    If the smallest step is under 7/365, then no additional age steps
    are returned.
    """
    minimum_step_size = 7 / 365
    geometric_growth = 2
    minimum_step_count = np.log(smallest_step / minimum_step_size) / np.log(geometric_growth)
    chosen_step_count = math.ceil(minimum_step_count)
    delta = smallest_step / geometric_growth**chosen_step_count
    added = np.array([delta * geometric_growth**i for i in range(chosen_step_count)])
    MATHLOG.info(f"Adding ages to ODE integration: {added}")
    return added


def prepare_data(execution_context, settings):
    if execution_context.parameters.tier == 3:
        freeze_bundle(execution_context, execution_context.parameters.bundle_id)

        if execution_context.parameters.add_csmr_cause is not None:
            MATHLOG.info(
                f"Cause {execution_context.parameters.add_csmr_cause} "
                "selected as CSMR source, freezing it's data if it has not already been frozen."
            )
            load_csmr_to_t3(execution_context)
        load_asdr_to_t3(execution_context)

    if execution_context.parameters.bundle_file:
        bundle = dataframe_from_disk(execution_context.parameters.bundle_file)
    else:
        bundle = normalized_bundle_from_database(
            execution_context,
            bundle_id=execution_context.parameters.bundle_id,
            tier=execution_context.parameters.tier
        )

    location_and_descendents = get_descendents(execution_context, include_parent=True)  # noqa: F841

    bundle = bundle.query("location_id in @location_and_descendents")
    MATHLOG.info(f"Filtering bundle to the current location and it's descendents. {len(bundle)} rows remaining.")

    stderr_mask = bundle.standard_error > 0
    if (~stderr_mask).sum() > 0:
        MATHLOG.warn(
            f"Filtering {(~stderr_mask).sum()} rows where standard error == 0 out of bundle. "
            f"{stderr_mask.sum()} rows remaining."
        )

    rr_mask = bundle.measure != "relrisk"
    mask = stderr_mask & rr_mask
    if (~rr_mask).sum() > 0:
        MATHLOG.info(
            f"Filtering {(~rr_mask).sum()} rows of relative risk data out of bundle. "
            f"{mask.sum()} rows remaining."
        )

    bundle = bundle[mask]

    measures_to_exclude = settings.model.exclude_data_for_param
    if measures_to_exclude:
        integrand_map = make_integrand_map()
        measures_to_exclude = [integrand_map[m].name for m in measures_to_exclude]
        mask = bundle.measure.isin(measures_to_exclude)
        if mask.sum() > 0:
            bundle = bundle[~mask]
            MATHLOG.info(
                f"Filtering {mask.sum()} rows of of data where the measure has been excluded. "
                f"Measures marked for exclusion: {measures_to_exclude}. "
                f"{len(bundle)} rows remaining."
            )

    return bundle


def model_context_from_settings(execution_context, settings):
    """
     1. Freeze the measurement bundle, which means that we make a copy of the
        input measurement data and study covariates for safe-keeping
        in case there is later question about what data was used.

     2. Retrieve the measurement bundle and its study covariates
        and convert it into data on the model, as described in
        :ref:`convert-bundle-to-measurement-data`.

     3. Add mortality data. This is cause-specific mortality
        data, and it is added as "mtspecific" in the Dismod-AT measurements.
        This data has a "measurement upper" and "measurement lower"
        which are converted into a standard error with a Gaussian
        prior.

     4. Add other-cause mortality as a constraint on the system, meaning
        the age-standardized death rate is used to construct both
        measurement data for mtall, with priors determined from
        "measurement upper" and "measurement lower", but also a constraint
        on omega, the underlying rate for other-cause mortality, so that
        Dismod-AT will accept this as a given in the problem.

     5. Create Average Integrand Cases, which are the list of
        desired outputs from Dismod-AT to show in graphs in EpiViz-AT.

     6. Construct all Fixed Effects. These are defined in
        https://bradbell.github.io/dismod_at/doc/model_variables.htm.

     7. Construct all Random Effects.
    """
    model_context = initial_context_from_epiviz(settings)

    bundle = prepare_data(execution_context, settings)

    observations = bundle_to_observations(model_context.parameters, bundle)
    model_context.input_data.observations = observations

    if execution_context.parameters.add_csmr_cause is not None:
        MATHLOG.info(f"Cause {execution_context.parameters.add_csmr_cause} selected as CSMR source, loading it's data.")
        add_mortality_data(model_context, execution_context, settings.model.drill_sex)
    else:
        MATHLOG.info(f"No cause selected as CSMR source so no CSMR data will be added to the bundle.")

    if settings.model.constrain_omega:
        add_omega_constraint(model_context, execution_context, settings.model.drill_sex)

    cases = make_average_integrand_cases_from_gbd(
        execution_context, [settings.model.drill_sex], include_birth_prevalence=bool(settings.model.birth_prev)
    )
    model_context.average_integrand_cases = cases

    model_context.parameters.additional_ode_steps = compute_age_steps(model_context.parameters.ode_step_size)
    fixed_effects_from_epiviz(model_context, execution_context, settings)
    random_effects_from_epiviz(model_context, settings)

    model_context.input_data.observations = model_context.input_data.observations.drop(columns="sex_id")
    model_context.average_integrand_cases = model_context.average_integrand_cases.drop(columns="sex_id")
    return model_context


def write_dismod_file(mc, ec, db_file_path):
    MATHLOG.info(f"Writing dismod database to {db_file_path}")
    dismod_file = model_to_dismod_file(mc, ec)
    dismod_file.engine = _get_engine(Path(db_file_path))
    dismod_file.flush()
    return dismod_file


def _get_dismod_db_path(dismod_file):
    dm_file_path = dismod_file.engine.url.database
    if dm_file_path == ":memory:":
        raise ValueError("Cannot run dismodat on an in-memory database")
    return dm_file_path


def _check_dismod_command(dismod_file, command):
    dismod_file.refresh()
    if f"end {command}" not in dismod_file.log.message.iloc[-1]:
        raise DismodATException(f"DismodAt failed to complete '{command}' command")


def run_dismod(dismod_file, command, *args):
    dm_file_path = _get_dismod_db_path(dismod_file)

    command_prefix = ["dmdismod", dm_file_path]

    run_and_watch(command_prefix + [command] + list(args), False, 1)

    _check_dismod_command(dismod_file, command)


@asyncio.coroutine
def async_run_dismod(dismod_file, command, *args):
    dm_file_path = _get_dismod_db_path(dismod_file)

    command_prefix = ["dmdismod", dm_file_path]

    yield from async_run_and_watch(command_prefix + [command] + list(args), False, 1)

    try:
        # FIXME: dismod damages the terminal charactersitics somehow when it's run concurrently.
        # It does this even if all output is supressed. Other programs don't cause this problem
        # even if they have lots of output. This is the least invasive way I've found
        # of ensuring that the environment is usable after this runs
        yield from async_run_and_watch(["stty", "sane"], False, 1)
    except DismodATException:
        # in some environments (inside a qsub) stty fails but in those
        # environments the problem with mangled terminals doesn't come
        # up, so just ignore it
        pass

    _check_dismod_command(dismod_file, command)


def run_dismod_fit(dismod_file, with_random_effects):
    random_or_fixed = "both" if with_random_effects else "fixed"

    run_dismod(dismod_file, "fit", random_or_fixed)


def run_dismod_predict(dismod_file):
    run_dismod(dismod_file, "predict", "fit_var")


def make_fixed_effect_samples(execution_context, num_samples):
    run_dismod(execution_context.dismodfile, "set", "truth_var", "fit_var")
    run_dismod(execution_context.dismodfile, "simulate", str(num_samples))


@asyncio.coroutine
def _fit_and_predict_fixed_effect_sample(db_path, sample_id, sem):
    yield from sem.acquire()
    try:
        with TemporaryDirectory() as d:
            temp_dm_path = Path(d) / "sample.db"
            shutil.copy2(db_path, temp_dm_path)
            dismod_file = DismodFile(_get_engine(temp_dm_path))
            yield from async_run_dismod(dismod_file, "set", "start_var", "truth_var")
            yield from async_run_dismod(dismod_file, "fit", "fixed", str(sample_id))
            yield from async_run_dismod(dismod_file, "predict", "fit_var")

            fit = dismod_file.fit_var
            fit["sample_index"] = sample_id

            predict = dismod_file.predict
            predict["sample_index"] = sample_id
        return (fit, predict)
    finally:
        sem.release()


@asyncio.coroutine
def _async_fit_and_predict_fixed_effect_samples(num_processes, dismodfile, samples):
    sem = asyncio.Semaphore(num_processes)
    jobs = []
    for sample in samples:
        jobs.append(_fit_and_predict_fixed_effect_sample(
            dismodfile,
            sample,
            sem
        ))
    log_level = logging.root.level
    math_root = logging.getLogger("cascade.math")
    math_log_level = math_root.level
    if num_processes > 1:
        logging.root.setLevel(logging.CRITICAL)
        math_root.setLevel(logging.CRITICAL)
    try:
        results = yield from asyncio.gather(*jobs)
    finally:
        logging.root.setLevel(log_level)
        logging.getLogger("cascade.math").setLevel(math_log_level)
    return results


def fit_and_predict_fixed_effect_samples(execution_context, num_processes):
    if execution_context.dismodfile.engine.url.database == ":memory:":
        raise ValueError("Cannot run fit_and_predict_fixed_effect_samples on an in-memory database")

    samples = execution_context.dismodfile.data_sim.simulate_index.unique()

    actual_processes = min(len(samples), num_processes)
    MATHLOG.info(f"Calculating {len(samples)} fixed effect samples")
    CODELOG.info(f"Starting parallel fixed effect sample generation using {actual_processes} processes")
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        _async_fit_and_predict_fixed_effect_samples(
            actual_processes,
            execution_context.dismodfile.engine.url.database,
            samples
        )
    )
    CODELOG.info("Done generating fixed effect samples")

    fit, predict = zip(*results)
    return pd.concat(fit), pd.concat(predict)


def has_random_effects(model):
    return any([bool(r.child_smoothings) for r in model.rates])


def main(args):
    """
     1. Parse arguments to the command line. The GUI calls a shell script
        which passes the model version ID to the EpiViz Runner.

     2. Construct an Execution Context, which tells EpiViz Runner where it can find
        directories, files, and databases on the cluster.

     3. Load Settings, which are the values you entered into EpiViz-AT on the web page.

     4. Using the Settings and the Execution Context, fill out the data and
        any parameters for Dismod-AT, as described in
        :ref:`build-model-from-epiviz-settings`.

     5. Put that data into a file and run Dismod-AT on that file.
    """
    ec = make_execution_context()

    settings = load_settings(ec, args.meid, args.mvid, args.settings_file)

    if settings.model.drill != "drill":
        raise NotImplementedError("Only 'drill' mode is currently supported")

    add_settings_to_execution_context(ec, settings)
    CascadePlan.from_epiviz_configuration(ec, settings)

    if args.skip_cache:
        ec.parameters.tier = 2
    else:
        ec.parameters.tier = 3
    for arg_name in ["db_only", "db_file_path", "no_upload", "bundle_file", "bundle_study_covariates_file"]:
        setattr(ec.parameters, arg_name, getattr(args, arg_name))

    one_location_set(ec, settings)

    MATHLOG.debug(f"Completed successfully")


def one_location_set(ec, settings):
    """Solve a parent with its children as random effects."""
    mc = model_context_from_settings(ec, settings)
    ec.dismodfile = write_dismod_file(mc, ec, ec.parameters.db_file_path)
    if not ec.parameters.db_only:
        run_dismod(ec.dismodfile, "init")
        run_dismod_fit(ec.dismodfile, has_random_effects(mc))
        MATHLOG.info(f"Successfully fit parent")

        num_samples = mc.policies["number_of_fixed_effect_samples"]
        make_fixed_effect_samples(ec, num_samples)
        sampled_fit, sampled_predict = fit_and_predict_fixed_effect_samples(ec,
                                                                            4)

        ec.dismodfile.predict = sampled_predict.drop("predict_id",
                                                     1).reset_index(drop=True)
        ec.dismodfile.flush()

        if not ec.parameters.no_upload:
            MATHLOG.debug(f"Uploading results to epiviz")
            save_model_results(ec)
        else:
            MATHLOG.debug(
                f"Skipping results upload because 'no-upload' was selected")
    else:
        MATHLOG.debug(
            f"Only creating the base db file because 'db-only' was selected")


def entry():
    readable_by_all = 0o0002
    os.umask(readable_by_all)

    parser = DMArgumentParser("Run DismodAT from Epiviz")
    parser.add_argument("db_file_path")
    parser.add_argument("--settings-file")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--db-only", action="store_true")
    parser.add_argument("-b", "--bundle-file")
    parser.add_argument("-s", "--bundle-study-covariates-file")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--pdb", action="store_true")
    args = parser.parse_args()

    CODELOG.debug(f"args: {args}")
    try:
        software_version = get_distribution("cascade").version
    except DistributionNotFound:
        # package is not installed
        software_version = "unavailable"
    MATHLOG.debug(f"Cascade version {software_version}.")
    if "JOB_ID" in os.environ:
        MATHLOG.info(f"Job id is {os.environ['JOB_ID']} on cluster {os.environ.get('SGE_CLUSTER_NAME', '')}")

    try:
        if args.skip_cache:
            args.no_upload = True

        main(args)
    except SettingsError as e:
        MATHLOG.error(str(e))
        CODELOG.error(f"Form data:{os.linesep}{pformat(e.form_data)}")
        error_lines = list()
        for error_spot, human_spot, error_message in e.form_errors:
            if args.settings_file is not None:
                error_location = error_spot
            else:
                error_location = human_spot
            error_lines.append(f"\t{error_location}: {error_message}")
        MATHLOG.error(f"Form validation errors:{os.linesep}{os.linesep.join(error_lines)}")
        exit(1)
    except BdbQuit:
        pass
    except Exception:
        if args.pdb:
            import pdb
            import traceback

            traceback.print_exc()
            pdb.post_mortem()
        else:
            MATHLOG.exception(f"Uncaught exception in {os.path.basename(__file__)}")
            raise


if __name__ == "__main__":
    entry()
