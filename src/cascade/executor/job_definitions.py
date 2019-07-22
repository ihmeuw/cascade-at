from contextlib import closing
from inspect import getmembers
import shelve

import pandas as pd

from cascade.core import getLoggers
from cascade.dismod import DismodATException
from cascade.executor.cascade_plan import recipe_graph_from_settings
from cascade.executor.construct_model import construct_model
from cascade.executor.covariate_description import create_covariate_specifications
from cascade.executor.estimate_location import retrieve_data, modify_input_data, compute_parent_fit_fixed, \
    compute_parent_fit, make_draws, save_outputs
from cascade.executor.priors_from_draws import set_priors_from_parent_draws
from cascade.input_data.configuration.raw_input import validate_input_data_types
from cascade.runner.data_passing import ShelfFile, PandasFile, DbFile
from cascade.runner.job_graph import CascadeJob, recipe_graph_to_job_graph

CODELOG, MATHLOG = getLoggers(__name__)


def is_pandas(maybe_df):
    return isinstance(maybe_df, (pd.DataFrame, pd.Series))


def save_global_data_to_hdf(path, global_data):
    dataframes = {name: df for (name, df) in getmembers(global_data, is_pandas)}

    not_written = set(global_data.__dict__) - set(dataframes)
    if not_written:
        CODELOG.info(f"The following members of global data were not written {not_written}.")
        CODELOG.info(f"The following members of global data were written {dataframes.keys()}.")
    else:
        CODELOG.info(f"All global data is dataframes.")

    CODELOG.info(f"HDFStore path {path}")
    with closing(pd.HDFStore(str(path), "w", complevel=9, complib="zlib")) as store:
        for name, df in dataframes.items():
            store.put(name, df, format="fixed", columns=True, dropna=False)

    return not_written


class GlobalPrepareData(CascadeJob):
    def __init__(self, recipe_id, local_settings, execution_context):
        super().__init__("global_prepare", recipe_id, local_settings, execution_context)
        global_location = 0
        self.outputs.update(dict(
            shared=ShelfFile(execution_context, "globalvars", global_location, "both", required_keys=[
                "covariate_multipliers", "covariate_data_spec",
            ]),
            data=PandasFile(execution_context, "globaldata.hdf", global_location, "both"),
        ))

    def run_under_mathlog(self):
        local_settings = self.local_settings
        execution_context = self.execution_context
        covariate_multipliers, covariate_data_spec = create_covariate_specifications(
            local_settings.settings.country_covariate, local_settings.settings.study_covariate
        )

        input_data = retrieve_data(execution_context, local_settings, covariate_data_spec)
        columns_wrong = validate_input_data_types(input_data)
        assert not columns_wrong, f"validation failed {columns_wrong}"
        modified_data = modify_input_data(input_data, local_settings, covariate_data_spec)
        not_written = save_global_data_to_hdf(self.outputs["data"].path, modified_data)

        with shelve.open(str(self.outputs["shared"].path)) as shared:
            shared["covariate_multipliers"] = covariate_multipliers
            shared["covariate_data_spec"] = covariate_data_spec

            for name in not_written:
                shared[name] = getattr(modified_data, name)


class FindSingleMAP(CascadeJob):
    """Run the fit without any pre-fit."""
    def __init__(self, recipe_id, local_settings, recipe_graph_neighbors, execution_context):
        super().__init__("find_single_maximum", recipe_id, local_settings, execution_context)
        self.inputs.update(dict(
            input_data=PandasFile(execution_context, "globaldata.hdf", 0, "both"),
        ))
        parent_location_id = local_settings.parent_location_id
        estimation_parent = [
            predecessor for predecessor in recipe_graph_neighbors["predecessors"]
            if predecessor.recipe == "estimate_location"
        ]
        if estimation_parent:
            grandparent_location = estimation_parent[0].location_id
            grandparent_sex = estimation_parent[0].sex
            self.inputs["grandparent"] = PandasFile(
                execution_context, "summary.hdf", grandparent_location, grandparent_sex)
        self.outputs["db_file"] = DbFile(
            execution_context, "fit.db", parent_location_id, recipe_id.sex
        )


class FindFixedMAP(CascadeJob):
    """Do a "fit fixed" which will precede the "fit both"."""
    def __init__(self, recipe_id, local_settings, neighbors, execution_context):
        super().__init__("find_maximum_fixed", recipe_id, local_settings, execution_context)
        self.inputs["input_data"] = PandasFile(execution_context, "globaldata.hdf", 0, "both")
        parent_location_id = local_settings.parent_location_id
        estimation_parent = [
            predecessor for predecessor in neighbors["predecessors"]
            if predecessor.recipe == "estimate_location"
        ]
        if estimation_parent:
            grandparent_location = estimation_parent[0].location_id
            grandparent_sex = estimation_parent[0].sex
            self.inputs["grandparent"] = PandasFile(
                execution_context, "summary.hdf", grandparent_location, grandparent_sex
            )
        self.outputs["db_file"] = DbFile(execution_context, "fit.db", parent_location_id, recipe_id.sex)


class FindBothMAP(CascadeJob):
    """Do a "fit both" assuming a "fit fixed" was done first."""
    def __init__(self, recipe_id, local_settings, execution_context):
        super().__init__("find_maximum_both", recipe_id, local_settings, execution_context)
        parent_location_id = local_settings.parent_location_id
        self.inputs.update(dict(
            input_data=PandasFile(execution_context, "globaldata.hdf", 0, "both"),
            db_file=DbFile(execution_context, "fit.db", parent_location_id, recipe_id.sex),
        ))
        self.outputs.update(dict(
            db_file=DbFile(execution_context, "fit.db", parent_location_id, recipe_id.sex),
        ))


class ConstructDraw(CascadeJob):
    """This one job has a task to do a fit for each draw."""
    def __init__(self, recipe_id, local_settings, execution_context):
        super().__init__("draw", recipe_id, local_settings, execution_context)
        parent_location_id = local_settings.parent_location_id
        self.inputs.update(dict(
            db_file=DbFile(execution_context, "fit.db", parent_location_id, recipe_id.sex),
        ))
        draw_cnt = local_settings.number_of_fixed_effect_samples
        for draw_idx in range(draw_cnt):
            draw_file = DbFile(execution_context, f"draw{draw_idx}.db", parent_location_id, recipe_id.sex)
            self.outputs[f"db_file{draw_idx}"] = draw_file


class Summarize(CascadeJob):
    """Gather results of draws."""
    def __init__(self, recipe_id, local_settings, neighbors, execution_context):
        super().__init__("summarize", recipe_id, local_settings, execution_context)
        parent_location_id = local_settings.parent_location_id
        draw_cnt = local_settings.number_of_fixed_effect_samples
        self.inputs.update(dict(
            input_data=PandasFile(execution_context, "globaldata.hdf", 0, "both"),
            db_file=DbFile(execution_context, "fit.db", parent_location_id, recipe_id.sex),
        ))
        for draw_idx in range(draw_cnt):
            draw_file = DbFile(execution_context, f"draw{draw_idx}.db", parent_location_id, recipe_id.sex)
            self.inputs[f"db_file{draw_idx}"] = draw_file
        self.outputs.update(dict(
            summary=PandasFile(execution_context, "summary.hdf", parent_location_id, recipe_id.sex)
        ))


class EstimateLocationPrepareData(CascadeJob):
    def __call__(self, execution_context):
        """
        Estimates rates for a single location in the location hierarchy.
        This does multiple fits and predictions in order to estimate uncertainty.

        Args:
            execution_context: Describes environment for this process.
            local_settings: A dictionary describing the work to do. This has
                a location ID corresponding to the location for this fit.
        """
        local_settings = self.local_settings
        covariate_multipliers, covariate_data_spec = create_covariate_specifications(
            local_settings.settings.country_covariate, local_settings.settings.study_covariate
        )
        shared = shelve.open(str(self.outputs["shared"].path(execution_context)))
        shared["covariate_multipliers"] = covariate_multipliers
        shared["covariate_data_spec"] = covariate_data_spec
        input_data = retrieve_data(execution_context, local_settings, covariate_data_spec)
        columns_wrong = validate_input_data_types(input_data)
        assert not columns_wrong, f"validation failed {columns_wrong}"
        grandparent = shelve.open(str(self.inputs["grandparent_shared"].path(execution_context)))
        modified_data = modify_input_data(input_data, local_settings, covariate_data_spec, grandparent)
        shared["prepared_input_data"] = modified_data


class EstimateLocationConstructModel(CascadeJob):
    def __call__(self, execution_context, local_settings, local_cache):
        covariate_multipliers = local_cache.get("covariate_multipliers:{local_settings.parent_location_id}")
        covariate_data_spec = local_cache.get("covariate_data_spec:{local_settings.parent_location_id}")
        modified_data = local_cache.get("prepared_input_data:{local_settings.parent_location_id}")
        model = construct_model(modified_data, local_settings, covariate_multipliers,
                                covariate_data_spec)
        set_priors_from_parent_draws(model, modified_data.draws)
        local_cache.set("prepared_model:{local_settings.parent_location_id}", model)


class EstimateLocationInitialGuess(CascadeJob):
    def __call__(self, execution_context, local_settings, local_cache):
        model = local_cache.get("prepared_model:{local_settings.parent_location_id}")
        input_data = local_cache.get("prepared_input_data:{local_settings.parent_location_id}")
        initial_guess = local_cache.get("parent_initial_guess:{local_settings.parent_location_id}")
        fit_result = compute_parent_fit_fixed(execution_context, local_settings, input_data, model, initial_guess)
        local_cache.set("parent_initial_guess:{local_settings.parent_location_id}", fit_result.fit)


class EstimateLocationInitialFit(CascadeJob):
    def __call__(self, execution_context, local_settings, local_cache):
        model = local_cache.get("prepared_model:{local_settings.parent_location_id}")
        input_data = local_cache.get("prepared_input_data:{local_settings.parent_location_id}")
        initial_guess = local_cache.get("parent_initial_guess:{local_settings.parent_location_id}")
        fit_result = compute_parent_fit(execution_context, local_settings, input_data, model, initial_guess)
        local_cache.set("parent_fit:{local_settings.parent_location_id}", fit_result)


class EstimateLocationComputeDraws(CascadeJob):
    def __call__(self, execution_context, local_settings, local_cache):
        model = local_cache.get("prepared_model:{local_settings.parent_location_id}")
        fit_result = local_cache.get("parent_fit:{local_settings.parent_location_id}")
        input_data = local_cache.get("prepared_input_data:{local_settings.parent_location_id}")
        draws = make_draws(
            execution_context,
            model,
            input_data,
            fit_result.fit,
            local_settings,
            execution_context.parameters.num_processes
        )
        if draws:
            draws, predictions = zip(*draws)
            local_cache.set(f"fit-draws:{local_settings.parent_location_id}", draws)
            local_cache.set(f"fit-predictions:{local_settings.parent_location_id}", predictions)
        else:
            raise DismodATException("Fit failed for all samples")


class EstimateLocationSavePredictions(CascadeJob):
    def __call__(self, execution_context, local_settings, local_cache):
        if not local_settings.run.no_upload:
            predictions = local_cache.get(f"fit-predictions:{local_settings.parent_location_id}")
            fit_result = local_cache.get("parent_fit:{local_settings.parent_location_id}")
            save_outputs(fit_result, predictions, execution_context, local_settings)


def recipe_to_jobs(recipe_identifier, local_settings, neighbors, execution_context):
    """Given a recipe, return a list of jobs that must be done in order.

    Args:
        recipe_identifier (RecipeIdentifier): A data struct that specifies
            what a modeler thinks of as one estimation.
        local_settings (Namespace|SimpleNamespace): These are settings that
            have been localized to apply to a particular location.

    Returns:
        List[Job]: A list of jobs to run in order. Could make it a graph,
        but that's unnecessary.
    """
    sub_jobs = list()
    if recipe_identifier.recipe == "bundle_setup":
        bundle_setup = GlobalPrepareData(
            recipe_identifier, local_settings, execution_context
        )
        sub_jobs.append(bundle_setup)
    elif recipe_identifier.recipe == "estimate_location":
        if local_settings.policies.fit_strategy == "fit_fixed_then_fit":
            sub_jobs.append(
                FindFixedMAP(
                    recipe_identifier, local_settings, neighbors, execution_context
                ))
            sub_jobs.append(
                FindBothMAP(
                    recipe_identifier, local_settings, execution_context
                ))
        else:
            sub_jobs.append(
                FindSingleMAP(
                    recipe_identifier, local_settings, neighbors, execution_context
                ))
        sub_jobs.extend([
            ConstructDraw(recipe_identifier, local_settings, execution_context),
            Summarize(recipe_identifier, local_settings, neighbors, execution_context),
        ])
    else:
        raise RuntimeError(f"Unknown recipe identifier {recipe_identifier}")
    return sub_jobs


def job_graph_from_settings(locations, settings, args, execution_context):
    recipe_graph = recipe_graph_from_settings(locations, settings, args)
    add_job_list(recipe_graph, execution_context)
    return recipe_graph_to_job_graph(recipe_graph)


def add_job_list(recipe_graph, execution_context):
    for node in recipe_graph:
        predecessors = recipe_graph.predecessors(node)
        successors = recipe_graph.successors(node)
        neighbors = dict(predecessors=predecessors, successors=successors)
        jobs = recipe_to_jobs(
            node, recipe_graph.nodes[node]["local_settings"], neighbors, execution_context
        )
        recipe_graph.nodes[node]["job_list"] = jobs
