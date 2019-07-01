from cascade.core import getLoggers
from cascade.dismod import DismodATException
from cascade.executor.cascade_plan import recipe_graph_from_settings
from cascade.executor.construct_model import construct_model
from cascade.executor.covariate_description import create_covariate_specifications
from cascade.executor.estimate_location import retrieve_data, modify_input_data, compute_parent_fit_fixed, \
    compute_parent_fit, make_draws, save_outputs
from cascade.executor.priors_from_draws import set_priors_from_parent_draws
from cascade.input_data.configuration.raw_input import validate_input_data_types
from cascade.runner.job_graph import Job, recipe_graph_to_job_graph

CODELOG, MATHLOG = getLoggers(__name__)


class EstimateLocationPrepareData(Job):
    def __call__(self, execution_context, local_settings, local_cache):
        """
        Estimates rates for a single location in the location hierarchy.
        This does multiple fits and predictions in order to estimate uncertainty.

        Args:
            execution_context: Describes environment for this process.
            local_settings: A dictionary describing the work to do. This has
                a location ID corresponding to the location for this fit.
        """
        covariate_multipliers, covariate_data_spec = create_covariate_specifications(
            local_settings.settings.country_covariate, local_settings.settings.study_covariate
        )
        local_cache.set("covariate_multipliers:{local_settings.parent_location_id}", covariate_multipliers)
        local_cache.set("covariate_data_spec:{local_settings.parent_location_id}", covariate_data_spec)
        input_data = retrieve_data(execution_context, local_settings, covariate_data_spec, local_cache)
        columns_wrong = validate_input_data_types(input_data)
        assert not columns_wrong, f"validation failed {columns_wrong}"
        modified_data = modify_input_data(input_data, local_settings, covariate_data_spec)
        local_cache.set("prepared_input_data:{local_settings.parent_location_id}", modified_data)


class EstimateLocationConstructModel(Job):
    def __call__(self, execution_context, local_settings, local_cache):
        covariate_multipliers = local_cache.get("covariate_multipliers:{local_settings.parent_location_id}")
        covariate_data_spec = local_cache.get("covariate_data_spec:{local_settings.parent_location_id}")
        modified_data = local_cache.get("prepared_input_data:{local_settings.parent_location_id}")
        model = construct_model(modified_data, local_settings, covariate_multipliers,
                                covariate_data_spec)
        set_priors_from_parent_draws(model, modified_data.draws)
        local_cache.set("prepared_model:{local_settings.parent_location_id}", model)


class EstimateLocationInitialGuess(Job):
    def __call__(self, execution_context, local_settings, local_cache):
        model = local_cache.get("prepared_model:{local_settings.parent_location_id}")
        input_data = local_cache.get("prepared_input_data:{local_settings.parent_location_id}")
        initial_guess = local_cache.get("parent_initial_guess:{local_settings.parent_location_id}")
        fit_result = compute_parent_fit_fixed(execution_context, local_settings, input_data, model, initial_guess)
        local_cache.set("parent_initial_guess:{local_settings.parent_location_id}", fit_result.fit)


class EstimateLocationInitialFit(Job):
    def __call__(self, execution_context, local_settings, local_cache):
        model = local_cache.get("prepared_model:{local_settings.parent_location_id}")
        input_data = local_cache.get("prepared_input_data:{local_settings.parent_location_id}")
        initial_guess = local_cache.get("parent_initial_guess:{local_settings.parent_location_id}")
        fit_result = compute_parent_fit(execution_context, local_settings, input_data, model, initial_guess)
        local_cache.set("parent_fit:{local_settings.parent_location_id}", fit_result)


class EstimateLocationComputeDraws(Job):
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


class EstimateLocationSavePredictions(Job):
    def __call__(self, execution_context, local_settings, local_cache):
        if not local_settings.run.no_upload:
            predictions = local_cache.get(f"fit-predictions:{local_settings.parent_location_id}")
            fit_result = local_cache.get("parent_fit:{local_settings.parent_location_id}")
            save_outputs(fit_result, predictions, execution_context, local_settings)


def recipe_to_jobs(recipe_identifier, local_settings):
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
        bundle_setup = Job("bundle_setup", recipe_identifier, local_settings)
        sub_jobs.append(bundle_setup)
    elif recipe_identifier.recipe == "estimate_location":
        if local_settings.policies.fit_strategy == "fit_fixed_then_fit":
            sub_jobs.append(Job("fit_fixed_then_fit", recipe_identifier, local_settings))
        sub_jobs.extend([
            Job(job_name, recipe_identifier, local_settings)
            for job_name in [
                "compute_initial_fit",
                "compute_draws_from_parent_fit",
                "save_predictions"
            ]])
    else:
        raise RuntimeError(f"Unknown recipe identifier {recipe_identifier}")
    return sub_jobs


def job_graph_from_settings(locations, settings, args):
    recipe_graph = recipe_graph_from_settings(locations, settings, args)
    for node in recipe_graph:
        jobs = recipe_to_jobs(node, recipe_graph.nodes[node]["local_settings"])
        recipe_graph.nodes[node]["job_list"] = jobs
    return recipe_graph_to_job_graph(recipe_graph)
