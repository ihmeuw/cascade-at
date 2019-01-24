"""
Entry point for running a the work of a single location in an EpiViz-AT cascade.
"""
from types import SimpleNamespace


def main(model_identifier, cascade_task_identifier):
    execution_context = get_execution_context()
    settings = get_settings(model_identifier)
    plan = CascadePlan(settings)
    this_location_work = plan.work_for(cascade_task_identifier)

    executor = DismodelExecutor(execution_context, this_location_work)
    executor.run()


def entry():
    # parse arguments, set up logging, and report exceptions.
    args = SimpleNamespace()
    main(args.mvid, args.task)


if __name__ == "__main__":
    entry()
