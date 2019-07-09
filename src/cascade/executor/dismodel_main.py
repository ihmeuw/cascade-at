import json
import pickle
from pathlib import Path
from textwrap import fill

import networkx as nx

from cascade.core import getLoggers
from cascade.executor.execution_context import make_execution_context
from cascade.executor.job_definitions import job_graph_from_settings
from cascade.input_data.db.configuration import load_settings
from cascade.input_data.db.locations import location_hierarchy
from cascade.runner.entry import entry

CODELOG, MATHLOG = getLoggers(__name__)


class Application:
    """
    Responsible for management of settings and creation of job graphs.
    """
    def __init__(self):
        self.locations = None
        self.settings = None
        self.execution_context = None

    def add_arguments(self, parser):
        parser.add_argument(
            "--meid", type=int,
            help="Modelable entity ID. This identifies the cause of disease.",
        )
        parser.add_argument(
            "--mvid", type=int,
            help=("Model version ID. There are multiple model versions for "
                  "each model entity"),
        )
        context_parser = parser.add_argument_group(
            "context",
            "Settings to change the process directories and environment."
        )
        context_parser.add_argument(
            "--infrastructure", action="store_true",
            help=fill("Whether we are running as infrastructure component. "
                      "Controls whether this tries to store files in the "
                      "base directory or whether it works in the"
                      "current directory."),
        )
        context_parser.add_argument(
            "--base-directory", type=Path, default=".",
            help="Directory in which to find and store files.",
        )

        pdb_parser = parser.add_argument_group(
            "debug",
            "These affect how this executes, for debugging."
        )
        pdb_parser.add_argument(
            "--no-upload", action="store_true",
            help=fill("This turns off all writing to databases, so that a run "
                      "will not affect outputs"),
        )
        pdb_parser.add_argument(
            "--db-only", action="store_true",
            help=fill("Run until it creates a Dismod-AT db file, and then quit. "
                      "This may make less sense when talking about a large "
                      "graph of computations."),
        )

        data_parser = parser.add_argument_group(
            "data",
            "Parsers to change what data is used for input."
        )
        data_parser.add_argument(
            "--settings-file", type=Path,
            help="Read settings from this file.",
        )
        data_parser.add_argument(
            "-b", "--bundle-file", type=Path,
            help=fill("The bundle normally comes from the databases but this "
                      "lets you specify a particular file as input. If this "
                      "isn't given, it will look for the bundle in the db or "
                      "in a known input directory."),
        )
        data_parser.add_argument(
            "-s", "--bundle-study-covariates-file", type=Path,
            help=fill("Read the study covariates from a separate file. "
                      "If this is not specified, then they are read from "
                      "either the databases or from the known file location."),
        )

        graph_parser = parser.add_argument_group(
            "graph",
            "Changes to the graph of work that is done."
        )
        graph_parser.add_argument(
            "--skip-cache", action="store_true",
            help=fill("Don't save bundle data to tier 3. Instead, read "
                      "it directly from tier 2."),
        )
        graph_parser.add_argument("--num-samples", type=int, help="Override number of samples.")

        sub_graph = parser.add_argument_group(
            "sub_graph",
            "These arguments select a subset of nodes to run."
        )
        sub_graph.add_argument("--location-id", type=int, help="location ID for this work")
        sub_graph.add_argument("--sex", type=str, help="sex as male, female, both")
        sub_graph.add_argument("--recipe", type=str, help="name of the recipe")
        sub_graph.add_argument("--name", type=str, help="job within the recipe")
        return parser

    def create_settings(self, args):
        # We need a sort-of-correct execution context when we first load
        # and then it gets refined after settings are loaded.
        execution_context = make_execution_context(
            gbd_round_id=6, num_processes=args.num_processes
        )
        self.settings = load_settings(
            execution_context, args.meid, args.mvid, args.settings_file)
        self.locations = location_hierarchy(
            location_set_version_id=self.settings.location_set_version_id,
            gbd_round_id=self.settings.gbd_round_id
        )
        configure_execution_context(execution_context, args, self.settings)
        self.execution_context = execution_context

    def load_settings(self, args):
        # The execution context isn't part of the settings, so it is
        # rebuilt here when settings are loaded.
        execution_context = make_execution_context(
            gbd_round_id=6, num_processes=args.num_processes
        )
        base = execution_context.model_base_directory(0)
        self.settings = json.load(base / "settings.json")
        self.locations = pickle.load((base / "locations.pickle").open("rb"))
        configure_execution_context(execution_context, args, self.settings)
        self.execution_context = execution_context

    def save_settings(self):
        base = self.execution_context.model_base_directory(0)
        json.dump(self.settings, base / "settings.json", indent=4)
        pickle.dump(self.locations, (base / "locations.pickle").open("wb"))

    def graph_of_jobs(self, args):
        return job_graph_from_settings(self.locations, self.settings, args)

    def sub_graph_to_run(self, args):
        job_graph = self.graph_of_jobs(args)
        nodes = job_graph.nodes

        for search in ["location_id", "recipe", "sex", "name"]:
            if search in args:
                nodes = [n for n in nodes if getattr(n, search) == getattr(args, search)]

        sub_graph = nx.subgraph(job_graph, nodes)
        sub_graph.graph["execution_context"] = self.execution_context
        return sub_graph


def configure_execution_context(execution_context, args, settings):
    if args.infrastructure:
        execution_context.parameters.organizational_mode = "infrastructure"
    else:
        execution_context.parameters.organizational_mode = "local"

    execution_context.parameters.base_directory = args.base_directory

    for param in ["modelable_entity_id", "model_version_id"]:
        setattr(execution_context.parameters, param, getattr(settings.model, param))


if __name__ == "__main__":
    app = Application()
    entry(app)
