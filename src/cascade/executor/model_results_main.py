"""This module aids in comparing results from Dismod AT and Dismod ODE.
"""
from argparse import ArgumentParser
import logging
import os
from pathlib import Path
import sys

import pandas as pd

from cascade.core.db import connection


CODELOG = logging.getLogger(__name__)


def _get_model_results(model_version_id, model_type):
    """Downloads the model results data for a Dismod model of type AT or ODE.
    The data lives in the epi.model_estimate_fit table which has columns:
    model_version_id, year_id, location_id, sex_id, age_group_id, measure_id,
    mean, upper, lower

    If saves_results data does not exist for model_version_id, an empty dataframe
    is returned.  This includes the case of a user supplying text, such as 'three',
    as a model_version_id.

    Args:
        model_version_id (int): identifies the data to retrieve
        model_type (str): identifies the Dismod model type, AT or ODE

    Returns:
        pandas.DataFrame: containing all columns in the "save results" upload table
    """

    if model_type.upper() == "AT":
        database = "dismod-at-dev"
    elif model_type.upper() == "ODE":
        database = "epi"
    else:
        raise ValueError(f"model must be of type 'AT' or 'ODE', not {model_type}")

    table = "epi.model_estimate_fit"

    query = f"""
    SELECT * FROM {table}
    WHERE
        model_version_id = %(model_version_id)s
    """

    with connection(database=database) as c:
        model_results_data = pd.read_sql(query, c, params={"model_version_id": model_version_id})
        CODELOG.debug(f"""Downloaded {len(model_results_data)} lines of dismod {model_type}
                      data for model_version_id {model_version_id} from '{database}'""")

    return model_results_data


def _write_model_results(model_results, model_version_id, model_type, output_dir):
    """ Writes the model_results dataframe as a csv file to the output dir.
    """
    file_name = Path(output_dir) / f"{model_type.lower()}_{model_version_id}.csv"
    model_results.to_csv(file_name, index=False)


def entry():
    """This is the entry that setuptools turns into an installed program.

    If the user does not provide an at_mvid, an empty dataframe is written as at_None.csv;
    similarly if no ode_mvid is provided, an empty ode_None.csv is written.
    """
    parser = ArgumentParser("Writes two csv files, one for Dismod AT results and one for Dismod ODE results.")
    parser.add_argument("--at-mvid", help="model_version_id for AT results")
    parser.add_argument("--ode-mvid", help="model_version_id for ODE results")
    parser.add_argument("--output-dir", default=".", help="output directory for csv files")
    parser.add_argument("-v", help="increase debugging verbosity", action="store_true")
    args, _ = parser.parse_known_args()
    if args.v:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)

    # check if output_directory exists
    if not os.path.isdir(args.output_dir):
        sys.exit(f"""Error: Directory provided: {args.output_dir} does not exist.
                            Please create first, or check the spelling.""")

    at_results = _get_model_results(args.at_mvid, "AT")
    ode_results = _get_model_results(args.ode_mvid, "ODE")

    _write_model_results(at_results, args.at_mvid, "AT", args.output_dir)
    _write_model_results(ode_results, args.ode_mvid, "ODE", args.output_dir)


if __name__ == "__main__":
    entry()
