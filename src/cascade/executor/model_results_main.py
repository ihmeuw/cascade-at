"""This module aids in comparing results from Dismod AT and Dismod ODE.
"""
from argparse import ArgumentParser
import itertools as it
import logging
import os
from pathlib import Path
import sys

import pandas as pd

from cascade.core.db import connection


CODELOG = logging.getLogger(__name__)


dbs = {"epi-dev": "epi-test",
       "epi-prod": "epi",
       "at-dev": "dismod-at-dev",
       "at-prod": "dismod-at-prod"}

dbs_easy_names = {"epi-test": "epi-dev",
                  "epi": "epi-prod",
                  "dismod-at-dev": "at-dev",
                  "dismod-at-prod": "at-prod"}

tables = {"fit": "epi.model_estimate_fit", "final": "epi.model_estimate_final"}

tables_easy_names = {"epi.model_estimate_fit": "fit", "epi.model_estimate_final": "final"}


def _retrieve_from_database(model_version_id, db, table):
    """Connect to the database and retrieve the results.

    Args:
        db (str): nickname of the IHME database, as used in a .odbc.ini file
        table (str): name of the table in the database

    Returns:
        pandas.DataFrame() of the retrieved results
    """

    query = f"""
    SELECT * FROM {table}
    WHERE
        model_version_id = %(model_version_id)s
    """

    with connection(database=db) as c:
        model_results_data = pd.read_sql(query, c, params={"model_version_id": model_version_id})
        CODELOG.debug(f"""Downloaded {len(model_results_data)} lines of
                      data for model_version_id {model_version_id} from '{db}'""")

    return model_results_data


def _get_model_results(model_version_id, db, table):
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

    if db and table:
        if db in dbs.keys() and table in tables.keys():
            model_results_data = _retrieve_from_database(model_version_id, dbs[db], tables[table])
            return model_results_data
        else:
            sys.exit(f"""Unknown db and table combination.  Acceptable databases are
                     {dbs.keys()} and acceptable tables are {tables.keys()}""")
    else:
        # check all, if find more than one report it, else return one
        model_results = None
        model_results_empty = None
        model_results_db_and_table = []
        for db, table in it.product(dbs.values(), tables.values()):
            model_results_retrieved = _retrieve_from_database(model_version_id, db, table)
            if not model_results_retrieved.empty:
                model_results_db_and_table.append((dbs_easy_names[db], tables_easy_names[table]))
                model_results = model_results_retrieved
            else:
                model_results_empty = model_results_retrieved
        if len(model_results_db_and_table) > 1:
            # exit and report
            sys.exit(f"""Found {model_version_id} in multiple (database, table) locations
                     {model_results_db_and_table}
                     Please identify the one you want and try again.""")
        elif len(model_results_db_and_table) == 1:
            return model_results[0]
        else:
            # returns an empty database, if none of the db/table combinations contains the model_version_id
            return model_results_empty


def _write_model_results(model_results, model_version_id, model_type, output_dir):
    """ Writes the model_results dataframe as a csv file to the output dir.
    """
    file_name = Path(output_dir) / f"{model_type.lower()}_{model_version_id}.csv"
    model_results.to_csv(file_name, index=False)


def entry():
    """This is the entry that setuptools turns into an installed program.

    If the user does not provide an at_mvid, an empty dataframe is written as at_None.csv;
    similarly if no ode_mvid is provided, an empty ode_None.csv is written.

    If no values are provided for db and table, a selected set of db's and tables are checked.  If
    the data is found in more than one place, the multiple locations are reported, no data is written for
    that mvid, and the program exits.
    """
    parser = ArgumentParser("Writes two csv files, one for Dismod AT results and one for Dismod ODE results.")
    parser.add_argument("--at-mvid", help="model_version_id for AT results")
    parser.add_argument("--ode-mvid", help="model_version_id for ODE results")
    parser.add_argument("--at-db", help="db name for AT results")
    parser.add_argument("--ode-db", help="db name for ODE results")
    parser.add_argument("--at-table", help="db table for AT results")
    parser.add_argument("--ode-table", help="db table for ODE results")
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

    at_results = _get_model_results(args.at_mvid, args.at_db, args.at_table)
    ode_results = _get_model_results(args.ode_mvid, args.ode_db, args.ode_table)

    _write_model_results(at_results, args.at_mvid, "AT", args.output_dir)
    _write_model_results(ode_results, args.ode_mvid, "ODE", args.output_dir)


if __name__ == "__main__":
    entry()
