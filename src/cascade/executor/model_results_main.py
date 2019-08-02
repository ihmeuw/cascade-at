"""
This module aids in comparing results from Dismod AT and Dismod ODE.
"""

import logging
import os
import sys
from argparse import ArgumentParser
from itertools import product

import pandas as pd

from cascade.core.db import connection
from cascade.saver.model_output import write_model_results

CODELOG = logging.getLogger(__name__)


DATABASES = ["epi", "epi-test", "dismod-at-dev", "dismod-at-prod"]

TABLES = {"fit": "epi.model_estimate_fit", "final": "epi.model_estimate_final"}

TABLES_EASY_NAMES = {"epi.model_estimate_fit": "fit", "epi.model_estimate_final": "final"}


def _retrieve_from_database(model_version_id, db, table):
    """Connect to the database and retrieve the results from the table.

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
    The data lives in the model_estimate_fit or model_estimate_final table which has columns:
    model_version_id, year_id, location_id, sex_id, age_group_id, measure_id,
    mean, upper, lower

    If saves_results data does not exist for model_version_id, the program exits and reports this.
    This includes the case of a user supplying text, such as 'three', as a model_version_id.

    Args:
        model_version_id (int): identifies the data to retrieve
        db (str): nickname of the IHME database, as used in a .odbc.ini file
        table (str): as easy name of the table for modelers, not necessarily the same as
            the name of the table used in the database

    Returns:
        pandas.DataFrame: containing all columns in the "save results" upload table
    """

    if db and table:
        # modeler provides a db and table to check for this model_version_id
        model_results_data = _retrieve_from_database(model_version_id, db, TABLES[table])
        if not model_results_data.empty:
            return model_results_data
        else:
            msg = f"""No data exists for {model_version_id} in database {db} and table {table}."""
            raise ValueError(msg)
    else:
        # check all, if find more than one report it, else return one, or exit if none are found
        model_results = []
        model_results_db_and_table = []
        for db, table in product(DATABASES, TABLES.values()):
            model_results_retrieved = _retrieve_from_database(model_version_id, db, table)
            if not model_results_retrieved.empty:
                model_results_db_and_table.append((db, TABLES_EASY_NAMES[table]))
                model_results.append(model_results_retrieved)
            # else:
                # model_version_id not in this db/table
        if len(model_results_db_and_table) > 1:
            # exit and report
            msg = f"""Found {model_version_id} in multiple (database, table) locations
                     {model_results_db_and_table}
                     Please identify the one you want and try again."""
            raise ValueError(msg)
        elif len(model_results) == 1:
            return model_results[0]
        else:
            # exit, and report that no data was found
            msg = f"""Model mvid={model_version_id} was not found in any of these tables
                  {list(TABLES.keys())}
                  in these databases
                  {DATABASES}."""
            raise ValueError(msg)


def entry():
    """This is the entry that setuptools turns into an installed program.

    If the user does not provide a model_version_id for at-mvid and/or ode-mvid, the program exits and
    reports that no data was found.

    If no values are provided for db and table, a selected set of db's and tables are checked.  If
    the data is found in more than one place, the multiple locations are reported, no data is written for
    that mvid, and the program exits.
    """
    parser = ArgumentParser("Writes two csv files, one for Dismod AT results and one for Dismod ODE results.")
    parser.add_argument("--at-mvid", help="model_version_id for AT results")
    parser.add_argument("--ode-mvid", help="model_version_id for ODE results")
    parser.add_argument("--at-db", help="db name for AT results", choices=DATABASES)
    parser.add_argument("--ode-db", help="db name for ODE results", choices=DATABASES)
    parser.add_argument("--at-table", help="db table for AT results", choices=TABLES.keys())
    parser.add_argument("--ode-table", help="db table for ODE results", choices=TABLES.keys())
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

    try:

        at_results = _get_model_results(args.at_mvid, args.at_db, args.at_table)
        write_model_results(at_results, args.at_mvid, "AT", args.output_dir)

    except ValueError as ve:
        print(ve)

    try:

        ode_results = _get_model_results(args.ode_mvid, args.ode_db, args.ode_table)
        write_model_results(ode_results, args.ode_mvid, "ODE", args.output_dir)

    except ValueError as ve:
        print(ve)


if __name__ == "__main__":
    entry()
