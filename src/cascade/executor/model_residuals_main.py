"""This module makes Dismod AT model residuals available in a file.
"""
import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from cascade.core.log import getLoggers
from cascade.dismod.db.wrapper import DismodFile, get_engine
from cascade.saver.model_output import write_model_results

CODELOG = getLoggers(__name__)


def _get_residuals(dm_file):
    """Retrieve the residual values from the dismod output tables in the dismod file.

    Args:
        dm_file (DismodFile): a db file imported into a DismodFile object

    Results:
        tuple of 2 pandas.DataFrame's, one for fit_var table, one for fit_data_subset table

    """
    # get fit_var table from dm_file
    # fit_var columns: fit_var_id, variable_value, residual_value, residual_dage,
    # residual_dtime, lagrange_value, lagrange_dage, lagrange_dtime
    fv_residuals = dm_file.fit_var

    # get fit_data_subset table from dm_file
    # fit_data_subset columns: fit_data_subset_id, avg_integrand, weighted_residual
    fds_residuals = dm_file.fit_data_subset

    return fv_residuals, fds_residuals


def entry():
    """This is the entry that setuptools turns into an installed program.
    """
    parser = ArgumentParser(f"""Writes two csv files, for Dismod AT residuals from
                            the dismod file's fit_var table and fit_data_subset table.""")
    parser.add_argument("--mvid", default="mvid", help="model version id for the dismod file")
    parser.add_argument("--dm-file", default=".", help="the Dismod db file containing the residuals")
    parser.add_argument("--output-dir", default=".", help="output directory for csv files")
    parser.add_argument("-v", help="increase debugging verbosity", action="store_true")
    args, _ = parser.parse_known_args()
    if args.v:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)

    # check if dismod file exists
    if not os.path.isfile(args.dm_file):
        sys.exit(f"""Error: File provided: {args.dm_file} does not exist.
                            Please locate the file, or check the spelling.""")

    # check if output_directory exists
    if not os.path.isdir(args.output_dir):
        sys.exit(f"""Error: Directory provided: {args.output_dir} does not exist.
                            Please create first, or check the spelling.""")

    model_version_id = args.mvid

    try:

        dm_file = DismodFile(get_engine(Path(args.dm_file)))

        fv_residuals, fds_residuals = _get_residuals(dm_file)

        write_model_results(fv_residuals, model_version_id, "resids_fv", args.output_dir)
        write_model_results(fds_residuals, model_version_id, "resids_fds", args.output_dir)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    entry()
