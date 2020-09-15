"""
The results of a Cascade-AT model need to be saved to the IHME epi databases.
This module wrangles the draw files from a completed model and uploads summaries
to the epi databases for visualization in EpiViz.

Eventually, this module should be replaced by something like ``save_results_at``.
"""

import os
from pathlib import Path
import pandas as pd
from typing import List

from cascade_at.core.db import db_tools
from cascade_at.core.log import get_loggers
from cascade_at.core import CascadeATError
from cascade_at.dismod.api.dismod_extractor import ExtractorCols

LOG = get_loggers(__name__)


VALID_TABLES = [
    'model_estimate_final',
    'model_estimate_fit',
    'model_prior'
]


class UiCols:
    MEAN = 'mean'
    LOWER = 'lower'
    UPPER = 'upper'
    LOWER_QUANTILE = 0.025
    UPPER_QUANTILE = 0.975


class ResultsError(CascadeATError):
    """Raised when there is an error with uploading or validating the results."""
    pass


class ResultsHandler:
    """
    Handles all of the DisMod-AT results including draw saving
    and uploading to the epi database.
    """
    def __init__(self):
        """
        Attributes
        ----------
        self.draw_keys
            The keys of the draw data frames
        self.summary_cols
            The columns that need to be present in all summary files
        """
        self.draw_keys: List[str] = ['measure_id', 'year_id', 'age_group_id',
                                     'location_id', 'sex_id', 'model_version_id']
        self.summary_cols: List[str] = [UiCols.MEAN, UiCols.LOWER, UiCols.UPPER]

    def _validate_results(self, df: pd.DataFrame) -> None:
        """
        Validates the input draw files. Put any additional
        validations here.

        Parameters
        ----------
        df
            An input data frame with draws
        """
        missing_cols = [x for x in self.draw_keys if x not in df.columns]
        if missing_cols:
            raise ResultsError(f"Missing id columns {missing_cols} for saving the results.")

    def _validate_summaries(self, df: pd.DataFrame) -> None:
        missing_cols = [x for x in self.summary_cols if x not in df.columns]
        if missing_cols:
            raise ResultsError(f"Missing summary columns {missing_cols} for saving the results.")

    def summarize_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarizes results from either mean or draw cols to get
        mean, upper, and lower cols.

        Parameters
        ----------
        df
            A data frame with draw columns or just a mean column
        """
        if ExtractorCols.VALUE_COL_FIT in df.columns:
            df[UiCols.MEAN] = df[ExtractorCols.VALUE_COL_FIT]
            df[UiCols.LOWER] = df[ExtractorCols.VALUE_COL_FIT]
            df[UiCols.UPPER] = df[ExtractorCols.VALUE_COL_FIT]
        else:
            draw_cols = [col for col in df.columns if col.startswith(ExtractorCols.VALUE_COL_SAMPLES)]
            df[UiCols.MEAN] = df[draw_cols].mean(axis=1)
            df[UiCols.LOWER] = df[draw_cols].quantile(q=UiCols.LOWER_QUANTILE, axis=1)
            df[UiCols.UPPER] = df[draw_cols].quantile(q=UiCols.UPPER_QUANTILE, axis=1)

        return df[self.draw_keys + [UiCols.MEAN, UiCols.LOWER, UiCols.UPPER]]

    def save_draw_files(self, df: pd.DataFrame, model_version_id: int,
                        directory: Path, add_summaries: bool) -> None:
        """
        Saves a data frame by location and sex in .csv files.
        This currently saves the summaries, but when we get
        save_results working it will save draws and then
        summaries as part of that.

        Parameters
        ----------
        df
            Data frame with the following columns:
                ['location_id', 'year_id', 'age_group_id', 'sex_id',
                'measure_id', 'mean' OR 'draw']
        model_version_id
            The model version to attach to the data
        directory
            Path to save the files to
        add_summaries
            Save an additional file with summaries to upload
        """
        LOG.info(f"Saving results to {directory.absolute()}")

        df['model_version_id'] = model_version_id
        self._validate_results(df=df)

        for loc in df.location_id.unique().tolist():
            os.makedirs(str(directory / str(loc)), exist_ok=True)
            for sex in df.sex_id.unique().tolist():
                subset = df.loc[
                    (df.location_id == loc) &
                    (df.sex_id == sex)
                ].copy()
                subset.to_csv(directory / str(loc) / f'{loc}_{sex}.csv')
                if add_summaries:
                    summary = self.summarize_results(df=subset)
                    self.save_summary_files(
                        df=summary, model_version_id=model_version_id, directory=directory
                    )

    def save_summary_files(self, df: pd.DataFrame, model_version_id: int, directory: Path) -> None:
        """
        Saves a data frame with summaries by location and sex in summary.csv files.

        Parameters
        ----------
        df
            Data frame with the following columns:
                ['location_id', 'year_id', 'age_group_id', 'sex_id',
                'measure_id', 'mean', 'lower', and 'upper']
        model_version_id
            The model version to attach to the data
        directory
            Path to save the files to
        """
        LOG.info(f"Saving results to {directory.absolute()}")

        df['model_version_id'] = model_version_id
        self._validate_results(df=df)
        self._validate_summaries(df=df)

        for loc in df.location_id.unique().tolist():
            os.makedirs(str(directory / str(loc)), exist_ok=True)
            for sex in df.sex_id.unique().tolist():
                subset = df.loc[
                    (df.location_id == loc) &
                    (df.sex_id == sex)
                    ].copy()
                subset.to_csv(directory / str(loc) / f'{loc}_{sex}_summary.csv')

    @staticmethod
    def upload_summaries(directory: Path, conn_def: str, table: str) -> None:
        """
        Uploads results from a directory to the model_estimate_final
        table in the Epi database specified by the conn_def argument.

        In the future, this will probably be replaced by save_results_dismod
        but we don't have draws to work with so we're just uploading summaries
        for now directly.

        Parameters
        ----------
        directory
            Directory where files are saved
        conn_def
            Connection to a database to be used with db_tools.ezfuncs
        table
            which table to upload to
        """
        if table not in VALID_TABLES:
            raise ResultsError("Don't know how to upload to table "
                               f"{table}. Valid tables are {VALID_TABLES}.")

        session = db_tools.ezfuncs.get_session(conn_def=conn_def)
        loader = db_tools.loaders.Infiles(table=table, schema='epi', session=session)

        generic_file = (directory / '*' / '*summary.csv').absolute()
        LOG.info(f"Loading all files to {conn_def} that match {generic_file} glob.")
        loader.indir(path=str(generic_file), commit=True, with_replace=True)
