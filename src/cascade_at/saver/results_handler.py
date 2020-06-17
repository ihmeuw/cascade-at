import os
from pathlib import Path
import pandas as pd

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
        self.draw_keys = ['measure_id', 'year_id', 'age_group_id',
                          'location_id', 'sex_id', 'model_version_id']

    def _validate_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the input draw files. Put any additional
        validations here.

        Parameters
        ----------
        df
        """
        missing_cols = [x for x in self.draw_keys if x not in df.columns]
        if missing_cols:
            raise ResultsError(f"Missing id columns {missing_cols} for saving the results.")
        return df

    def summarize_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarizes results from either mean or draw cols to get
        mean, upper, and lower cols.
        """
        if ExtractorCols.VALUE_COL_FIT in df.columns:
            df[UiCols.MEAN] = df[ExtractorCols.VALUE_COL_FIT]
            df[UiCols.LOWER] = df[ExtractorCols.VALUE_COL_FIT]
            df[UiCols.UPPER] = df[ExtractorCols.VALUE_COL_FIT]
        else:
            DRAW_COLS = [col for col in df.columns if col.startswith(ExtractorCols.VALUE_COL_SAMPLES)]
            df[UiCols.MEAN] = df[DRAW_COLS].mean(axis=1)
            df[UiCols.LOWER] = df[DRAW_COLS].quantile(q=UiCols.LOWER_QUANTILE, axis=1)
            df[UiCols.UPPER] = df[DRAW_COLS].quantile(q=UiCols.UPPER_QUANTILE, axis=1)

        return df[self.draw_keys + [UiCols.MEAN, UiCols.LOWER, UiCols.UPPER]]

    def save_draw_files(self, df: pd.DataFrame, model_version_id: int,
                        directory: Path, add_summaries: bool):
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
        validated_df = self._validate_results(df=df)

        for loc in validated_df.location_id.unique().tolist():
            os.makedirs(str(directory / str(loc)), exist_ok=True)
            for sex in validated_df.sex_id.unique().tolist():
                subset = validated_df.loc[
                    (validated_df.location_id == loc) &
                    (validated_df.sex_id == sex)
                ].copy()
                subset.to_csv(directory / str(loc) / f'{loc}_{sex}.csv')
                self.summarize_results(df=subset).to_csv(directory / str(loc) / f'{loc}_{sex}_summary.csv')

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
