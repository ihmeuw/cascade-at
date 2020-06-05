import os

from cascade_at.core.db import db_tools
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


class ResultsHandler:
    """
    Handles all of the DisMod-AT results including draw saving
    and uploading to the epi database.
    """
    def __init__(self, model_version_id):
        self.model_version_id = model_version_id
        self.draw_keys = ['measure_id', 'year_id', 'age_group_id',
                          'location_id', 'sex_id', 'model_version_id']

    def validate_results(self, df):
        """
        Validates the input draw files. Put any additional
        validations here.
        Args:
            df: (pd.DataFrame)

        Returns:

        """
        missing_cols = [x for x in self.draw_keys if x not in df.columns]
        if missing_cols:
            raise RuntimeError(f"Missing id columns {missing_cols} for saving the results.")
        return df

    def save_draw_files(self, df, directory):
        """
        Saves a data frame by location and sex in .h5 files.
        This currently saves the summaries, but when we get
        save_results working it will save draws and then
        summaries as part of that.

        Args:
            df: (pd.DataFrame)
            directory: (pathlib.Path)

        Returns:

        """
        LOG.info(f"Saving results to {directory.absolute()}")

        df['model_version_id'] = self.model_version_id
        validated_df = self.validate_results(df=df)

        for loc in validated_df.location_id.unique().tolist():
            os.makedirs(directory / str(loc), exist_ok=True)
            for sex in validated_df.sex_id.unique().tolist():
                subset = validated_df.loc[
                    (validated_df.location_id == loc) &
                    (validated_df.sex_id == sex)
                ].copy()
                subset.to_csv(directory / str(loc) / f'{loc}_{sex}.csv')

    @staticmethod
    def upload_summaries(directory, conn_def):
        """
        Uploads results from a directory to the model_estimate_final
        table in the Epi database specified by the conn_def argument.

        In the future, this will probably be replaced by save_results_dismod
        but we don't have draws to work with so we're just uploading summaries
        for now directly.

        Args:
            directory: (pathlib.Path)
            conn_def: (str)

        Returns: None
        """
        session = db_tools.ezfuncs.get_session(conn_def=conn_def)
        loader = db_tools.loaders.Infiles(table='model_estimate_final', schema='epi', session=session)

        generic_file = (directory / '*' / '*.csv').absolute()
        LOG.info(f"Loading all files to {conn_def} that match {generic_file} glob.")
        loader.indir(path=str(generic_file), commit=True, with_replace=True)
