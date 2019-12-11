import pandas as pd

from cascade_at.core.log import get_loggers
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.api.fill_extract_helpers import reference_tables, data_tables, grid_tables

LOG = get_loggers(__name__)


class DismodExtractor(DismodIO):
    """
    Sits on top of the DismodIO class,
    and takes everything from the collector module
    and puts them into the Dismod database tables
    in the correct construction.
    """
    def __init__(self, path):
        super().__init__(path=path)

    def extract_predictions(self):
        """
        Gets the predictions from the predict table and transforms them
        into the GBD ids that we expect.
        :return:
        """
        pass