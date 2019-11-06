from cascade_at.core.log import get_loggers
from cascade_at.dismod.api.dismod_io import DismodIO

LOG = get_loggers(__name__)

class DismodAlchemy(DismodIO):
    """
    Sits on top of the DismodIO class,
    and takes inputs from the model construction,
    putting them into the Dismod database tables
    in the correct construction.
    """
    def __init__(self, engine):
        super().__init__(engine=engine)
    
    