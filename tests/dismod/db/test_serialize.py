from cascade.core.context import ModelContext
from cascade.model.grids import PriorGrid, AgeTimeGrid
from cascade.model.rates import Smooth
from cascade.model.priors import GaussianPrior
from cascade.dismod.db.serialize import dismodfile_from_model_context
from cascade.dismod.db.wrapper import DismodFile, _get_engine


def test_development_target():
    context = ModelContext()

    grid = AgeTimeGrid.uniform(age_start=0, age_end=120, age_step=1, time_start=1990, time_end=2018, time_step=1)

    d_time = PriorGrid(grid)
    d_time[:, :].prior = GaussianPrior(0, 0.1)
    d_age = PriorGrid(grid)
    d_age[:, :].prior = GaussianPrior(0, 0.1)
    value = PriorGrid(grid)
    value[:, :].prior = GaussianPrior(0, 0.1)

    smooth = Smooth()
    smooth.d_time_priors = d_time
    smooth.d_age_priors = d_age
    smooth.value_priors = value

    context.rates["iota"].parent_smooth = smooth

    dm = dismodfile_from_model_context(context)
    e = _get_engine(None)
    dm.engine = e
    dm.flush()
    dm2 = DismodFile(e, {}, {})
    print(dm2.smooth_grid)
