from cascade_at.core.log import get_loggers
from cascade_at.dismod.api.dismod_sqlite import DismodSQLite

LOG = get_loggers(__name__)


class DismodIO(DismodSQLite):
    """
    This class is a verbose getter and setter for tables in the dismod file.
    If you want to set one of the tables in the dismod file, you should
    just be able to say, e.g. dmfile.data = pd.DataFrame({...}) as the 'setter', and it will
    automatically write it. Likewise, if you want to get one of the tables,
    then you can just do df = dmfile.data as the 'getter' and it will automatically read it.
    """
    def __init__(self, path):
        super().__init__(path=path)

    # AGE TABLE
    @property
    def age(self):
        return self.read_table('age')

    @age.setter
    def age(self, df):
        self.write_table('age', df)

    # AVGINT TABLE
    @property
    def avgint(self):
        return self.read_table('avgint')

    @avgint.setter
    def avgint(self, df):
        self.write_table('avgint', df)

    # COVARIATE TABLE
    @property
    def covariate(self):
        return self.read_table('covariate')

    @covariate.setter
    def covariate(self, df):
        self.write_table('covariate', df)

    # CONSTRAINT TABLE
    @property
    def constraint(self):
        return self.read_table('constraint')

    @constraint.setter
    def constraint(self, df):
        self.write_table('constraint', df)

    # DATA TABLE
    @property
    def data(self):
        return self.read_table('data')

    @data.setter
    def data(self, df):
        self.write_table('data', df)

    # DATA SUBSET TABLE
    @property
    def data_subset(self):
        return self.read_table('data_subset')

    @data_subset.setter
    def data_subset(self, df):
        raise RuntimeError("Cannot set data_subset table.")

    # DATA SIM TABLE
    @property
    def data_sim(self):
        return self.read_table('data_sim')

    @data_sim.setter
    def data_sim(self, df):
        raise RuntimeError("Cannot set data_sim table.")

    # DENSITY TABLE
    @property
    def density(self):
        return self.read_table('density')

    @density.setter
    def density(self, df):
        self.write_table('density', df)

    # DEPEND VAR TABLE
    @property
    def depend_var(self):
        return self.read_table('depend_var')

    @depend_var.setter
    def depend_var(self, df):
        self.write_table('depend_var', df)

    # FIT VAR TABLE
    @property
    def fit_var(self):
        return self.read_table('fit_var')

    @fit_var.setter
    def fit_var(self, df):
        self.write_table('fit_var', df)

    # FIT DATA SUBSET TABLE
    @property
    def fit_data_subset(self):
        return self.read_table('fit_data_subset')

    @fit_data_subset.setter
    def fit_data_subset(self, df):
        raise RuntimeError("Cannot set fit_data_subset table.")

    # INTEGRAND TABLE
    @property
    def integrand(self):
        return self.read_table('integrand')

    @integrand.setter
    def integrand(self, df):
        self.write_table('integrand', df)

    # LOG TABLE
    @property
    def log(self):
        return self.read_table('log')

    @log.setter
    def log(self, df):
        raise RuntimeError("Cannot set log table.")

    # MULCOV TABLE
    @property
    def mulcov(self):
        return self.read_table('mulcov')

    @mulcov.setter
    def mulcov(self, df):
        self.write_table('mulcov', df)

    # NODE TABLE
    @property
    def node(self):
        return self.read_table('node')

    @node.setter
    def node(self, df):
        self.write_table('node', df)

    # OPTION TABLE
    @property
    def option(self):
        return self.read_table('option')

    @option.setter
    def option(self, df):
        self.write_table('option', df)

    # PRIOR TABLE
    @property
    def prior(self):
        return self.read_table('prior')

    @prior.setter
    def prior(self, df):
        self.write_table('prior', df)

    # PRIOR SIM TABLE
    @property
    def prior_sim(self):
        return self.read_table('prior_sim')

    @prior_sim.setter
    def prior_sim(self, df):
        raise RuntimeError("Cannot set prior_sim table.")

    # PREDICT TABLE
    @property
    def predict(self):
        return self.read_table('predict')

    @predict.setter
    def predict(self, df):
        raise RuntimeError("Cannot set predict table.")

    # RATE TABLE
    @property
    def rate(self):
        return self.read_table('rate')

    @rate.setter
    def rate(self, df):
        self.write_table('rate', df)

    # NSLIST TABLE
    @property
    def nslist(self):
        return self.read_table('nslist')

    @nslist.setter
    def nslist(self, df):
        self.write_table('nslist', df)

    # NSLIST PAIR TABLE
    @property
    def nslist_pair(self):
        return self.read_table('nslist_pair')

    @nslist_pair.setter
    def nslist_pair(self, df):
        self.write_table('nslist_pair', df)

    # SAMPLE TABLE
    @property
    def sample(self):
        return self.read_table('sample')

    @sample.setter
    def sample(self, df):
        self.write_table('sample', df)

    # SCALE VAR TABLE
    @property
    def scale_var(self):
        return self.read_table('scale_var')

    @scale_var.setter
    def scale_var(self, df):
        self.write_table('scale_var', df)

    # SMOOTH TABLE
    @property
    def smooth(self):
        return self.read_table('smooth')

    @smooth.setter
    def smooth(self, df):
        self.write_table('smooth', df)

    # SMOOTH GRID TABLE
    @property
    def smooth_grid(self):
        return self.read_table('smooth_grid')

    @smooth_grid.setter
    def smooth_grid(self, df):
        self.write_table('smooth_grid', df)

    # START VAR TABLE
    @property
    def start_var(self):
        return self.read_table('start_var')

    @start_var.setter
    def start_var(self, df):
        self.write_table('start_var', df)

    # SUBGROUP TABLE
    @property
    def subgroup(self):
        return self.read_table('subgroup')

    @subgroup.setter
    def subgroup(self, df):
        self.write_table('subgroup', df)

    # TIME TABLE
    @property
    def time(self):
        return self.read_table('time')

    @time.setter
    def time(self, df):
        self.write_table('time', df)

    # TRUTH VAR TABLE
    @property
    def truth_var(self):
        return self.read_table('truth_var')

    @truth_var.setter
    def truth_var(self, df):
        self.write_table('truth_var', df)

    # VAR TABLE
    @property
    def var(self):
        return self.read_table('var')

    @var.setter
    def var(self, df):
        raise RuntimeError("Cannot set var table.")

    # WEIGHT
    @property
    def weight(self):
        return self.read_table('weight')

    @weight.setter
    def weight(self, df):
        self.write_table('weight', df)

    # WEIGHT GRID
    @property
    def weight_grid(self):
        return self.read_table('weight_grid')

    @weight_grid.setter
    def weight_grid(self, df):
        self.write_table('weight_grid', df)

    # HESSIAN FIXED TABLE
    @property
    def hes_fixed(self):
        return self.read_table('hes_fixed')

    @hes_fixed.setter
    def hes_fixed(self, df):
        self.write_table('hes_fixed', df)

    # HESSIAN RANDOM TABLE
    @property
    def hes_random(self):
        return self.read_table('hes_random')

    @hes_random.setter
    def hes_random(self, df):
        self.write_table('hes_random', df)
