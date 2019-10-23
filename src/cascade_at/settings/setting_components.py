from cascade_at.settings import priors


class Model:
    def __init__(self):
        self.modelable_entity_id = None
        self.bundle_id = None
        self.crosswalk_version_id = None
        self.decomp_step_id = None
        self.research_area = None
        self.title = None
        self.description = None

        self.random_seed = None
        self.default_age_grid = None
        self.default_time_grid = None
        self.add_calc_emr = None
        self.birth_prev = None
        self.ode_step_size = None
        self.minimum_meas_cv = None
        self.rate_case = None
        self.data_density = None
        self.constrain_omega = None
        self.add_csmr_cause = None


class MaxNumIter:
    def __init__(self):
        self.fixed = None
        self.random = None


class PrintLevel:
    def __init__(self):
        self.fixed = None
        self.random = None


class AcceptAfterMaxSteps:
    def __init__(self):
        self.fixed = None
        self.random = None


class StudentsDOF:
    def __init__(self):
        self.priors = None
        self.data = None


class LogStudentsDOF:
    def __init__(self):
        self.priors = None
        self.data = None


class Eta:
    def __init__(self):
        self.priors = None
        self.data = None


class MinCV:
    def __init__(self):
        self.cascade_level_id = None
        self.value = None


class MinCVByRate:
    def __init__(self):
        self.rate_measure_id = None
        self.cascade_level_id = None
        self.value = None


class Covariate:
    def __init__(self):
        self.age_time_specific = None
        self.mulcov_type: None
        self.measure_id = None
        self.covariate_id = None
        self.transformation = None

    def configure(self):
        return self


class Rate:
    def __init__(self):
        self.age_time_specific = None
        self.age_grid = None
        self.time_grid = None
        self.rate = None
        self.age_grid = None
        self.time_grid = None

    def configure(self):
        return self


class RandomEffect:
    def __init__(self):
        self.age_time_specific = None
        self.rate = None
        self.age_grid = None
        self.time_grid = None

    def configure(self):
        return self


class Tolerance:
    def __init__(self):
        self.fixed = None
        self.random = None


