from cascade.core.form import Form, IntField, FloatField, StrField, StringListField, OptionField, FormList, Dummy


class Distribution(Form):
    density = OptionField(["uniform", "gaussian", "laplace", "students", "log_gaussian", "log_laplace", "log_students"])
    min = FloatField(nullable=True)
    mean = FloatField(nullable=True)
    max = FloatField(nullable=True)
    std = FloatField(nullable=True)


class SmoothingPrior(Form):
    prior_type = OptionField(["dage", "dtime", "value"])
    age_lower = FloatField(nullable=True)
    age_upper = FloatField(nullable=True)
    time_lower = FloatField(nullable=True)
    time_upper = FloatField(nullable=True)
    density = OptionField(["uniform", "gaussian", "laplace", "students", "log_gaussian", "log_laplace", "log_students"])
    min = FloatField(nullable=True)
    mean = FloatField(nullable=True)
    max = FloatField(nullable=True)
    std = FloatField(nullable=True)


class SmoothingPriorGroup(Form):
    dage = SmoothingPrior(name_field="prior_type")
    dtime = SmoothingPrior(name_field="prior_type")
    value = SmoothingPrior(name_field="prior_type")


class Smoothing(Form):
    rate = IntField()
    age_grid = StringListField(constructor=float)
    time_grid = StringListField(constructor=float)
    default = SmoothingPriorGroup()
    mulstd = SmoothingPriorGroup(nullable=True)
    detail = FormList(SmoothingPrior, nullable=True)

    custom_age_grid = Dummy()
    custom_time_grid = Dummy()


class Model(Form):
    modelable_entity_id = IntField()
    title = StrField()
    description = StrField()
    bundle_id = IntField(nullable=True)
    drill = OptionField(["cascade", "drill"])
    drill_location = IntField()
    drill_sex = OptionField([1, 2], nullable=True)
    default_age_grid = StringListField(constructor=float)
    default_time_grid = StringListField(constructor=float)


class Configuration(Form):
    model = Model()
    gbd_round_id = IntField()
    csmr_cod_output_version_id = IntField()
    csmr_mortality_output_version_id = IntField()
    location_set_version_id = IntField()
    random_effect = FormList(Smoothing, nullable=True)
    rate = FormList(Smoothing)

    min_cv = FormList(Dummy)
    min_cv_by_rate = FormList(Dummy)
    re_bound_location = FormList(Dummy)
    study_covariate = Dummy()
    country_covariate = Dummy()
    derivative_test = Dummy()
    max_num_iter = Dummy()
    print_level = Dummy()
    accept_after_max_steps = Dummy()
    tolerance = Dummy()
    students_dof = Dummy()
    log_students_dof = Dummy()
    eta = Dummy()
    data_eta_by_integrand = Dummy()
    data_density_by_integrand = Dummy()
    config_version = Dummy()
