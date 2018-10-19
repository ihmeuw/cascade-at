"""A representation of the configuration form we expect to receive from EpiViz.
The hope is that this form will do as much validation and precondition checking
as is feasible within the constraint that it must be able to validate a full
EpiViz parameter document in significantly less than one second. This is
because it will be used as part of a web service which gates EpiViz submissions
and must return in near real time.

The Configuration class is the root of the form.

"""
import numpy as np

from cascade.core.form import Form, IntField, FloatField, StrField, StringListField, OptionField, FormList, Dummy
from cascade.model import priors

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


class SmoothingPrior(Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_object = None

    prior_type = OptionField(["dage", "dtime", "value"])
    age_lower = FloatField(nullable=True, display="Age lower")
    age_upper = FloatField(nullable=True, display="Age upper")
    time_lower = FloatField(nullable=True, display="Time lower")
    time_upper = FloatField(nullable=True, display="Time upper")
    density = OptionField(["uniform", "gaussian", "laplace", "students", "log_gaussian", "log_laplace", "log_students"], display="Density")
    min = FloatField(nullable=True, default=float("-inf"), display="Min")
    mean = FloatField(nullable=True, display="Mean")
    max = FloatField(nullable=True, default=float("inf"), display="Max")
    std = FloatField(nullable=True, display="Std")
    nu = FloatField(nullable=True)
    eta = FloatField(nullable=True)

    def _full_form_validation(self, root):
        errors = []

        if not self.is_field_unset("age_lower") and not self.is_field_unset("age_lower"):
            if self.age_lower > self.age_upper:
                errors.append("age_lower must be less than or equal to age_upper")
        if not self.is_field_unset("time_lower") and not self.is_field_unset("time_lower"):
            if self.time_lower > self.time_upper:
                errors.append("time_lower must be less than or equal to time_upper")

        try:
            lower = self.min
            upper = self.max
            mean = self.mean
            if mean is None:
                if np.isinf(lower) or np.isinf(upper):
                    mean = max(lower, 0)
            std = self.std
            nu = self.nu
            if self.eta is None:
                if not root.is_field_unset("eta"):
                    eta = root.eta.priors
                else:
                    eta = None
            else:
                eta = self.eta

            if self.density == "uniform":
                self.prior_object = priors.Uniform(lower, upper, mean)
            elif self.density == "gaussian":
                self.prior_object = priors.Gaussian(mean, std, lower, upper)
            elif self.density == "laplace":
                self.prior_object = priors.Laplace(mean, std, lower, upper)
            elif self.density == "students":
                self.prior_object = priors.StudentsT(mean, std, nu, lower, upper, eta)
            elif self.density == "log_gaussian":
                self.prior_object = priors.LogGaussian(mean, std, eta, lower, upper)
            elif self.density == "log_laplace":
                self.prior_object = priors.LogLaplace(mean, std, eta, lower, upper)
            elif self.density == "log_students":
                self.prior_object = priors.LogStudentsT(mean, std, nu, eta, lower, upper)
            else:
                errors.append(f"Unknown density '{self.density}'")
        except priors.PriorError as e:
            errors.append(f"Parameters incompatible with density '{self.density}': {str(e)}")

        return errors


class SmoothingPriorGroup(Form):
    dage = SmoothingPrior(name_field="prior_type", nullable=True, display="Age diff")
    dtime = SmoothingPrior(name_field="prior_type", nullable=True, display="Time diff")
    value = SmoothingPrior(name_field="prior_type", display="Values")


class Smoothing(Form):
    rate = OptionField(["pini", "iota", "rho", "chi", "omega"], "Rate")
    location = IntField(nullable=True)
    age_grid = StringListField(constructor=float, nullable=True, display="Age grid")
    time_grid = StringListField(constructor=float, nullable=True, display="Time grid")
    default = SmoothingPriorGroup(display="Defaults")
    mulstd = SmoothingPriorGroup(nullable=True, display="MulStd")
    detail = FormList(SmoothingPrior, nullable=True, display="Detail")

    custom_age_grid = Dummy()
    custom_time_grid = Dummy()


class StudyCovariate(Form):
    # Haven't seen if this is a string or an ID for the column in the bundle.
    study_covariate = StringListField(nullable=True, display="Study covariates")

    measure_id = IntField(display="Measure")
    mulcov_type = OptionField(["rate_value", "meas_value", "meas_std"], display="Multiplier type")
    transformation = IntField(display="Transformation")
    at_dependence = IntField(display="AT dependence")

    age_grid = StringListField(constructor=float, nullable=True, display="Age grid")
    time_grid = StringListField(constructor=float, nullable=True, display="Time grid")
    default = SmoothingPriorGroup(display="Defaults")
    mulstd = SmoothingPriorGroup(nullable=True, display="MulStd")
    detail = FormList(SmoothingPrior, nullable=True, display="Detail")

    custom_age_grid = Dummy()
    custom_time_grid = Dummy()


class CountryCovariate(Form):
    country_covariate_id = IntField(display="Covariate")

    measure_id = IntField(display="Measure")
    mulcov_type = OptionField(["rate_value", "meas_value", "meas_std"], display="Multiplier type")
    transformation = IntField(display="Transformation")
    at_dependence = IntField(display="AT dependence")

    age_grid = StringListField(constructor=float, nullable=True, display="Age grid")
    time_grid = StringListField(constructor=float, nullable=True, display="Time grid")
    default = SmoothingPriorGroup(display="Defaults")
    mulstd = SmoothingPriorGroup(nullable=True, display="MulStd")
    detail = FormList(SmoothingPrior, nullable=True, display="Detail")

    custom_age_grid = Dummy()
    custom_time_grid = Dummy()


class Model(Form):
    modelable_entity_id = IntField()
    model_version_id = IntField(nullable=True)
    minimum_meas_cv = FloatField(nullable=True)
    add_csmr_cause = IntField(nullable=True, display="CSMR cause")
    title = StrField(nullable=True, display="Title")
    description = StrField(nullable=True, display="Description")
    bundle_id = IntField(nullable=True, display="Data bundle")
    drill = OptionField(["cascade", "drill"], display="Drill")
    drill_location = IntField(display="Drill location")
    drill_sex = OptionField([1, 2], constructor=int, nullable=True, display="Drill sex")
    default_age_grid = StringListField(constructor=float, display="(Cascade) Age grid")
    default_time_grid = StringListField(constructor=float, display="(Cascade) Time grid")
    rate_case = OptionField(
        ["iota_zero_rho_pos", "iota_pos_rho_zero", "iota_zero_rho_zero", "iota_pos_rho_pos"],
        nullable=True,
        default="iota_pos_rho_zero",
        display="(Advanced) Rate case",
    )


class Eta(Form):
    priors = FloatField(nullable=True)
    data = FloatField(nullable=True)


class Configuration(Form):
    """ The root Form of the whole configuration tree.

    Example:
        >>> input_data = json.loads(json_blob)
        >>> form = Configuration(input_data)
        >>> errors = form.validate_and_normalize()
        >>> if errors:
                print(errors)
                raise Exception("Woops")
            else:
                print(f"Ready to configure a model for {form.model.modelable_entity_id}")

    """
    model = Model(display="Model")
    gbd_round_id = IntField(display="GBD Round ID")
    random_effect = FormList(Smoothing, nullable=True, display="Random effects")
    rate = FormList(Smoothing, display="Rates")
    study_covariate = FormList(StudyCovariate, display="Study covariates")
    country_covariate = FormList(CountryCovariate, display="Country covariates")
    eta = Eta()

    csmr_cod_output_version_id = Dummy()
    csmr_mortality_output_version_id = Dummy()
    location_set_version_id = Dummy()
    min_cv = FormList(Dummy)
    min_cv_by_rate = FormList(Dummy)
    re_bound_location = FormList(Dummy)
    derivative_test = Dummy()
    max_num_iter = Dummy()
    print_level = Dummy()
    accept_after_max_steps = Dummy()
    tolerance = Dummy()
    students_dof = Dummy()
    log_students_dof = Dummy()
    data_eta_by_integrand = Dummy()
    data_density_by_integrand = Dummy()
    config_version = Dummy()
