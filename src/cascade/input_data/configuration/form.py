"""A representation of the configuration form we expect to receive from EpiViz.
The hope is that this form will do as much validation and precondition checking
as is feasible within the constraint that it must be able to validate a full
EpiViz parameter document in significantly less than one second. This is
because it will be used as part of a web service which gates EpiViz submissions
and must return in near real time.

The Configuration class is the root of the form.

"""
import numpy as np

from cascade.core.form import (
    Form,
    BoolField,
    IntField,
    FloatField,
    StrField,
    StringListField,
    ListField,
    OptionField,
    FormList,
    Dummy,
)
from cascade.model import priors

from cascade.core.log import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


class SmoothingPrior(Form):
    """Priors for smoothing."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_object = None

    prior_type = OptionField(["dage", "dtime", "value"])
    age_lower = FloatField(nullable=True, display="Age lower")
    age_upper = FloatField(nullable=True, display="Age upper")
    time_lower = FloatField(nullable=True, display="Time lower")
    time_upper = FloatField(nullable=True, display="Time upper")
    born_lower = FloatField(nullable=True, display="Born lower")
    born_upper = FloatField(nullable=True, display="Born upper")
    density = OptionField(
        ["uniform", "gaussian", "laplace", "students", "log_gaussian", "log_laplace", "log_students"], display="Density"
    )
    min = FloatField(nullable=True, default=float("-inf"), display="Min")
    mean = FloatField(nullable=True, display="Mean")
    max = FloatField(nullable=True, default=float("inf"), display="Max")
    std = FloatField(nullable=True, display="Std")
    nu = FloatField(nullable=True)
    eta = FloatField(nullable=True)

    def _full_form_validation(self, root):  # noqa: C901 too complex
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
            if mean is None and (np.isinf(lower) or np.isinf(upper)):
                mean = max(lower, 0)
            std = self.std

            if self.nu is None:
                if self.density == "students" and not root.is_field_unset("students_dof"):
                    nu = root.students_dof.priors
                elif self.density == "log_students" and not root.is_field_unset("log_students_dof"):
                    nu = root.log_students_dof.priors
                else:
                    nu = None
            else:
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
                self.prior_object = priors.StudentsT(mean, std, nu, lower, upper)
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
    value = SmoothingPrior(name_field="prior_type", nullable=True, display="Values")


class Smoothing(Form):
    rate = OptionField(["pini", "iota", "rho", "chi", "omega"], "Rate")
    location = IntField(nullable=True)
    age_grid = StringListField(constructor=float, nullable=True, display="Age grid")
    time_grid = StringListField(constructor=float, nullable=True, display="Time grid")
    default = SmoothingPriorGroup(display="Defaults")
    mulstd = SmoothingPriorGroup(nullable=True, display="MulStd")
    detail = FormList(SmoothingPrior, nullable=True, display="Detail")
    age_time_specific = IntField(display="Age and Time specific", nullable=True)

    custom_age_grid = Dummy()
    custom_time_grid = Dummy()

    def _full_form_validation(self, root):
        errors = []

        if self.rate == "pini":
            if not self.is_field_unset("age_grid") and len(self.age_grid) != 1:
                errors.append("Pini must have exactly one age point")
        else:
            age_grid = self.age_grid or root.model.default_age_grid
            if len(age_grid) > 1 and self.default.is_field_unset("dage"):
                errors.append("You must supply a default age diff prior if the smoothing has extent over age")

        time_grid = self.time_grid or root.model.default_time_grid
        if len(time_grid) > 1 and self.default.is_field_unset("dtime"):
            errors.append("You must supply a default time diff prior if the smoothing has extent over time")

        if self._container._name == "rate":
            # This validation only makes sense for Fixed Effects not Random Effects
            # TODO This repeats validation logic in cascade.model.rates but I don't see a good way to bring that in here
            is_negative = True
            is_positive = True
            for prior in [self.default.value] + [p for p in self.detail or [] if p.prior_type == "value"]:
                is_negative = is_negative and prior.min == 0 and prior.max == 0
                is_positive = is_positive and prior.min > 0
                if prior.min < 0:
                    errors.append("Rates must be constrained to be >= 0 at all points. Add or correct the lower bound")
                    break

            if self.rate in ["iota", "rho"]:
                if not (is_negative or is_positive):
                    errors.append(f"Rate {self.rate} must be either fully positive or constrained to zero")

        return errors


class StudyCovariate(Form):
    # Haven't seen if this is a string or an ID for the column in the bundle.
    study_covariate_id = IntField(display="Covariate")

    measure_id = IntField(display="Measure")
    mulcov_type = OptionField(["rate_value", "meas_value", "meas_std"], display="Multiplier type")
    transformation = IntField(display="Transformation")
    age_time_specific = IntField(display="Age and Time specific")

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
    age_time_specific = IntField(display="Age and Time specific")

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
    random_seed = IntField()
    minimum_meas_cv = FloatField(nullable=True, display="Data CV floor")
    add_csmr_cause = IntField(nullable=True, display="CSMR cause")
    title = StrField(nullable=True, display="Title")
    description = StrField(nullable=True, display="Description")
    bundle_id = IntField(nullable=True, display="Data bundle")
    drill = OptionField(["cascade", "drill"], display="Drill")
    drill_location = IntField(display="Drill location", nullable=True)
    drill_location_start = IntField(display="Drill location start", nullable=True)
    drill_location_end = IntField(display="Drill location end", nullable=True)
    drill_sex = OptionField([1, 2], constructor=int, nullable=True, display="Drill sex")
    birth_prev = OptionField([0, 1], constructor=int, nullable=True, default=0, display="Prevalence at birth")
    default_age_grid = StringListField(constructor=float, display="(Cascade) Age grid")
    default_time_grid = StringListField(constructor=float, display="(Cascade) Time grid")
    constrain_omega = OptionField([0, 1], constructor=int, nullable=False, display="Constrain other cause mortality")
    exclude_data_for_param = ListField(constructor=int, nullable=True, display="Exclude data for parameter")
    ode_step_size = FloatField(display="ODE step size")
    additional_ode_steps = StringListField(constructor=float, nullable=True,
                                           display="Advanced additional ODE steps")
    split_sex = OptionField(["most_detailed", "1", "2", "3", "4", "5"], display="Split sex (Being used as Drill Start)")
    quasi_fixed = OptionField([0, 1], default=0, constructor=int, nullable=True)
    zero_sum_random = ListField(nullable=True, display="Zero-sum random effects")
    bound_frac_fixed = FloatField(
        default=1e-2, nullable=True,
        display="allowed modification to point to move it within bounds"
    )
    bound_random = FloatField(
        nullable=True,
        display="allowed modification to point to move it within bounds"
    )
    rate_case = Dummy()
    data_density = StrField(nullable=True, display="Data density")

    def _full_form_validation(self, root):
        errors = []

        if self.drill == "drill":
            if self.is_field_unset("drill_sex"):
                errors.append("For a drill, please specify Drill sex.")

        return errors


class Eta(Form):
    priors = FloatField(nullable=True)
    data = FloatField(nullable=True)


class DataEta(Form):
    integrand_measure_id = IntField(nullable=True)
    value = FloatField(nullable=True)


class DataDensity(Form):
    value = StrField(nullable=True)
    integrand_measure_id = IntField(nullable=True)


class StudentsDOF(Form):
    priors = FloatField(nullable=True, default=5)
    data = FloatField(nullable=True, default=5)


class DerivativeTest(Form):
    fixed = OptionField(
        ["none", "first-order", "second-order", "only-second-order",
         "adaptive", "trace-adaptive"],
        default="none",
        display="test for these derivatives",
        nullable=True
    )
    random = OptionField(
        ["none", "first-order", "second-order", "only-second-order",
         "adaptive", "trace-adaptive"],
        default="none",
        display="test for these derivatives",
        nullable=True
    )


class FixedRandomInt(Form):
    fixed = IntField(nullable=True)
    random = IntField(nullable=True)


class FixedRandomFloat(Form):
    fixed = FloatField(nullable=True)
    random = FloatField(nullable=True)


class RandomEffectBound(Form):
    location = IntField(nullable=True)
    value = FloatField(nullable=True)


class Policies(Form):
    estimate_emr_from_prevalence = OptionField(
        [0, 1], constructor=int, default=0, display="Estimate EMR from prevalance", nullable=True
    )
    use_weighted_age_group_midpoints = OptionField([1, 0], default=1, constructor=int, nullable=True)
    number_of_fixed_effect_samples = IntField(default=30, nullable=True)
    with_hiv = BoolField(default=True, nullable=True, display="Whether to get ASDR with HIV deaths.")
    age_group_set_id = IntField(default=12, nullable=True, display="Age groups for analysis work.")
    exclude_relative_risk = OptionField([1, 0], default=1, constructor=int, nullable=True)
    meas_std_effect = OptionField(
        ["add_std_scale_all", "add_std_scale_log", "add_var_scale_all", "add_var_scale_log"],
        default="add_var_scale_log",
        display="Measurement standard deviation effect",
        nullable=True
    )
    limited_memory_max_history_fixed = IntField(
        default=30, nullable=True,
        display="number of most recent iterations taken into account for quasi-Newton"
    )
    fit_strategy = OptionField(["fit", "fit_fixed_then_fit"], default="fit", constructor=int, nullable=True)
    decomp_step = StrField(nullable=True, default="step1")
    gbd_round_id = IntField(nullable=True, default=6)


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

    model = Model(display="Model", validation_priority=5)
    policies = Policies(display="Policies")
    gbd_round_id = IntField(display="GBD Round ID")
    random_effect = FormList(Smoothing, nullable=True, display="Random effects")
    rate = FormList(Smoothing, display="Rates")
    study_covariate = FormList(StudyCovariate, display="Study covariates")
    country_covariate = FormList(CountryCovariate, display="Country covariates")
    eta = Eta(validation_priority=5)
    students_dof = StudentsDOF(validation_priority=5)
    log_students_dof = StudentsDOF(validation_priority=5)
    csmr_cod_output_version_id = IntField()
    # Unclear how this differs from csmr_cod_output_version_id. Has same value.
    csmr_mortality_output_version_id = Dummy()
    location_set_version_id = IntField(default=429, nullable=True)
    min_cv = FormList(Dummy)
    min_cv_by_rate = FormList(Dummy)
    re_bound_location = FormList(RandomEffectBound)
    derivative_test = DerivativeTest(display="Derivative test")
    max_num_iter = FixedRandomInt(display="Max ipopt iterations")
    print_level = FixedRandomInt(display="Print level")
    accept_after_max_steps = FixedRandomInt(display="Max backtracking")
    tolerance = FixedRandomFloat(display="Desired relative convergence tolerance")
    data_eta_by_integrand = FormList(DataEta)
    data_density_by_integrand = FormList(DataDensity)
    config_version = IntField(nullable=True, display="Settings version")
