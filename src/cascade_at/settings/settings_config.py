import numpy as np

from cascade_at.core.form.fields import (
    Form,
    BoolField,
    IntField,
    FloatField,
    StrField,
    NativeListField,
    StringListField,
    ListField,
    OptionField,
    FormList,
    Dummy,
)
from cascade_at.model import priors

from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


class SmoothingPrior(Form):
    """Priors for smoothing."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_object = None

    prior_type: OptionField = OptionField(["dage", "dtime", "value"])
    age_lower: FloatField = FloatField(nullable=True, display="Age lower")
    age_upper: FloatField = FloatField(nullable=True, display="Age upper")
    time_lower: FloatField = FloatField(nullable=True, display="Time lower")
    time_upper: FloatField = FloatField(nullable=True, display="Time upper")
    born_lower: FloatField = FloatField(nullable=True, display="Born lower")
    born_upper: FloatField = FloatField(nullable=True, display="Born upper")
    density: OptionField = OptionField(
        ["uniform", "gaussian", "laplace", "students",
         "log_gaussian", "log_laplace", "log_students"], display="Density"
    )
    min: FloatField = FloatField(nullable=True, default=float("-inf"), display="Min")
    mean: FloatField = FloatField(nullable=True, display="Mean")
    max: FloatField = FloatField(nullable=True, default=float("inf"), display="Max")
    std: FloatField = FloatField(nullable=True, display="Std")
    nu: FloatField = FloatField(nullable=True)
    eta: FloatField = FloatField(nullable=True)

    def _full_form_validation(self, root):
        errors = []

        if not self.is_field_unset("age_lower") and not self.is_field_unset("age_lower"):
            if self.age_lower > self.age_upper:
                errors.append(
                    "age_lower must be less than or equal to age_upper")
        if not self.is_field_unset("time_lower") and not self.is_field_unset("time_lower"):
            if self.time_lower > self.time_upper:
                errors.append(
                    "time_lower must be less than or equal to time_upper")

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
                self.prior_object = priors.StudentsT(
                    mean, std, nu, lower, upper)
            elif self.density == "log_gaussian":
                self.prior_object = priors.LogGaussian(
                    mean, std, eta, lower, upper)
            elif self.density == "log_laplace":
                self.prior_object = priors.LogLaplace(
                    mean, std, eta, lower, upper)
            elif self.density == "log_students":
                self.prior_object = priors.LogStudentsT(
                    mean, std, nu, eta, lower, upper)
            else:
                errors.append(f"Unknown density '{self.density}'")
        except priors.PriorError as e:
            errors.append(
                f"Parameters incompatible with density '"
                f"{self.density}': {str(e)}")

        return errors


class SmoothingPriorGroup(Form):
    dage: SmoothingPrior = SmoothingPrior(
        name_field="prior_type", nullable=True, display="Age diff")
    dtime: SmoothingPrior = SmoothingPrior(
        name_field="prior_type", nullable=True, display="Time diff")
    value: SmoothingPrior = SmoothingPrior(
        name_field="prior_type", nullable=True, display="Values")


class Smoothing(Form):
    rate: OptionField = OptionField(["pini", "iota", "rho", "chi", "omega"], "Rate")
    location: IntField = IntField(nullable=True)
    age_grid: StringListField = StringListField(
        constructor=float, nullable=True, display="Age grid")
    time_grid: StringListField = StringListField(
        constructor=float, nullable=True, display="Time grid")
    default: SmoothingPriorGroup = SmoothingPriorGroup(display="Defaults")
    mulstd: SmoothingPriorGroup = SmoothingPriorGroup(nullable=True, display="MulStd")
    detail: FormList = FormList(SmoothingPrior, nullable=True, display="Detail")
    age_time_specific: IntField = IntField(
        display="Age and Time specific", nullable=True)

    custom_age_grid: Dummy = Dummy()
    custom_time_grid: Dummy = Dummy()

    def _full_form_validation(self, root):
        errors = []

        if self.rate == "pini":
            if not self.is_field_unset("age_grid") and len(self.age_grid) != 1:
                errors.append("Pini must have exactly one age point")
        else:
            age_grid = self.age_grid or root.model.default_age_grid
            if len(age_grid) > 1 and self.default.is_field_unset("dage"):
                errors.append(
                    "You must supply a default age diff prior if the "
                    "smoothing has extent over age")

        time_grid = self.time_grid or root.model.default_time_grid
        if len(time_grid) > 1 and self.default.is_field_unset("dtime"):
            errors.append(
                "You must supply a default time diff prior if the smoothing "
                "has extent over time")

        if self._container._name == "rate":
            # This validation only makes sense for Fixed Effects not
            # Random Effects
            # TODO This repeats validation logic in cascade.model.rates but I
            # don't see a good way to bring that in here
            is_negative = True
            is_positive = True
            for prior in ([self.default.value] + [p for p in self.detail or
                          [] if p.prior_type == "value"]):
                is_negative = is_negative and prior.min == 0 and prior.max == 0
                is_positive = is_positive and prior.min > 0
                if prior.min < 0:
                    errors.append(
                        "Rates must be constrained to be >= 0 at all points. "
                        "Add or correct the lower bound")
                    break

            if self.rate in ["iota", "rho"]:
                if not (is_negative or is_positive):
                    errors.append(
                        f"Rate {self.rate} must be either fully positive or "
                        f"constrained to zero")

        return errors


class StudyCovariate(Form):
    study_covariate_id: IntField = IntField(display="Covariate")

    measure_id: StrField = StrField(display="Measure")
    mulcov_type: OptionField = OptionField(
        ["rate_value", "meas_value", "meas_noise"], display="Multiplier type")
    transformation: IntField = IntField(display="Transformation")
    age_time_specific: IntField = IntField(display="Age and Time specific")

    age_grid: StringListField = StringListField(
        constructor=float, nullable=True, display="Age grid")
    time_grid: StringListField = StringListField(
        constructor=float, nullable=True, display="Time grid")
    default: SmoothingPriorGroup = SmoothingPriorGroup(display="Defaults")
    mulstd: SmoothingPriorGroup = SmoothingPriorGroup(nullable=True, display="MulStd")
    detail: FormList = FormList(SmoothingPrior, nullable=True, display="Detail")

    custom_age_grid: Dummy = Dummy()
    custom_time_grid: Dummy = Dummy()


class CountryCovariate(Form):
    country_covariate_id: IntField = IntField(display="Covariate")

    measure_id: StrField = StrField(display="Measure")
    mulcov_type: OptionField = OptionField(
        ["rate_value", "meas_value", "meas_noise"], display="Multiplier type")
    transformation: IntField = IntField(display="Transformation")
    age_time_specific: IntField = IntField(display="Age and Time specific")

    age_grid: StringListField = StringListField(
        constructor=float, nullable=True, display="Age grid")
    time_grid: StringListField = StringListField(
        constructor=float, nullable=True, display="Time grid")
    default: SmoothingPriorGroup = SmoothingPriorGroup(display="Defaults")
    mulstd: SmoothingPriorGroup = SmoothingPriorGroup(nullable=True, display="MulStd")
    detail: SmoothingPrior = FormList(SmoothingPrior, nullable=True, display="Detail")

    custom_age_grid: Dummy = Dummy()
    custom_time_grid: Dummy = Dummy()


class Model(Form):
    modelable_entity_id: IntField = IntField()
    decomp_step_id: IntField = IntField()
    model_version_id: IntField = IntField(nullable=True)
    random_seed: IntField = IntField()
    minimum_meas_cv: FloatField = FloatField(nullable=True, display="Data CV floor")
    add_csmr_cause: IntField = IntField(nullable=True, display="CSMR cause")
    title: StrField = StrField(nullable=True, display="Title")
    description: StrField = StrField(nullable=True, display="Description")
    crosswalk_version_id: IntField = IntField(
        nullable=False, display="Crosswalk version ID")
    bundle_id: IntField = IntField(nullable=True, display="Data bundle")
    drill: OptionField = OptionField(["cascade", "drill"], display="Drill")
    drill_location: IntField = IntField(display="Drill location", nullable=True)
    drill_location_start: IntField = IntField(
        display="Drill location start", nullable=True)
    drill_location_end: NativeListField = NativeListField(
        display="Drill location end", nullable=True)
    drill_sex: OptionField = OptionField(
        [1, 2], constructor=int, nullable=True, display="Drill sex")
    birth_prev: OptionField = OptionField(
        [0, 1], constructor=int, nullable=True, default=0,
        display="Prevalence at birth")
    default_age_grid: StringListField = StringListField(
        constructor=float, display="(Cascade) Age grid")
    default_time_grid: StringListField = StringListField(
        constructor=float, display="(Cascade) Time grid")
    constrain_omega: OptionField = OptionField(
        [0, 1], constructor=int, nullable=False,
        display="Constrain other cause mortality")
    exclude_data_for_param: ListField = ListField(
        constructor=int, nullable=True, display="Exclude data for parameter")
    ode_step_size: FloatField = FloatField(display="ODE step size")
    addl_ode_stpes: StringListField = StringListField(
        constructor=float, nullable=True,
        display="Advanced additional ODE steps")
    split_sex: OptionField = OptionField(
        ["most_detailed", "1", "2", "3", "4", "5"],
        display="Split sex (Being used as Drill Start)")
    quasi_fixed: OptionField = OptionField(
        [0, 1], default=0, constructor=int, nullable=True)
    zero_sum_random: ListField = ListField(
        nullable=True, display="Zero-sum random effects")
    bound_frac_fixed: FloatField = FloatField(
        default=1e-2, nullable=True,
        display="allowed modification to point to move it within bounds"
    )
    bound_random: FloatField = FloatField(
        nullable=True,
        display="allowed modification to point to move it within bounds"
    )
    rate_case: StrField = StrField(nullable=False, display="The rate case")
    data_density: StrField = StrField(nullable=True, display="Data density")
    relabel_incidence: IntField = IntField(nullable=False, display="Relabel incidence")
    midpoint_approximation: NativeListField = NativeListField(
        nullable=True, display="Midpoint approximation"
    )

    def _full_form_validation(self, root):
        errors = []

        if self.drill == "drill":
            if self.is_field_unset("drill_sex"):
                errors.append("For a drill, please specify Drill sex.")

        return errors


class Eta(Form):
    priors: FloatField = FloatField(nullable=True)
    data: FloatField = FloatField(nullable=True)


class DataCV(Form):
    integrand_measure_id: IntField = IntField(nullable=True)
    value: FloatField = FloatField(nullable=True)


class MinCV(Form):
    cascade_level_id: StrField = StrField(nullable=True)
    value: FloatField = FloatField(nullable=True)


class MinCVRate(Form):
    cascade_level_id: StrField = StrField(nullable=True)
    rate_measure_id: StrField = StrField(nullable=True)
    value: FloatField = FloatField(nullable=True)


class DataEta(Form):
    integrand_measure_id: IntField = IntField(nullable=True)
    value: FloatField = FloatField(nullable=True)


class DataDensity(Form):
    value: StrField = StrField(nullable=True)
    integrand_measure_id: IntField = IntField(nullable=True)


class StudentsDOF(Form):
    priors: FloatField = FloatField(nullable=True, default=5)
    data: FloatField = FloatField(nullable=True, default=5)


class DerivativeTest(Form):
    fixed: OptionField = OptionField(
        ["none", "first-order", "second-order", "only-second-order",
         "adaptive", "trace-adaptive"],
        default="none",
        display="test for these derivatives",
        nullable=True
    )
    random: OptionField = OptionField(
        ["none", "first-order", "second-order", "only-second-order",
         "adaptive", "trace-adaptive"],
        default="none",
        display="test for these derivatives",
        nullable=True
    )


class FixedRandomInt(Form):
    fixed: IntField = IntField(nullable=True)
    random: IntField = IntField(nullable=True)


class FixedRandomFloat(Form):
    fixed: FloatField = FloatField(nullable=True)
    random: FloatField = FloatField(nullable=True)


class RandomEffectBound(Form):
    location: IntField = IntField(nullable=True)
    value: FloatField = FloatField(nullable=True)


class Policies(Form):
    estimate_emr_from_prevalence: OptionField = OptionField(
        [0, 1], constructor=int, default=0,
        display="Estimate EMR from prevalance", nullable=True
    )
    use_weighted_age_group_midpoints: OptionField = OptionField(
        [1, 0], default=1, constructor=int, nullable=True)
    number_of_fixed_effect_samples: IntField = IntField(default=30, nullable=True)
    with_hiv: BoolField = BoolField(
        default=True, nullable=True,
        display="Whether to get ASDR with HIV deaths.")
    age_group_set_id: IntField = IntField(
        default=12, nullable=True, display="Age groups for analysis work.")
    exclude_relative_risk: OptionField = OptionField(
        [1, 0], default=1, constructor=int, nullable=True)
    meas_noise_effect: OptionField = OptionField(
        ["add_std_scale_all", "add_std_scale_log", "add_var_scale_all",
         "add_var_scale_log"],
        default="add_var_scale_log",
        display="Measurement standard deviation effect",
        nullable=True
    )
    limited_memory_max_history_fixed: IntField = IntField(
        default=30, nullable=True,
        display=("number of most recent iterations taken into account "
                 "for quasi-Newton")
    )
    gbd_round_id: IntField = IntField(nullable=True, default=6)


class SettingsConfig(Form):
    """ The root Form of the whole settings inputs tree.
    This collects all settings from EpiViz-AT and adds default
    values when they are missing.

    A representation of the configuration form we expect to receive from EpiViz.
    The hope is that this form will do as much validation and precondition checking
    as is feasible within the constraint that it must be able to validate a full
    EpiViz parameter document in significantly less than one second. This is
    because it will be used as part of a web service which gates EpiViz submissions
    and must return in near real time.

    The Configuration class is the root of the form.

    Example
    -------
        >>> import json
        >>> input_data = json.loads(json_blob)
        >>> form = SettingsConfig(input_data)
        >>> errors = form.validate_and_normalize()
    """
    model: Model = Model(display="Model", validation_priority=5)
    policies: Policies = Policies(display="Policies")
    gbd_round_id: IntField = IntField(display="GBD Round ID")
    random_effect: FormList = FormList(
        Smoothing, nullable=True, display="Random effects")
    rate: FormList = FormList(Smoothing, display="Rates")
    country_covariate: FormList = FormList(
        CountryCovariate, display="Country covariates")
    study_covariate: FormList = FormList(
        StudyCovariate, display="Study covariates", nullable=True)
    eta: Eta = Eta(validation_priority=5)
    students_dof: StudentsDOF = StudentsDOF(validation_priority=5)
    log_students_dof: StudentsDOF = StudentsDOF(validation_priority=5)
    location_set_version_id: IntField = IntField(default=429, nullable=True)
    csmr_cod_output_version_id: IntField = IntField()
    csmr_mortality_output_version_id: Dummy = Dummy()
    min_cv: FormList = FormList(MinCV, display="Minimum coefficient of variation", nullable=False)
    min_cv_by_rate: FormList = FormList(
        MinCVRate, display="Minimum coefficient of variation by rate", nullable=True)
    re_bound_location: FormList = FormList(RandomEffectBound)
    derivative_test: DerivativeTest = DerivativeTest(display="Derivative test")
    max_num_iter: FixedRandomInt = FixedRandomInt(display="Max ipopt iterations")
    print_level: FixedRandomInt = FixedRandomInt(display="Print level")
    accept_after_max_steps: FixedRandomInt = FixedRandomInt(display="Max backtracking")
    tolerance: FixedRandomFloat = FixedRandomFloat(
        display="Desired relative convergence tolerance")
    data_cv_by_integrand: FormList = FormList(DataCV)
    data_eta_by_integrand: FormList = FormList(DataEta)
    data_density_by_integrand: FormList = FormList(DataDensity)
    config_version: StrField = StrField(nullable=True, display="Settings version")
