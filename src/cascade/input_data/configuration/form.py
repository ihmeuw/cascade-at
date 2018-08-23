from cascade.input_data.configuration.abstract_form import (
    _Form,
    IntegerField,
    FloatField,
    StringField,
    OptionField,
    FormList,
    BooleanField,
)


class Distribution(_Form):
    density = OptionField(["uniform", "gaussian", "laplace", "students", "log_gaussian", "log_laplace", "log_students"])
    min = FloatField(nullable=True)
    mean = FloatField(nullable=True)
    max = FloatField(nullable=True)
    std = FloatField(nullable=True)


class SmoothingPrior(_Form):
    prior_type = OptionField(["dage", "dtime", "value"])
    distribution = Distribution()
    age_lower = FloatField()
    age_upper = FloatField()
    time_lower = FloatField()
    time_upper = FloatField()


class SmoothingPriorGroup(_Form):
    dage = SmoothingPrior(name_field="prior_type")
    dtime = SmoothingPrior(name_field="prior_type")
    value = SmoothingPrior(name_field="prior_type")


class RandomEffect(_Form):
    default = SmoothingPriorGroup()
    mulstd = SmoothingPrior()
    detail = FormList(SmoothingPrior)
    rate = IntegerField()
    custom_age_grid = StringField()
    custom_time_grid = StringField()


class Model(_Form):
    modelable_entity_id = IntegerField()
    title = StringField()
    description = StringField()
    bundle_id = IntegerField()
    drill = BooleanField(nullable=True)
    drill_sex = OptionField(["male", "female"], nullable=True)


class Configuration(_Form):
    model = Model()
    random_effect = FormList(RandomEffect)
