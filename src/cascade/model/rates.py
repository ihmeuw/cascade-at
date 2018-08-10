class Smooth:
    def __init__(self, value_priors=None, d_age_priors=None, d_time_priors=None):
        self.value_priors = value_priors
        self.d_age_priors = d_age_priors
        self.d_time_priors = d_time_priors


class Rate:
    def __init__(self, name, parent_smooth=None, child_smooth=None):
        self.name = name
        self.parent_smooth = parent_smooth
        self.child_smooth = child_smooth
