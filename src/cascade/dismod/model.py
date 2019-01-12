from math import isnan

from cascade.dismod.dismod_groups import DismodGroups
from cascade.dismod.var import Var


class Model(DismodGroups):
    """
    Uses locations as given and translates them into nodes for Dismod-AT.
    Uses ages and times as given and translates them into ``age_id``
    and ``time_id`` for Dismod-AT.
    """
    def __init__(self, nonzero_rates, parent_location, child_location):
        """
        >>> locations = location_hierarchy(execution_context)
        >>> m = Model(["chi", "omega", "iota"], locations, 6)
        """
        super().__init__()
        self.nonzero_rates = nonzero_rates
        self.location_id = parent_location
        self.child_location = child_location
        self.covariates = list()  # of class Covariate
        self.weights = dict()

    def write(self, writer):
        writer.start_model(self.nonzero_rates, self.child_location)
        for group in self.values():
            for grid in group.values():
                writer.write_ages_and_times(grid.ages, grid.times)
        for weight_value in self.weights.values():
            writer.write_ages_and_times(weight_value.ages, weight_value.times)

        writer.write_covariate(self.covariates)
        for name, weight in self.weights.items():
            writer.write_weight(name, weight)

        for group_name, group in self.items():
            if group_name == "rate":
                for rate_name, grid in group.items():
                    writer.write_rate(rate_name, grid)
            elif group_name == "random_effect":
                for (covariate, rate_name), grid in group.items():
                    writer.write_random_effect(covariate, rate_name, grid)
            elif group_name in {"alpha", "beta", "gamma"}:
                for (covariate, target), grid in group.items():
                    writer.write_mulcov(group_name, covariate, target, grid)
            else:
                raise RuntimeError(f"Unknown kind of field {group_name}")

    @property
    def model_variables(self):
        parts = DismodGroups()
        for rate, rate_rf in self.rate.items():
            parts.rate[rate] = Var(rate_rf.age_time)
        for (re_rate, re_location), re_rf in self.random_effect.items():
            # There will always be model variables for every child, even if
            # there is one smoothing for all children.
            if re_location is None or isnan(re_location):
                for child in self.child_location:
                    parts.random_effect[(re_rate, child)] = Var(re_rf.age_time)
            else:
                parts.random_effect[(re_rate, re_location)] = Var(re_rf.age_time)
        for alpha, alpha_rf in self.alpha.items():
            parts.alpha[alpha] = Var(alpha_rf.age_time)
        for beta, beta_rf in self.alpha.items():
            parts.beta[beta] = Var(beta_rf.age_time)
        for gamma, gamma_rf in self.alpha.items():
            parts.gamma[gamma] = Var(gamma_rf.age_time)
        return parts