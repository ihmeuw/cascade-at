from collections import defaultdict

import numpy as np

from cascade_at.core.log import get_loggers
from cascade_at.model.model import Model
from cascade_at.model.utilities.grid_helpers import smooth_grid_from_smoothing_form

LOG = get_loggers(__name__)


class ModelConstruct:
    def __init__(self, settings):
        """
        An object initialized with model settings from
        cascade.settings.configuration.Configuration that can be used
        to construct parent-child location-specific models with
        the attribute ModelConstruct.construct().

        :param settings:
        """
        self.settings = settings
        self.age_time_grid = self.construct_age_time_grid()
        self.single_age_time_grid = self.construct_single_age_time_grid()

        self.model = None

    def construct_age_time_grid(self):
        """
        Construct a DEFAULT age-time grid,
        to be updated when we initialize the model.

        :return:
        """
        default_age_time = dict()
        default_age_time["age"] = np.linspace(0, 100, 21)
        default_age_time["time"] = np.linspace(1990, 2015, 6)

        for kind in ["age", "time"]:
            default_grid = getattr(self.settings.model, f"default_{kind}_grid")
            if default_grid is not None:
                default_age_time[kind] = np.sort(np.array(default_grid, dtype=np.float))

        return default_age_time

    def construct_single_age_time_grid(self):
        """
        Construct a single age-time grid.
        Use this age and time when a smooth grid doesn't depend on age and time.

        :return:
        """
        single_age = self.age_time_grid["age"][:1]
        single_time = [self.age_time_grid["time"][len(self.age_time_grid["time"]) // 2]]
        single_age_time = (single_age, single_time)
        return single_age_time

    def construct_for_parent_location(self, location_dag, parent_location_id):
        """
        Construct a Model object for a parent location and its children.

        :param location_dag: (cascade.inputs.locations.LocationDAG)
        :param parent_location_id: (int)
        :return: self
        """
        children = list(location_dag.dag.successors(parent_location_id))
        model = Model(
            nonzero_rates=self.settings.rate,
            parent_location=parent_location_id,
            child_location=children,
            covariates=None,
            weights=None
        )

        for smooth in self.settings.rate:
            rate_grid = smooth_grid_from_smoothing_form(
                default_age_time=self.age_time_grid,
                single_age_time=self.single_age_time_grid,
                smooth=smooth
            )
            model.rate[smooth.rate] = rate_grid

        if self.settings.random_effect:
            random_effect_by_rate = defaultdict(list)
            for smooth in self.settings.random_effect:
                re_grid = smooth_grid_from_smoothing_form(
                    default_age_time=self.age_time_grid,
                    single_age_time=self.single_age_time_grid,
                    smooth=smooth
                )
                if not smooth.is_field_unset("location") and smooth.location in model.child_location:
                    location = smooth.location
                else:
                    location = None
                model.random_effect[(smooth.rate, location)] = re_grid
                random_effect_by_rate[smooth.rate].append(location)

            for rate_to_check, locations in random_effect_by_rate.items():
                if locations != [None] and set(locations) != set(model.child_location):
                    raise RuntimeError(f"Random effect for {rate_to_check} does not have "
                                       f"entries for all child locations, only {locations} "
                                       f"instead of {model.child_location}.")

        # TODO: Construct covariates.

        return model

