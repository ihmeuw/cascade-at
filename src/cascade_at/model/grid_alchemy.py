from collections import defaultdict
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import itertools

from cascade_at.core.log import get_loggers
from cascade_at.model.model import Model
from cascade_at.settings.settings import SettingsConfig
from cascade_at.inputs.locations import LocationDAG
from cascade_at.inputs.covariate_specs import CovariateSpecs
from cascade_at.model.utilities.grid_helpers import smooth_grid_from_smoothing_form
from cascade_at.model.utilities.grid_helpers import rectangular_data_to_var
from cascade_at.model.utilities.grid_helpers import constraint_from_rectangular_data
from cascade_at.model.utilities.grid_helpers import estimate_grid_from_draws
from cascade_at.settings.settings_config import Smoothing
from cascade_at.model.var import Var
from cascade_at.model.smooth_grid import SmoothGrid, _PriorGrid
from cascade_at.model.priors import _Prior

LOG = get_loggers(__name__)


MOST_DETAILED_CASCADE_LEVEL = 'most_detailed'


class Alchemy:
    def __init__(self, settings: SettingsConfig):
        """
        An object initialized with model settings from
        cascade.settings.configuration.Configuration that can be used
        to construct parent-child location-specific models with
        the attribute ModelConstruct.construct_two_level_model().

        Examples
        --------
        >>> from cascade_at.settings.base_case import BASE_CASE
        >>> from cascade_at.settings.settings import load_settings
        >>> from cascade_at.inputs.measurement_inputs import MeasurementInputsFromSettings
        
        >>> settings = load_settings(BASE_CASE)
        >>> mc = Alchemy(settings)
        
        >>> i = MeasurementInputsFromSettings(settings)
        >>> i.get_raw_inputs()

        >>> mc.construct_two_level_model(location_dag=i.location_dag,
        >>>                              parent_location_id=102,
        >>>                              covariate_specs=i.covariate_specs)
        """
        self.settings = settings
        self.age_time_grid = self.construct_age_time_grid()
        self.single_age_time_grid = self.construct_single_age_time_grid()

        self.model = None

    def construct_age_time_grid(self) -> Dict[str, np.ndarray]:
        """
        Construct a DEFAULT age-time grid,
        to be updated when we initialize the model.
        """
        default_age_time = dict()
        default_age_time["age"] = np.linspace(0, 100, 21)
        default_age_time["time"] = np.linspace(1990, 2015, 6)

        for kind in ["age", "time"]:
            default_grid = getattr(self.settings.model, f"default_{kind}_grid")
            if default_grid is not None:
                default_age_time[kind] = np.sort(np.array(default_grid, dtype=float))

        return default_age_time

    def construct_single_age_time_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct a single age-time grid.
        Use this age and time when a smooth grid doesn't depend on age and time.
        """
        single_age = self.age_time_grid["age"][:1]
        single_time = np.array([self.age_time_grid["time"][len(self.age_time_grid["time"]) // 2]])
        single_age_time = (single_age, single_time)
        return single_age_time

    def get_smoothing_grid(self, rate: Smoothing) -> SmoothGrid:
        """
        Construct a smoothing grid for any rate in the model.

        Parameters
        ----------
        rate
            Some smoothing form for a rate.

        Returns
        -------
            The rate translated into a SmoothGrid based on the model settings'
            default age and time grids.

        """
        return smooth_grid_from_smoothing_form(
            default_age_time=self.age_time_grid,
            single_age_time=self.single_age_time_grid,
            smooth=rate
        )

    def get_all_rates_grids(self) -> Dict[str, SmoothGrid]:
        """
        Get a dictionary of all the rates and their grids in the model.
        """
        return {c.rate: self.get_smoothing_grid(rate=c) for c in self.settings.rate}

    @staticmethod
    def override_priors(rate_grid: SmoothGrid, update_dict=Dict[str, np.ndarray],
                        new_prior_distribution: Optional[str] = 'gaussian'):
        """
        Override priors for rates. This is used
        when we want to do posterior to prior, so we are
        overriding the global settings with location-specific
        settings based on parent posteriors.

        Parameters
        ----------
        rate_grid
            SmoothGrid object for a rate
        update_dict
            Dictionary with ages and times vectors and draws for values, dage, and dtime
            to use in overriding the prior.
        new_prior_distribution
            The new prior distribution to override the existing priors.
        """
        # Check that the prior grid lines up with this rate
        # grid. If it doesn't, we have a problem.
        assert (update_dict['ages'] == rate_grid.ages).all()
        assert (update_dict['times'] == rate_grid.times).all()
        # For each of the types of priors, update rate_grid
        # with the new prior information from the update_prior
        # object that has info from a different model fit
        if 'value' in update_dict:
            estimate_grid_from_draws(
                grid_priors=rate_grid.value, draws=update_dict['value'],
                ages=rate_grid.ages, times=rate_grid.times,
                new_prior_distribution=new_prior_distribution
            )
        if 'dage' in update_dict:
            estimate_grid_from_draws(
                grid_priors=rate_grid.dage, draws=update_dict['dage'],
                ages=rate_grid.ages[:-1], times=rate_grid.times,
                new_prior_distribution=new_prior_distribution
            )
        if 'dtime' in update_dict:
            estimate_grid_from_draws(
                grid_priors=rate_grid.dtime, draws=update_dict['dtime'],
                ages=rate_grid.ages, times=rate_grid.times[:-1],
                new_prior_distribution=new_prior_distribution
            )

    @staticmethod
    def apply_min_cv_to_prior_grid(prior_grid: _PriorGrid, min_cv: float, min_std: float = 1e-10) -> None:
        """
        Applies the minimum coefficient of variation to a _PriorGrid
        to enforce that minCV across all variables in the grid.
        Updates the _PriorGrid in place.
        """
        prior_grid.grid['std'] = prior_grid.grid.apply(
            lambda row: max(min_std, row['std'], np.abs(row['mean']) * min_cv ), axis=1
        )

    def construct_two_level_model(self, location_dag: LocationDAG, parent_location_id: int,
                                  covariate_specs: CovariateSpecs,
                                  weights: Optional[Dict[str, Var]] = None,
                                  omega_df: Optional[pd.DataFrame] = None,
                                  update_prior: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                                  min_cv: Optional[Dict[str, Dict[str, float]]] = None,
                                  update_mulcov_prior: Optional[Dict[Tuple[str, str, str], _Prior]] = None):
        """
        Construct a Model object for a parent location and its children.

        Parameters
        ----------
        location_dag
            Location DAG specifying the location hierarchy
        parent_location_id
            Parent location to build the model for
        covariate_specs
            covariate specifications, specifically will use covariate_specs.covariate_multipliers
        weights
        omega_df
            data frame with omega values in it (other cause mortality)
        update_prior
            dictionary of dictionary for prior updates to rates
        update_mulcov_prior
            dictionary of mulcov prior updates
        min_cv
            dictionary (can be defaultdict) for minimum coefficient of variation
            keyed by cascade level, then by rate
        """
        children = location_dag.children(parent_location_id)
        cascade_level = str(location_dag.depth(parent_location_id)) # min_cv lookup expects a string key
        is_leaf = location_dag.is_leaf(parent_location_id)
        if is_leaf:
            cascade_level = MOST_DETAILED_CASCADE_LEVEL

        model = Model(
            nonzero_rates=self.settings.rate,
            parent_location=parent_location_id,
            child_location=children,
            covariates=covariate_specs.covariate_list,
            weights=weights
        )

        # First construct the rate grid, and update with prior
        # information from a parent for value, dage, and dtime.
        for smooth in self.settings.rate:
            rate_grid = self.get_smoothing_grid(rate=smooth)
            if update_prior is not None:
                if smooth.rate in update_prior:
                    self.override_priors(rate_grid=rate_grid, update_dict=update_prior[smooth.rate])
                    if min_cv is not None:
                        self.apply_min_cv_to_prior_grid(
                            prior_grid=rate_grid.value, min_cv=min_cv[cascade_level][smooth.rate]
                        )
            model.rate[smooth.rate] = rate_grid
        
        # Second construct the covariate grids
        for mulcov in covariate_specs.covariate_multipliers:
            grid = smooth_grid_from_smoothing_form(
                    default_age_time=self.age_time_grid,
                    single_age_time=self.single_age_time_grid,
                    smooth=mulcov.grid_spec
                )
            if update_mulcov_prior is not None and (mulcov.group, *mulcov.key) in update_mulcov_prior:
                ages = grid.ages
                times = grid.times
                for age, time in itertools.product(ages, times):
                    lb = grid.value[age, time].lower
                    ub = grid.value[age, time].upper
                    update_mulcov_prior[(mulcov.group, *mulcov.key)].lower = lb
                    update_mulcov_prior[(mulcov.group, *mulcov.key)].upper = ub
                    grid.value[age, time] = update_mulcov_prior[(mulcov.group, *mulcov.key)] 
            model[mulcov.group][mulcov.key] = grid

        # Construct the random effect grids, based on the parent location
        # specified.
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

        # Lastly, constrain omega for the parent and the random effects for the children.
        if self.settings.model.constrain_omega:
            LOG.info("Adding the omega constraint.")
            
            if omega_df is None:
                raise RuntimeError("Need an omega data frame in order to constrain omega.")
            
            parent_omega = omega_df.loc[omega_df.location_id == parent_location_id].copy()
            if parent_omega.empty:
                raise RuntimeError(f"No omega values for location {parent_location_id}.")

            omega = rectangular_data_to_var(gridded_data=parent_omega)
            model.rate["omega"] = constraint_from_rectangular_data(
                rate_var=omega,
                default_age_time=self.age_time_grid
            )
            
            locations = set(omega_df.location_id.unique().tolist())
            children_without_omega = set(children) - set(locations)
            if children_without_omega:
                LOG.warning(f"Children of {parent_location_id} missing omega {children_without_omega}"
                            f"so not including child omega constraints")
            else:
                for child in children:
                    child_omega = omega_df.loc[omega_df.location_id == child].copy()
                    assert not child_omega.empty
                    child_rate = rectangular_data_to_var(gridded_data=child_omega)

                    def child_effect(age, time):
                        return np.log(child_rate(age, time) / omega(age, time))
                    
                    model.random_effect[("omega", child)] = constraint_from_rectangular_data(
                        rate_var=child_effect,
                        default_age_time=self.age_time_grid
                    )
        return model

