import pandas as pd

from cascade_at.core.log import get_loggers
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.api.fill_extract_helpers import reference_tables, data_tables, grid_tables

LOG = get_loggers(__name__)


class DismodFiller(DismodIO):
    """
    Sits on top of the DismodIO class,
    and takes everything from the collector module
    and puts them into the Dismod database tables
    in the correct construction.

    Parameters:
        path: (pathlib.Path)
        settings_configuration: (cascade_at.collector.settings_configuration.SettingsConfiguration)
        measurement_inputs: (cascade_at.collector.measurement_inputs.MeasurementInputs)
        grid_alchemy: (cascade_at.collector.grid_alchemy.GridAlchemy)
        parent_location_id: (int) which parent location to construct the database for
        sex_id: (int) the sex that this database will be run for

    Attributes:
        self.parent_child_model: (cascade_at.model.model.Model) that was constructed from grid_alchemy parameter
            for one specific parent and its descendents

    Example:
        >>> from pathlib import Path
        >>> from cascade_at.model.grid_alchemy import Alchemy
        >>> from cascade_at.inputs.measurement_inputs import MeasurementInputsFromSettings
        >>> from cascade_at.settings.base_case import BASE_CASE
        >>> from cascade_at.settings.settings import load_settings

        >>> settings = load_settings(BASE_CASE)
        >>> inputs = MeasurementInputsFromSettings(settings)
        >>> inputs.demographics.location_id = [102, 555] # subset the locations to make it go faster
        >>> inputs.get_raw_inputs()
        >>> inputs.configure_inputs_for_dismod(settings)
        >>> alchemy = Alchemy(settings)

        >>> da = DismodFiller(path=Path('temp.db'),
        >>>                    settings_configuration=settings,
        >>>                    measurement_inputs=inputs,
        >>>                    grid_alchemy=alchemy,
        >>>                    parent_location_id=1,
        >>>                    sex_id=3)
        >>> da.fill_for_parent_child()
    """
    def __init__(self, path, settings_configuration, measurement_inputs, grid_alchemy, parent_location_id, sex_id,
                 child_prior=None):
        super().__init__(path=path)

        self.settings = settings_configuration
        self.inputs = measurement_inputs
        self.alchemy = grid_alchemy
        self.parent_location_id = parent_location_id
        self.sex_id = sex_id
        self.child_prior = child_prior

        self.omega_df = self.get_omega_df()
        self.covariate_reference_specs = self.calculate_reference_covariates()
        self.parent_child_model = self.get_parent_child_model()

        self.min_age = self.inputs.dismod_data.age_lower.min()
        self.max_age = self.inputs.dismod_data.age_upper.max()

        self.min_time = self.inputs.dismod_data.time_lower.min()
        self.max_time = self.inputs.dismod_data.time_upper.max()

    def get_omega_df(self):
        """
        Get the correct omega data frame for this two-level model.

        :return: pd.DataFrame
        """
        if self.inputs.omega is not None:
            omega_df = self.inputs.omega.loc[self.inputs.omega.sex_id == self.sex_id].copy()
        else:
            omega_df = None
        return omega_df

    def get_parent_child_model(self):
        """
        Construct a two-level model that corresponds to this parent location ID
        and its children.

        :return: (cascade_at.model.model.Model)
        """
        return self.alchemy.construct_two_level_model(
            location_dag=self.inputs.location_dag,
            parent_location_id=self.parent_location_id,
            covariate_specs=self.covariate_reference_specs,
            omega_df=self.omega_df,
            update_prior=self.child_prior
        )

    def calculate_reference_covariates(self):
        """
        Calculates reference covariate values based on the input object
        and the parent/sex we have in the two-level model.
        Modifies the baseline covariate specs object.

        :return: (cascade_at.inputs.covariate_specs.CovariateSpecs)
        """
        return self.inputs.calculate_country_covariate_reference_values(
            parent_location_id=self.parent_location_id,
            sex_id=self.sex_id
        )

    def fill_for_parent_child(self, **additional_option_kwargs):
        """
        Fills the Dismod database with inputs
        and a model construction for a parent location
        and its descendents.

        Pass in some optional keyword arguments to fill the option
        table with additional info or to over-ride the defaults.
        """
        LOG.info(f"Filling tables in {self.path.absolute()}")
        self.fill_reference_tables()
        self.fill_grid_tables()
        self.fill_data_tables()
        self.option = self.construct_option_table(**additional_option_kwargs)

    def node_id_from_location_id(self, location_id):
        """
        Get the node ID from a location ID in an already created node table.
        """
        loc_df = self.node.loc[self.node.c_location_id == location_id]
        if len(loc_df) > 1:
            raise RuntimeError("Problem with the node table -- should only be one node-id for each location_id.")
        return loc_df['node_id'].iloc[0]

    def fill_reference_tables(self):
        """
        Fills all of the reference tables including density, node, covariate, age, and time.

        :return: self
        """
        self.density = reference_tables.construct_density_table()
        self.node = reference_tables.construct_node_table(location_dag=self.inputs.location_dag)
        self.covariate = reference_tables.construct_covariate_table(covariates=self.parent_child_model.covariates)
        self.age = reference_tables.construct_age_time_table(
            variable_name='age', variable=self.parent_child_model.get_age_array(),
            data_min=self.min_age, data_max=self.max_age
        )
        self.time = reference_tables.construct_age_time_table(
            variable_name='time', variable=self.parent_child_model.get_time_array(),
            data_min=self.min_time, data_max=self.max_time
        )
        self.integrand = reference_tables.construct_integrand_table(
            data_cv_from_settings=self.inputs.data_cv_from_settings(settings=self.settings)
        )
        return self

    def fill_data_tables(self):
        """
        Fills the data tables including data and avgint.

        :return: self
        """
        self.data = data_tables.construct_data_table(
            df=self.inputs.dismod_data,
            node_df=self.node,
            covariate_df=self.covariate,
            ages=self.parent_child_model.get_age_array(),
            times=self.parent_child_model.get_time_array()
        )
        avgint_df = self.inputs.to_gbd_avgint(
            parent_location_id=self.parent_location_id,
            sex_id=self.sex_id
        )
        self.avgint = data_tables.construct_gbd_avgint_table(
            df=avgint_df,
            node_df=self.node,
            covariate_df=self.covariate,
            integrand_df=self.integrand,
            ages=self.parent_child_model.get_age_array(),
            times=self.parent_child_model.get_time_array()
        )
        return self

    def fill_grid_tables(self):
        """
        Fills the grid-like tables including weight,
        rate, smooth, smooth_grid, prior, integrand,
        mulcov, nslist, nslist_pair.

        :return: self
        """
        self.weight, self.weight_grid = grid_tables.construct_weight_grid_tables(
            weights=self.parent_child_model.get_weights(),
            age_df=self.age, time_df=self.time
        )
        model_tables = grid_tables.construct_model_tables(
            model=self.parent_child_model,
            location_df=self.node,
            age_df=self.age, time_df=self.time,
            covariate_df=self.covariate
        )
        self.rate = model_tables['rate']
        self.smooth = model_tables['smooth']
        self.smooth_grid = model_tables['smooth_grid']
        self.prior = model_tables['prior']
        self.mulcov = model_tables['mulcov']
        self.nslist = model_tables['nslist']
        self.nslist_pair = model_tables['nslist_pair']
        self.subgroup = model_tables['subgroup']

        # Initialize empty tables that need to be there that may or may not
        # be filled with relevant info, if they're currently empty.
        for name in ["nslist", "nslist_pair", "mulcov", "smooth_grid", "smooth"]:
            if getattr(self, name).empty:
                setattr(self, name, self.empty_table(table_name=name))

    def construct_option_table(self, **kwargs):
        """
        Construct the option table with the default arguments,
        and if needed can pass in some kwargs to update the dictionary
        with new options or over-ride old options.
        """
        LOG.info("Filling option table.")

        option_dict = {
            'parent_node_id': self.node_id_from_location_id(location_id=self.parent_location_id),
            'random_seed': self.settings.model.random_seed,
            'ode_step_size': self.settings.model.ode_step_size,
            'rate_case': self.settings.model.rate_case,
            'meas_noise_effect': self.settings.policies.meas_std_effect
        }
        for kind in ['fixed', 'random']:
            for opt in ['derivative_test', 'max_num_iter', 'print_level', 'accept_after_max_steps', 'tolerance']:
                if hasattr(self.settings, opt):
                    setting = getattr(self.settings, opt)
                    if not setting.is_field_unset(kind):
                        option_dict.update({f'{opt}_{kind}': getattr(setting, kind)})
        if not self.settings.model.is_field_unset("addl_ode_stpes"):
            option_dict.update({'age_avg_split': " ".join(str(a) for a in self.settings.model.addl_ode_stpes)})
        if not self.settings.model.is_field_unset("quasi_fixed"):
            option_dict.update({'quasi_fixed': str(self.settings.model.quasi_fixed == 1).lower()})
            option_dict.update({'bound_frac_fixed': self.settings.model.bound_frac_fixed})
        if not self.settings.model.is_field_unset("zero_sum_random"):
            option_dict.update({'zero_sum_child_rate': " ".join(self.settings.model.zero_sum_random)})
        if not self.settings.policies.is_field_unset("limited_memory_max_history_fixed"):
            option_dict.update(
                {'limited_memory_max_history_fixed': self.settings.policies.limited_memory_max_history_fixed}
            )
        option_dict.update(**kwargs)

        df = pd.DataFrame()
        df = df.append([pd.Series({'option_name': k, 'option_value': v}) for k, v in option_dict.items()])
        df['option_id'] = df.index
        df['option_value'] = df['option_value'].astype(str)

        return df
