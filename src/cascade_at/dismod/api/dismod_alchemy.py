import numpy as np
import pandas as pd

from cascade_at.core.log import get_loggers
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.constants import DensityEnum, IntegrandEnum, INTEGRAND_TO_WEIGHT, WeightEnum

LOG = get_loggers(__name__)


class DismodAlchemy(DismodIO):
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

    Attributes:
        self.parent_child_model: (cascade_at.model.model.Model) that was constructed from grid_alchemy parameter
            for one specific parent and its descendents

    Example:
        >>> from pathlib import Path
        >>> from cascade_at.collector.grid_alchemy import Alchemy
        >>> from cascade_at.collector.measurement_inputs import MeasurementInputsFromSettings
        >>> from cascade_at.settings.base_case import BASE_CASE
        >>> from cascade_at.settings.settings import load_settings

        >>> settings = load_settings(BASE_CASE)
        >>> inputs = MeasurementInputsFromSettings(settings)
        >>> inputs.demographics.location_id = [102, 555] # subset the locations to make it go faster
        >>> inputs.get_raw_inputs()
        >>> inputs.configure_inputs_for_dismod(settings)
        >>> alchemy = Alchemy(settings)

        >>> da = DismodAlchemy(path=Path('temp.db'),
        >>>                    settings_configuration=settings,
        >>>                    measurement_inputs=inputs,
        >>>                    grid_alchemy=alchemy,
        >>>                    parent_location_id=1)
        >>> da.fill_for_parent_child()
    """
    def __init__(self, path, settings_configuration, measurement_inputs, grid_alchemy, parent_location_id):
        super().__init__(path=path)
        self.settings = settings_configuration
        self.inputs = measurement_inputs
        self.alchemy = grid_alchemy
        self.parent_location_id = parent_location_id

        self.parent_child_model = self.alchemy.construct_two_level_model(
            location_dag=self.inputs.location_dag,
            parent_location_id=self.parent_location_id,
            covariate_specs=self.inputs.covariate_specs
        )
    
    def fill_for_parent_child(self, **option_kwargs):
        """
        Fills the Dismod database with inputs
        and a model construction for a parent location
        and its descendents.

        Pass in some optional keyword arguments to fill the option
        table with additional info or to over-ride the defaults.
        """
        LOG.info(f"Filling tables in {self.path.absolute()}")
        self.age = self.construct_age_time_table(
            variable_name='age', variable=self.parent_child_model.get_age_array()
        )
        self.time = self.construct_age_time_table(
            variable_name='time', variable=self.parent_child_model.get_time_array()
        )

        self.age_dict = dict(zip(self.age.age_id, self.age.age))
        self.time_dict = dict(zip(self.time.time_id, self.time.time))

        self.node = self.construct_node_table(location_dag=self.inputs.location_dag)
        self.data = self.construct_data_table(df=self.inputs.dismod_data, node=self.node)
        self.weight, self.weight_grid = self.construct_weight_grid_tables(
            weights=self.parent_child_model.get_weights(),
            age_dict=self.age_dict,
            time_dict=self.time_dict
        )
        
        self.option = self.construct_option_table(**option_kwargs)
       
    def construct_option_table(self, **kwargs):
        """
        Construct the option table with the default arguments,
        and if needed can pass in some kwargs to update the dictionary
        with new options or over-ride old options.
        """
        LOG.info("Filling option table.")

        option_dict = {
            'parent_location_id': self.parent_location_id,
            'random_seed': self.settings.model.random_seed,
            'ode_step_size': self.settings.model.ode_step_size,
            'max_num_iter_fixed': self.settings.max_num_iter.fixed,
            'max_num_iter_random': self.settings.max_num_iter.random,
            'print_level_fixed': self.settings.print_level.fixed,
            'print_level_random': self.settings.print_level.random,
            'accept_after_max_steps_fixed': self.settings.print_level.fixed,
            'accept_after_max_steps_random': self.settings.print_level.random,
            'tolerance_fixed': self.settings.tolerance.fixed,
            'tolerance_random': self.settings.tolerance.random,
            'rate_case': self.parent_child_model.get_nonzero_rates()
        }
        option_dict.update(**kwargs)

        df = pd.DataFrame()
        df = df.append([pd.Series({'option_name': k, 'option_value': v}) for k, v in option_dict.items()])
        df['option_id'] = df.index
        return df

    @staticmethod
    def construct_age_time_table(variable_name, variable):
        """
        Constructs the age or time table with age_id and age or time_id and time.
        Has unique identifiers for each.

        Parameters:
            variable_name: (str) one of 'age' or 'time'
            variable: (np.array) array of ages or times
        """
        LOG.info(f"Constructing {variable_name} table.")
        variable = variable[np.unique(variable.round(decimals=14), return_index=True)[1]]
        variable.sort()
        if variable[-1] - variable[0] < 1:
            variable = np.append(variable, variable[-1] + 1)
        df = pd.DataFrame(dict(id=range(len(variable)), var=variable))
        df.rename(columns={'id': f'{variable_name}_id', 'var': variable_name}, inplace=True)
        return df

    @staticmethod
    def construct_node_table(location_dag):
        """
        Constructs the node table from a location
        DAG's to_dataframe() method.

        Parameters:
            location_dag: (cascade_at.inputs.locations.LocationDAG)
        """
        LOG.info("Constructing node table.")
        node = location_dag.to_dataframe()
        node = node.reset_index(drop=True)
        node["node_id"] = node.index
        p_node = node[["node_id", "location_id"]].rename(
            columns={"location_id": "parent_id", "node_id": "parent"}
        )
        node = node.merge(p_node, on="parent_id")
        node.rename(columns={
            "name": "node_name",
            "location_id": "c_location_id"
        }, inplace=True)
        node = node[['node_id', 'node_name', 'parent', 'c_location_id']]
        return node

    @staticmethod
    def construct_data_table(df, node):
        """
        Constructs the data table from input df.

        Parameters:
            df: (pd.DataFrame) data frame of inputs that have been prepped for dismod
            node: (pd.DataFrame) the dismod node table
        """
        LOG.info("Constructing data table.")
        data = df.copy()
        data.rename(columns={
            "location_id": "c_location_id",
            "location": "c_location"
        }, inplace=True)
        data['c_location_id'] = data['c_location_id'].astype(int)
        data = data.merge(
            node[["node_id", "c_location_id"]],
            on=["c_location_id"]
        )
        data["density_id"] = data["density"].apply(lambda x: DensityEnum[x].value)
        data["integrand_id"] = data["measure"].apply(lambda x: IntegrandEnum[x].value)
        data["weight_id"] = data["measure"].apply(lambda x: INTEGRAND_TO_WEIGHT[x].value)
        data.drop(['measure', 'density'], axis=1, inplace=True)

        data.reset_index(inplace=True, drop=True)
        data["data_name"] = data.index.astype(str)

        data = data[[
            'data_name', 'integrand_id', 'density_id', 'node_id', 'weight_id',
            'hold_out', 'meas_value', 'meas_std', 'eta', 'nu',
            'age_lower', 'age_upper', 'time_lower', 'time_upper'
        ]]

        return data

    @staticmethod
    def construct_weight_grid_tables(weights, age_dict, time_dict):
        """
        Constructs the weight and weight_grid tables."

        Parameters:
            weights (Dict[str, Var]): There are four kinds of weights:
                "constant", "susceptible", "with_condition", and "total".
                No other weights are used.
            age_dict (Dict[int, float]): dictionary of age ID to age
            time_dict (Dict[int, float]): dictionary of time ID to time
        
        Returns:
            (pd.DataFrame, pd.DataFrame) the weight table and the weight grid table
        """
        LOG.info("Constructing weight and weight grid tables.")

        names = [w.name for w in WeightEnum]
        weight = pd.DataFrame({
            'weight_id': [w.value for w in WeightEnum],
            'weight_name': names,
            'n_age': [len(weights[name].ages) for name in names],
            'n_time': [len(weights[name].times) for name in names]
        })
        weight_grid = []
        for w in WeightEnum:
            LOG.info(f"Writing weight {w.name}.")
            one_grid = weights[w.name].grid[["age", "time", "mean"]].rename(columns={"mean": "weight"})
            one_grid["weight_id"] = w.value
            weight_grid.append(one_grid)
        weight_grid = pd.concat(weight_grid).reset_index(drop=True)
        weight_grid["age_id"] = weight_grid["age"].map({v: k for k, v in age_dict.items()})
        weight_grid["time_id"] = weight_grid["time"].map({v: k for k, v in time_dict.items()})
        weight_grid["weight_grid_id"] = weight_grid.index
        weight_grid.drop(columns=["age", "time"], inplace=True, axis=1)
        return weight, weight_grid

    @staticmethod
    def construct_covariate_table():
        pass

    @staticmethod
    def construct_mulcov_table():
        pass
    
    @staticmethod
    def construct_avgint_table():
        pass
    
    @staticmethod
    def construct_density_table():
        pass

    @staticmethod
    def construct_constraint_table():
        pass
    
    @staticmethod
    def construct_depend_var_table():
        pass
    
    @staticmethod
    def construct_fit_var_table():
        pass
    
    @staticmethod
    def construct_integrand_table():
        pass
    
    @staticmethod
    def construct_smooth_grid_prior_tables():

        pass
    
    @staticmethod
    def construct_rate_table():
        pass
    
    @staticmethod
    def construct_nslist_table():
        pass
    
    @staticmethod
    def construct_nslist_pair_table():
        pass
    
    @staticmethod
    def construct_sample_table():
        pass
    
    @staticmethod
    def construct_start_var_table():
        pass
    
    @staticmethod
    def construct_truth_var_table():
        pass
    
    