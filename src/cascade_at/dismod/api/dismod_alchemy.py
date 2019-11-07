import numpy as np
import pandas as pd

from cascade_at.core.log import get_loggers
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.dismod_ids import DensityEnum, IntegrandEnum, INTEGRAND_TO_WEIGHT

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
    """
    def __init__(self, path, settings_configuration, measurement_inputs, grid_alchemy, parent_location_id):
        super().__init__(path=path)
        self.settings_configuration = settings_configuration
        self.measurement_inputs = measurement_inputs
        self.grid_alchemy = grid_alchemy
        self.parent_location_id = parent_location_id

        self.parent_child_model = self.grid_alchemy.construct_two_level_model(
            location_dag=self.measurement_inputs.location_dag,
            parent_location_id=self.parent_location_id,
            covariate_specs=self.measurement_inputs.CovariateSpecs
        )
    
    def fill_for_parent_child(self):
        """
        Fills the Dismod database with inputs
        and a model construction for a parent location
        and its descendents.
        """
        self.age = self.construct_age_time_table(
            variable_name='age', variable=self.parent_child_model.get_grid_ages()
        )
        self.time = self.construct_age_time_table(
            variable_name='time', variable=self.parent_child_model.get_grid_times()
        )
        self.node = self.construct_node_table(location_dag=self.measurement_inputs.location_dag)
        self.data = self.construct_data_table(df=self.measurement_inputs.dismod_data, node=self.node)

    @staticmethod
    def construct_age_time_table(variable_name, variable):
        """
        Constructs the age or time table with age_id and age or time_id and time.
        Has unique identifiers for each.

        :param variable_name: (str)
        :param variable: ()
        :return:
        """
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
        node = location_dag.to_dataframe()
        node.rename(columns={
            "name": "node_name",
            "location_id": "c_location_id"
        })
        node = node.reset_index(drop=True)
        node["node_id"] = node.index
        return node

    @staticmethod
    def construct_data_table(df, node):
        """
        Constructs the data table from input df.

        Parameters:
            df: (pd.DataFrame)
            node: (pd.DataFrame) the dismod node table
        """
        data = df.copy()
        data.rename(columns={
            "location_id": "c_location_id",
            "location": "c_location"
        }, inplace=True)
        data["c_location_id"] = data["c_location_id"].astype(str)
        data = data.merge(
            node[["node_id", "c_location_id"]],
            on=["c_location_id"]
        )
        data["density_id"] = data["density"].apply(lambda x: DensityEnum[x].value)
        data["integrand_id"] = data["integrand"].apply(lambda x: IntegrandEnum[x].value)
        data["weight_id"] = data["integrand"].apply(lambda x: INTEGRAND_TO_WEIGHT[x].value)
        data.drop(['integrand', 'density'], axis=1, inplace=True)

        data.reset_index(inplace=True, drop=True)
        data["data_name"] = data.index.astype(str)

        return data
