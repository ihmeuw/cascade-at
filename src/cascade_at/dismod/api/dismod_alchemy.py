from cascade_at.core.log import get_loggers
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.dismod_ids import DensityEnum, IntegrandEnum, INTEGRAND_TO_WEIGHT

LOG = get_loggers(__name__)


class DismodAlchemy(DismodIO):
    """
    Sits on top of the DismodIO class,
    and takes inputs from the model construction,
    putting them into the Dismod database tables
    in the correct construction.
    """
    def __init__(self, engine):
        super().__init__(engine=engine)
    
    def initialize(self, inputs, model_construct):
        """
        Initializes the Dismod database with inputs
        and a model construction for a parent location
        and its descendents.
        """
        self.construct_node_table(location_dag=inputs.location_dag)
        self.construct_data_table(df=inputs.dismod_data)

    def construct_node_table(self, location_dag):
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
        self.node = node
        
    def construct_data_table(self, df):
        """
        Constructs the data table from input df.

        Parameters:
            df: (pd.DataFrame)
        """
        data = df.copy()
        data.rename(columns={
            "location_id": "c_location_id",
            "location": "c_location"
        }, inplace=True)
        data["c_location_id"] = data["c_location_id"].astype(str)
        data = data.merge(
            self.node[["node_id", "c_location_id"]],
            on=["c_location_id"]
        )
        data["density_id"] = data["density"].apply(lambda x: DensityEnum[x].value)
        data["integrand_id"] = data["integrand"].apply(lambda x: IntegrandEnum[x].value)
        data["weight_id"] = data["integrand"].apply(lambda x: INTEGRAND_TO_WEIGHT[x].value)
        data.drop(['integrand', 'density'], axis=1, inplace=True)

        data.reset_index(inplace=True, drop=True)
        data["data_name"] = data.index.astype(str)

        self.data = data
