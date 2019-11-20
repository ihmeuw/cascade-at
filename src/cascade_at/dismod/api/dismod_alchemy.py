import numpy as np
import pandas as pd
from numbers import Real

from cascade_at.core.log import get_loggers
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.constants import DensityEnum, IntegrandEnum, \
    INTEGRAND_TO_WEIGHT, WeightEnum, MulCovEnum, RateEnum, enum_to_dataframe

LOG = get_loggers(__name__)

DEFAULT_DENSITY = ["uniform", 0, -np.inf, np.inf]


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

        self.age_dict = None
        self.time_dict = None
    
    def fill_for_parent_child(self, **additional_option_kwargs):
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
        self.density = self.construct_density_table()

        self.node = self.construct_node_table(location_dag=self.inputs.location_dag)
        self.data = self.construct_data_table(df=self.inputs.dismod_data, node=self.node)
        self.covariate = self.construct_covariate_table(covariates=self.parent_child_model.covariates)
        covariate_index = dict(self.covariate[["covariate_name", "covariate_id"]].to_records(index=False))

        self.weight, self.weight_grid = self.construct_weight_grid_tables(
            weights=self.parent_child_model.get_weights(),
            age_dict=self.age_dict,
            time_dict=self.time_dict
        )

        interconnected_tables = self.construct_interconnected_tables(
            model=self.parent_child_model,
            location_df=self.inputs.location_dag.to_dataframe()
        )
        self.rate = interconnected_tables['rate']
        self.smooth = interconnected_tables['smooth']
        self.smooth_grid = interconnected_tables['smooth_grid']
        self.prior = interconnected_tables['prior']
        self.integrand = interconnected_tables['integrand']
        self.mulcov = interconnected_tables['mulcov']
        self.nslist = interconnected_tables['nslist']
        self.nslist_pair = interconnected_tables['nslist_pair']

        self.option = self.construct_option_table(**additional_option_kwargs)
       
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
    def construct_covariate_table(covariates):
        """
        Constructs the covariate table from a list of Covariate objects.

        :param covariates: List(cascade_at.model.covariate.Covariate)
        :return: pd.DataFrame()
        """
        covariates_reordered = list()
        lookup = {search.name: search for search in covariates}
        for special in ["sex", "one"]:
            if special in lookup:
                covariates_reordered.append(lookup[special])
                del lookup[special]
        for remaining in sorted(lookup.keys()):
            covariates_reordered.append(lookup[remaining])
        LOG.info(f"Writing covariates {', '.join(c.name for c in covariates_reordered)}")

        null_references = list()
        for check_ref_col in covariates_reordered:
            if not isinstance(check_ref_col.reference, Real):
                null_references.append(check_ref_col.name)
        if null_references:
            raise ValueError(f"Covariate columns without reference values {null_references}.")

        covariate_rename = dict()
        for covariate_idx, covariate_obj in enumerate(covariates_reordered):
            covariate_rename[covariate_obj.name] = f"x_{covariate_idx}"

        covariate_table = pd.DataFrame({
            "covariate_id": np.arange(len(covariates_reordered)),
            "covariate_name": [col.name for col in covariates_reordered],
            "reference": np.array([col.reference for col in covariates_reordered], dtype=np.float),
            "max_difference": np.array([col.max_difference for col in covariates_reordered], dtype=np.float)
        })
        return covariate_table
    
    @staticmethod
    def construct_avgint_table():
        pass
    
    @staticmethod
    def construct_density_table():
        return pd.DataFrame({
            'density_name': [x.name for x in DensityEnum]
        })

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
    def add_prior_smooth_entries(grid_name, grid, num_existing_priors, num_existing_grids,
                                 age_dict, time_dict):
        """
        Returns:
            (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        """
        age_count, time_count = (len(grid.ages), len(grid.times))
        prior_df = grid.priors
        assert len(prior_df) == (age_count * time_count + 1) * 3

        # Get the densities for the priors
        prior_df.loc[prior_df.density.isnull(), ["density", "mean", "lower", "upper"]] = DEFAULT_DENSITY
        prior_df["density_id"] = prior_df["density"].apply(lambda x: DensityEnum[x].value)
        prior_df["prior_id"] = prior_df.index + num_existing_priors
        prior_df["assigned"] = prior_df.density.notna()

        prior_df.rename(columns={"name": "prior_name"}, inplace=True)

        # Assign names to each of the priors
        null_names = prior_df.prior_name.isnull()
        prior_df.loc[~null_names, "prior_name"] = (
            prior_df.loc[~null_names, "prior_name"].astype(str) + "    " +
            prior_df.loc[~null_names, "prior_id"].astype(str)
        )
        prior_df.loc[null_names, "prior_name"] = prior_df.loc[null_names, "prior_name"].apply(
            lambda pid: f"{grid_name}_{pid}"
        )

        # TODO: EDIT THIS SO THAT THE PRIOR DF ONLY KEEPS NECESSARY COLUMNS
        #  BUT THAT THE AGES ARE FIXED FOR THE GRID DF!

        # Create the simple smooth data frame
        smooth_df = pd.DataFrame({
            "smooth_name": [grid_name],
            "n_age": [age_count],
            "n_time": [time_count]
        })

        # Create the grid entries
        long_table = prior_df.loc[prior_df.age_id.notna()][["age_id", "time_id", "prior_id", "kind"]]
        grid_df = long_table[["age_id", "time_id"]].sort_values(["age_id", "time_id"]).drop_duplicates()

        for kind in ["value", "dage", "dtime"]:
            grid_values = long_table.loc[long_table.kind == kind].drop("kind", axis="columns")
            grid_values.rename(columns={"prior_id": f"{kind}_prior_id"}, inplace=True)
            grid_df = grid_df.merge(grid_values, on=["age_id", "time_id"])

        grid_df = grid_df.sort_values(["age_id", "time_id"], axis=0).reindex()
        grid_df["const_value"] = np.nan
        grid_df["smooth_grid_id"] = grid_df.index + num_existing_grids

        return prior_df, smooth_df, grid_df

    @staticmethod
    def construct_interconnected_tables(model, location_df):
        """
        Loops through the items from a model object, which include
        rate, random_effect, alpha, beta, and gamma.

        Each of these are "grid" vars, so they need entries in prior,
        smooth, and smooth_grid. This function returns those tables.

        It also constructs the rate, integrand, and mulcov tables (alpha, beta, gamma),
        plus nslist and nslist_pair tables.

        Parameters:
            model: cascade_at.model.model.Model
            covariate_index: Dict #TODO: add this!
            location_df: pd.DataFrame

        Returns:
            Dict
        """
        smooth_table = pd.DataFrame()
        prior_table = pd.DataFrame()
        grid_table = pd.DataFrame()

        nslist = {}
        nslist_pair_table = pd.DataFrame()

        integrand_table = pd.DataFrame({"integrand_name": enum_to_dataframe(IntegrandEnum)["name"]})
        integrand_table["minimum_meas_cv"] = 0.0

        rate_table = pd.DataFrame({
            'rate_id': [rate.value for rate in RateEnum],
            'rate_name': [rate.name for rate in RateEnum],
            'parent_smooth_id': np.nan,
            'child_smooth_id': np.nan,
            'child_nslist_id': np.nan
        })

        if "rate" in model:
            for rate_name, grid in model["rate"].items():
                """
                Loop through each of the rates and add entries into the
                prior, and smooth tables. Also put an entry in the rate table so we know the
                parent smooth ID.
                """
                prior, smooth, grid = DismodAlchemy.add_prior_smooth_entries(
                    grid_name=rate_name, grid=grid,
                    num_existing_priors=len(prior_table),
                    num_existing_grids=len(grid_table)
                )

                smooth_id = len(smooth_table)
                smooth['smooth_id'] = smooth_id
                grid['smooth_id'] = smooth_id

                smooth_table = smooth_table.append(smooth)
                prior_table = prior_table.append(prior)
                grid_table = grid_table.append(grid)

                rate_table.loc[rate_table.rate_id == RateEnum[rate_name].value, "parent_smooth_id"] = smooth_id
        
        if "random_effect" in model:
            for (rate_name, child_location), grid in model["random_effect"].items():
                """
                Loop through each of the random effects and add entries
                into the prior and smooth tables.
                """
                grid_name = f"{rate_name}_re"
                if child_location is not None:
                    grid_name = grid_name + f"_{child_location}"
                
                prior, smooth, grid = DismodAlchemy.add_prior_smooth_entries(
                    grid_name=grid_name, grid=grid,
                    num_existing_priors=len(prior_table),
                    num_existing_grids=len(grid_table)
                )

                smooth_id = len(smooth_table)
                smooth["smooth_id"] = smooth_id
                grid["smooth_id"] = smooth_id

                smooth_table = smooth_table.append(smooth)
                prior_table = prior_table.append(prior)
                grid_table = grid_table.append(grid)

                if child_location is None:
                    rate_table.loc[rate_table.rate_id == RateEnum[rate_name].value, "child_smooth_id"] = smooth_id
                else:
                    # If we are doing this for a child location, then we want to make entries in the
                    # nslist and nslist_pair tables
                    node_id = location_df[location_df.location_id == child_location].node_id.iloc[0]
                    if rate_name not in nslist:
                        ns_id = len(nslist)
                        nslist[rate_name] = ns_id
                    else:
                        ns_id = nslist[rate_name]
                        nslist_pair_table.append(pd.DataFrame({
                            'nslist_id': ns_id,
                            'node_id': node_id,
                            'smooth_id': smooth_id
                        }))

        mulcov_table = []
        potential_mulcovs = ["alpha", "beta", "gamma"]
        mulcovs = [x for x in potential_mulcovs if x in model]

        for m in mulcovs:
            for (covariate, rate_or_integrand), grid in model[m].items():
                grid_name = f"{m}_{rate_or_integrand}_{covariate}"

                prior, smooth, grid = DismodAlchemy.add_prior_smooth_entries(
                    grid_name=grid_name, grid=grid,
                    num_existing_priors=len(prior_table),
                    num_existing_grids=len(grid_table)
                )
                smooth_id = len(smooth_table)
                smooth["smooth_id"] = smooth_id
                grid["smooth_id"] = smooth_id

                prior_table = prior_table.append(prior)
                smooth_table = smooth_table.append(smooth)
                grid_table = grid_table.append(grid)

                mulcov = pd.DataFrame({
                    "mulcov_type": [MulCovEnum[m].value],
                    "rate_id": [np.nan],
                    "integrand_id": [np.nan],
                    "covariate_id": [covariate],
                    "smooth_id": smooth_id
                })
                if m == "alpha":
                    mulcov["rate_id"] = RateEnum[rate_or_integrand].value
                elif m in ["beta", "gamma"]:
                    mulcov["integrand_id"] = IntegrandEnum[rate_or_integrand].value
                else:
                    raise RuntimeError(f"Unknown mulcov type {m}.")
                mulcov_table.append(mulcov)

        mulcov_table = pd.concat(mulcov_table).reset_index(drop=True)
        mulcov_table["mulcov_id"] = mulcov_table.index

        nslist_table = pd.DataFrame.from_records(
            data=list(nslist.items()),
            columns=["nslist_name", "nslist_id"]
        )

        return {
            'rate': rate_table,
            'integrand': integrand_table,
            'prior': prior_table,
            'smooth': smooth_table,
            'mulcov': mulcov_table,
            'nslist': nslist_table,
            'nslist_pair': nslist_pair_table
        }

    @staticmethod
    def construct_sample_table():
        pass
    
    @staticmethod
    def construct_start_var_table():
        pass
    
    @staticmethod
    def construct_truth_var_table():
        pass

