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
        sex_id: (int) the sex that this database will be run for

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
    def __init__(self, path, settings_configuration, measurement_inputs, grid_alchemy, parent_location_id, sex_id):
        super().__init__(path=path)
        self.settings = settings_configuration
        self.inputs = measurement_inputs
        self.alchemy = grid_alchemy
        self.parent_location_id = parent_location_id
        self.sex_id = sex_id

        self.covariate_specs_with_valid_reference = self.inputs.calculate_country_covariate_reference_values(
            parent_location_id=self.parent_location_id,
            sex_id=self.sex_id
        )
        self.parent_child_model = self.alchemy.construct_two_level_model(
            location_dag=self.inputs.location_dag,
            parent_location_id=self.parent_location_id,
            covariate_specs=self.covariate_specs_with_valid_reference
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
        self.age = self.construct_age_time_table(
            variable_name='age', variable=self.parent_child_model.get_age_array()
        )
        self.time = self.construct_age_time_table(
            variable_name='time', variable=self.parent_child_model.get_time_array()
        )
        self.density = self.construct_density_table()
        self.node = self.construct_node_table(location_dag=self.inputs.location_dag)
        self.covariate = self.construct_covariate_table(covariates=self.parent_child_model.covariates)
        self.data = self.construct_data_table(
            df=self.inputs.dismod_data,
            node=self.node,
        )

        self.weight, self.weight_grid = self.construct_weight_grid_tables(
            weights=self.parent_child_model.get_weights(),
            age_df=self.age, time_df=self.time
        )
        interconnected_tables = self.construct_interconnected_tables(
            model=self.parent_child_model,
            location_df=self.inputs.location_dag.to_dataframe(),
            age_df=self.age, time_df=self.time,
            covariate_df=self.covariate
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
    def convert_age_time_to_id(df, age_df, time_df):
        """
        Converts the times and ages to IDs based on a dictionary passed
        that should be made from the age or time table. Gets the "closest"
        age or time.

        :param df: pd.DataFrame
        :param age_df: pd.DataFrame
        :param time_df: pdDataFrame
        :return:
        """
        at_tables = {'age': age_df, 'time': time_df}
        assert "age" in df.columns
        assert "time" in df.columns
        df = df.assign(save_idx=df.index)
        for dat in ["age", "time"]:
            col_id = f"{dat}_id"
            sort_by = df.sort_values(dat)
            in_grid = sort_by[dat].notna()
            at_table = at_tables[dat]
            aged = pd.merge_asof(sort_by[in_grid], at_table, on=dat, direction="nearest")
            df = df.merge(aged[["save_idx", col_id]], on="save_idx", how="left")
        assert "age_id" in df.columns
        assert "time_id" in df.columns
        return df.drop(["save_idx", "age", "time"], axis=1)

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
    def construct_weight_grid_tables(weights, age_df, time_df):
        """
        Constructs the weight and weight_grid tables."

        Parameters:
            weights (Dict[str, Var]): There are four kinds of weights:
                "constant", "susceptible", "with_condition", and "total".
                No other weights are used.
            age_df (pd.DataFrame)
            time_df (pd.DataFrame)
        
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

        weight_grid = DismodAlchemy.convert_age_time_to_id(
            df=weight_grid, age_df=age_df, time_df=time_df
        )
        weight_grid["weight_grid_id"] = weight_grid.index
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
    def construct_density_table():
        return pd.DataFrame({
            'density_name': [x.name for x in DensityEnum]
        })

    @staticmethod
    def add_prior_smooth_entries(grid_name, grid, num_existing_priors, num_existing_grids,
                                 age_df, time_df):
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

        # Convert to age and time ID for prior table
        prior_df = DismodAlchemy.convert_age_time_to_id(
            df=prior_df, age_df=age_df, time_df=time_df
        )

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

        prior_df = prior_df[[
            'prior_id', 'prior_name', 'lower', 'upper',
            'mean', 'std', 'eta', 'nu', 'density_id'
        ]].sort_values(by='prior_id').reset_index(drop=True)

        return prior_df, smooth_df, grid_df

    @staticmethod
    def default_integrand_table():
        return pd.DataFrame({
            "integrand_name": enum_to_dataframe(IntegrandEnum)["name"],
            "minimum_meas_cv": 0.0
        })

    @staticmethod
    def default_rate_table():
        return pd.DataFrame({
            'rate_id': [rate.value for rate in RateEnum],
            'rate_name': [rate.name for rate in RateEnum],
            'parent_smooth_id': np.nan,
            'child_smooth_id': np.nan,
            'child_nslist_id': np.nan
        })

    @staticmethod
    def construct_interconnected_tables(model, location_df, age_df, time_df, covariate_df):
        """
        Loops through the items from a model object, which include
        rate, random_effect, alpha, beta, and gamma.

        Each of these are "grid" vars, so they need entries in prior,
        smooth, and smooth_grid. This function returns those tables.

        It also constructs the rate, integrand, and mulcov tables (alpha, beta, gamma),
        plus nslist and nslist_pair tables.

        Parameters:
            model: cascade_at.model.model.Model
            location_df: pd.DataFrame
            age_df: pd.DataFrame
            time_df: pd.DataFrame
            covariate_df: pd.DataFrame

        Returns:
            Dict
        """
        nslist = {}
        smooth_table = pd.DataFrame()
        prior_table = pd.DataFrame()
        grid_table = pd.DataFrame()
        mulcov_table = pd.DataFrame()
        nslist_pair_table = pd.DataFrame()

        integrand_table = DismodAlchemy.default_integrand_table()
        rate_table = DismodAlchemy.default_rate_table()

        covariate_index = dict(covariate_df[["covariate_name", "covariate_id"]].to_records(index=False))

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
                    num_existing_grids=len(grid_table),
                    age_df=age_df, time_df=time_df
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
                    num_existing_grids=len(grid_table),
                    age_df=age_df, time_df=time_df
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

        potential_mulcovs = ["alpha", "beta", "gamma"]
        mulcovs = [x for x in potential_mulcovs if x in model]

        for m in mulcovs:
            import pdb; pdb.set_trace()
            for (covariate, rate_or_integrand), grid in model[m].items():
                grid_name = f"{m}_{rate_or_integrand}_{covariate}"

                prior, smooth, grid = DismodAlchemy.add_prior_smooth_entries(
                    grid_name=grid_name, grid=grid,
                    num_existing_priors=len(prior_table),
                    num_existing_grids=len(grid_table),
                    age_df=age_df, time_df=time_df
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
                    "covariate_id": [covariate_index[covariate]],
                    "smooth_id": [smooth_id]
                })
                if m == "alpha":
                    mulcov["rate_id"] = RateEnum[rate_or_integrand].value
                elif m in ["beta", "gamma"]:
                    mulcov["integrand_id"] = IntegrandEnum[rate_or_integrand].value
                else:
                    raise RuntimeError(f"Unknown mulcov type {m}.")
                mulcov_table.append(mulcov)

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
            'smooth_grid': grid_table,
            'mulcov': mulcov_table,
            'nslist': nslist_table,
            'nslist_pair': nslist_pair_table
        }

    @staticmethod
    def construct_avgint_table():
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
    def construct_sample_table():
        pass
    
    @staticmethod
    def construct_start_var_table():
        pass
    
    @staticmethod
    def construct_truth_var_table():
        pass

