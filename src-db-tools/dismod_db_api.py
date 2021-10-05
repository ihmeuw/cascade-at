#! /usr/bin/env python

"""
It seems the best practice for caching is to just let sqlite3 do the work (e.g., do not have an python object of pandas dataframes that represent the database content.)

"""

import sys
import os
import sqlite3
import pandas as pd; pd.set_option('expand_frame_repr', False)
import numpy as np
import pdb
from pdb import set_trace
from collections import OrderedDict

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dismod_db_api')
logger.setLevel(logging.WARNING)

try:
    from cascade_at_gma.lib.cached_property import cached_property
except ImportError:
    class cached_property(object):
        """
        A property that is only computed once per instance and then replaces itself
        with an ordinary attribute. Deleting the attribute resets the property.
        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

        def __init__(self, func):
            self.__doc__ = getattr(func, '__doc__')
            self.func = func

        def __get__(self, obj, cls):
            if obj is None:
                return self
            value = obj.__dict__[self.func.__name__] = self.func(obj)
            return value

try:
    from cascade_at_gma.lib.utilities import flatten, startswith_filter
except ImportError:
    def startswith_filter(startswith, strings):
        def tst(c):
            return c.startswith(startswith)
        return list(filter(tst, strings))

    def flatten(l):
        "Flatten a nested sequence into the elements of the sequence."
        for el in l:
            if isinstance(el, collections.Iterable) and not isinstance(el, str):
                for sub in flatten(el):
                    yield sub
            else:
                yield el

nan_int = -1
_isolation_level_ = None    # autocommit
_isolation_level_ = 'exclusive'

def set_option(DB, key, value):
    option = DB.option
    if key in option.option_name.values:
        option.loc[option.option_name == key, 'option_value'] = value
    else:
        option = option.append(dict(option_name = key, option_value = value), ignore_index=True)
    DB.option=option

def rename_covariates(df):
    from warnings import warn
    # warn("FIXME -- Remove the covariate name changer in the API once the covariate table meta information has stabilized.")
    return df
    def _rename(n):
        if 'sex' in n: return 'x_sex'
        if 'one' in n: return 'x_one'
        return f"x_{n.rsplit('_', 1)[0]}"
    df['covariate_name'] = df['covariate_name'].map(_rename)
    return df

class DismodDbAPI_backing_store(object):
    """
    # FIXME -- pick a better name than backing_store for something that can be either an in-object cache, or an external sqlite database

    This class manages "table" access switching between an in-object cache, and an external sqlite database.

    If use_cache is true, this class builds an dictionary to store sqlite-like tables as pandas dataframes. 
    Currently a database filename is required, and when a table is accessed for the first time, that table
    is pulled from the database file into the cache. Hence, one can easily use the cache with a preexisting database.

    If use_cache is false, all table reads and writes are performed by sqlite. When use_cache is false, and isolation_level
    is None, then the sql isolation level is 'SERIALIZABLE', otherwise it is 'READ UNCOMMITTED'.

    Transactions can be None, deferred (default), immediate, or exclusive. 

    None implies autocommit. FIXME -- describe what autocommit is.

    Immediate means RESERVED locks are acquired on the database without waiting for the database to be used. Other database connections
    can read the databases, but none will be able to write to the database. 

    Deferred means that no locks are acquired on the database until the database is first accessed, then the first read operation
    creates a SHARED lock and the first write operation creates a RESERVED lock. Because the acquisition of locks is deferred 
    until they are needed, it is possible that another thread or process could create a separate transaction and write to the database 
    during current thread execution.

    Exclusive causes EXCLUSIVE locks to be acquired on the databases. Only read_uncommitted connections will be able to read the database,
    and no other connection will be able to write until the transaction is complete.

    Timing test results:
      None: 58.6 sec
      immediate: 6.275 sec
      deferred: 6.22 sec
      exclusive: 6.17 sec
      python object: 3.00 sec

    """

    def __init__(self, filename_or_DB, use_cache = False, isolation_level = _isolation_level_):
        isolation_levels = (None, "deferred", "immediate", "exclusive")
        self.filename = filename_or_DB.filename if isinstance(filename_or_DB, DismodDbAPI) else filename_or_DB
        if use_cache: self._model_context = {}
        assert isolation_level in isolation_levels, f"isolation_level must be one of {isolation_levels} -- None implies autocommit."
        self.isolation_level = isolation_level

    def from_sql(self, table_name):
        if self.filename is not None:
            with sqlite3.connect(self.filename, isolation_level = self.isolation_level) as con:
                return pd.read_sql_query(f"SELECT * FROM '{table_name}'", con)
        else:
            logger.warning ("FIXME -- make this class work for null filename")
            pass
        
    def to_sql(self, df, table_name, sql_type):
        with sqlite3.connect(self.filename, isolation_level = self.isolation_level) as con:
            df.to_sql(table_name, con, if_exists='replace', index=False, dtype=sql_type)

    def drop_table(self, table_name):
        with sqlite3.connect(self.filename, isolation_level = self.isolation_level) as con:
            cursor = con.cursor()
            cursor.execute("DROP TABLE IF EXISTS '%s'" % table_name)
            cursor.close()
            
    def copy_table(self, old_table, new_table):
        with sqlite3.connect(self.filename, isolation_level = self.isolation_level) as con:
            cursor = con.cursor()
            cursor.execute("CREATE TABLE '%s' AS SELECT * FROM '%s'" % (new_table, old_table))
            cursor.close()

    def rename_table(self, old_table, new_table):
        with sqlite3.connect(self.filename, isolation_level = self.isolation_level) as con:
            cursor = con.cursor()
            cursor.execute("ALTER TABLE '%s' RENAME TO '%s'" % (old_table, new_table))
            cursor.close()

    def flush(self):
        """ 
        Flush all the in-memory cache tables to the sqlite database
        """
        if hasattr(self, '_model_context'):
            for table_name, df in self._model_context.items():
                if __debug__: print(f'Writing table "{table_name}" to file {self.filename}')
                sql_type = getattr(self, f'{table_name}_meta', None)
                self.to_sql(df, table_name, sql_type)

    def _get(self, table_name):
        if hasattr(self, '_model_context'):
            if table_name in self._model_context:
                from warnings import warn
                warn('FIXME -- if dismod_at modifies a table that has been previously cached, there is currently no detector to determine if the cache should be refreshed.')
            else:
                self._model_context[table_name] = self.from_sql(table_name)
            return self._model_context[table_name]
        else:
            return self.from_sql(table_name)

    def _set(self, df, table_name, sql_type):
        index_name = "%s_id" % table_name
        if index_name not in df.columns:
            print("Adding dismod-required column %s to table %s." % (index_name, table_name))
            df[index_name] = range(len(df))

        c_cols = [col for col in df if col.startswith('c_')]
        if df.empty:
            df = pd.DataFrame([], columns=sql_type)
        else:
            df = df.drop([col for col in df if col not in sql_type.keys() and col not in c_cols], axis=1)
            missing = tuple(sorted(set(sql_type.keys()) - set(df.columns)))
            if missing:
                logger.error("Table %s is missing columns: %s." % (table_name, missing))
        keys = list(sql_type.keys())
        keys += [k for k in c_cols if k not in keys]
        df = df.reindex(keys, axis=1)
        self.mark_nan_ids(df, sql_type, inverse = True)
        if hasattr(self, '_model_context'):
            self._model_context[table_name] = df
        else:
            self.to_sql(df, table_name, sql_type)

    def _delete(self, table_name):
        if hasattr(self, '_model_context'):
            if table_name in self._model_context:
                del self._model_context[table_name]
        else:
            self.drop_table(table_name)

    def _copy(self, from_table, to_table):
        if hasattr(self, '_model_context'):
            self._model_context[to_table] = self._model_context[from_table]
        else:
            self.copy_table(from_table, to_table)

    def _rename(self, from_table, to_table):
        if hasattr(self, '_model_context'):
            self._model_context[to_table] = self._model_context[from_table]
            del self._model_context[from_table]
        else:
            self.rename_table(from_table, to_table)

class DismodDbAPI_core(DismodDbAPI_backing_store):
    """
    This class manages access to the dismod_at sqlite file tables.
    """
    def __init__(self, filename_or_DB, log_level = logging.warning, **kwds):
        assert isinstance(filename_or_DB, (str, DismodDbAPI)), "Input must be either a DismodDbAPI database, or a filename."
        logging.basicConfig(level=log_level)
        super(DismodDbAPI_core, self).__init__(filename_or_DB, **kwds)

    @staticmethod
    def mark_nan_ids(table, table_meta, inverse=False):
        id_cols = [k for k,v in table_meta.items() if k.endswith('_id') and v == 'integer']
        if not inverse:
            table.loc[:, id_cols] = table.loc[:, id_cols].fillna(nan_int)
        else:
            table.loc[:, id_cols] = table.loc[:, id_cols].replace(nan_int, np.NaN)
        return table

    def ordered_dtypes(self, seq):
        rtn = OrderedDict()
        for k,v in seq: rtn[k] = v
        return rtn

    def children(self, node):
        return self.node.loc[self.node.parent == node, :]

    def nested_descendants(self, node):
        def recurse(node):
            children = self.node.loc[self.node.parent == node, :]
            if children.empty:
                return [node]
            else:
                return [node] + [recurse(child) for child in children.loc[:, 'node_id']]
        rtn = recurse(node)
        rtn.remove(node)
        return rtn

    def descendants(self, node):
        return [flatten(_) for _ in self.nested_descendants(node)]


    @property
    def x_covs(self):
        if 'data' in self.tables:
            with sqlite3.connect(self.filename) as con:
                columns = pd.read_sql_query("PRAGMA table_info(data)", con).name
            return startswith_filter('x_', columns)
        elif 'covariate' in self.tables:
            return ['x_%d' % _ for _ in self.covariate.covariate_id]
        else:
            raise Exception("Either the data or covariate tables must be initialized prior to requesting covariate (x_cov) names.")

    # -- Tables --
    @property
    def tables(self):
        "Get the table names in the sqlite database."
        with sqlite3.connect(self.filename) as con:
            return tuple(sorted(pd.read_sql_query("SELECT name FROM 'sqlite_master'"
                                                  "  WHERE type='table'", con).T.values[0]))
    # -- Age --
    @property
    def age_meta(self):
        return self.ordered_dtypes([('age_id', 'integer primary key'),
                                    ('age', 'real')])
    @property
    def age(self):
        return self._get('age')
    @age.setter
    def age(self, df):
        self._set(df, 'age', self.age_meta)

    # -- age_avg --
    @property
    def age_avg(self):
        return self._get('age_avg')

    # -- Avgint --
    @property
    def avgint_meta(self):
        types = self.ordered_dtypes([('avgint_id', 'integer primary key'),
                                     ('integrand_id', 'integer'),
                                     ('node_id', 'integer'),
                                     ('weight_id', 'integer'),
                                     ('subgroup_id', 'integer'),
                                     ('age_lower', 'real'),
                                     ('age_upper', 'real'),
                                     ('time_lower', 'real'),
                                     ('time_upper', 'real')])
        for cov in self.x_covs: types[cov] = 'real'
        return types
    @property
    def avgint(self):
        return self._get('avgint')
    @avgint.setter
    def avgint(self, avgint):
        self._set(avgint, 'avgint', self.avgint_meta)
    @avgint.deleter
    def avgint(self):
        self._delete('avgint')
        # Apparently, the table must exist for dismod_at.init to run
        self._set(pd.DataFrame([]), 'avgint', self.avgint_meta)

    # -- Covariate --
    @property
    def covariate_meta(self):
        return self.ordered_dtypes([('covariate_id', 'integer primary key'),
                                    ('covariate_name', 'text unique'),
                                    ('reference', 'real'),
                                    ('max_difference', 'real')])
    @property
    def covariate(self):
        return rename_covariates(self._get('covariate'))
    @covariate.setter
    def covariate(self, covariate):
        self._set(covariate, 'covariate', self.covariate_meta)
    @covariate.deleter
    def covariate(self):
        self._delete('covariate')

    def covariate_name2id(self, covariate_name):
        return int(self.covariate.loc[self.covariate.covariate_name == covariate_name, 'covariate_id'])

    # -- Constraint --
    @property
    def constraint_meta(self):
        types = self.ordered_dtypes([('constraint_id', 'integer primary key'),
                                     ('integrand_id', 'integer'),
                                     ('density_id', 'integer'),
                                     ('node_id', 'integer'),
                                     ('weight_id', 'integer'),
                                     ('hold_out', 'integer'),
                                     ('meas_value', 'real'),
                                     ('meas_std', 'real'),
                                     ('age_lower', 'real'),
                                     ('age_upper', 'real'),
                                     ('time_lower', 'real'),
                                     ('time_upper', 'real')])
        for cov in self.x_covs: types[cov] = 'real'
        return types
    @property
    def constraint(self):
        with sqlite3.connect(self.filename) as con:
            try:
                return self.mark_nan_ids(self._get('constraint'), self.constraint_meta)
            except:
                return pd.DataFrame([], columns=list(self.constraint_meta.keys()))
    @constraint.setter
    def constraint(self, constraint):
        self._set(constraint, 'constraint', self.constraint_meta)
    @constraint.deleter
    def constraint(self):
        self._delete('constraint')

    # -- Data --
    @property
    def data_meta(self):
        types = self.ordered_dtypes([('data_id', 'integer primary key'),
                                     ('data_name', 'text'),
                                     ('integrand_id', 'integer'),
                                     ('density_id', 'integer'),
                                     ('node_id', 'integer'),
                                     ('weight_id', 'integer'),
                                     ('subgroup_id', 'integer'),
                                     ('hold_out', 'integer'),
                                     ('meas_value', 'real'),
                                     ('meas_std', 'real'),
                                     ('sample_size', 'real'),
                                     ('age_lower', 'real'),
                                     ('age_upper', 'real'),
                                     ('time_lower', 'real'),
                                     ('time_upper', 'real'),
                                     ('eta', 'real'),
                                     ('nu', 'real')])
        if self.x_covs is not None:
            for cov in self.x_covs: types[cov] = 'real'
        return types

    @property
    def data(self):
        df = self._get('data')
        keys = list(self.data_meta.keys())
        missing_keys = list(set(keys).difference(df.columns))
        if missing_keys:
            logger.warning("The data table is missing columns %s -- adding them." % missing_keys)
            for k in missing_keys:
                df[k] = None
        return self.mark_nan_ids(df[keys], self.data_meta)
    @data.setter
    def data(self, data):
        data_meta = self.data_meta
        for cov in startswith_filter('x_', data.columns):
            data_meta[cov] = 'real'
        for comment in startswith_filter('c_', data.columns):
            data_meta[comment] = 'object'
        self._set(data, 'data', data_meta)
    @data.deleter
    def data(self):
        self._delete('data')

    # -- Data_subset --
    @property
    def data_subset_meta(self):
        return self.ordered_dtypes([('data_subset_id', 'integer primary key'),
                                    ('data_id', 'integer unique')])
    @property
    def data_subset(self):
        return self._get('data_subset')
    if __debug__:
        # Data_subset an output table -- user should never set it, the init command sets it.
        # That said -- sometimes it is useful to set it
        @data_subset.setter
        def data_subset(self, data_subset):
            logger.warning("Setting data_subset should probably only done by dismod_AT.")
            self._set(data_subset, 'data_subset', self.data_subset_meta)
    @data_subset.deleter
    def data_subset(self):
        self._delete('data_subset')

    # -- Density --
    @property
    def density_meta(self):
        return self.ordered_dtypes([('density_id', 'integer primary key'),
                                    ('density_name', 'text unique')])
    @property
    def density(self):
        return self._get('density')
    @density.setter
    def density(self, density):
        self._set(density, 'density', self.density_meta)

    def density_name2id(self, density_name):
        return int(self.density.loc[self.density.density_name == density_name, 'density_id'])

    # -- Subgroup --
    @property
    def subgroup_meta(self):
        return self.ordered_dtypes([('subgroup_id', 'integer primary key'),
                                    ('subgroup_name', 'text unique'),
                                    ('group_id', 'integer'),
                                    ('group_name', 'text')])
    @property
    def subgroup(self):
        return self._get('subgroup')
    @subgroup.setter
    def subgroup(self, subgroup):
        self._set(subgroup, 'subgroup', self.subgroup_meta)

    def subgroup_name2id(self, subgroup_name):
        return int(self.subgroup.loc[self.subgroup.subgroup_name == subgroup_name, 'subgroup_id'])

    # -- Depend_var --
    @property
    def depend_var_meta(self):
        return self.ordered_dtypes([('depend_var_id', 'integer primary key'),
                                    ('data_depend', 'integer'),
                                    ('prior_depend', 'integer')])
    @property
    def depend_var(self):
        return self._get('depend_var')
    # Depend_var is an output table -- user should never set it, the init command sets it.
    # @depend_var.setter
    # def depend_var(self, depend_var):
    #     logger.warning("Setting depend_var should probably only done by dismod_AT.")
    #     self._set(depend_var, 'depend_var', self.depend_var_meta)
    @depend_var.deleter
    def depend_var(self):
        self._delete('depend_var')

    # -- Fit_var --
    @property
    def fit_var_meta(self):
        return self.ordered_dtypes([('fit_var_id', 'integer primary key'),
                                    ('fit_var_value', 'real'),
                                    ('residual_value', 'real'),
                                    ('residual_dage', 'real'),
                                    ('residual_dtime', 'real'),
                                    ('lagrange_value', 'real'),
                                    ('lagrange_dage', 'real'),
                                    ('lagrange_dtime',' real')])
    @property
    def fit_var(self):
        return self._get('fit_var')
    if __debug__:
        # Fit_var is an output table -- user should never set it, the init command sets it.
        # That said -- sometimes it is useful to set it
        @fit_var.setter
        def fit_var(self, fit_var):
            logger.warning("Setting fit_var should probably only done by dismod_AT.")
            self._set(fit_var, 'fit_var', self.fit_var_meta)
    @fit_var.deleter
    def fit_var(self):
        self._delete('fit_var')

    # -- Fit_data_subset --
    @property
    def fit_data_subset_meta(self):
        return self.ordered_dtypes([('fit_data_subset_id', 'integer primary key'),
                                    ('data_subset_id', 'integer unique'),
                                    ('avg_integrand', 'real'),
                                    ('weighted_residual', 'real')])
    @property
    def fit_data_subset(self):
        return self._get('fit_data_subset')
    if __debug__:
        # Fit_data_subset is an output table -- user should never set it, the init command sets it.
        # That said -- sometimes it is useful to set it
        @fit_data_subset.setter
        def fit_data_subset(self, fit_data_subset):
            logger.warning("Setting fit_data_subset should probably only done by dismod_AT.")
            self._set(fit_data_subset, 'fit_data_subset', self.fit_data_subset_meta)
    @fit_data_subset.deleter
    def fit_data_subset(self):
        self._delete('fit_data_subset')

    # -- Hessian Fixed --
    @property
    def hes_fixed_meta(self):
        return self.ordered_dtypes([('hes_fixed_id', 'integer primary key'),
                                    ('row_var_id', 'integer'),
                                    ('col_var_id', 'integer'),
                                    ('hes_fixed_value', 'real')])
    @property
    def hes_fixed(self):
        return self._get('hes_fixed')
    @hes_fixed.setter
    def hes_fixed(self, hes_fixed):
        self._set(hes_fixed, 'hes_fixed', self.hes_fixed_meta)

    def hes_random_meta(self):
        return self.ordered_dtypes([('hes_random_id', 'integer primary key'),
                                    ('row_var_id', 'integer'),
                                    ('col_var_id', 'integer'),
                                    ('hes_random_value', 'real')])

    # -- Hessian Random --
    @property
    def hes_random(self):
        return self._get('hes_random')
    @hes_random.setter
    def hes_random(self, hes_random):
        self._set(hes_random, 'hes_random', self.hes_random_meta)


    # -- Integrand --
    @property
    def integrand_meta(self):
        return self.ordered_dtypes([('integrand_id', 'integer primary key'),
                                    ('integrand_name', 'text unique'),
                                    ('minimum_meas_cv', 'real')])
    @property
    def integrand(self):
        return self._get('integrand')
    @integrand.setter
    def integrand(self, integrand):
        self._set(integrand, 'integrand', self.integrand_meta)

    def integrand_name2id(self, integrand_name):
        return int(self.integrand.loc[self.integrand.integrand_name == integrand_name, 'integrand_id'])

    # -- Log --
    @property
    def log(self):
        return self._get('log')
    @log.setter
    def log(self, log):
        raise Exception('Setter for the log table is not implemented.')

    # -- Mulcov --
    @property
    def mulcov_meta(self):
        return self.ordered_dtypes([('mulcov_id', 'integer primary key'),
                                    ('mulcov_type', 'text'),
                                    ('rate_id', 'integer'),
                                    ('integrand_id', 'integer'),
                                    ('covariate_id', 'integer'),
                                    ('group_smooth_id', 'integer'),
                                    ('group_id', 'integer'),
                                    ('subgroup_smooth_id', 'integer')])
    @property
    def mulcov(self):
        return self.mark_nan_ids(self._get('mulcov'), self.mulcov_meta).fillna(nan_int)
    @mulcov.setter
    def mulcov(self, mulcov):
        self._set(mulcov, 'mulcov', self.mulcov_meta)

    # -- Node --
    @property
    def node_meta(self):
        return self.ordered_dtypes([('node_id', 'integer primary key'),
                                    ('node_name', 'text'),
                                    ('parent', 'integer'),
                                    ('c_location_id', 'integer unique')])
    @property
    def node(self):
        return self._get('node').fillna(nan_int)
    @node.setter
    def node(self, node):
        node['parent'] = node.parent.replace({-1: None})
        self._set(node, 'node', self.node_meta)

    # Node names are augmented with the location id -- these methods fetch the node_id from the name or location.
    def _get_node_location(self, name_with_id):
        if ('(' in name_with_id) and (')' in name_with_id):
            return int(name_with_id[name_with_id.index('(')+1 : name_with_id.index(')')].strip())
        else:
            return int(name_with_id)

    def _get_node_name(self, name_with_id):
        if '(' not in name_with_id:
            return name_with_id
        elif name_with_id.strip()[0] == '(':
            return name_with_id[name_with_id.index(')')+1 :].strip()
        elif name_with_id.strip()[-1] == ')':
            return name_with_id[: name_with_id.index('(')].strip()
    @cached_property
    def _node_names_split(self):
        df = self.node[:]
        df['location'] = df['node_name'].apply(self._get_node_location)
        df['stripped_name'] = df['node_name'].apply(self._get_node_name)
        return df
    def node_id(self, name):
        name = self._get_node_name(name)
        return int(int(self._node_names_split.loc[self._node_names_split.stripped_name == name, 'node_id'].squeeze()))
    node_name2id = node_id
    def node_location(self, name):
        name = self._get_node_name(name)
        return int(self._node_names_split.loc[self._node_names_split.stripped_name == name, 'location'].squeeze())
    def node_name(self, name):
        name = self._get_node_name(name)
        return self._node_names_split.loc[self._node_names_split.stripped_name == name, 'stripped_name'].squeeze()

    # -- Option --
    @property
    def option_meta(self):
        return self.ordered_dtypes([('option_id', 'integer primary key'),
                                    ('option_name', 'text unique'),
                                    ('option_value', 'text')])
    @property
    def option(self):
        return self._get('option')
    @option.setter
    def option(self, option):
        self._set(option, 'option', self.option_meta)
    @property
    def options(self):
        """
        Provide object access (e.g., option.option_name) to the option table.
        Get options like so:
        DB.options.option_name
        """
        option_dct = dict([(k,v) for k,v in self.option.loc[:, ['option_name', 'option_value']].values])
        return type('Options', (object,), option_dct)()
    @options.setter
    def options(self, kwds):
        """
        Set items in the option table like so:
        DB.options = dict(option_name = option_value, ...)
        """
        assert isinstance(kwds, dict), "The options.setter argument must be a dictionary"
        option = self.option
        for k,v in kwds.items():
            assert k in option.option_name.values, "Option name %s is invalid" % k
            option.loc[option.option_name == k, 'option_value'] = v
        self.option = option

    # -- Prior --
    @property
    def prior_meta(self):
        return self.ordered_dtypes([('prior_id', 'integer primary key'),
                                    ('prior_name', 'text unique'),
                                    ('lower', 'real'),
                                    ('upper', 'real'),
                                    ('mean', 'real'),
                                    ('std', 'real'),
                                    ('eta', 'real'),
                                    ('nu', 'real'),
                                    ('density_id', 'integer')])
    @property
    def prior(self):
        df = self._get('prior')
        keys = list(self.prior_meta.keys())
        missing_keys = list(set(keys).difference(df.columns))
        if missing_keys:
            logger.warning("The prior table is missing columns %s -- adding them." % list(missing_keys))
            for k in missing_keys:
                df[k] = None if k != 'nu' else 2
        return self.mark_nan_ids(df[keys], self.prior_meta)
    @prior.setter
    def prior(self, prior):
        self._set(prior, 'prior', self.prior_meta)

    # -- Predict --
    @property
    def predict_meta(self):
        return self.ordered_dtypes([('predict_id', 'integer primary key'),
                                    ('sample_index', 'integer'),
                                    ('avgint_id', 'integer'),
                                    ('avg_integrand', 'real')])
    @property
    def predict(self):
        return self._get('predict')
    # Predict is an output table -- user should never set it.
    # @predict.setter
    # def predict(self, predict):
    #     self._set(predict, 'predict', self.predict_meta)
    @predict.deleter
    def predict(self):
        self._delete('predict')

    # -- Rate --
    @property
    def rate_meta(self):
        return self.ordered_dtypes([('rate_id', 'integer primary key'),
                                    ('rate_name', 'text unique'),
                                    ('parent_smooth_id', 'integer'),
                                    ('child_smooth_id', 'integer'),
                                    ('child_nslist_id', 'integer')])
    @property
    def rate(self):
        return self.mark_nan_ids(self._get('rate'), self.rate_meta)
    @rate.setter
    def rate(self, rate):
        self._set(rate, 'rate', self.rate_meta)

    def rate_name2id(self, rate_name):
        return int(self.rate.loc[self.rate.rate_name == rate_name, 'rate_id'])

    # -- Nslist --
    @property
    def nslist_meta(self):
        return self.ordered_dtypes([('nslist_id', 'integer primary key'),
                                    ('nslist_name', 'text')])
    @property
    def nslist(self):
        return self.mark_nan_ids(self._get('nslist'), self.nslist_meta)
    @nslist.setter
    def nslist(self, nslist):
        self._set(nslist, 'nslist', self.nslist_meta)

    # -- Nslist_pair --
    @property
    def nslist_pair_meta(self):
        return self.ordered_dtypes([('nslist_pair_id', 'integer primary key'),
                                    ('nslist_id', 'integer'),
                                    ('node_id', 'integer'),
                                    ('smooth_id', 'integer')])
    @property
    def nslist_pair(self):
        return self.mark_nan_ids(self._get('nslist_pair'), self.nslist_pair_meta)
    @nslist_pair.setter
    def nslist_pair(self, nslist_pair):
        self._set(nslist_pair, 'nslist_pair', self.nslist_pair_meta)

    # -- Sample --
    @property
    def sample_meta(self):
        return self.ordered_dtypes([('sample_id', 'integer primary key'),
                                    ('sample_index', 'integer'),
                                    ('var_id', 'integer'),
                                    ('var_value', 'real')])
    @property
    def sample(self):
        return self._get('sample')
    # Sample is generally an dismod output table, but there is the special case where bootstrap fits are run in parallel and the user needs to set it.
    @sample.setter
    def sample(self, sample):
        self._set(sample, 'sample', self.sample_meta)
    @sample.deleter
    def sample(self):
        self._delete('sample')
        
    # -- Scale_var
    @property
    def scale_var_meta(self):
        return self.ordered_dtypes([('scale_var_id', 'integer primary key'),
                                    ('scale_var_value', 'real')])
    @property
    def scale_var(self):
        return self.mark_nan_ids(self._get('scale_var'), self.scale_var_meta)
    @scale_var.setter
    def scale_var(self, scale_var):
        self._set(scale_var, 'scale_var', self.scale_var_meta)

    # -- Smooth --
    @property
    def smooth_meta(self):
        return self.ordered_dtypes([('smooth_id', 'integer primary key'),
                                    ('smooth_name', 'text unique'),
                                    ('n_age','integer'),
                                    ('n_time','integer'),
                                    ('mulstd_value_prior_id', 'integer'),
                                    ('mulstd_dage_prior_id', 'integer'),
                                    ('mulstd_dtime_prior_id', 'integer')])
    @property
    def smooth(self):
        return self.mark_nan_ids(self._get('smooth'), self.smooth_meta)
    @smooth.setter
    def smooth(self, smooth):
        self._set(smooth, 'smooth', self.smooth_meta)

    # -- Smooth_grid --
    @property
    def smooth_grid_meta(self):
        return self.ordered_dtypes([('smooth_grid_id', 'integer primary key'),
                                    ('smooth_id', 'integer'),
                                    ('age_id', 'integer'),
                                    ('time_id', 'integer'),
                                    ('value_prior_id', 'integer'),
                                    ('dage_prior_id', 'integer'),
                                    ('dtime_prior_id','integer'),
                                    ('const_value', 'real')])
    @property
    def smooth_grid(self):
        return self.mark_nan_ids(self._get('smooth_grid'), self.smooth_grid_meta)
    @smooth_grid.setter
    def smooth_grid(self, smooth_grid):
        self._set(smooth_grid, 'smooth_grid', self.smooth_grid_meta)

    # -- Start_var --
    @property
    def start_var_meta(self):
        return self.ordered_dtypes([('start_var_id', 'integer primary key'),
                                    ('start_var_value', 'real')])
    @property
    def start_var(self):
        return self._get('start_var')
    @start_var.setter
    def start_var(self, start_var):
        self._set(start_var, 'start_var', self.start_var_meta)
    @start_var.deleter
    def start_var(self):
        self._delete('start_var')

    # -- Time --
    @property
    def time_meta(self):
        return self.ordered_dtypes([('time_id', 'integer primary key'),
                                    ('time', 'real')])
    @property
    def time(self):
        return self._get('time')
    @time.setter
    def time(self, time):
        self._set(time, 'time', self.time_meta)

    # -- Truth_var --
    @property
    def truth_var_meta(self):
        return self.ordered_dtypes([('truth_var_id', 'integer primary key'),
                                    ('truth_var_value', 'real')])
    @property
    def truth_var(self):
        return self._get('truth_var')
    @truth_var.setter
    def truth_var(self, truth_var):
        self._set(truth_var, 'truth_var', self.truth_var_meta)
    @truth_var.deleter
    def truth_var(self):
        self._delete('truth_var')

    # -- Data Simulate --
    @property
    def data_sim_meta(self):
        return self.ordered_dtypes([('data_sim_id', 'integer primary key'),
                                    ('simulate_index', 'integer'),
                                    ('data_subset_id', 'integer'),
                                    ('data_sim_value', 'real'),
                                    ('data_sim_stdcv', 'real'),
                                    ('data_sim_delta', 'real')])
    @property
    def data_sim(self):
        return self._get('data_sim')
    if __debug__:
        # Data_Sim is an output table -- user should never set it, but it sometimes has meas_value < 0 so it needs to be set.
        @data_sim.setter
        def data_sim(self, data_sim):
            self._set(data_sim, 'data_sim', self.data_sim_meta)
    @data_sim.deleter
    def data_sim(self):
        self._delete('data_sim')

    # -- Prior Simulate --
    @property
    def prior_sim_meta(self):
        return self.ordered_dtypes([('prior_sim_id', 'integer primary key'),
                                    ('simulate_index', 'integer'),
                                    ('var_id', 'integer'),
                                    ('prior_sim_value', 'real'),
                                    ('prior_sim_dage', 'real'),
                                    ('prior_sim_dtime', 'real')])
    @property
    def prior_sim(self):
        return self._get('prior_sim')
    if __debug__:
        # Prior_Sim is an output table -- user should never set it, but it sometimes has meas_value < 0 so it needs to be set.
        @prior_sim.setter
        def prior_sim(self, prior_sim):
            self._set(prior_sim, 'prior_sim', self.prior_sim_meta)
    @prior_sim.deleter
    def prior_sim(self):
        self._delete('prior_sim')

    # -- Var --
    @property
    def var_meta(self):
        return self.ordered_dtypes([('var_id', 'integer primary key'),
                                    ('var_type', 'text'),
                                    ('smooth_id', 'integer'),
                                    ('age_id', 'integer'),
                                    ('time_id', 'integer'),
                                    ('node_id', 'integer'),
                                    ('rate_id', 'integer'),
                                    ('integrand_id', 'integer'),
                                    ('covariate_id', 'integer'),
                                    ('mulcov_id', 'integer'),
                                    ('group_id', 'integer'),
                                    ('subgroup_id', 'integer')])
    @property
    def var(self):
        return self.mark_nan_ids(self._get('var'), self.var_meta)
    # Var is an output table -- user should never set it.
    if __debug__:
        @var.setter
        def var(self, var):
            self._set(var, 'var', self.var_meta)

    # -- Weight --
    @property
    def weight_meta(self):
        return self.ordered_dtypes([('weight_id', 'integer primary key'),
                                    ('weight_name', 'text unique'),
                                    ('n_age', 'integer'),
                                    ('n_time', 'integer')])
    @property
    def weight(self):
        return self._get('weight')
    @weight.setter
    def weight(self, weight):
        self._set(weight, 'weight', self.weight_meta)

    # -- Weight_grid --
    @property
    def weight_grid_meta(self):
        return self.ordered_dtypes([('weight_grid_id', 'integer primary key'),
                                    ('weight_id', 'integer'),
                                    ('age_id', 'integer'),
                                    ('time_id', 'integer'),
                                    ('weight', 'real')])
    @property
    def weight_grid(self):
        return self.mark_nan_ids(self._get('weight_grid'), self.weight_grid_meta)
    @weight_grid.setter
    def weight_grid(self, weight_grid):
        self._set(weight_grid, 'weight_grid', self.weight_grid_meta)

class CascadeOption(object):
    """
    CascadeOption adds a table to store cascade options (e.g. not dismod_at options)
    """
    @property
    def cascade_option_meta(self):
        return self.ordered_dtypes([('cascade_option_id', 'integer primary key'),
                                    ('cascade_option_name', 'text unique'),
                                    ('cascade_option_value', 'text')])
    @property
    def cascade_option(self):
        try:
            return self._get('cascade_option')
        except:
            return pd.DataFrame(columns = ['cascade_option_name', 'cascade_option_value'])
    @cascade_option.setter
    def cascade_option(self, option):
        self._set(option, 'cascade_option', self.cascade_option_meta)
    @property
    def cascade_options(self):
        """
        Provide object access (e.g., cascade_option.cascade_option_name) to the cascade_option table.
        Get cascade_options like so:
        DB.cascade_options.cascade_option_name
        """
        option_dct = dict([(k,v) for k,v in self.cascade_option.loc[:, ['cascade_option_name', 'cascade_option_value']].values])
        return type('CascadeOption', (object,), option_dct)()
    @cascade_options.setter
    def cascade_options(self, kwds):
        """
        Set items in the cascade_option table like so:
        DB.cascade_options = dict(cascade_option_name = cascade_option_value, ...)
        """
        assert isinstance(kwds, dict), "The cascade_options.setter argument must be a dictionary"
        option = self.cascade_option
        for k,v in kwds.items():
            assert k in option.cascade_option_name.values, "Option name %s is invalid" % k
            option.loc[option.cascade_option_name == k, 'cascade_option_value'] = v
        self.cascade_option = option

class DismodDbAPI(DismodDbAPI_core, CascadeOption):
    def __init__(self, *args, **kwds):
        super(DismodDbAPI, self).__init__(*args, **kwds)
    @property
    def fit(self, merge_age_priors=False, merge_time_priors = False):
        """
        Optimization Results for Variables

        The fit_var table contains the maximum likelihood estimate for the model_variables corresponding to data in table data.meas_value. 
        A new table fit_var is created each time the dismod fit command is executed. The table contains:

        fit_var_id:     Starts at 0 and increments by 1. The fit_var.fit_var_id is foreign key for the var table;
                        (i.e., fit_var.fit_var_id == var.var_id)
        fit_var_value: The model variable values. To be specific, the fixed effects (theta), maximize the total objective L(theta)
                        and the corresponding optimal random effects U(theta)
        residual_value: The weighted residual corresponding to the value_prior_id for this variable. 
                        If there is no such residual, (e.g., if the corresponding density is uniform) this value will be null. 
        residual_dage:  The weighted residual corresponding to the dage_prior_id for this variable. 
                        If dage_prior_id is null, this value will be null.
        residual_dtime: The weighted residual corresponding to the dtime_prior_id for this variable. 
                        If dtime_prior_id is null, this value will be null
        """
        try:
            parent_node_id = int(self.options.parent_node_id)
        except:
            parent_node_id = int(self.node.loc[self.node.node_name == self.options.parent_node_name, 'node_id'])
        prior_df = self.prior.merge(self.density, how='left')
        df = (self.var
              .merge(self.node, how='left')
              .merge(self.rate, how='left')
              .merge(self.integrand, how='left')
              .merge(self.covariate, how='left')
              .merge(self.age, how='left')
              .merge(self.time, how='left')
              .merge(self.smooth_grid, how='left', on=['age_id', 'time_id', 'smooth_id']))
        if 'start_var' in self.tables:
            df = df.merge(self.start_var, left_on = 'var_id', right_on = 'start_var_id')
        if 'fit_var' in self.tables:
            df = df.merge(self.fit_var, left_on = 'var_id', right_on = 'fit_var_id')
        if 'truth_var' in self.tables:
            df = df.merge(self.truth_var, left_on = 'var_id', right_on = 'truth_var_id')
        # if 0:
        #     # Merging with sample causes the fit table to grow by the number of samples -- Don't want this
        #     if 'sample' in self.tables:
        #         df = df.merge(self.sample, how='left')
        df.rename(columns={'eta':'integrand_eta'}, inplace=True)
        node_id = df.node_id.fillna('nan')
        df['fixed'] = ~((node_id != parent_node_id) & (df.var_type == 'rate'))
        # pdf = prior_df.rename(columns={'lower':'lower_value','upper':'upper_value','mean':'mean_value','std':'std_value','eta':'eta_value','density_name':'density_value'})
        pdf = prior_df
        df = df.merge(pdf, how='left', left_on='value_prior_id', right_on='prior_id')
        if merge_age_priors:
            pdf = prior_df.rename(columns={'lower':'lower_age','upper':'upper_age','mean':'mean_age','std':'std_age','eta':'eta_age','density_name':'density_age'})
            df = df.merge(pdf, how='left', left_on='dage_prior_id', right_on='prior_id')
        if merge_time_priors:
            pdf = prior_df.rename(columns={'lower':'lower_time','upper':'upper_time','mean':'mean_time','std':'std_time','eta':'eta_time','density_name':'density_time'})
            df = df.merge(pdf, how='left', left_on='dtime_prior_id', right_on='prior_id')
        return df

    def _covariates(self, substr):
        df = rename_covariates(self._get('covariate'))
        if 1:
            # Fix Brad's bullshit
            df.loc[df.covariate_name == 'ones', 'c_covariate_name'] = 's_ones'
        print(df)
        cov_name = 'c_covariate_name' if 'x_0' in df.covariate_name.tolist() else 'covariate_name'
        cov_df = pd.DataFrame([dict(covariate_id = row.covariate_id,
                                    xcov_name = 'x_%d' % row.covariate_id,
                                    covariate_name_short = getattr(row, cov_name)[len(substr):])
                               for row in df.itertuples()
                               if getattr(row, cov_name) and getattr(row, cov_name).startswith(substr)
                               and ('one' not in getattr(row, cov_name))])
        return cov_df
        
    @property
    def sex_covariate(self):
        "Country covariate and x_cov names"
        df = rename_covariates(self._get('covariate'))
        if not df.empty:
            if 'c_covariate_name' in df:
                df = df.loc[[('sex' in n if n else False) for n in df.c_covariate_name], ['covariate_id', 'c_covariate_name']].rename(columns = {'c_covariate_name': 'covariate_name'})
            else:
                df = df.loc[[('sex' in n if n else False) for n in df.covariate_name], ['covariate_id', 'covariate_name']]
                if not df.empty:
                    df['xcov_name'] = 'x_%d' % df.covariate_id
        return df

    @property
    def one_covariate(self):
        "Country covariate and x_cov names"
        df = rename_covariates(self._get('covariate'))
        if not df.empty:
            if 'c_covariate_name' in df:
                df = df.loc[[('one' in n if n else False) for n in df.c_covariate_name], ['covariate_id', 'c_covariate_name']].rename(columns = {'c_covariate_name': 'covariate_name'})
            else:
                df = df.loc[[('one' in n if n else False) for n in df.covariate_name], ['covariate_id', 'covariate_name']]
                if not df.empty:
                    df['xcov_name'] = 'x_%d' % df.covariate_id
        return df

    @property
    def country_covariates(self):
        "Country covariate and x_cov names"
        a = self._covariates(substr = 'x_c_')
        if not a.empty: return a
        b = self._covariates(substr = 'c_')
        return b
    @property
    def study_covariates(self):
        "Study covariate and x_cov names"
        a = self._covariates(substr = 'x_s_')
        if not a.empty: return a
        b = self._covariates(substr = 's_')
        return b
    @property
    def data_merge(self):
        "Summary of the data/fit_data_subset tables merged with their relations"
        data = (self.data
                .merge(self.integrand, how='left')
                .merge(self.density, how='left')
                .merge(self.node, how='left')
                .merge(self.node.drop(columns='parent').rename(columns={'node_name': 'parent_name'}),
                       how='left', left_on='parent', right_on='node_id'))
        if 'fit_data_subset' in self.tables:
            data = self.fit_data_subset.merge(data, how='left', left_on='data_subset_id', right_on='data_id')
        return data

    @property
    def var_merge(self):
        "Summary of the var/fit_var tables merged with their relations"
        fit = (self.var
               .merge(self.age, how='left')
               .merge(self.time, how='left')
               .merge(self.rate, how='left')
               .merge(self.smooth_grid, how='left')
               .merge(self.smooth, how='left'))

        for name in ['value', 'dage', 'dtime', 'mulstd_value', 'mulstd_dage', 'mulstd_dtime']: 
            fit = fit.merge(self.prior, how='left', left_on='%s_prior_id' % name, right_on='prior_id').merge(self.density, how='left')
            fit.rename(columns = {'prior_name': '%s_prior_name' % name,
                                  'mean': '%s_prior_mean' % name,
                                  'lower': '%s_prior_lower' % name,
                                  'upper': '%s_prior_upper' % name,
                                  'std': '%s_prior_std' % name,
                                  'nu': '%s_prior_nu' % name,
                                  'eta': '%s_prior_eta' % name,
                                  'density': '%s_prior_density' % name}, inplace=True)
            fit.drop(columns = ['prior_id', 'density_id'], inplace=True)
        if 'fit_var' in self.tables:
            fit = fit.merge(self.fit_var, how='left', left_on='var_id', right_on='fit_var_id')
        return fit

if (__name__ == '__main__'):
    _cache_db_ = False

    import shutil
    sqlite_filename = os.path.join(os.environ['CASCADE_AT_GMA_DATA'], '100667/full/1/both/1990_1995_2000_2005_2010_2015/100667.db')
    shutil.copy2(sqlite_filename, os.path.expanduser('~/tmp.db'))
    self = DismodDbAPI(os.path.expanduser('~/tmp.db'), use_cache = _cache_db_, isolation_level = 'immediate')

    print('Testing')

    print ('Testing covariate properties.')
    assert not self.sex_covariate.empty
    assert not self.country_covariates.empty
    assert not self.study_covariates.empty
    import time

    t0 = time.time()
    print ('Testing copy.')
    data = self.data
    self._copy('data', 'data.1')
    self._delete('data')
    self._copy('data.1', 'data')
    self._delete('data.1')
    assert np.alltrue(data.fillna(-999) == self.data.fillna(-999))

    print ('Testing rename.')
    self._rename('data', 'data.1')
    self._delete('data')
    self._rename('data.1', 'data')
    assert np.alltrue(data.fillna(-999) == self.data.fillna(-999))

    # Test the option setter
    options = self.options
    options.max_num_iter_fixed = -999
    self.options = dict(max_num_iter_fixed = -999)
    assert int(self.options.max_num_iter_fixed) == -999, "Options setting test failed."
    
    # Test the node code
    assert sorted(self.node.node_id) == sorted(list(flatten(self.descendants(0))) + [0]), "Node descendants test failed"

    settable = ['integrand', 'density', 'node', 'weight', 'weight_grid', 'covariate', 'option', 'cascade_option' , 'avgint', 'age', 
                'time', 'data', 'rate', 'smooth', 'smooth_grid', 'prior', 'mulcov', 'start_var', 'truth_var', 'nslist', 'nslist_pair']
    not_settable = ['log', 'var', 'data_subset', 'fit_var', 'fit_data_subset', 'data_sim', 'prior_sim', 'sample', 'predict']

    # Make sure none of the integer id columns in the pandas dataframes have NA values
    for table in settable + not_settable:
        print ('Checking', table, "for null in the id columns.")
        try:
            t = getattr(self, table)
            m = getattr(self, table+'_meta')
            id_cols = [k for k,v in m.items() if k.endswith('_id') and v == 'integer']
            if id_cols:
                if np.any(t[id_cols].isna()): set_trace()
                assert not np.any(t[id_cols].isna()), "One of the columns %s had a null id" % id_cols

                q = self._get(table)
                q[id_cols] = q[id_cols].fillna(nan_int)
                assert np.all(q.fillna(-999) == t.fillna(-999)), "Some of the _id columns were not correctly masked."

        except (pd.io.sql.DatabaseError, AttributeError) as ex:
            print (ex)

    _test_ = True
    if _test_:
        # Check the node name and location code
        nid = self.node_id('HI')
        node_name = self.node.loc[self.node.node_id==nid, 'node_name'].squeeze()
        assert self.node_id('HI') == nid
        assert self.node_name('HI') == 'HI'
        assert str(self.node_location('HI')) in node_name

        for table in settable + not_settable:
            print ('Testing', table, 'table.')

            if table not in self.tables:
                print ("** WARNING ** table %s is missing." % table)
            else:
                a = getattr(self, table)
                try:
                    if table in settable:
                        setattr(self, table, a)
                        b = getattr(self, table)

                        mask = a.isnull().values
                        A = np.asarray(a.values)
                        A[mask] = 0
                        B = np.asarray(b.values)
                        B[mask] = 0
                        tst = (A == B)
                        assert np.alltrue(tst)

                        self._set(a, table+'_new', getattr(self, table+'_meta'))
                    else:
                        print ('Table %s is a dismod_at output table' % table)
                except Exception as ex:
                    print ('** ERROR **', table, ex)

    if _cache_db_:
        print ('Testing flush')
        self.flush()
    else:
        self.commit()
    print (f"Execution time: {time.time() - t0}")
