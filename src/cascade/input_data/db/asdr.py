"""
DISMODAT-50   Promote mtall data to T3
DISMODAT-257  Promote ASDR data to T3

ASDR: Age-Specific Death/Mortality Rate

With a given model_version_id, checks if the ASDR data set has not already
been pushed to the t3_model_version_asdr table.
If so, then extracts the asdr data for the parent node
and inserts it into the t3_model_version_asdr table.

The objective is to make the promoted data available to EpiViz for plotting.
"""

import logging


try:
    from db_queries import get_envelope
except ImportError:

    class DummyGetEnvelope:
        def __getattr__(self, name):
            raise ImportError(
                f"Required package db_queries.get_envelope not found")

    get_envelope = DummyGetEnvelope()

try:
    from db_queries import get_population
except ImportError:

    class DummyGetPopulation:
        def __getattr__(self, name):
            raise ImportError(
                f"Required package db_queries.get_population not found")

    get_population = DummyGetPopulation()

try:
    from db_tools import ezfuncs
except ImportError:

    class DummyEZFuncs:
        def __getattr__(self, name):
            raise ImportError(f"Required package db_tools.ezfuncs not found")

    ezfuncs = DummyEZFuncs()

from cascade.core.db import (
    cursor,)


CODELOG = logging.getLogger(__name__)


def exists_model_version_asdr_t3(execution_context):
    """
    Checks if the asdr data for the model_version_id
    already exists in the epi.t3_model_version_asdr.

    Args:
        execution_context (ExecutionContext): context manager object

    Returns:
        (Bool): True if there exists ASDR data for the model version id.
    """

    model_version_id = execution_context.parameters.model_version_id

    query = f"""
    select exists(
            select * from epi.t3_model_version_asdr
            where model_version_id={model_version_id}
           ) """

    with cursor(execution_context) as c:
        c.execute(query)
        exists = c.fetchone()[0]

    if exists != 1:
        CODELOG.info(
           f"The asdr data for MVID {model_version_id}"
           f" has already been promoted to t3")
        # raise ValueError

    return exists == 1


def get_age_groups(execution_context):
    """
    Directly query the shared.age_group table, to get 3 columns of all rows.

    Args:
        execution_context (ExecutionContext): context manager object

    Returns:
        A dataframe of three columns: age_group_id, age_lower, age_upper
    """

    # How does this query() differ from, say the:
    #  get_demographics(gbd_team="epi", gbd_round_id=5)["age_group_id"]
    # which gets epi 2017 demographics stuff?

    query = """
    SELECT age_group_id,
     age_group_years_start,
     age_group_years_end
     FROM shared.age_group """

    with cursor(execution_context) as c:
        c.execute(query)
        all_age_groups = c.fetchone()

    return all_age_groups


def _get_asdr_t2_data(execution_context):
    """
    Gather and prepare Tier 2 ASDR data.

    Args:
        execution_context (ExecutionContext): context manager object

    Returns:
        (DataFrame): of Tier 2 ASDR data of a model_version_id.
        The data is ready to be promoted to the Tier 3 table.
    """

    try:
        # Download from T2 to file
        # (can't cross-insert from different DB servers)

        ages = get_age_groups(execution_context)

        mortality_age_grid = list(range(2, 22))
        # mo_vid = self.model_version_meta.csmr_mortality_output_version_id
        # mo_vid = mo_vid.values[0]

        asdr = get_envelope(location_id=-1,
                            # gma location_set=9,
                            location_set_id=9,
                            age_group_id=mortality_age_grid,
                            year_id=[1985, 1990, 1995, 2000, 2005, 2010,
                                     2016],
                            sex_id=[1, 2],
                            with_hiv=1)
        if 'run_id' in asdr:
            run_id_asdr = asdr.run_id.unique().squeeze()
            logging.info("ASDR run_id is {}".format(run_id_asdr))
            # TODO: record these run_ids for reproducibility verification

        pop = get_population(location_id=-1,
                             # gma location_set=9,
                             location_set_id=9,
                             age_group_id=mortality_age_grid,
                             year_id=[1985, 1990, 1995, 2000, 2005, 2010,
                                      2016],
                             sex_id=[1, 2])
        if 'run_id' in pop:
            run_id_pop = pop.run_id.unique().squeeze()
            logging.info("Pop run_id is {}".format(run_id_pop))
            # TODO: record these run_ids for reproducibility verification

        asdr = asdr.merge(
            pop, on=['location_id', 'year_id', 'age_group_id', 'sex_id'])
        asdr['meas_value'] = asdr['mean']/asdr['population']
        asdr['meas_lower'] = asdr['lower']/asdr['population']
        asdr['meas_upper'] = asdr['upper']/asdr['population']
        asdr = asdr.merge(ages, on='age_group_id')
        asdr.rename(columns={'year_id': 'time_lower',
                             'sex_id': 'x_sex',
                             'age_group_years_start': 'age_lower',
                             'age_group_years_end': 'age_upper'},
                    inplace=True)

        df = asdr[['location_id', 'time_lower', 'age_group_id', 'x_sex',
                   'age_lower', 'age_upper', 'meas_value', 'meas_lower',
                   'meas_upper']]

        # Drop NULLs, typically the result of mismatches between
        # COD and MORT
        df = df[df.meas_value.notnull()]

        # Write to disk, for testing
        # asdr_file = f"{HOME}/asdr.csv"
        # df.to_csv(asdr_file, index=False)

        return df

    except Exception as e:
        logging.error(f"GMA: {e}")


def _upload_asdr_data_to_tier_3(cursor, model_version_id, asdr_data):
    """Uploads asdr data to tier 3 attached to the current model_version_id.

    Args:
        cursor (db.cursor): object generated from core.db.cursor

        model_version_id (Str):

        asdr_data (Dataframe):
    """

    insert_query = f"""
     INSERT INTO epi.t3_model_version_asdr (
         model_version_id,
         year_id,
         location_id,
         sex_id,
         age_group_id,
         age_upper,
         age_lower,
         mean,
         upper,
         lower
     ) VALUES (
         {model_version_id}, {", ".join(["%s"]*9)}
     )"""

    cursor.executemany(insert_query, asdr_data.values)

    CODELOG.debug(f"uploaded {len(asdr_data)} lines of asdr data")


def promote_asdr_t2_to_tier_3(execution_context) -> bool:
    """
    The user can request via a setting, add_asdr_cause, to include
    asdr data for a given model.  If the user has requested that and
    the data have not already been uploaded to t3, upload it to
    t3_model_version_asdr table.

    If there is any data in the tier 3 table for the model_version_id
    then no data will not be promoted, and the function returns False.

    Args:
        execution_context (ExecutionContext): context manager object

    Returns:
        (Bool): True if the data for the model version id is promoted
        to the Tier 3 table.
    """

    model_version_id = execution_context.parameters.model_version_id

    if not execution_context.parameters.add_asdr_cause:

        CODELOG.info(f"User did not request that asdr data be added for"
                     f" model_version_id {model_version_id}")
        return False

    else:
        CODELOG.info(f"User requested asdr data be added for"
                     f" model_version_id {model_version_id}")

        # database = execution_context.parameters.bundle_database  # ??
        database = execution_context.parameters.asdr_database  # ??

        if exists_model_version_asdr_t3(execution_context):
            CODELOG.info(f"asdr tier 3 data for MVID {model_version_id}"
                         f" on '{database}' already exist, doing nothing.")

            return False
        else:
            CODELOG.info(f"Uploading asdr data for model_version_id"
                         f" {model_version_id} on '{database}'")

            asdr_data = _get_asdr_t2_data(execution_context)

            with cursor(execution_context) as c:
                _upload_asdr_data_to_tier_3(c, model_version_id, asdr_data)

            CODELOG.info("promote_asdr_t2_to_tier_3 finished normally")

            return True
