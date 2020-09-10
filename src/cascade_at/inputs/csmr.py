import pandas as pd

from cascade_at.core.db import db_queries as db
from cascade_at.core.db import gbd, db_tools

from cascade_at.core.log import get_loggers
from cascade_at.inputs.base_input import BaseInput
from cascade_at.dismod.constants import IntegrandEnum
from cascade_at.inputs.uncertainty import bounds_to_stdev

LOG = get_loggers(__name__)


def get_best_cod_correct(gbd_round_id: int) -> int:
    run_query = f"""
            SELECT MAX(co.output_version_id) AS version
            FROM cod.output_version co
            JOIN shared.decomp_step ds USING (decomp_step_id)
            WHERE co.is_best = 1
            AND co.best_end IS NULL
            AND ds.gbd_round_id = {gbd_round_id}
            """
    run_id = db_tools.ezfuncs.query(
        run_query, conn_def='cod'
    ).version.astype(int).squeeze()
    if run_id is None:
        raise RuntimeError(f"Cannot find a best codcorrect output for gbd round ID {gbd_round_id}.")
    LOG.info(f"Found run ID {run_id}.")
    proc_query = f"""
            SELECT
                val AS codcorrect_version,
                gbd_process_version_id,
                gbd_process_version_status_id,
                gbd_round_id,
                decomp_step_id
            FROM gbd_process_version_metadata
            JOIN
                gbd_process_version USING (gbd_process_version_id)
            JOIN
                metadata_type USING (metadata_type_id)
            WHERE
                metadata_type = 'CodCorrect Version'
                and gbd_process_id = 3
                and gbd_process_version_status_id = 1
                and val = {run_id}
            ORDER BY gbd_process_version_id DESC
            """
    process_version_id = db_tools.ezfuncs.query(
        proc_query, conn_def='gbd'
    ).codcorrect_version.astype(int).squeeze()
    if process_version_id is None:
        raise RuntimeError(f"Cannot find process version ID for run ID {run_id}.")
    LOG.info(f"Found process version ID {process_version_id}.")
    return process_version_id


class CSMR(BaseInput):
    def __init__(self, cause_id: int, demographics,
                 decomp_step: str, decomp_step_id: int, gbd_round_id: int):
        """
        Get cause-specific mortality rate
        for demographic groups from a specific
        CodCorrect output version.

        :param cause_id: (int)
        :param demographics (cascade_at.inputs.demographics.Demographics)
        :param decomp_step: (str)
        :param decomp_step_id: (int)
        :param gbd_round_id: (int)
        """
        super().__init__(gbd_round_id=gbd_round_id)
        self.cause_id = cause_id
        self.demographics = demographics
        self.decomp_step = decomp_step
        self.decomp_step_id = decomp_step_id
        self.gbd_round_id = gbd_round_id

        self.raw = None
        self.process_version_id = None

    def get_raw(self):
        """
        Pulls the raw CSMR and assigns it to
        this class.
        :return: self
        """
        if self.cause_id:
            self.process_version_id = get_best_cod_correct(
                gbd_round_id=self.gbd_round_id,
            )
            LOG.info(f"Getting CSMR from process version ID {self.process_version_id}")
            self.raw = db.get_outputs(
                topic='cause',
                cause_id=self.cause_id,
                metric_id=gbd.constants.metrics.RATE,
                measure_id=gbd.constants.measures.DEATH,
                year_id=self.demographics.year_id,
                location_id=self.demographics.drill_locations,
                sex_id=self.demographics.sex_id,
                age_group_id=self.demographics.age_group_id,
                gbd_round_id=self.gbd_round_id,
                process_version_id=self.process_version_id
            )
        else:
            LOG.info("There is no CSMR cause to pull from.")
            self.raw = pd.DataFrame()
        return self

    def attach_to_model_version_in_db(self, model_version_id, conn_def):
        """
        Uploads the CSMR for this model and attaches
        it to the model version so that it can be
        viewed in EpiViz.

        Returns: None
        """
        df = self.raw[['year_id', 'location_id', 'sex_id', 'age_group_id', 'val', 'upper', 'lower']]
        df['model_version_id'] = model_version_id
        df.rename(columns={'val': 'mean'}, inplace=True)

        session = db_tools.ezfuncs.get_session(conn_def=conn_def)
        loader = db_tools.loaders.Inserts(
            table='t3_model_version_csmr',
            schema='epi',
            insert_df=df
        )
        loader.insert(session=session, commit=True)

    def configure_for_dismod(self, hold_out=0):
        """
        Configures CSMR for DisMod.

        :param hold_out: (int) hold-out value for Dismod. 0 means it will be fit, 1 means held out
        :return: (pd.DataFrame)
        """
        if self.cause_id:
            df = self.raw.rename(columns={
                "val": "meas_value",
                "year_id": "time_lower"
            })
            df["time_upper"] = df["time_lower"] + 1
            df = self.convert_to_age_lower_upper(df)
            df['integrand_id'] = IntegrandEnum.mtspecific.value
            df['measure'] = IntegrandEnum.mtspecific.name

            df["meas_std"] = bounds_to_stdev(df.lower, df.upper)
            df = self.keep_only_necessary_columns(df)
            df["hold_out"] = hold_out
        else:
            df = pd.DataFrame()
        return df

