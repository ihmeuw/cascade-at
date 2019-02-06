import os
import json

from cascade.core.db import cursor

"""
CREATE TABLE epi.dismod_resource_usage (
    run_id varchar(32),
    jobid varchar(255),
    job_key text,
    process_key text,
    scale_parameters text,
    inserted_at datetime,
    summary text
)
"""


def write_summary_to_db(execution_context, run_id, job_key, process_key, scale_parameters, summary):
    q = """insert into dismod_resource_usage (run_id, jobid, job_key, process_key, scale_parameters, inserted_at, summary)
           values (%(run_id)s, %(jobid)s, %(job_key)s, %(process_key)s, %(scale_parameters)s, now(), %(summary)s)"""

    jobid = None
    if os.environ.get("JOB_ID"):
        jobid = os.environ.get("JOB_ID")
        if os.environ.get("UGE_TASK_ID"):
            jobid += "." + os.environ.get("UGE_TASK_ID")
    summary = json.dumps(summary, sort_keys=True)
    job_key = json.dumps(job_key, sort_keys=True)
    process_key = json.dumps(process_key, sort_keys=True)
    scale_parameters = json.dumps(scale_parameters, sort_keys=True)

    with cursor(execution_context) as c:
        c.execute(
            q,
            args={
                "run_id": run_id.hex,
                "jobid": jobid,
                "job_key": c.connection.escape_string(job_key),
                "process_key": c.connection.escape_string(process_key),
                "scale_parameters": c.connection.escape_string(scale_parameters),
                "summary": c.connection.escape_string(summary),
            },
        )
