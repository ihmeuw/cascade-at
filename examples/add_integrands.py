"""
This takes the timings files and adds some information
from the db files that seems necessary to interpret
results.
"""
import json
from argparse import ArgumentParser
from os import linesep
from pathlib import Path

import numpy as np

from cascade.dismod.db.wrapper import DismodFile, get_engine
from cascade.dismod.metrics import data_records
from cascade.dismod.constants import INTEGRAND_COHORT_COST

INTEGRANDS = list(INTEGRAND_COHORT_COST.keys())


def json_translate(o):
    """This exists because we try to write an np.int64 to JSON."""
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def reread_metrics(metrics_in):
    db_file_path = Path(metrics_in["db_file"])
    if set(metrics_in.keys()) & set(INTEGRANDS):
        print(f"no need for {db_file_path}")
        return None
    print(db_file_path)
    db_file = DismodFile(get_engine(db_file_path))
    new_values = data_records(db_file)
    metrics_in.update(new_values)
    assert len(metrics_in) > 10
    return metrics_in


def over_json_do(timing, retime, generate):
    print(f"From {timing} to {retime}")
    if not retime.exists():
        retime.mkdir(exist_ok=True, parents=True)
    for json_file in timing.glob("*.json"):
        with json_file.open() as json_input:
            old_dict = json.load(json_input)
        new_dict = generate(old_dict)
        if new_dict:
            new_timing = retime / json_file.name
            with new_timing.open("w") as json_output:
                json.dump(
                    new_dict,
                    json_output,
                    default=json_translate,
                    indent=2
                )
                json_output.write(linesep)


def entry():
    parser = ArgumentParser()
    parser.add_argument(
        "--timing", type=Path, default=Path("timing"),
        help="Read json files from this diretory",
    )
    parser.add_argument(
        "--retime", type=Path, default=Path("retime"),
        help="Put new files in this directory",
    )
    args = parser.parse_args()
    over_json_do(args.timing, args.retime, reread_metrics)


if __name__ == "__main__":
    entry()
