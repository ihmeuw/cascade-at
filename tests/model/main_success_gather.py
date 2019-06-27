import csv
import json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from cascade.dismod.constants import INTEGRAND_TO_WEIGHT


RENAME_HEADER = {
    "wall clock": "wall",
    "System time": "sys",
    "User time": "user",
    "Maximum resident": "rss",
    "avgint": None,
    "n_children": None,
    "data_cnt": None,
    "data_extent_cohort": None,
    "data_extent_primary": None,
    "data_point_cohort": None,
    "data_point_primary": None,
    "dismod_at command": "effect",
    "derivative_test_fixed": None,
    "topology_choice": None,
    "ipopt iterations": "iterations",
    "quasi_fixed": None,
    "random_effect_points": None,
    "rate_count": None,
    "smooth_count": None,
    "variables": None,
    "zero_sum_random": None,
    "processor type": "processor",
    "Exit status": "exit_status",
    "Percent of CPU this job got": "percent_cpu",
    "Maximum resident set size (kbytes)": "max_rss_kb",
    "Major (requiring I/O) page faults": "major_page_faults",
    "Minor (reclaiming a frame) page faults": "minor_page_faults",
    "Voluntary context switches": "voluntary_context_switches",
    "Involuntary context switches": "involuntary_context_switches",
    "File system inputs": "file_system_inputs",
    "File system outputs": "file_system_outputs",
    "Signals delivered": "signals_delivered",
    **{integrand: None for integrand in INTEGRAND_TO_WEIGHT.keys()},
}
"""
None means the name is fine.
If it's missing, it doesn't get used.
"""


def load_records(directory):
    records = list()
    for json_name in Path(directory).glob("*.json"):
        parameters = json.load(json_name.open())
        records.append(parameters)
    return records


def shorter_names(records):
    columns = set()
    for parameters in records:
        columns |= set(parameters)

    rename = dict()
    did_not_match = list()
    for search_key, rename_value in RENAME_HEADER.items():
        match = [col for col in columns if search_key in col]
        if len(match) == 1:
            if rename_value:
                rename[match[0]] = rename_value
            else:
                rename[match[0]] = search_key
        else:
            did_not_match.append(search_key)
    print(f"Search keys did not match {did_not_match}")
    return [
        {rename[k]: v for (k, v) in record.items() if k in rename}
        for record in records
    ]


def timing_to_seconds(time_str):
    splitted = time_str.split(":")
    if len(splitted) == 2:
        return 60 * int(splitted[0]) + float(splitted[1])
    elif len(splitted) == 3:
        return 3600 * int(splitted[0]) + 60 * int(splitted[1]) + int(splitted[2])
    else:
        assert len(splitted) not in {2, 3}


def individual_corrections(df):
    df = df.assign(
        wall=df.wall.apply(timing_to_seconds),
        zero_sum_random=~df.zero_sum_random.isnull(),
        effect=df.effect.str[4:],
        avgint=df.avgint.astype(int),
        data_cnt=df.data_cnt.astype(int),
        data_extent_cohort=df.data_extent_cohort.astype(int),
        data_extent_primary=df.data_extent_primary.astype(int),
        data_point_cohort=df.data_point_cohort.astype(int),
        data_point_primary=df.data_point_primary.astype(int),
        exit_status=df.exit_status.astype(int),
        file_system_inputs=df.file_system_inputs.astype(int),
        file_system_outputs=df.file_system_outputs.astype(int),
        involuntary_context_switches=df.involuntary_context_switches.astype(int),
        iterations=df.iterations.astype(int),
        major_page_faults=df.major_page_faults.astype(int),
        max_rss_kb=df.max_rss_kb.astype(int),
        minor_page_faults=df.minor_page_faults.astype(int),
        n_children=df.n_children.astype(int),
        fraction_cpu=df.percent_cpu.str[:-1].astype(float) / 100,
        random_effect_points=df.random_effect_points.astype(int),
        rate_count=df.rate_count.astype(int),
        signals_delivered=df.signals_delivered.astype(int),
        smooth_count=df.smooth_count.astype(int),
        sys=df.sys.astype(float),
        user=df.user.astype(float),
        variables=df.variables.astype(int),
        voluntary_context_switches=df.voluntary_context_switches.astype(int),
        Sincidence=df.Sincidence.fillna(0).astype(int),
        Tincidence=df.Tincidence.fillna(0).astype(int),
        mtall=df.mtall.fillna(0).astype(int),
        mtexcess=df.mtexcess.fillna(0).astype(int),
        mtother=df.mtother.fillna(0).astype(int),
        mtspecific=df.mtspecific.fillna(0).astype(int),
        mtstandard=df.mtstandard.fillna(0).astype(int),
        prevalence=df.prevalence.fillna(0).astype(int),
        relrisk=df.relrisk.fillna(0).astype(int),
        remission=df.remission.fillna(0).astype(int),
        susceptible=df.susceptible.fillna(0).astype(int),
        withC=df.withC.fillna(0).astype(int),
    )
    df.loc[df.effect == "fixed", "random_effect_points"] = 0
    return df


def load_and_transform(directory):
    records = load_records(directory)
    renamed = shorter_names(records)
    df = pd.DataFrame(renamed)
    return individual_corrections(df)


def entry():
    parser = ArgumentParser()
    parser.add_argument(
        "--timing", type=Path, help="Directory containing timings")
    args = parser.parse_args()
    df = load_and_transform(args.timing)
    df.to_csv("timings.csv", index=False)


def write_csv(records, columns):
    headers = list(sorted(columns))
    with open("timings.csv", "w") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow([header for header in headers])
        for record in records:
            writer.writerow(
                [record[col] if col in record else "" for col in headers])


if __name__ == "__main__":
    entry()
