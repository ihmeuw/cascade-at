import json
import sys
import csv
from pathlib import Path
import pandas as pd


RENAME_HEADER = {
    "wall clock": "wall",
    "System time": "sys",
    "User time": "user",
    "Maximum resident": "rss",
    "avgint": None,
    "children": None,
    "data_cnt": None,
    "data_extent_cohort": None,
    "data_extent_primary": None,
    "data_point_cohort": None,
    "data_point_primary": None,
    "dismod_at command": "effect",
    "derivative_test_fixed": None,
    "ipopt iterations": "iterations",
    "quasi_fixed": None,
    "random_effect_points": None,
    "rate_count": None,
    "smooth_count": None,
    "variables": None,
    "zero_sum_random": None,
}


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
    for search_key, rename_value in RENAME_HEADER.items():
        match = [col for col in columns if search_key in col]
        assert len(match) == 1
        if rename_value:
            rename[match[0]] = rename_value
        else:
            rename[match[0]] = search_key
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
    )
    df.loc[df.effect == "fixed", "random_effect_points"] = 0
    return df


def load_and_transform(directory):
    records = load_records(directory)
    renamed = shorter_names(records)
    df = pd.DataFrame(renamed)
    return individual_corrections(df)


def entry():
    directory = Path(sys.argv[1])
    df = load_and_transform(directory)
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
