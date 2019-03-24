import argparse
import json
from textwrap import dedent

from cascade.input_data.db.configuration import load_raw_settings_mvid, load_raw_settings_meid
from cascade.executor.execution_context import make_execution_context

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def entry():
    parser = argparse.ArgumentParser(dedent("""
    EpiViz-AT saves all information for any model version into a settings
    document. This downloads that settings document and saves it as
    a JSON file that can be edited and run by dmcascade.
    """))
    parser.add_argument("output_file", type=argparse.FileType("w"))
    parser.add_argument("--meid", type=int,
                        help="If you specify a modelable entity ID, this "
                        "will download the most recent model version ID for "
                        " that MEID.")
    parser.add_argument("--mvid", type=int,
                        help="Model version ID as seen in EpiViz-AT.")
    args = parser.parse_args()

    ec = make_execution_context()
    if args.mvid:
        raw_settings, found_mvid = load_raw_settings_mvid(ec, args.mvid)
    elif args.meid:
        raw_settings, found_mvid = load_raw_settings_meid(ec, args.meid)
    else:
        print("Need either an meid or an mvid to retrieve settings.")
        exit()

    json.dump(raw_settings, args.output_file, indent=4)


if __name__ == "__main__":
    entry()
