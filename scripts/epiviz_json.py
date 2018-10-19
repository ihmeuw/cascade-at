import argparse
import json

from cascade.input_data.db.configuration import load_raw_settings_mvid, load_raw_settings_meid
from cascade.testing_utilities import make_execution_context

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def main():
    parser = argparse.ArgumentParser("Download the JSON settings for an epiviz model")
    parser.add_argument("output_file", type=argparse.FileType("w"))
    parser.add_argument("--meid", type=int)
    parser.add_argument("--mvid", type=int)
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
    main()
