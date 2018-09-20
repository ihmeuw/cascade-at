import argparse
import json

from cascade.input_data.db.configuration import settings_for_model


def main():
    parser = argparse.ArgumentParser("Download the JSON settings for an epiviz model")
    parser.add_argument("output_file", type=argparse.FileType("w"))
    parser.add_argument("--meid", type=int)
    parser.add_argument("--mvid", type=int)
    args = parser.parse_args()

    raw_settings = settings_for_model(args.meid, args.mvid)

    json.dump(raw_settings, args.output_file, indent=4)


if __name__ == "__main__":
    main()
