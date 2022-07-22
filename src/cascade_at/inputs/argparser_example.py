import sys
import argparse


def main(age_group_set_id = 4, mvid = 3, cause_id = 2):
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", help="increase output verbosity")
    parser.add_argument("-r", "--root-node-path", type = str, default = '',
                        help = f"Age Group Set ID -- default ''")
    parser.add_argument("-mvid", "--model_version_id", type = int, default = mvid,
                        help = f"Model Version ID -- default = {mvid}")
    parser.add_argument("-c", "--cause_id", type = int, default = cause_id,
                        help = f"Cause ID -- default = {cause_id}")
    parser.add_argument("-a", "--age_group_set_id", type = int, default = age_group_set_id,
                        help = "Age Group Set ID -- default {age_group_set_id}")
    args = parser.parse_args()
    return args

args = main()
if args.verbose:
    print("verbosity turned on")
    print (args)
