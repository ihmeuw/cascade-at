#!/usr/bin/env python
import json
import logging
import sys

# from typing import Optional

from cascade_at.executor.args.arg_utils import ArgumentList
from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.executor.args.args import ModelVersionID, BoolArg, LogLevel, StrArg

LOG = get_loggers(__name__)

ARG_LIST = ArgumentList([
    ModelVersionID(),
    BoolArg('--make', help='whether or not to make the file structure for the cascade'),
    BoolArg('--configure', help='whether or not to configure for the IHME cluster'),
    LogLevel(),
    StrArg('--json-file', help='for testing, pass a json file directly by filepath'
                               'instead of referencing a model version ID.'),
    StrArg('--test-dir', help='if set, will save files to the directory specified.'
                              'Invalidated if --configure is set.')
])


def main():

    args = ARG_LIST.parse_args(sys.argv[1:])
    logging.basicConfig(level=LEVELS[args.log_level])

    from cascade_at.executor.configure_inputs import configure_inputs
    global context, inputs
    context, inputs = configure_inputs(
        model_version_id = args.model_version_id,
        make = False,
        configure = False,
        test_dir=args.test_dir,
        json_file=args.json_file,
    )

    from cascade_at.executor.dismod_db import dismod_db
    dismod_db(model_version_id = args.model_version_id,
              parent_location_id=inputs.drill_location_start,
              fill=True,
              test_dir=args.test_dir,
              save_fit = False,
              save_prior = False)

    from cascade_at.executor.run import run
    run(model_version_id = args.model_version_id,
        jobmon = False,
        make = False,
        skip_configure = True,
        json_file = args.json_file,
        test_dir = args.test_dir,
        execute_dag = False)


if __name__ == '__main__':
    if not sys.argv[0]:
        mvid = 475588
        # json_cmd = f'--json-file /Users/gma/ihme/epi/at_cascade/data/{mvid}_settings-USA.json'
        # cmd = f'dismod_ihme_input --model-version-id {mvid} --configure {json_cmd}'

        mvid = 475746
        json_cmd = f'--json-file /Users/gma/ihme/epi/at_cascade/data/{mvid}_settings-HighIncome.json'
        cmd = f'dismod_ihme_input --model-version-id {mvid} --configure {json_cmd}'

        mvid = 475527
        json_cmd = f'--json-file /Users/gma/ihme/epi/at_cascade/data/{mvid}_settings-SLatinAmerica.json'
        cmd = f'dismod_ihme_input --model-version-id {mvid} {json_cmd} --test-dir /tmp'

        print (cmd)
        sys.argv = cmd.split()

    main()
