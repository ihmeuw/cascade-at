#!/usr/bin/env python
import json
import logging
import sys
import pandas as pd
import numpy as np
from functools import reduce

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


class CovariateReference:
    def __init__(self, inputs):
        self.inputs = inputs
        self.cov_ids = {c.covariate_id: c.name
                        for c in inputs.covariate_specs.covariate_specs
                        if c.study_country == 'country'}
        self.loc_df = inputs.location_dag.df
        self.pop_df = inputs.population.configure_for_dismod()
    def configure_for_dismod(self, covariate_data):
        from cascade_at.inputs.utilities.covariate_weighting import CovariateInterpolator
        def handle_exception(**kwds):
            try: return cov.interpolate(**kwds)
            except: return None
        cov_df = covariate_data.configure_for_dismod(self.pop_df, self.loc_df)
        cov = CovariateInterpolator(cov_df, self.pop_df)
        cov_id = covariate_data.covariate_id
        cov_name = self.cov_ids[cov_id]
        cov_df = pd.DataFrame([{'location_id': loc_id, 'sex_id': sex_id,
                                cov_name: handle_exception(loc_id = loc_id,sex_id = sex_id, age_lower=0, age_upper=100, time_lower = 1970, time_upper = 2020)}
                              for loc_id in sorted(cov_df.location_id.unique())
                              for sex_id in (1,2,3)])
        self.inputs.transform_country_covariates(cov_df)
        return cov_df

def all_locations(inputs, settings):
    import json
    import dill
    from cascade_at.inputs.measurement_inputs import MeasurementInputs
    covariate_id = [i.country_covariate_id for i in settings.country_covariate]
    inputs2 = MeasurementInputs(model_version_id=settings.model.model_version_id,
                                gbd_round_id=settings.gbd_round_id,
                                decomp_step_id=settings.model.decomp_step_id,
                                csmr_cause_id = settings.model.add_csmr_cause,
                                crosswalk_version_id=settings.model.crosswalk_version_id,
                                country_covariate_id=covariate_id,
                                conn_def='epi',
                                location_set_version_id=settings.location_set_version_id)
    inputs2.get_raw_inputs()
    inputs2.configure_inputs_for_dismod(settings)

    return inputs2

    

def main():

    args = ARG_LIST.parse_args(sys.argv[1:])
    logging.basicConfig(level=LEVELS[args.log_level])

    from cascade_at.settings.settings import load_settings

    with open(args.json_file) as f:
        settings_json = json.load(f)
    settings = load_settings(settings_json=settings_json)

    if 0:
        from cascade_at.executor.configure_inputs import configure_inputs
        global context, inputs
        context, inputs = configure_inputs(
            model_version_id = args.model_version_id,
            make = False,
            configure = False,
            test_dir=args.test_dir,
            json_file=args.json_file,
        )

        inputs2 = all_locations(inputs, settings)

        for d in inputs, inputs2:
            print()
            for integrand in sorted(d.dismod_data.measure.unique()):
                print (integrand, len(d.dismod_data[d.dismod_data.measure == integrand]), 'locations', len(d.dismod_data.loc[d.dismod_data.measure == integrand].location_id.unique()))

        if 1:
            import shutil
            import dill
            with open(f'/tmp/cascade_dir/data/{args.model_version_id}/inputs/inputs1.p', 'wb') as stream:
                dill.dump(inputs, stream)
            with open(f'/tmp/cascade_dir/data/{args.model_version_id}/inputs/inputs2.p', 'wb') as stream:
                dill.dump(inputs2, stream)
            shutil.copy2(f'/tmp/cascade_dir/data/{args.model_version_id}/inputs/inputs2.p', f'/tmp/cascade_dir/data/{args.model_version_id}/inputs/inputs.p')

        from cascade_at.executor.dismod_db import dismod_db
        # It seems that dismod_db gets mtall/mtspecific from inputs.p for just the parent and the parents children
        # And it seems that the entire set of locations is in inputs.p for mtall and mtspecific.
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

    else:
            
        import dill

        with open(f'/tmp/cascade_dir/data/{args.model_version_id}/inputs/inputs1.p', 'rb') as stream:
            inputs = dill.load(stream)
        global covariate_reference, data, asdr, csmr

        cov_ref = CovariateReference(inputs)
        covariate_reference = reduce(lambda x, y: pd.merge(x, y),
                                     [cov_ref.configure_for_dismod(c) for c in inputs.covariate_data])

        data = inputs.data.configure_for_dismod(relabel_incidence=settings.model.relabel_incidence)
        data = inputs.add_covariates_to_data(data)

        asdr = inputs.asdr.configure_for_dismod()
        csmr = inputs.csmr.configure_for_dismod()
        
        if __debug__:
            asdr_grps = asdr.groupby(['sex_id', 'location_id'])
            csmr_grps = csmr.groupby(['sex_id', 'location_id'])
            import numpy as np
            assert np.all(asdr_grps.count() == csmr_grps.count())



if __name__ == '__main__':
    if not sys.argv[0]:
        mvid = 475746
        json_cmd = f'--json-file /Users/gma/ihme/epi/at_cascade/data/{mvid}_settings-HighIncome.json'
        cmd = f'dismod_ihme_input --model-version-id {mvid} --configure {json_cmd}'

        mvid = 475527
        json_cmd = f'--json-file /Users/gma/ihme/epi/at_cascade/data/{mvid}_settings-SLatinAmerica.json'
        cmd = f'dismod_ihme_input --model-version-id {mvid} {json_cmd} --test-dir /tmp'
        
        mvid = 475588
        json_cmd = f'--json-file /Users/gma/ihme/epi/at_cascade/data/{mvid}_settings-USA.json'
        cmd = f'dismod_ihme_input --model-version-id {mvid} --configure {json_cmd} --test-dir /tmp'

        mvid = 475588
        json_cmd = f'--json-file /Users/gma/ihme/epi/at_cascade/data/{mvid}_settings-world.json'
        cmd = f'dismod_ihme_input --model-version-id {mvid} --configure {json_cmd} --test-dir /tmp'

        mvid = 475588
        json_cmd = f'--json-file /Users/gma/ihme/epi/at_cascade/data/{mvid}_settings-Alabama.json'
        cmd = f'dismod_ihme_input --model-version-id {mvid} --configure {json_cmd} --test-dir /tmp'

        print (cmd)
        sys.argv = cmd.split()

    cmd = f'dismod_ihme_input --model-version-id mvid'
    print ('ERROR this command with no json and no test-dir is not working')
    print (cmd)


    main()
