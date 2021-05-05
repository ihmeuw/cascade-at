import sys
from pathlib import Path
import numpy as np
import pandas as pd
import sqlalchemy
from cascade_at.dismod.api.dismod_io import DismodIO
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
plt.interactive(1)

def parse_args():
    import argparse
    from distutils.util import strtobool as str2bool
    parser = argparse.ArgumentParser()
    name_string = "-filename" if sys.argv[0] == '' else "filename"
    parser.add_argument(name_string, type=str, help="Dismod_AT cascade dbs directory")
    parser.add_argument("-l", "--location-ids", type = int, nargs='+', default = None,
                        help="Location ids to plot, default = all")
    parser.add_argument("-i", "--integrands", type = str, nargs='+', default = None,
                        help = f"Integrands to plot, default = all")
    parser.add_argument("-d", "--disease", type = str, default = '', help="Disease name (for plot title)")
    args = parser.parse_args()
    return args

def plot(residuals, path, mvid, integrands = None, disease = ''):
    if integrands:
        integrand_names = integrands
    else:
        integrand_names = residuals.integrand_name.unique()
    pdf_file = path / 'weighted_residuals.pdf'
    pdf = PdfPages(pdf_file)
    print (f"Saving plot {pdf_file}")
    for i, n in enumerate(integrand_names):
        integrand = residuals.loc[residuals.integrand_name == n]
        l = len(residuals.c_parent_location_id.unique())
        m = len(integrand.c_parent_location_id.unique())
        if m:
            # fig, ax = plt.subplots(1, 1, tight_layout=True)
            fig = plt.figure(i)
            ax = plt.gca()
            plt.title(f"{disease} {n.capitalize()} MVID {mvid} \n({m} of {l} total locations have data)")
            r = integrand.weighted_residual
            plt.xlabel(f'Weighted Residual (range = [{np.round(r.min(), 1)}, {np.round(r.max(),1)}])')
            plt.ylabel('Count')
            n_bins = max(100, int(len(r)/100))
            ax.hist(r, bins=n_bins)
            pdf.savefig( fig )
            plt.close( fig.number )
    pdf.close()

def collect(dbs, location_ids = None):
    residuals = pd.DataFrame()
    i = -1
    if location_ids:
        dbs = [p for p in dbs if int(p.parts[-3]) in location_ids]
    for p in dbs:
        global db
        db = DismodIO(p)
        try:
            db.option
            loc,sex = map(int, p.parts[-3:-1])
            fit = (db.data_subset.merge(db.data, how='left')
                   .merge(db.fit_data_subset, left_on = 'data_subset_id', right_on = 'fit_data_subset_id')
                   .merge(db.node, how='left')
                   .merge(db.integrand, how='left'))
            cov_names = {f'x_{row.covariate_id}': row.c_covariate_name
                         for i, row in db.covariate[['covariate_id', 'c_covariate_name']].iterrows()}
            fit.rename(columns = cov_names, inplace=True)
            fit['c_parent_location_id'] = loc
            cols = (['c_parent_location_id', 'c_location_id', 'integrand_name', 'data_name',
                     'age_lower', 'age_upper', 'time_lower', 'time_upper', 'weighted_residual']
                    + list(cov_names.values()))
            residuals = residuals.append(fit[cols])
            i += 1
            print (i, f'sex: {sex}, location: {loc}')
        except:
            continue
    return residuals

def main():
    args = parse_args()
    path = Path(args.filename).expanduser()
    assert path.is_dir()

    dbs = list(path.glob('**/*/dismod.db'))
    mvid,dbs_str = dbs[0].parts[-5:-3]
    assert dbs_str == 'dbs', f'The path {path} does not specify the root of the cascade data structure for mvid {mvid}'
    print (f"Accumulating the fit residuals for the entire cascade execution of mvid {mvid}.")

    residuals = collect(dbs, location_ids = args.location_ids)
    plot_path = path.parent / 'plots'
    if not plot_path.exists():
        plot_path.mkdir(parents=True, exist_ok=True)
    plot(residuals, plot_path, mvid, integrands = args.integrands, disease = args.disease)

if __name__ == '__main__':
    if not sys.argv[0]:
        mvid_path = '/Users/gma/ihme/epi/at_cascade/data/475588/dbs'
        superregions = [str(i) for i in [4, 31, 64, 103, 137, 158, 166]]
        superregions = [str(i) for i in ['1']]
        sys.argv = ['plot_residuals', mvid_path, '--location-ids'] + superregions
    residuals = main()


       
