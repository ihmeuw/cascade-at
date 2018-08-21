DMCSV2DB Tool
=============

This is a script that reads CSV files and runs Dismod-AT to fit and predict
from them. It presents basic functionality from Dismod-AT. It puts input
data into a Dismod-AT formatted sqlite file in order for Dismod-AT to perform
a fit on the data.

There is a first version of this in Dismod-AT. The version installed with
Cascade is something the team can augment as needed.

The input is a CSV file called ``measure.csv`` with the following columns.

 *  integrand (str): One of ``remission``, ``mtexcess``, ``prevalence``, ``mtall``, ``mtother``.
 *  age_lower (float): Lower age of observation age-span and time-span.
 *  age_upper (float): Upper age of observation age-span and time-span.
 *  time_lower (float): Lower time of observation age-span and time-span.
 *  time_upper (float): Upper time of observation age-span and time-span.
 *  meas_value (float): The value of the integrand for that age-time-span.
 *  meas_std (float): The standard deviation of the observation error.
 *  hold_out (int): 0 or 1, where 1 says to use this data as the constraint, mtother.

The ``mtother`` data is special. The script assumes that this is entirely known
and treat it as a constraint on the fit. The ``mtother`` data must,
as a consequence, be defined over the whole computational grid.
For instance, if it is defined for ages (0.0, 10.0, 50.0) and years
(1990, 1995, 2000), then it must be defined for all nine combinations of
those ages and times. In addition, the fit will be found on those nine
points for all of the rates of incidence, remission, and excess mortality.

As an example, assuming you are running on the cluster, you could run::

    # Reads measure.csv and writes fit.db
    dmcsv2db measure.csv fit.db
    dmdismod fit.db set option quasi_fixed false
    dmdismod fit.db set option ode_step_size 1
    dmdismod fit.db init
    # Given measured data, finds underlying rates.
    dmdismod fit.db fit fixed
    # Now that rates are found, tell us what measurements would be.
    dmdismod fit.db predict fit_var
    # Translates db contents into CSV files.
    dmdismodpy fit.db db2csv

