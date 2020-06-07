import pandas as pd


def calculate_omega(asdr: pd.DataFrame, csmr: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates other cause mortality (omega) from ASDR (mtall -- all-cause
    mortality) and CSMR (mtspecific -- cause-specific mortality). For most
    diseases, mtall is a good approximation to omega, but we calculate
    omega = mtall - mtspecific in case it isn't. For diseases without CSMR
    (self.csmr_cause_id = None), then omega = mtall.
    """
    join_columns = ['location_id', 'time_lower', 'time_upper',
                    'age_lower', 'age_upper', 'sex_id']
    mtall = asdr[join_columns + ['meas_value']].copy()
    mtall.rename(columns={'meas_value': 'mtall'}, inplace=True)

    if csmr.empty:
        omega = mtall.copy()
        omega.rename(columns={'mtall': 'mean'}, inplace=True)
    else:
        mtspecific = csmr[join_columns + ['meas_value']].copy()
        mtspecific.rename(
            columns={'meas_value': 'mtspecific'}, inplace=True)
        omega = mtall.merge(mtspecific, on=join_columns)
        omega['mean'] = omega['mtall'] - omega['mtspecific']
        omega.drop(columns=['mtall', 'mtspecific'], inplace=True)

    negative_omega = omega['mean'] < 0
    if any(negative_omega):
        raise ValueError("There are negative values for omega. Must fix.")

    return omega
