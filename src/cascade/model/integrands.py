from cascade.input_data.db.demographics import get_age_groups, get_years


def integrand_grids_from_gbd(model_context, execution_context):
    gbd_age_groups = get_age_groups(execution_context)
    age_ranges = [(r.age_group_years_start, r.age_group_years_end) for _, r in gbd_age_groups.iterrows()]
    time_ranges = [(y, y) for y in get_years(execution_context)]

    for integrand in model_context.outputs.integrands:
        integrand.age_ranges = age_ranges
        integrand.time_ranges = time_ranges
