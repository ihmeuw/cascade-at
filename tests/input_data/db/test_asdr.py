from cascade.input_data.db.asdr import get_asdr_data


def test_asdr_columns(ihme):
    asdr = get_asdr_data(6, "step1", [101], with_hiv=True)
    assert not asdr.duplicated(["age_group_id", "location_id", "year_id", "sex_id"]).any()
    assert (asdr.location_id == 101).all()
