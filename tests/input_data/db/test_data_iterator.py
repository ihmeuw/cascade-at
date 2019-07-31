import pytest

from cascade.input_data.db.data_iterator import grouped_by_count


@pytest.mark.parametrize("in_string,count", [
    (list(range(11)), 5),
    (list(range(10)), 5),
    (list(range(9)), 5),
    (list(), 5),
    ([chr(x + ord("a")) for x in range(23)], 7),
])
def test_grouped_by_count_happy(in_string, count):
    result = list()
    total = set()
    under = 0
    for group in grouped_by_count(in_string, count):
        result.extend(group)
        total |= set(group)
        assert len(group) <= count
        assert len(group) > 0
        if len(group) < count:
            under += 1
    assert len(total) == len(result)
    assert len(total) == len(in_string)
    assert under <= 1
