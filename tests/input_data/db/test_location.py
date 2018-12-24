import pytest

from cascade.testing_utilities import make_execution_context

from cascade.input_data.db.locations import get_descendents, location_id_from_location_and_level


class MockLocation:
    def __init__(self, node_id, nodes):
        self._node_id = node_id
        self._nodes = nodes
        self._node_parents = {v: k for k, vs in nodes.items() for v in vs}

    def get_node_by_id(self, node_id):
        return MockLocation(node_id, self._nodes)

    def get_nodelvl_by_id(self, node_id):
        def ancestors(node):
            parent = node.parent
            if parent is None:
                return 0
            else:
                return 1 + ancestors(parent)

        return ancestors(self.get_node_by_id(node_id))

    @property
    def id(self):
        return self._node_id

    @property
    def parent(self):
        parent_id = self._node_parents.get(self._node_id)
        if parent_id is not None:
            return self.get_node_by_id(parent_id)
        else:
            return None

    @property
    def children(self):
        return [self.get_node_by_id(child_id) for child_id in self._nodes[self.id]]

    def all_descendants(self):
        def recursive_children(node):
            result = []
            for child in node.children:
                result.append(child)
                result.extend(recursive_children(child))
            return result

        return recursive_children(self)


@pytest.fixture
def mock_locations(mocker):
    locations = mocker.patch("cascade.input_data.db.locations.get_location_hierarchy_from_gbd")
    nodes = {0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [], 4: [], 5: [], 6: [7], 7: []}
    locations.return_value = MockLocation(0, nodes)
    return locations


def test_get_descendents__all_descendents(mock_locations):
    ec = make_execution_context(location_id=0)
    # descendents of global
    assert set(get_descendents(ec)) == set(range(1, 8))

    ec = make_execution_context(location_id=7)
    # descendents of a leaf (ie. nothing)
    assert set(get_descendents(ec)) == set()


def test_get_descendents__only_children(mock_locations):
    ec = make_execution_context(location_id=0)
    # children of global
    assert set(get_descendents(ec, children_only=True)) == {1, 2}

    ec = make_execution_context(location_id=5)
    # children of a leaf
    assert set(get_descendents(ec, children_only=True)) == set()


def test_get_descendents__include_parent(mock_locations):
    ec = make_execution_context(location_id=0)
    # descendents of global and iteslf
    assert set(get_descendents(ec, include_parent=True)) == set(range(0, 8))
    # children of global and iteslf
    assert set(get_descendents(ec, children_only=True, include_parent=True)) == {0, 1, 2}

    ec = make_execution_context(location_id=5)
    # descendents of a leaf and itself
    assert set(get_descendents(ec, include_parent=True, children_only=True)) == {5}
    # children of a leaf and itself
    assert set(get_descendents(ec, include_parent=True, children_only=True)) == {5}


def test_location_id_from_location_and_level__happy_path(mock_locations):
    ec = make_execution_context()
    assert location_id_from_location_and_level(ec, 0, 1) == [0]
    assert location_id_from_location_and_level(ec, 7, 1) == [0]
    assert location_id_from_location_and_level(ec, 7, 2) == [2]
    assert location_id_from_location_and_level(ec, 4, 2) == [1]
    assert location_id_from_location_and_level(ec, 7, "most_detailed") == [7]


def test_location_id_from_location_and_level__too_low(mock_locations):
    ec = make_execution_context()
    with pytest.raises(Exception):
        location_id_from_location_and_level(ec, 0, "most_detailed")

    with pytest.raises(Exception):
        location_id_from_location_and_level(ec, 2, 3)
