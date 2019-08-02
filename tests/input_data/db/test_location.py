import networkx as nx
import pytest

from cascade.input_data.db.locations import (
    get_descendants, location_id_from_location_and_level, location_hierarchy,
    location_id_from_start_and_finish, all_locations_with_these_parents
)


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
def sample_locations():
    G = nx.DiGraph()
    G.add_nodes_from(list(range(8)))
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (6, 7)])
    assert len(G.nodes) == 8
    return G


@pytest.mark.parametrize("parents,expected", [
    ([0], [0, 1, 2]),
    ([0, 1], [0, 1, 2, 3, 4]),
    ([2], [2, 5, 6]),
    ([2, 5], [2, 5, 6]),
    ([2, 6], [2, 5, 6, 7]),
    ([6], [6, 7]),
    ([7], [7]),
])
def test_all_with_parents__happy(sample_locations, parents, expected):
    result = all_locations_with_these_parents(sample_locations, parents)
    result.sort()
    assert result == expected


def test_get_descendants__all_descendants(sample_locations):
    parent_location_id = 0
    # descendants of global
    assert set(get_descendants(sample_locations, parent_location_id)) == set(range(1, 8))

    parent_location_id = 7
    # descendants of a leaf (ie. nothing)
    assert set(get_descendants(sample_locations, parent_location_id)) == set()


def test_get_descendants__only_children(sample_locations):
    parent_location_id = 0
    # children of global
    assert set(get_descendants(sample_locations, parent_location_id, children_only=True)) == {1, 2}

    parent_location_id = 5
    # children of a leaf
    assert set(get_descendants(sample_locations, parent_location_id, children_only=True)) == set()


def test_get_descendants__include_parent(sample_locations):
    parent_location_id = 0
    # descendants of global and iteslf
    assert set(get_descendants(sample_locations, parent_location_id, include_parent=True)) == set(range(0, 8))
    # children of global and iteslf
    assert set(get_descendants(
        sample_locations, parent_location_id, children_only=True, include_parent=True)) == {0, 1, 2}

    parent_location_id = 5
    # descendants of a leaf and itself
    assert set(get_descendants(sample_locations, parent_location_id, include_parent=True, children_only=True)) == {5}
    # children of a leaf and itself
    assert set(get_descendants(sample_locations, parent_location_id, include_parent=True, children_only=True)) == {5}


def test_location_id_from_location_and_level__happy_path(sample_locations):
    assert location_id_from_location_and_level(sample_locations, 0, 1)[0] == 0
    assert location_id_from_location_and_level(sample_locations, 7, 1)[0] == 0
    assert location_id_from_location_and_level(sample_locations, 7, 2)[0] == 2
    assert location_id_from_location_and_level(sample_locations, 4, 2)[0] == 1
    assert location_id_from_location_and_level(sample_locations, 7, "most_detailed")[0] == 7


def test_drill_from_location_and_level__happy_path(sample_locations):
    assert location_id_from_location_and_level(sample_locations, 0, 1) == [0]
    assert location_id_from_location_and_level(sample_locations, 7, 1) == [0, 2, 6, 7]
    assert location_id_from_location_and_level(sample_locations, 7, 2) == [2, 6, 7]
    assert location_id_from_location_and_level(sample_locations, 4, 2) == [1, 4]
    assert location_id_from_location_and_level(sample_locations, 7, "most_detailed") == [7]


@pytest.mark.parametrize("start,finish,ans", [
    (0, 1, [0, 1]),
    (None, 1, [0, 1]),
    (None, 4, [0, 1, 4]),
    (None, "4", [0, 1, 4]),
    (2, 7, [2, 6, 7]),
    (None, 7, [0, 2, 6, 7]),
])
def test_drill_from_location_and_level__no_start(sample_locations, start, finish, ans):
    assert location_id_from_start_and_finish(sample_locations, start, finish) == ans


def test_location_id_from_location_and_level__too_low(sample_locations):
    with pytest.raises(Exception):
        location_id_from_location_and_level(sample_locations, 0, "most_detailed")

    with pytest.raises(Exception):
        location_id_from_location_and_level(sample_locations, 2, 3)


def test_location_hierarchy_networkx(ihme):
    locs = location_hierarchy(6, location_set_id=35)
    assert nx.is_directed_acyclic_graph(locs)
    assert nx.dag_longest_path_length(locs) == 6
    assert nx.dag_longest_path(locs)[0] == 1
    assert locs.nodes[1]["level"] == 0
    assert locs.nodes[13]["location_name"] == "Malaysia"


def test_ancestors_level(ihme):
    locs = location_hierarchy(6, location_set_id=35)
    drill = list(nx.topological_sort(nx.subgraph(locs, nbunch=nx.ancestors(locs, 491))))
    assert drill == [1, 4, 5, 6]
