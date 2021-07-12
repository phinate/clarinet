import pytest
from pydantic import ValidationError

from clarinet import DiscreteNode

name = "test_name"
prob_table = [[0.3, 0.7], [0.7, 0.3]]


def test_basic_functionality():
    states = ["c1", "c2"]
    parents, children = ["A", "B"], ["C", "D"]
    x = DiscreteNode(
        name=name,
        parents=parents,
        children=children,
        prob_table=prob_table,
        states=states,
    )
    assert x.states == ("c1", "c2")
    x.json()


@pytest.mark.parametrize(
    "states",
    (
        pytest.param([], id="empty"),
        pytest.param(["e"], id="only one state"),
        pytest.param("fagfdsgfd", id="string"),
    ),
)
def test_states_validerror(states):
    with pytest.raises(ValidationError):
        DiscreteNode(name=name, states=states)
