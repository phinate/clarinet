import json

import pytest

from clarinet import BayesNet
# make sure to use every node type in every example where possible
# add prob tables later

normal_dict = {
    "raining": {
        "parents": ["cloudy"],
        "children": ["wet grass"],
        "categories": ["raining", "not raining"],
    },
    "cloudy": {
        "children": ["raining"],
        "categories": ["cloudy", "clear"],
    },
    "wet grass": {
        "parents": ["raining"],
        "categories": ["wet", "dry"],
    },
}

more_complex_dict = {
    'O': {'name': 'O', 'parents': ['E'], 'children': ['T']},
    'S': {'name': 'S', 'parents': [], 'children': ['E']},
    'R': {'name': 'R', 'parents': ['E'], 'children': ['T']},
    'A': {'name': 'A', 'parents': [], 'children': ['E']},
    'E': {'name': 'E', 'parents': ['A', 'S'], 'children': ['O', 'R']},
    'T': {'name': 'T', 'parents': ['O', 'R'], 'children': []}
}


vc = 'very_complex_dict.json'

with open(vc) as file:
    very_complex_dict = json.loads(file.read())

very_complex_string = (
    "[Age][Mileage][SocioEcon|Age][GoodStudent|Age:SocioEcon]" +
    "[RiskAversion|Age:SocioEcon][OtherCar|SocioEcon]" +
    "[VehicleYear|SocioEcon:RiskAversion]" +
    "[MakeModel|SocioEcon:RiskAversion][SeniorTrain|Age:RiskAversion]" +
    "[HomeBase|SocioEcon:RiskAversion][AntiTheft|SocioEcon:RiskAversion]" +
    "[RuggedAuto|VehicleYear:MakeModel][Antilock|VehicleYear:MakeModel]" +
    "[DrivingSkill|Age:SeniorTrain][CarValue|VehicleYear:MakeModel:Mileage]" +
    "[Airbag|VehicleYear:MakeModel][DrivQuality|RiskAversion:DrivingSkill]" +
    "[Theft|CarValue:HomeBase:AntiTheft][Cushioning|RuggedAuto:Airbag]" +
    "[DrivHist|RiskAversion:DrivingSkill]" +
    "[Accident|DrivQuality:Mileage:Antilock]" +
    "[ThisCarDam|RuggedAuto:Accident][OtherCarCost|RuggedAuto:Accident]" +
    "[MedCost|Age:Accident:Cushioning][ILiCost|Accident]" +
    "[ThisCarCost|ThisCarDam:Theft:CarValue]" +
    "[PropCost|ThisCarCost:OtherCarCost]"
)

cycle_dict = {
    "raining": {
        "parents": ["cloudy"],
        "children": ["wet grass"],
        "categories": ["raining", "not raining"],
    },
    "cloudy": {
        "children": ["raining"],
        "categories": ["cloudy", "clear"],
    },
    "wet grass": {
        "parents": ["raining"],
        "children": ["cloudy"],  # cycle
        "categories": ["wet", "dry"],
    },
}

isolated_dict = {
    "raining": {
        "parents": ["cloudy"],
        "children": ["wet grass"],
        "categories": ["raining", "not raining"],
    },
    "cloudy": {
        "children": ["raining"],
        "categories": ["cloudy", "clear"],
    },
    "wet grass": {
        "parents": ["raining"],
        "categories": ["wet", "dry"],
    },
    "yeet skeet": {
        "parents": [],
    },
}

missing_dict = {
    "raining": {
        "parents": ["cloudy"],
        "children": ["wet grass"],
        "categories": ["raining", "not raining"],
    },
    "cloudy": {
        "children": ["raining"],
        "categories": ["cloudy", "clear"],
    },
    "wet grass": {
        "parents": ["raining", "yeet skeet"],
        "categories": ["wet", "dry"],
    },
}


@pytest.mark.parametrize(
    ('params', 'expected'),
    (
        pytest.param(
            dict(nodes=normal_dict),
            normal_dict,
            id="simple dag structure with categories",
        ),
        pytest.param(
            dict(nodes=more_complex_dict),
            more_complex_dict,
            id="more complex dag structure",
        ),
        pytest.param(
            dict(nodes=very_complex_dict),
            very_complex_dict,
            id="very complex dag structure",
        ),
    )
)
def test_net_instantiation(params, expected):
    x = BayesNet(**params)
    assert dict(x.dict()["nodes"]) == dict(x.nodes) == expected
    x.json()

# isolated case may be changed later
# (e.g. case of learned structure? undirected models?)


@pytest.parametrize(
    'params',
    (
        pytest.param(dict(nodes=cycle_dict), id='cyclic dag'),
        pytest.param(dict(nodes=isolated_dict), id='dag with isolated node'),
        pytest.param(
            dict(nodes=missing_dict),
            id='dag with parents/children that arent nodes'
        ),
    ),
)
def test_net_instantiation_failure_cases(params):
    with pytest.raises(AssertionError):
        BayesNet(**params)


@pytest.parametrize(
    ('string', 'expected'),
    (
        pytest.param(
            very_complex_string,
            very_complex_dict,
            id="complex modelstring"
        )
    )
)
def test_from_modelstring(string, expected):
    x = BayesNet.from_modelstring(string)
    assert dict(x.nodes) == expected
#     assert x.modelstring = string


def test_from_modelstring_invalid(string): pass


def test_add_node(node, expected): pass


def test_modify_nodes(node_types, node_kwargs, expected): pass
