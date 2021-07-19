# import numpy as np
# import pytest

# from clarinet import BaseDiscrete, Node

# name = "test_name"


# @pytest.mark.parametrize(
#     ("prob_table", "expected_prob_table"),
#     (
#         pytest.param([], np.array([]), id="empty list"),
#         pytest.param(
#             [[0.3, 0.7], [0.7, 0.3]],
#             np.array([[0.3, 0.7], [0.7, 0.3]]),
#             id="non-empty list",
#         ),
#         pytest.param(
#             np.array([[0.3, 0.7], [0.7, 0.3]]),
#             np.array([[0.3, 0.7], [0.7, 0.3]]),
#             id="non-empty array",
#         ),
#     ),
# )
# def test_basic_functionality(prob_table, expected_prob_table):
#     parents, children = ["A", "B"], ["C", "D"]
#     x = BaseDiscrete(
#         name=name, parents=parents, children=children, prob_table=prob_table
#     )
#     assert np.allclose(x.dict()["prob_table"], expected_prob_table)
#     x.json()


# def test_prob_table():
#     pass  # for when shape checks and sum(probs)=1 checks exist


# def test_from_node():
#     n = Node(name=name, parents=["F"])  # arbitrary
#     prob_table = np.array([[0.3, 0.7], [0.7, 0.3]])
#     BaseDiscrete.from_node(n, prob_table=prob_table)
