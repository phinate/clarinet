import pytest
from pydantic import BaseModel
from pydantic import ValidationError

from clarinet import Modelstring


@pytest.mark.parametrize(
    "string",
    ("[A", "[]", "[A][B|A|C][C]", "[A| ]", "[A][|B]", "[A][B|:A]", [3]),
)
def test_from_modelstring_invalid(string):
    class test(BaseModel):
        s: Modelstring

    with pytest.raises(ValidationError):
        test(s=string)
