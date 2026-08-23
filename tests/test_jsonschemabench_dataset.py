import json

from inspect_ai.model import ResponseSchema
from jsonschema import Draft202012Validator

from openbench.datasets.jsonschemabench.jsonschemabench import _adapt_schema


def test_adapt_schema_normalizes_nullable_type_arrays_for_response_schema():
    original = {
        "type": "object",
        "properties": {
            "values": {
                "type": ["array", "null"],
                "items": {"type": ["string", "null"]},
            }
        },
        "required": ["values"],
    }

    adapted = json.loads(_adapt_schema(json.dumps(original)))

    ResponseSchema(name="nullable_array", json_schema=adapted, strict=False)
    validator = Draft202012Validator(adapted)
    validator.validate({"values": None})
    validator.validate({"values": ["one", None]})
    assert not validator.is_valid({"values": [1]})

    values_schema = adapted["properties"]["values"]
    assert values_schema["anyOf"][0]["type"] == "array"
    assert values_schema["anyOf"][1] == {"type": "null"}
    assert values_schema["anyOf"][0]["items"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
