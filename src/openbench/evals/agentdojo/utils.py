from typing import Any, Callable, Sequence, TypeAlias

import yaml
from pydantic import BaseModel

FunctionReturnType: TypeAlias = (
    BaseModel
    | Sequence["FunctionReturnType"]
    | dict[str, Any]
    | str
    | int
    | float
    | bool
    | None
)
"""Union of valid return types for functions. The default [FunctionsRuntime][agentdojo.functions_runtime.FunctionsRuntime]
is not guaranteed to work with other types."""


def tool_result_to_str(
    tool_result: FunctionReturnType,
    dump_fn: Callable[
        [dict[str, Any] | list[str | dict[str, Any]]], str
    ] = yaml.safe_dump,
) -> str:
    """
    Basic tool output formatter with YAML dump by default.

    Could work with `json.dumps` as `dump_fn`.
    """
    if isinstance(tool_result, BaseModel):
        return dump_fn(tool_result.model_dump()).strip()

    if isinstance(tool_result, list):
        res_items: list[str | dict[str, Any]] = []
        for item in tool_result:
            if type(item) in [str, int]:
                res_items.append(str(item))
            elif isinstance(item, BaseModel):
                res_items.append(item.model_dump())
            else:
                raise TypeError(
                    "Not valid type for item tool result: " + str(type(item))
                )

        # If type checking passes, this is guaranteed to be a list of BaseModel
        final_str = dump_fn(res_items).strip()
        return final_str

    return str(tool_result)
