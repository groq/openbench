"""Structured-output schema helpers."""

from typing import Any

from inspect_ai.model import ResponseSchema
from pydantic import Field


class RawResponseSchema(ResponseSchema):
    """Response schema that retains keywords Inspect does not model.

    Inspect's ``JSONSchema`` model intentionally exposes only a portable subset
    of JSON Schema. Providers that support a larger surface need the original
    document as well, otherwise assertions such as ``oneOf``, ``pattern``, and
    ``minLength`` are silently discarded during model validation.
    """

    raw_json_schema: dict[str, Any] = Field(exclude=True)
