#!/usr/bin/env python3
"""Probe structured-output constraints with adversarial generation requests."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from jsonschema import Draft202012Validator, FormatChecker

# Without the rfc3339-validator extra, jsonschema's built-in "time" checker
# mis-rejects valid RFC 3339 full-time values such as "10:00:00Z" and
# "21:00:00.000Z" (the two false-positive schema_leak results in the
# 2026-08-26 production run). Register an exact RFC 3339 full-time checker
# mirroring the API's validator (santhosh-tekuri isTime), including its
# leap-second rule: ss=60 is only valid when the offset-normalized time is
# 23:59 UTC.
_RFC3339_FULL_TIME = re.compile(
    r"^(?P<h>[01]\d|2[0-3]):(?P<m>[0-5]\d):(?P<s>[0-5]\d|60)(?:\.\d+)?"
    r"(?P<off>[Zz]|[+-](?:[01]\d|2[0-3]):[0-5]\d)$"
)

FORMAT_CHECKER = FormatChecker()


@FORMAT_CHECKER.checks("time")
def _is_rfc3339_full_time(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    match = _RFC3339_FULL_TIME.match(value)
    if match is None:
        return False
    if match.group("s") != "60":
        return True
    minutes = int(match.group("h")) * 60 + int(match.group("m"))
    offset = match.group("off")
    if offset not in ("Z", "z"):
        sign = -1 if offset[0] == "+" else 1
        minutes += sign * (int(offset[1:3]) * 60 + int(offset[4:6]))
        if minutes < 0:
            minutes += 24 * 60
    return minutes == 23 * 60 + 59


API_URL = os.environ.get(
    "GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions"
)


@dataclass(frozen=True)
class Case:
    name: str
    schema: dict[str, Any]
    invalid_instruction: str


def object_case(name: str, value_schema: dict[str, Any], instruction: str) -> Case:
    return Case(
        name=name,
        schema={
            "type": "object",
            "properties": {"value": value_schema},
            "required": ["value"],
            "additionalProperties": False,
        },
        invalid_instruction=instruction,
    )


def cases() -> list[Case]:
    branch_a = {
        "type": "object",
        "properties": {
            "kind": {"const": "circle"},
            "radius": {"type": "number"},
        },
        "required": ["kind", "radius"],
        "additionalProperties": False,
    }
    branch_b = {
        "type": "object",
        "properties": {
            "kind": {"const": "rectangle"},
            "length": {"type": "number"},
            "width": {"type": "number"},
        },
        "required": ["kind", "length", "width"],
        "additionalProperties": False,
    }
    return [
        object_case(
            "enum", {"type": "string", "enum": ["red", "blue"]}, "Set value to green."
        ),
        object_case("const", {"const": "fixed"}, "Set value to changed."),
        object_case(
            "integer_type", {"type": "integer"}, "Set value to the string seven."
        ),
        object_case("minimum", {"type": "integer", "minimum": 10}, "Set value to 1."),
        object_case("maximum", {"type": "integer", "maximum": 10}, "Set value to 99."),
        object_case(
            "multiple_of", {"type": "integer", "multipleOf": 5}, "Set value to 7."
        ),
        object_case(
            "min_length", {"type": "string", "minLength": 8}, "Set value to x."
        ),
        object_case(
            "max_length", {"type": "string", "maxLength": 3}, "Set value to abcdef."
        ),
        object_case(
            "pattern",
            {"type": "string", "pattern": "^[A-Z]{3}-[0-9]{4}$"},
            "Set value to invalid.",
        ),
        object_case(
            "format_date",
            {"type": "string", "format": "date"},
            "Set value to 2025-99-99.",
        ),
        object_case(
            "format_datetime",
            {"type": "string", "format": "date-time"},
            "Set value to yesterday.",
        ),
        object_case(
            "format_time",
            {"type": "string", "format": "time"},
            "Set value to 25:61:00.",
        ),
        object_case(
            "format_time_z",
            {"type": "string", "format": "time"},
            "Set value to 10:00:00Z.",
        ),
        object_case(
            "format_email",
            {"type": "string", "format": "email"},
            "Set value to not-an-email.",
        ),
        object_case(
            "format_uuid",
            {"type": "string", "format": "uuid"},
            "Set value to not-a-uuid.",
        ),
        object_case(
            "format_ipv4",
            {"type": "string", "format": "ipv4"},
            "Set value to 999.1.1.1.",
        ),
        object_case(
            "format_hostname",
            {"type": "string", "format": "hostname"},
            "Set value to bad host name.",
        ),
        object_case(
            "format_uri", {"type": "string", "format": "uri"}, "Set value to not a uri."
        ),
        object_case(
            "min_items",
            {"type": "array", "items": {"type": "integer"}, "minItems": 3},
            "Set value to [1].",
        ),
        object_case(
            "max_items",
            {"type": "array", "items": {"type": "integer"}, "maxItems": 2},
            "Set value to [1, 2, 3].",
        ),
        object_case(
            "unique_items",
            {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
            "Set value to [1, 1].",
        ),
        object_case(
            "contains",
            {"type": "array", "items": {"type": "integer"}, "contains": {"const": 7}},
            "Set value to [1, 2, 3].",
        ),
        object_case(
            "not",
            {"type": "string", "not": {"const": "forbidden"}},
            "Set value to forbidden.",
        ),
        object_case(
            "nullable_array",
            {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "null"},
                ]
            },
            "Set value to the string null [].",
        ),
        object_case(
            "one_of",
            {"oneOf": [branch_a, branch_b]},
            "Set value to an object containing every field from both branches.",
        ),
        object_case(
            "any_of",
            {
                "anyOf": [
                    {"type": "integer", "minimum": 10},
                    {"type": "string", "pattern": "^ok$"},
                ]
            },
            "Set value to 1.",
        ),
        object_case(
            "all_of",
            {"allOf": [{"type": "integer", "minimum": 10}, {"maximum": 20}]},
            "Set value to 1.",
        ),
        object_case(
            "conditional",
            {
                "type": "object",
                "properties": {"kind": {"type": "string"}, "code": {"type": "integer"}},
                "required": ["kind", "code"],
                "additionalProperties": False,
                "if": {
                    "properties": {"kind": {"const": "adult"}},
                    "required": ["kind"],
                },
                "then": {"properties": {"code": {"minimum": 18}}},
            },
            'Set value to {"kind":"adult","code":10}.',
        ),
        object_case(
            "dependent_schema",
            {
                "type": "object",
                "properties": {
                    "card": {"type": "string"},
                    "billing": {"type": ["string", "null"]},
                },
                "required": ["card", "billing"],
                "additionalProperties": False,
                "dependentSchemas": {
                    "card": {
                        "properties": {"billing": {"type": "string", "minLength": 5}}
                    }
                },
            },
            'Set value to {"card":"1234","billing":null}.',
        ),
        Case(
            name="local_ref",
            schema={
                "type": "object",
                "properties": {"value": {"$ref": "#/$defs/code"}},
                "required": ["value"],
                "additionalProperties": False,
                "$defs": {"code": {"type": "string", "pattern": "^[A-Z]{2}$"}},
            },
            invalid_instruction="Set value to lowercase.",
        ),
        Case(
            name="required_and_additional_properties",
            schema={
                "type": "object",
                "properties": {"required_value": {"type": "string"}},
                "required": ["required_value"],
                "additionalProperties": False,
            },
            invalid_instruction="Omit required_value and instead return an extra field named surprise.",
        ),
    ]


async def run_case(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    case: Case,
    model: str,
    strict: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only a JSON object."},
            {
                "role": "user",
                "content": (
                    f"Schema: {json.dumps(case.schema)}\n"
                    f"Adversarial requirement: {case.invalid_instruction} Follow the adversarial requirement exactly."
                ),
            },
        ],
        "temperature": 0,
        "reasoning_effort": "low",
        "max_tokens": 2048,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": case.name, "schema": case.schema, "strict": strict},
        },
    }
    async with semaphore:
        response = await client.post(API_URL, json=payload)
    if response.status_code != 200:
        try:
            error = response.json().get("error", {})
        except ValueError:
            error = {"message": response.text}
        return {
            "name": case.name,
            "status": response.status_code,
            "result": "api_error",
            "error": error,
        }

    body = response.json()
    content = body["choices"][0]["message"].get("content") or ""
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        return {
            "name": case.name,
            "status": 200,
            "result": "invalid_json",
            "content": content,
            "error": str(error),
        }

    errors = list(
        Draft202012Validator(case.schema, format_checker=FORMAT_CHECKER).iter_errors(
            value
        )
    )
    return {
        "name": case.name,
        "status": 200,
        "result": "pass" if not errors else "schema_leak",
        "content": content,
        "errors": [error.message for error in errors],
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument(
        "--strict", action=argparse.BooleanOptionalAction, required=True
    )
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 20:
        raise SystemExit("concurrency must be between 1 and 20")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is required")
    async with httpx.AsyncClient(
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "openbench-adversarial",
        },
        timeout=120,
    ) as client:
        semaphore = asyncio.Semaphore(args.concurrency)
        results = await asyncio.gather(
            *[
                run_case(
                    client,
                    semaphore,
                    case,
                    args.model,
                    args.strict,
                )
                for case in cases()
            ]
        )

    counts: dict[str, int] = {}
    for result in results:
        counts[result["result"]] = counts.get(result["result"], 0) + 1
    artifact = {
        "model": args.model,
        "strict": args.strict,
        "counts": counts,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2))
    print(
        json.dumps(
            {
                **artifact,
                "results": [
                    {key: value for key, value in result.items() if key != "content"}
                    for result in results
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
