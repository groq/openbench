from typing import Dict, List, Tuple
from datasets import load_dataset  # type: ignore[import-untyped]
from inspect_ai.dataset import Dataset, Sample, MemoryDataset
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    ChatMessageAssistant,
    ChatMessageTool,
)

JSONSCHEMABENCH_SYSTEM_PROMPT = (
    "You need to generate a JSON object that matches the schema below."
)

FEWSHOT_EXAMPLES: Dict[Tuple[str, ...], List[Tuple[str, str]]] = {
    ("Snowplow",): [
        (
            '{\n    "additionalProperties": false,\n    "description": "Schema for a JSON Paths file for loading Redshift from JSON or Avro, http://docs.aws.amazon.com/redshift/latest/dg/copy-parameters-data-format.html#copy-json-jsonpaths",\n    "properties": {\n        "jsonpaths": {\n            "items": {\n                "type": "string"\n            },\n            "minItems": 1,\n            "type": "array"\n        }\n    },\n    "required": [\n        "jsonpaths"\n    ],\n    "self": {\n        "format": "jsonschema",\n        "name": "jsonpaths_file",\n        "vendor": "com.amazon.aws.redshift",\n        "version": "1-0-0"\n    },\n    "type": "object"\n}',
            '{"jsonpaths": ["$.user.id", "$.user.name", "$.user.address.street"]}',
        ),
        (
            '{\n    "additionalProperties": false,\n    "description": "Schema for a Google Analytics enhanced e-commerce product impression custom metric entity",\n    "properties": {\n        "customMetricIndex": {\n            "maximum": 200,\n            "minimum": 1,\n            "type": "integer"\n        },\n        "listIndex": {\n            "maximum": 200,\n            "minimum": 1,\n            "type": "integer"\n        },\n        "productIndex": {\n            "maximum": 200,\n            "minimum": 1,\n            "type": "integer"\n        },\n        "value": {\n            "type": [\n                "integer",\n                "null"\n            ]\n        }\n    },\n    "self": {\n        "format": "jsonschema",\n        "name": "product_impression_custom_metric",\n        "vendor": "com.google.analytics.measurement-protocol",\n        "version": "1-0-0"\n    },\n    "type": "object"\n}',
            '{"customMetricIndex": 120, "listIndex": 45, "productIndex": 10, "value": 300}',
        ),
    ],
    ("Github_easy", "Github_hard", "Github_medium", "Github_trivial", "Github_ultra"): [
        (
            '{\n    "$schema": "http://json-schema.org/draft-04/schema#",\n    "definitions": {\n        "address1": {"type": "string"},\n        "address2": {"type": "string"},\n        "city": {"type": "string"},\n        "country": {"type": "string"},\n        "postalCode": {"type": "string"},\n        "state": {"type": "string"}\n    },\n    "description": "A simple address schema",\n    "properties": {\n        "address1": {"$ref": "#/definitions/address1"},\n        "address2": {"$ref": "#/definitions/address2"},\n        "city": {"$ref": "#/definitions/city"},\n        "country": {"$ref": "#/definitions/country"},\n        "postalCode": {"$ref": "#/definitions/postalCode"},\n        "state": {"$ref": "#/definitions/state"}\n    },\n    "type": "object"\n}',
            '{"address1": "123 Main Street", "address2": "Apt 4B", "city": "Seattle", "country": "USA", "postalCode": "98101", "state": "WA"}',
        ),
        (
            '{\n    "$schema": "http://json-schema.org/draft-06/schema#",\n    "definitions": {\n        "ElementType": {\n            "enum": ["component", "directive"],\n            "type": "string"\n        },\n        "SelectorChange": {\n            "properties": {\n                "remove": {\n                    "description": "Remove directive/component",\n                    "type": "boolean"\n                },\n                "replaceWith": {\n                    "description": "Replace original selector with new one",\n                    "type": "string"\n                },\n                "selector": {\n                    "description": "Original selector to apply change to",\n                    "type": "string"\n                },\n                "type": {\n                    "$ref": "#/definitions/ElementType",\n                    "description": "Type of selector the change applies to - either component or directive"\n                }\n            },\n            "required": ["selector", "type"],\n            "type": "object"\n        }\n    },\n    "properties": {\n        "changes": {\n            "description": "An array of changes to component/directive selectors",\n            "items": {\n                "$ref": "#/definitions/SelectorChange"\n            },\n            "type": "array"\n        }\n    },\n    "required": ["changes"],\n    "type": "object"\n}',
            '{\n  "changes": [\n    {\n      "selector": "app-root",\n      "type": "component",\n      "remove": false,\n      "replaceWith": "new-root"\n    },\n    {\n      "selector": "my-directive",\n      "type": "directive",\n      "remove": true,\n      "replaceWith": "new-directive"\n    }\n  ]\n}',
        ),
    ],
    ("Glaiveai2K",): [
        (
            '{"properties": {"username": {"description": "The user\'s username", "type": "string"}, "email": {"description": "The user\'s email address", "type": "string"}, "age": {"description": "The user\'s age", "type": "integer"}, "is_active": {"description": "Whether the user is active", "type": "boolean"}}, "required": ["username", "email"], "type": "object"}',
            '{"username": "johndoe", "email": "john@example.com", "age": 30, "is_active": true} ',
        ),
        (
            '{"properties": {"product_id": {"description": "The ID of the product", "type": "string"}, "rating": {"description": "The rating given by the user", "type": "integer"}, "comments": {"description": "Additional comments about the product", "type": "string"}}, "required": ["product_id", "rating"], "type": "object"}',
            '{"product_id": "12345", "rating": 5, "comments": "Excellent product! Highly recommend."} ',
        ),
    ],
    ("JsonSchemaStore",): [
        (
            '{\n  "$id": "https://json.schemastore.org/minecraft-trim-pattern.json",\n  "$schema": "http://json-schema.org/draft-07/schema#",\n  "description": "A trim pattern for a Minecraft data pack config schema",\n  "properties": {\n    "asset_id": {\n      "type": "string"\n    },\n    "description": {\n      "properties": {\n        "color": {\n          "type": "string"\n        },\n        "translate": {\n          "type": "string"\n        }\n      },\n      "required": ["translate"],\n      "type": "object"\n    },\n    "template_item": {\n      "type": "string"\n    }\n  },\n  "required": ["asset_id", "description", "template_item"],\n  "title": "Minecraft Data Pack Trim Pattern",\n  "type": "object"\n}',
            '{\n  "asset_id": "minecraft:trim_pattern",\n  "description": {\n    "color": "#FFAA00",\n    "translate": "trim_pattern.description"\n  },\n  "template_item": "minecraft:template_item"\n}',
        ),
        (
            '{\n  "$comment": "https://minecraft.fandom.com/wiki/Data_Pack",\n  "$id": "https://json.schemastore.org/minecraft-damage-type.json",\n  "$schema": "http://json-schema.org/draft-07/schema#",\n  "description": "A damage type\'s for a Minecraft data pack config schema",\n  "properties": {\n    "death_message_type": {\n      "enum": ["default", "fall_variants", "intentional_game_design"],\n      "type": "string"\n    },\n    "effects": {\n      "enum": ["hurt", "thorns", "drowning", "burning", "poking", "freezing"],\n      "type": "string"\n    },\n    "exhaustion": {\n      "type": "number"\n    },\n    "message_id": {\n      "type": "string"\n    },\n    "scaling": {\n      "enum": ["never", "always", "when_caused_by_living_non_player"],\n      "type": "string"\n    }\n  },\n  "required": ["message_id", "scaling", "exhaustion"],\n  "title": "Minecraft Data Pack Damage Type",\n  "type": "object"\n}',
            '{\n  "message_id": "minecraft:damage.message",\n  "scaling": "always",\n  "exhaustion": 0.3,\n  "death_message_type": "default",\n  "effects": "hurt"\n}',
        ),
    ],
    ("Kubernetes",): [
        (
            '{\n  "description": "A topology selector requirement is a selector that matches given label. This is an alpha feature and may change in the future.",\n  "properties": {\n    "key": {\n      "description": "The label key that the selector applies to.",\n      "type": ["string", "null"]\n    },\n    "values": {\n      "description": "An array of string values. One value must match the label to be selected. Each entry in Values is ORed.",\n      "items": {\n        "type": ["string", "null"]\n      },\n      "type": ["array", "null"]\n    }\n  },\n  "required": ["key", "values"],\n  "type": "object"\n}',
            '{\n  "key": "region",\n  "values": ["us-west-1", "us-east-1"]\n}',
        ),
        (
            '{\n  "description": "HostAlias holds the mapping between IP and hostnames that will be injected as an entry in the pod\'s hosts file.",\n  "properties": {\n    "hostnames": {\n      "description": "Hostnames for the above IP address.",\n      "items": {\n        "type": ["string", "null"]\n      },\n      "type": ["array", "null"]\n    },\n    "ip": {\n      "description": "IP address of the host file entry.",\n      "type": ["string", "null"]\n    }\n  },\n  "type": "object"\n}',
            '{\n  "ip": "192.168.1.1",\n  "hostnames": ["example.com", "test.com"]\n}',
        ),
    ],
    ("WashingtonPost",): [
        (
            '{\n  "additionalProperties": false,\n  "description": "Models a auxiliary used in targeting a piece of content.",\n  "properties": {\n    "_id": {\n      "description": "The unique identifier for this auxiliary.",\n      "type": "string"\n    },\n    "name": {\n      "description": "The general name for this auxiliary.",\n      "type": "string"\n    },\n    "uid": {\n      "description": "A short identifier for this auxiliary. Usually used in cases where a long form id cannot work.",\n      "type": "string"\n    }\n  },\n  "required": ["_id", "uid"],\n  "title": "Auxiliary",\n  "type": "object"\n}',
            '{\n  "_id": "12345",\n  "uid": "aux123",\n  "name": "Sample Auxiliary"\n}',
        ),
        (
            '{\n  "additionalProperties": {},\n  "definitions": {\n    "trait_additional_properties_json": {\n      "$schema": "http://json-schema.org/draft-04/schema#",\n      "additionalProperties": {},\n      "description": "A grab-bag object for non-validatable data.",\n      "title": "Has additional properties",\n      "type": "object"\n    }\n  },\n  "description": "Comment configuration data",\n  "properties": {\n    "additional_properties": {\n      "$ref": "#/definitions/trait_additional_properties_json"\n    },\n    "allow_comments": {\n      "description": "If false, commenting is disabled on this content.",\n      "type": "boolean"\n    },\n    "comments_period": {\n      "description": "How long (in days) after publish date until comments are closed.",\n      "type": "integer"\n    },\n    "display_comments": {\n      "description": "If false, do not render comments on this content.",\n      "type": "boolean"\n    },\n    "moderation_required": {\n      "description": "If true, comments must be moderator-approved before being displayed.",\n      "type": "boolean"\n    }\n  },\n  "title": "Comments",\n  "type": "object"\n}',
            '{\n  "allow_comments": true,\n  "comments_period": 30,\n  "display_comments": true,\n  "moderation_required": false,\n  "additional_properties": {}\n}',
        ),
    ],
    ("default",): [],
}


def _find_examples_for_subset(subset: str | None) -> List[Tuple[str, str]]:
    """Find few-shot examples for a subset."""
    for key, examples in FEWSHOT_EXAMPLES.items():
        if subset in key:
            return examples
    return FEWSHOT_EXAMPLES[("default",)]


def _get_few_shot_examples(subset: str | None, num_shots: int) -> List[Tuple[str, str]]:
    """Get first N few-shot examples for a subset."""
    examples = _find_examples_for_subset(subset)
    if num_shots > len(examples):
        raise ValueError(
            f"Not enough {subset} examples to prompt with {num_shots} shots"
        )
    return examples[:num_shots]


def _build_messages(
    schema: str, examples: List[Tuple[str, str]]
) -> List[ChatMessageSystem | ChatMessageUser | ChatMessageAssistant | ChatMessageTool]:
    """Build message list with few-shot examples."""
    messages: List[
        ChatMessageSystem | ChatMessageUser | ChatMessageAssistant | ChatMessageTool
    ] = [ChatMessageSystem(content=JSONSCHEMABENCH_SYSTEM_PROMPT)]

    for schema_str, json_str in examples:
        messages.append(ChatMessageUser(content=schema_str))
        messages.append(ChatMessageAssistant(content=json_str))

    messages.append(ChatMessageUser(content=schema))
    return messages


def record_to_sample(
    record: dict, num_shots: int = 0, subset: str | None = None
) -> Sample:
    """Convert HuggingFace dataset record to Inspect Sample with few-shot prompting."""
    schema = record["json_schema"]
    unique_id = record["unique_id"]

    # Build few-shot prompt if requested
    examples = _get_few_shot_examples(subset, num_shots)
    messages = _build_messages(schema, examples)

    return Sample(
        input=messages,
        target="",
        metadata={
            "schema": schema,
            "unique_id": unique_id,
            "num_shots": num_shots,
        },
    )


def get_dataset(
    subset: str | None = None, split: str = "all", num_shots: int = 0
) -> Dataset:
    """Load JSONSchemaBench dataset from HuggingFace with few-shot examples.

    Args:
        subset: Dataset subset (e.g., "Github_easy", "Kubernetes", "Snowplow")
        split: Dataset split ("test", "val", "train", or "all")
        num_shots: Number of few-shot examples (0 for zero-shot, paper used 2)
    """
    split_param = {
        "test": "test",
        "val": "val",
        "train": "train",
        "all": "train[:]+val[:]+test[:]",
    }
    config = subset if subset else "default"
    name = f"jsonschemabench_{config}_{split}_{num_shots}shot"
    dataset = load_dataset(
        "epfl-dlab/JSONSchemaBench", config, split=split_param[split]
    )
    samples = [
        record_to_sample(record, num_shots=num_shots, subset=subset)
        for record in dataset
    ]
    return MemoryDataset(samples=samples, name=name)
