"""Yutori Navigator provider implementation.

OpenAI-compatible wrapper for the Yutori Navigator Chat Completions API.

Environment variables:
  - YUTORI_API_KEY: Yutori API key (required)
  - YUTORI_BASE_URL: Override the default base URL (defaults to
    https://api.yutori.com/v1)

Example model strings: ``yutori/n1.5-latest``, ``yutori/n1.5-20260428``.

Website: https://yutori.com
API Docs: https://docs.yutori.com
"""

import json
import os
from typing import Any

from inspect_ai.model import GenerateConfig
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI


class YutoriAPI(OpenAICompatibleAPI):
    """Yutori Navigator provider - OpenAI-compatible Chat Completions."""

    DEFAULT_BASE_URL = "https://api.yutori.com/v1"

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        tool_set: str | None = None,
        disable_tools: list[str] | str | None = None,
        json_schema: dict[str, Any] | str | None = None,
        max_completion_tokens: int | None = None,
        **model_args: Any,
    ) -> None:
        model_name_clean = model_name.replace("yutori/", "", 1)
        base_url = base_url or os.environ.get("YUTORI_BASE_URL", self.DEFAULT_BASE_URL)
        api_key = api_key or os.environ.get("YUTORI_API_KEY")

        if not api_key:
            raise ValueError(
                "Yutori API key not found. Set the YUTORI_API_KEY environment variable."
            )

        # Navigator-specific request fields ride along via extra_body. CLI -M
        # flags arrive as strings, so coerce list/dict forms here.
        extras: dict[str, Any] = {}
        if tool_set is not None:
            extras["tool_set"] = tool_set
        if disable_tools is not None:
            if isinstance(disable_tools, str):
                disable_tools = [
                    t.strip() for t in disable_tools.split(",") if t.strip()
                ]
            extras["disable_tools"] = disable_tools
        if json_schema is not None:
            if isinstance(json_schema, str):
                try:
                    json_schema = json.loads(json_schema)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON for -M json_schema=...: {e}") from e
            extras["json_schema"] = json_schema
        self._extra_body: dict[str, Any] = extras
        self._max_completion_tokens = max_completion_tokens

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="yutori",
            service_base_url=self.DEFAULT_BASE_URL,
            **model_args,
        )

    def completion_params(self, config: GenerateConfig, tools: bool) -> dict[str, Any]:
        params = super().completion_params(config=config, tools=tools)

        # Navigator rejects max_tokens; remap so openbench --max-tokens works.
        # An explicit -M max_completion_tokens=... wins over the remapped value.
        if "max_tokens" in params:
            params.setdefault("max_completion_tokens", params.pop("max_tokens"))
        if self._max_completion_tokens is not None:
            params["max_completion_tokens"] = self._max_completion_tokens

        # User-supplied extra_body (via GenerateConfig.extra_body) wins over
        # provider-default Navigator extras from -M flags.
        if self._extra_body:
            existing = params.get("extra_body") or {}
            params["extra_body"] = {**self._extra_body, **existing}

        return params

    def service_model_name(self) -> str:
        return self.model_name
