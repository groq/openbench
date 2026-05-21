"""Unit tests for the Yutori Navigator provider."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from inspect_ai.model import GenerateConfig

from openbench.model._providers.yutori import YutoriAPI
from openbench.provider_config import PROVIDER_CONFIGS, ProviderType


@pytest.fixture
def patched_client():
    """Patch the AsyncOpenAI client so __init__ doesn't need network/credentials."""
    with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
        yield


def _make_api(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> YutoriAPI:
    """Build a YutoriAPI with sane defaults for tests."""
    monkeypatch.delenv("YUTORI_BASE_URL", raising=False)
    kwargs: dict[str, Any] = {
        "model_name": "yutori/n1.5-latest",
        "api_key": "test-key",
    }
    kwargs.update(overrides)
    return YutoriAPI(**kwargs)


class TestYutoriProviderInit:
    """Construction-time behavior: prefix stripping, env resolution, errors."""

    def test_strips_yutori_prefix(self, patched_client, monkeypatch):
        api = _make_api(monkeypatch)
        assert api.service_model_name() == "n1.5-latest"

    def test_reads_api_key_from_env(self, patched_client, monkeypatch):
        monkeypatch.setenv("YUTORI_API_KEY", "env-key")
        api = YutoriAPI(model_name="yutori/n1.5-latest")
        assert api.api_key == "env-key"

    def test_missing_api_key_raises(self, patched_client, monkeypatch):
        monkeypatch.delenv("YUTORI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="YUTORI_API_KEY"):
            YutoriAPI(model_name="yutori/n1.5-latest")

    def test_default_base_url(self, patched_client, monkeypatch):
        api = _make_api(monkeypatch)
        assert api.base_url == "https://api.yutori.com/v1"

    def test_base_url_env_override(self, patched_client, monkeypatch):
        monkeypatch.setenv("YUTORI_BASE_URL", "https://staging.yutori.com/v1")
        api = YutoriAPI(model_name="yutori/n1.5-latest", api_key="test-key")
        assert api.base_url == "https://staging.yutori.com/v1"

    def test_explicit_base_url_wins_over_env(self, patched_client, monkeypatch):
        monkeypatch.setenv("YUTORI_BASE_URL", "https://staging.yutori.com/v1")
        api = YutoriAPI(
            model_name="yutori/n1.5-latest",
            api_key="test-key",
            base_url="https://override.yutori.com/v1",
        )
        assert api.base_url == "https://override.yutori.com/v1"


class TestYutoriCompletionParams:
    """completion_params() must remap max_tokens and inject Navigator extras."""

    def test_max_tokens_remapped(self, patched_client, monkeypatch):
        api = _make_api(monkeypatch)
        params = api.completion_params(GenerateConfig(max_tokens=512), tools=False)
        assert "max_tokens" not in params
        assert params["max_completion_tokens"] == 512

    def test_no_remap_when_max_tokens_unset(self, patched_client, monkeypatch):
        api = _make_api(monkeypatch)
        params = api.completion_params(GenerateConfig(), tools=False)
        assert "max_tokens" not in params
        # No spurious max_completion_tokens injection either.
        assert "max_completion_tokens" not in params

    def test_tool_set_in_extra_body(self, patched_client, monkeypatch):
        api = _make_api(monkeypatch, tool_set="browser_tools_expanded-20260403")
        params = api.completion_params(GenerateConfig(), tools=False)
        assert params["extra_body"]["tool_set"] == "browser_tools_expanded-20260403"

    def test_disable_tools_string_parsed_to_list(self, patched_client, monkeypatch):
        api = _make_api(monkeypatch, disable_tools="hold_key, drag ,  ")
        params = api.completion_params(GenerateConfig(), tools=False)
        assert params["extra_body"]["disable_tools"] == ["hold_key", "drag"]

    def test_disable_tools_list_passthrough(self, patched_client, monkeypatch):
        api = _make_api(monkeypatch, disable_tools=["hold_key", "drag"])
        params = api.completion_params(GenerateConfig(), tools=False)
        assert params["extra_body"]["disable_tools"] == ["hold_key", "drag"]

    def test_json_schema_string_parsed(self, patched_client, monkeypatch):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        api = _make_api(monkeypatch, json_schema=json.dumps(schema))
        params = api.completion_params(GenerateConfig(), tools=False)
        assert params["extra_body"]["json_schema"] == schema

    def test_json_schema_dict_passthrough(self, patched_client, monkeypatch):
        schema = {"type": "object"}
        api = _make_api(monkeypatch, json_schema=schema)
        params = api.completion_params(GenerateConfig(), tools=False)
        assert params["extra_body"]["json_schema"] == schema

    def test_no_extra_body_when_no_extras(self, patched_client, monkeypatch):
        api = _make_api(monkeypatch)
        params = api.completion_params(GenerateConfig(), tools=False)
        # Either the key is absent or the value is falsy — both acceptable.
        assert not params.get("extra_body")

    def test_user_extra_body_wins_over_provider_defaults(
        self, patched_client, monkeypatch
    ):
        """A caller's GenerateConfig.extra_body overrides Navigator defaults on key collision."""
        api = _make_api(monkeypatch, tool_set="provider-default")
        # Simulate the base class returning an existing extra_body — happens when
        # a caller sets GenerateConfig.extra_body explicitly.
        original = api.completion_params

        def with_existing(config, tools):
            params = original(GenerateConfig(), tools=tools)
            return params

        # Monkeypatch the parent's completion_params to inject extra_body.
        from inspect_ai.model._providers import openai_compatible

        def fake_parent(self, config, tools):
            return {"extra_body": {"tool_set": "user-wins", "user_only": 1}}

        monkeypatch.setattr(
            openai_compatible.OpenAICompatibleAPI,
            "completion_params",
            fake_parent,
        )
        params = api.completion_params(GenerateConfig(), tools=False)
        assert params["extra_body"]["tool_set"] == "user-wins"
        assert params["extra_body"]["user_only"] == 1

    def test_explicit_max_completion_tokens_wins_over_remap(
        self, patched_client, monkeypatch
    ):
        """If both max_tokens and max_completion_tokens are set, the explicit one wins."""
        from inspect_ai.model._providers import openai_compatible

        def fake_parent(self, config, tools):
            return {"max_tokens": 100, "max_completion_tokens": 500}

        monkeypatch.setattr(
            openai_compatible.OpenAICompatibleAPI,
            "completion_params",
            fake_parent,
        )
        api = _make_api(monkeypatch)
        params = api.completion_params(GenerateConfig(), tools=False)
        assert "max_tokens" not in params
        assert params["max_completion_tokens"] == 500

    def test_invalid_json_schema_raises_actionable_error(
        self, patched_client, monkeypatch
    ):
        """A malformed -M json_schema=... fails loud-and-early with provider context."""
        with pytest.raises(ValueError, match="json_schema"):
            _make_api(monkeypatch, json_schema="{not valid json")

    def test_max_completion_tokens_kwarg_not_passed_to_async_openai(
        self, patched_client, monkeypatch
    ):
        """Regression: -M max_completion_tokens=N must not leak into AsyncOpenAI kwargs."""
        api = _make_api(monkeypatch, max_completion_tokens=500)
        assert "max_completion_tokens" not in api.model_args

    def test_explicit_max_completion_tokens_kwarg_wins_over_remap(
        self, patched_client, monkeypatch
    ):
        """An explicit -M max_completion_tokens=N overrides the --max-tokens remap."""
        from inspect_ai.model._providers import openai_compatible

        def fake_parent(self, config, tools):
            return {"max_tokens": 100}

        monkeypatch.setattr(
            openai_compatible.OpenAICompatibleAPI,
            "completion_params",
            fake_parent,
        )
        api = _make_api(monkeypatch, max_completion_tokens=500)
        params = api.completion_params(GenerateConfig(), tools=False)
        assert "max_tokens" not in params
        assert params["max_completion_tokens"] == 500


class TestYutoriProviderConfig:
    """The provider must be wired into the centralized PROVIDER_CONFIGS table."""

    def test_provider_config_registered(self):
        config = PROVIDER_CONFIGS[ProviderType.YUTORI]
        assert config.name == "yutori"
        assert config.api_key_env == "YUTORI_API_KEY"
        assert config.base_url_env == "YUTORI_BASE_URL"
        assert config.base_url == "https://api.yutori.com/v1"
        assert config.supports_vision is True
        assert config.supports_function_calling is True

    def test_env_vars_listed(self):
        config = PROVIDER_CONFIGS[ProviderType.YUTORI]
        env_vars = config.get_all_env_vars()
        assert "YUTORI_API_KEY" in env_vars
        assert "YUTORI_BASE_URL" in env_vars
