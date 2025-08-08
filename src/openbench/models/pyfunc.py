from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import os
from typing import Any, Callable

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    modelapi,
)


def _import_module_from_path(path: str):
    """Import a Python module from a filesystem path."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(abs_path):
        raise ImportError(f"pyfunc: file path does not exist: {path}")
    mod_name = f"_pyfunc_{abs(hash(abs_path))}"
    spec = importlib.util.spec_from_file_location(mod_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"pyfunc: unable to create spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_callable(spec: str) -> Callable[..., Any]:
    """
    Resolve a function from a spec of the form:
      - 'module.submodule:function'
      - '/abs/or/rel/path.py:function'
      - '~/path/to/file.py:function'

    Returns a callable. Raises on failure.
    """
    # Strip any accidental query, e.g., 'module:function?foo=bar'
    spec = spec.split("?", 1)[0]

    if ":" not in spec:
        raise ValueError(
            "pyfunc: model name must be '<module_or_file>:<function>', "
            "e.g., 'my_pkg.my_mod:gpt_5_pro_mode' or '~/proj/pro_mode.py:pro_mode'"
        )
    module_part, func_name = spec.split(":", 1)
    module_part_expanded = os.path.expanduser(module_part)

    # Heuristic: treat as path if it looks like a file path or ends with .py
    is_path = (
        module_part_expanded.endswith(".py")
        or module_part_expanded.startswith(".")
        or module_part_expanded.startswith("/")
        or os.path.sep in module_part_expanded
        or os.path.exists(module_part_expanded)
    )

    if is_path:
        module = _import_module_from_path(module_part_expanded)
    else:
        module = importlib.import_module(module_part_expanded)

    try:
        func = getattr(module, func_name)
    except AttributeError as e:
        raise AttributeError(
            f"pyfunc: function '{func_name}' not found in '{module_part}'"
        ) from e

    if not callable(func):
        raise TypeError(f"pyfunc: resolved object '{func_name}' is not callable")
    return func


def _extract_last_user_text(messages: list[dict[str, str]]) -> str:
    """Return the last user message text, else concatenate all texts."""
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    # Fallback: join everything
    return "\n\n".join(str(m.get("content", "")) for m in messages)


@modelapi("pyfunc")
class PythonFunctionModel(ModelAPI):
    """
    A custom Inspect AI provider that lets you use a Python function as if it were a chat model.

    Model string format (observe the required provider/model slash):
      --model pyfunc/<module_or_path>:<function>

    Examples:
      --model pyfunc/my_pkg.my_mod:gpt_5_pro_mode
      --model pyfunc/~/repo/pro_mode.py:pro_mode
      --model pyfunc/./local_file.py:run

    Pass extra kwargs to your function via `-M key=value` (see `bench eval -h`).
    Your function may use one of these signatures:
      - f(messages: List[Dict[str, str]], **kwargs) -> str | dict
      - f(prompt: str, **kwargs) -> str | dict
      - f(text: str, **kwargs) -> str | dict
      - f(input: str, **kwargs) -> str | dict

    If a dict is returned and contains 'final', that value is used as the completion.
    Otherwise the result is stringified.
    """

    def __init__(self, model_name: str, **model_args: Any):
        # model_name is the portion after 'pyfunc/' (e.g. 'pkg.mod:func' or '/path/file.py:func')
        self._spec = model_name
        self._func = _resolve_callable(model_name)
        # Arbitrary kwargs forwarded to the function
        self._func_kwargs = dict(model_args or {})
        # If CLI passed -M model=..., it was remapped to inner_model; map back for the callable.
        if "inner_model" in self._func_kwargs and "model" not in self._func_kwargs:
            self._func_kwargs["model"] = self._func_kwargs.pop("inner_model")

    @property
    def model_name(self) -> str:
        # Show the underlying python function spec (module:func)
        # Inspect AI reads this during display / task scheduling.
        return self._spec
    
    @property
    def base_url(self) -> str:
        # only used for logging/metadata by Inspect AI
        return getattr(self, "_base_url", "pyfunc://local")

    async def generate(  # type: ignore[override]
        self,
        input: list[ChatMessage],
        tools: Any = None,
        tool_choice: Any = None,
        config: GenerateConfig | None = None,
        **_: Any,
    ) -> ModelOutput:
        """
        Generate a completion by delegating to a user-provided Python function.
        The signature is intentionally liberal to tolerate Inspect AI API changes.
        """
        # -- Coerce Inspect ChatMessage objects to simple {role, content} dicts --
        def _msg_text(m: ChatMessage) -> str:
            # Prefer .text if present (pure-text message)
            if hasattr(m, "text") and m.text is not None:
                return str(m.text)
            # Fallback to .content if it's already a string
            c = getattr(m, "content", None)
            if isinstance(c, str):
                return c
            # Last resort: stringify the content
            return "" if c is None else str(c)

        msgs: list[dict[str, str]] = [
            {"role": getattr(m, "role", "user"), "content": _msg_text(m)} for m in input
        ]
        prompt = _extract_last_user_text(msgs)

        # Build kwargs for the user function
        kwargs = dict(self._func_kwargs)

        # If the function accepts temperature/top_p/max_tokens, forward them (when provided)
        sig = inspect.signature(self._func)
        params = sig.parameters
        cfg = config or GenerateConfig()

        if "temperature" in params and cfg.temperature is not None and "temperature" not in kwargs:
            kwargs["temperature"] = cfg.temperature
        if "top_p" in params and cfg.top_p is not None and "top_p" not in kwargs:
            kwargs["top_p"] = cfg.top_p

        # Map max_tokens to what the function expects
        if "max_completion_tokens" in params and cfg.max_tokens is not None and "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = cfg.max_tokens
        elif "max_tokens" in params and cfg.max_tokens is not None and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = cfg.max_tokens

        # Convenience: auto-inject API keys if parameters exist and not provided
        if "groq_api_key" in params and "groq_api_key" not in kwargs:
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                kwargs["groq_api_key"] = groq_key
        if "openai_api_key" in params and "openai_api_key" not in kwargs:
            openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
            if openai_key:
                kwargs["openai_api_key"] = openai_key

        # Drop unknown kwargs if function doesn't accept **kwargs
        accepts_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
        if not accepts_var_kw:
            kwargs = {k: v for k, v in kwargs.items() if k in params}

        # Decide how to call the user function
        async def _call_async():
            if "messages" in params:
                return await self._func(messages=msgs, **kwargs)  # type: ignore[misc]
            if "prompt" in params:
                return await self._func(prompt=prompt, **kwargs)  # type: ignore[misc]
            if "text" in params:
                return await self._func(text=prompt, **kwargs)  # type: ignore[misc]
            if "input" in params:
                return await self._func(input=prompt, **kwargs)  # type: ignore[misc]
            # Positional fallback
            return await self._func(prompt, **kwargs)  # type: ignore[misc]

        def _call_sync():
            if "messages" in params:
                return self._func(messages=msgs, **kwargs)
            if "prompt" in params:
                return self._func(prompt=prompt, **kwargs)
            if "text" in params:
                return self._func(text=prompt, **kwargs)
            if "input" in params:
                return self._func(input=prompt, **kwargs)
            # Positional fallback
            if len(params) > 0:
                return self._func(prompt, **kwargs)
            return self._func(**kwargs)

        # Execute and normalize output, surfacing any root-cause error verbosely
        try:
            if inspect.iscoroutinefunction(self._func):
                result = await _call_async()
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, _call_sync)
        except Exception as e:
            # Provide a concise but useful wrapper so ExceptionGroup doesn't hide the cause
            import traceback
            tb = "".join(traceback.format_exception_only(type(e), e)).strip()
            raise RuntimeError(
                f"pyfunc '{self._spec}' raised an error: {tb}. "
                f"If this comes from the underlying model/API call, verify your API keys and -M args."
            ) from e

        # Normalize to text
        if isinstance(result, dict) and "final" in result and isinstance(result["final"], str):
            content = result["final"]
        else:
            content = "" if result is None else str(result)

        # Identify as pyfunc with the underlying spec for logs
        model_id = f"pyfunc/{self._spec}"
        return ModelOutput.from_content(model=model_id, content=content)