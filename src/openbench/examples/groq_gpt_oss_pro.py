from __future__ import annotations

from typing import List, Dict, Any, Optional
import concurrent.futures as cf
import time
import os
import random
import re

# Lazy import with a helpful error if the groq client isn't installed
try:
    from groq import Groq  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "The 'groq' Python package is required for gpt_5_pro_mode(). "
        "Install it with: pip install groq"
    ) from e

try:  # GroqError may not exist in all client versions; fall back to Exception
    from groq import GroqError  # type: ignore
except Exception:  # pragma: no cover
    class GroqError(Exception):  # type: ignore
        pass


DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_MAX_COMPLETION_TOKENS = 30000


def _one_completion(
    client: Groq,
    messages: List[Dict[str, str]],
    *,
    model: str,
    temperature: float,
    max_completion_tokens: int,
    top_p: float = 1.0,
) -> str:
    """Single non-streamed completion with robust retry/backoff and jitter."""
    max_attempts = 6
    base_delay = 0.5  # seconds
    delay = base_delay

    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_completion_tokens,
                stream=False,
            )
            content = resp.choices[0].message.content
            return content or ""
        except Exception as e:  # Catch RateLimitError, API errors, transient network issues
            if attempt == max_attempts - 1:
                raise
            # If server suggests a wait (e.g., "Please try again in 247.83ms"), respect it.
            msg = str(e)
            sleep_s = delay
            m_ms = re.search(r"Please try again in\s*([\d\.]+)\s*ms", msg, re.IGNORECASE)
            m_s = re.search(r"Please try again in\s*([\d\.]+)\s*s", msg, re.IGNORECASE)
            if m_ms:
                try:
                    sleep_s = max(sleep_s, float(m_ms.group(1)) / 1000.0)
                except Exception:
                    pass
            elif m_s:
                try:
                    sleep_s = max(sleep_s, float(m_s.group(1)))
                except Exception:
                    pass

            # Add +/-20% jitter to avoid thundering herd
            jitter = 1.0 + random.uniform(-0.2, 0.2)
            time.sleep(sleep_s * jitter)
            delay = min(delay * 2, 30.0)  # exponential backoff with cap


def _build_synthesis_messages(candidates: List[str], question: str) -> List[Dict[str, str]]:
    """
    Build synthesis messages to merge candidate answers into a single final answer.
    The system prompt asks for decisive, clean synthesis with no mention of candidates.
    """
    numbered = "\n\n".join(
        f"<candidate {i+1}>\n{txt}\n</candidate {i+1}>" for i, txt in enumerate(candidates, start=1)
    )
    system = (
        "You are an expert editor. Synthesize ONE best answer from the candidate "
        "answers provided, merging strengths, correcting errors, and removing repetition. "
        "Do not mention the candidates or the synthesis process. Be decisive and clear."
    )
    user = (
        f"You are given {len(candidates)} candidate answers delimited by tags.\n\n"
        f"{numbered}\n\nReturn the single best final answer to the question: `{question}`"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def gpt_5_pro_mode(
    messages: List[Dict[str, str]],
    *,
    n_runs: int = 8,
    model: str = DEFAULT_MODEL,
    temperature_candidates: float = 0.9,
    temperature_synthesis: float = 0.2,
    top_p: float = 1.0,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    groq_api_key: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> str:
    """
    Fan out n_runs parallel generations (T≈0.9) then synthesize a single answer (T≈0.2).
    Arguments match common knobs and can be overridden via the pyfunc provider with -M.

    Parameters
    ----------
    messages : List[Dict[str, str]]
        Chat messages in OpenAI format: [{'role': 'system'|'user'|'assistant', 'content': '...'}, ...]
    n_runs : int
        Number of parallel candidate generations.
    model : str
        Groq model identifier (default: openai/gpt-oss-120b).
    temperature_candidates : float
        Temperature for candidate generations.
    temperature_synthesis : float
        Temperature for the synthesis pass.
    top_p : float
        Nucleus sampling parameter; usually 1.0 for GPT‑OSS.
    max_completion_tokens : int
        Max tokens for both candidate and synthesis generations.
    groq_api_key : Optional[str]
        If provided, used explicitly; otherwise GROQ_API_KEY env var is used.

    Returns
    -------
    str
        The synthesized final answer.

    Notes
    -----
    - Both candidate and synthesis requests include retries with exponential backoff and jitter,
      and honor server-provided retry-after hints when present.
    - You can reduce burstiness by lowering `max_workers` (default is min(n_runs, 16)).
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")

    # Early check for API key to fail fast with a clear message
    if groq_api_key is None and not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Set it in the environment or pass -M groq_api_key=..."
        )

    client = Groq(api_key=groq_api_key) if groq_api_key else Groq()

    # Parallel candidate generations (threaded; works well in most environments)
    max_workers = min(n_runs, max_workers if max_workers is not None else 16)
    candidates: List[Optional[str]] = [None] * n_runs  # preserve order

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_idx = {
            ex.submit(
                _one_completion,
                client,
                messages,
                model=model,
                temperature=temperature_candidates,
                top_p=top_p,
                max_completion_tokens=max_completion_tokens,
            ): i
            for i in range(n_runs)
        }
        for fut in cf.as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            candidates[i] = fut.result()

    # Ensure all candidates are present
    final_candidates = [c if isinstance(c, str) else "" for c in candidates]

    print(f"question: {messages[0]['content']}")

    # Synthesis pass (with retries)
    synth_messages = _build_synthesis_messages(final_candidates, messages[0]["content"])
    final = _one_completion(
        client,
        synth_messages,
        model=model,
        temperature=temperature_synthesis,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
    )
    return final


def pro_mode(
    prompt: str,
    *,
    n_runs: int = 8,
    model: str = DEFAULT_MODEL,
    temperature_candidates: float = 0.9,
    temperature_synthesis: float = 0.2,
    top_p: float = 1.0,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    groq_api_key: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> str:
    """
    Convenience wrapper to support a simple (prompt: str) signature.
    """
    messages = [{"role": "user", "content": prompt}]
    return gpt_5_pro_mode(
        messages,
        n_runs=n_runs,
        model=model,
        temperature_candidates=temperature_candidates,
        temperature_synthesis=temperature_synthesis,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        groq_api_key=groq_api_key,
        max_workers=max_workers,
    )


__all__ = ["gpt_5_pro_mode", "pro_mode"]