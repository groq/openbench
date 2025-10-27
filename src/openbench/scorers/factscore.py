"""Inspect AI scorer adapter powered by FactScoreLite."""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List

from inspect_ai.scorer import Score, mean, scorer, Target
from inspect_ai.solver import TaskState

from openbench.metrics.factscore import factscore_metrics

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from openbench.utils.factscore_wiki import WikipediaRetriever
from openbench.utils.factscore_cache import (
    cache_dir,
    knowledge_db_path,
    resolve_cache_root,
)

# Note: Avoid importing WikipediaRetriever / BM25-dependent utilities at module
# import time to ensure the optional 'rank-bm25' dependency is only imported when
# the scorer is actually invoked.

try:  # pragma: no cover - import guarded for optional dependency
    from FactScoreLite import (  # type: ignore[import-not-found, import-untyped]
        AtomicFactGenerator,
        FactScorer,
        configs as factscorelite_configs,
    )
except ImportError as exc:  # pragma: no cover - exercised when dependency missing
    raise RuntimeError(
        "FactScoreLite scorer requires the 'factscorelite' package."
    ) from exc


@dataclass
class FactScoreConfig:
    """Configuration for running FactScoreLite scoring."""

    openai_model: str = "gpt-4o-mini"
    knowledge_source: str | None = "enwiki-20230401"
    gamma: int = 10
    cache_root: str | None = None
    passages: int = 8


def _normalise_model_name(model_name: str) -> str:
    if model_name.lower() in {"retrieval+chatgpt", "chatgpt"}:
        return "gpt-4o-mini"
    return model_name


def _ensure_openai_key() -> None:
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is required but not set. "
            "Set it via: export OPENAI_API_KEY='your-key'"
        )


def _ensure_nltk_resources() -> None:
    try:
        import nltk  # type: ignore[import-not-found, import-untyped]
    except ImportError as exc:  # pragma: no cover - safeguarded by optional deps
        raise RuntimeError(
            "FactScoreLite scorer requires the 'nltk' package. Install with `pip install openbench[factscore]`."
        ) from exc

    required = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]

    for resource_path, download_name in required:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


def _resolve_db_path(cache_root: Path, knowledge_source: str | None) -> Path:
    if not knowledge_source:
        knowledge_source = "enwiki-20230401"

    candidate = Path(knowledge_source).expanduser()
    if candidate.suffix == ".db" and candidate.exists():
        return candidate.resolve()

    filename = candidate.name
    if not filename.endswith(".db"):
        filename = f"{filename}.db"

    return knowledge_db_path(cache_root, filename=filename)


# Global registry to track active FactScore runners for cleanup
_active_runners: List[_FactScoreLiteRunner] = []
_runners_lock = asyncio.Lock()


class _FactScoreLiteRunner:
    """Serialises FactScoreLite calls and manages shared state."""

    def __init__(self, cfg: FactScoreConfig):
        self.cfg = cfg
        self._init_lock = asyncio.Lock()
        self._run_lock = asyncio.Lock()
        self._initialized = False
        self._atomic_generator: AtomicFactGenerator | None = None
        self._fact_scorer: FactScorer | None = None
        self._retriever: WikipediaRetriever | None = None
        self._cache_root: Path | None = None
        self._registered = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self._initialize()
            self._initialized = True
            # Register for cleanup on first initialization
            if not self._registered:
                await _register_runner(self)
                self._registered = True

    def _initialize(self) -> None:
        _ensure_openai_key()
        _ensure_nltk_resources()

        cache_root = resolve_cache_root(self.cfg.cache_root)
        self._cache_root = cache_root

        # Configure FactScoreLite persistent state under cache root.
        lite_cache_dir = cache_dir(cache_root) / "factscorelite"
        lite_cache_dir.mkdir(parents=True, exist_ok=True)

        facts_path = lite_cache_dir / "facts.json"
        decisions_path = lite_cache_dir / "decisions.json"
        for path in (facts_path, decisions_path):
            if path.exists():
                path.unlink()

        factscorelite_configs.facts_db_path = str(facts_path)
        factscorelite_configs.decisions_db_path = str(decisions_path)
        factscorelite_configs.model_name = _normalise_model_name(self.cfg.openai_model)

        self._atomic_generator = AtomicFactGenerator()
        self._fact_scorer = FactScorer()

        db_path = _resolve_db_path(cache_root, self.cfg.knowledge_source)
        retrieval_cache = (
            cache_dir(cache_root) / "factscorelite" / "knowledge_cache.json"
        )
        # Import WikipediaRetriever lazily to avoid importing optional dependencies
        from openbench.utils.factscore_wiki import (
            WikipediaRetriever,  # type: ignore
        )

        self._retriever = WikipediaRetriever(
            db_path=db_path,
            cache_path=retrieval_cache,
            num_passages=self.cfg.passages,
        )

    async def score(
        self, topic: str, generation: str, query: str | None
    ) -> Dict[str, Any]:
        await self._ensure_initialized()
        assert self._atomic_generator is not None
        assert self._fact_scorer is not None
        assert self._retriever is not None

        async with self._run_lock:
            return await asyncio.to_thread(self._score_sync, topic, generation, query)

    def _score_sync(
        self, topic: str, generation: str, query: str | None
    ) -> Dict[str, Any]:
        assert self._atomic_generator is not None
        assert self._fact_scorer is not None
        assert self._retriever is not None

        knowledge_source = self._retriever.get_knowledge(topic, query)

        facts_by_sentence = self._atomic_generator.run(generation)
        facts: list[str] = [
            fact.strip()
            for _sentence, atomics in facts_by_sentence
            for fact in atomics
            if fact and fact.strip()
        ]

        if not facts:
            return {
                "facts": [],
                "decisions": [],
                "score": 0.0,
                "init_score": 0.0,
                "num_facts_per_response": 0.0,
            }

        decisions = self._fact_scorer.get_score(
            facts=facts, knowledge_source=knowledge_source
        )

        supported_values = [
            1.0 if decision.get("is_supported") else 0.0 for decision in decisions
        ]
        if not supported_values:
            init_score = 0.0
        else:
            init_score = float(sum(supported_values) / len(supported_values))

        score = init_score
        if self.cfg.gamma and len(decisions) > 0:
            if len(decisions) >= self.cfg.gamma:
                penalty = 1.0
            else:
                penalty = math.exp(1 - self.cfg.gamma / max(len(decisions), 1))
            score = penalty * init_score

        return {
            "facts": facts,
            "decisions": decisions,
            "score": score,
            "init_score": init_score,
            "num_facts_per_response": len(facts),
        }

    async def close(self) -> None:
        """Close and cleanup resources, ensuring cache is persisted."""
        async with self._init_lock:
            if self._retriever is not None:
                self._retriever.close()


async def _register_runner(runner: _FactScoreLiteRunner) -> None:
    """Register a runner for cleanup tracking."""
    async with _runners_lock:
        if runner not in _active_runners:
            _active_runners.append(runner)


async def _unregister_runner(runner: _FactScoreLiteRunner) -> None:
    """Unregister a runner after cleanup."""
    async with _runners_lock:
        if runner in _active_runners:
            _active_runners.remove(runner)


async def cleanup_factscore_runners() -> None:
    """Close and cleanup all active FactScore runners.

    This ensures that the Wikipedia cache is persisted to disk via WikipediaRetriever.close().
    Should be called at the end of evaluation to guarantee cache is saved.
    """
    async with _runners_lock:
        runners_to_close = list(_active_runners)

    for runner in runners_to_close:
        try:
            await runner.close()
        except Exception:
            # Silently ignore cleanup errors
            pass
        finally:
            await _unregister_runner(runner)


def _is_knowledge_source_error(exc: Exception) -> bool:
    """Return True if ``exc`` is a KnowledgeSourceError from wiki utils.

    Imported lazily to avoid importing optional dependencies until needed.
    """
    try:
        from openbench.utils.factscore_wiki import KnowledgeSourceError
    except Exception:
        return False
    return isinstance(exc, KnowledgeSourceError)


@scorer(metrics=[mean(), factscore_metrics()])  # type: ignore[arg-type]
def factscore_scorer(
    model_name: str = "retrieval+ChatGPT",
    knowledge_source: str | None = "enwiki-20230401",
    gamma: int = 10,
    cache_root: str | None = None,
    passages: int = 8,
) -> Callable[[TaskState, Target], Any]:
    """Create a scorer that evaluates generations using FactScoreLite."""

    cfg = FactScoreConfig(
        openai_model=_normalise_model_name(model_name),
        knowledge_source=knowledge_source,
        gamma=gamma,
        cache_root=cache_root,
        passages=passages,
    )

    runner = _FactScoreLiteRunner(cfg)

    async def score(state: TaskState, target: Target | None = None) -> Score:
        _ = target
        topic = None
        if isinstance(state.metadata, dict):
            topic = state.metadata.get("topic")
        if not topic and state.sample_id and hasattr(state, "sample_id"):
            if isinstance(state.sample_id, dict) and "metadata" in state.sample_id:
                topic = state.sample_id.get("metadata", {}).get("topic")  # type: ignore[union-attr]
        if not topic and state.sample_id:
            topic = state.sample_id

        topic = str(topic) if topic else ""
        if not topic.strip():
            raise RuntimeError(
                "FactScoreLite scorer requires the sample metadata to include a 'topic'."
            )

        generation = state.output.completion if state.output else ""
        generation = generation or ""

        if not generation.strip():
            return Score(
                value=0.0,
                answer=generation,
                metadata={
                    "topic": topic,
                    "responded": False,
                    "factscore": 0.0,
                    "init_score": 0.0,
                    "facts_per_response": 0.0,
                    "decisions": [],
                    "respond_ratio_sample": 0.0,
                    "respond_ratio_batch": None,
                    "raw_result": {
                        "facts": [],
                        "decisions": [],
                        "score": 0.0,
                        "init_score": 0.0,
                        "num_facts_per_response": 0.0,
                    },
                },
            )

        query = getattr(state, "input_text", None)

        try:
            result = await runner.score(topic=topic, generation=generation, query=query)
        except Exception as exc:
            if _is_knowledge_source_error(exc):
                raise RuntimeError(str(exc)) from exc
            raise

        raw_score = result.get("score", 0.0) or 0.0
        init_score = result.get("init_score", 0.0) or 0.0
        facts_per_response = result.get("num_facts_per_response", 0)
        decisions: List[Any] = result.get("decisions", [])
        responded = bool(decisions)

        metadata: Dict[str, Any] = {
            "topic": topic,
            "factscore": float(raw_score),
            "init_score": float(init_score),
            "responded": responded,
            "facts_per_response": float(facts_per_response),
            "decisions": decisions,
            "respond_ratio_sample": 1.0 if responded else 0.0,
            "respond_ratio_batch": None,
            "raw_result": result,
        }

        return Score(
            value=float(raw_score),
            answer=generation,
            metadata=metadata,
        )

    return score
