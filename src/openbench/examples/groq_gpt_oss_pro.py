from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures as cf
import time
import os
import random
import re
import json
from statistics import median

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


# NEW: input normalization helper
def _coerce_messages(messages_or_prompt: Any) -> List[Dict[str, str]]:
    """
    Accepts: str | List[str] | List[Dict[str,str]] and returns List[Dict[str,str]].
    - If List[str], join with double newlines into a single user message.
    """
    if isinstance(messages_or_prompt, str):
        return [{"role": "user", "content": messages_or_prompt}]
    if isinstance(messages_or_prompt, list):
        if not messages_or_prompt:
            return [{"role": "user", "content": ""}]
        first = messages_or_prompt[0]
        if isinstance(first, dict) and "content" in first:
            out: List[Dict[str, str]] = []
            for m in messages_or_prompt:
                if isinstance(m, dict):
                    role = m.get("role", "user") or "user"
                    content = m.get("content", "")
                    out.append({"role": role, "content": str(content)})
                else:
                    out.append({"role": "user", "content": str(m)})
            return out
        # List[str] or mixed
        joined = "\n\n".join(str(x) for x in messages_or_prompt)
        return [{"role": "user", "content": joined}]
    raise ValueError("messages must be str, List[str], or List[{'role','content'}] dicts")


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


def _get_client(groq_api_key: Optional[str]) -> Groq:
    """Create or reuse a Groq client, with a clear error if the API key is missing."""
    if groq_api_key is None and not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Set it in the environment or pass -M groq_api_key=..."
        )
    return Groq(api_key=groq_api_key) if groq_api_key else Groq()


def _gpt_5_pro_mode_impl(
    client: Groq,
    messages: List[Dict[str, str]],
    *,
    n_runs: int,
    model: str,
    temperature_candidates: float,
    temperature_synthesis: float,
    top_p: float,
    max_completion_tokens: int,
    max_workers: Optional[int],
) -> str:
    """
    Core implementation used by both gpt_5_pro_mode and the deeper mode.
    Runs `n_runs` high-temp candidates in parallel, then synthesizes once.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")

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


def gpt_5_pro_mode(
    messages: Any,
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
    max_workers : Optional[int]
        Max workers for the candidate fanout (default = min(n_runs, 16)).

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
    messages = _coerce_messages(messages)
    client = _get_client(groq_api_key)
    print(f"question: {messages[0]['content']}")
    return _gpt_5_pro_mode_impl(
        client,
        messages,
        n_runs=n_runs,
        model=model,
        temperature_candidates=temperature_candidates,
        temperature_synthesis=temperature_synthesis,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        max_workers=max_workers,
    )


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


# -----------------------------
# Deeper Pro Mode (multi-group)
# -----------------------------

def gpt_5_pro_mode_deeper(
    messages: Any,
    *,
    num_groups: int = 4,
    runs_per_group: int = 8,
    model: str = DEFAULT_MODEL,
    temperature_candidates: float = 0.9,
    temperature_synthesis: float = 0.2,
    # If None, reuse temperature_synthesis for the final cross-group synthesis
    temperature_final_synthesis: Optional[float] = None,
    top_p: float = 1.0,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    groq_api_key: Optional[str] = None,
    # Concurrency knobs
    max_workers_per_group: Optional[int] = None,
    max_group_workers: Optional[int] = None,
) -> str:
    """
    Run a *deeper* pro mode:
      - Split into `num_groups` groups.
      - Each group runs the standard pro mode fanout of `runs_per_group` candidates
        (high temperature), followed by a group-level synthesis (low temperature).
      - Take the `num_groups` synthesized group answers and synthesize them *again*
        into one final answer.

    Example:
      num_groups=10, runs_per_group=5  ->  10 groups × (5 candidates → 1 group synth) → final synth

    Parameters
    ----------
    messages : List[Dict[str, str]]
        Chat messages in OpenAI format.
    num_groups : int
        Number of parallel pro-mode groups to run (>= 1).
    runs_per_group : int
        Number of candidate generations per group (>= 1).
    model, temperature_candidates, temperature_synthesis, top_p, max_completion_tokens :
        Same semantics as in `gpt_5_pro_mode`.
    temperature_final_synthesis : Optional[float]
        Temperature for the final cross-group synthesis. Defaults to `temperature_synthesis`.
    groq_api_key : Optional[str]
        API key (or set GROQ_API_KEY env var).
    max_workers_per_group : Optional[int]
        Max workers used *within each group* for the candidate fanout.
        Default: min(runs_per_group, 16).
    max_group_workers : Optional[int]
        Max workers used to run groups in parallel.
        Default: min(num_groups, 16).

    Returns
    -------
    str
        The final synthesized answer across all groups.
    """
    messages = _coerce_messages(messages)
    if num_groups < 1:
        raise ValueError("num_groups must be >= 1")
    if runs_per_group < 1:
        raise ValueError("runs_per_group must be >= 1")

    client = _get_client(groq_api_key)
    print(f"question: {messages[0]['content']}")

    group_workers = min(num_groups, max_group_workers if max_group_workers is not None else 16)
    inner_workers = min(
        runs_per_group,
        max_workers_per_group if max_workers_per_group is not None else 16,
    )

    # Run groups in parallel; each group runs a regular pro-mode (fanout -> synth)
    group_results: List[Optional[str]] = [None] * num_groups

    def _run_one_group(idx: int) -> str:
        return _gpt_5_pro_mode_impl(
            client,
            messages,
            n_runs=runs_per_group,
            model=model,
            temperature_candidates=temperature_candidates,
            temperature_synthesis=temperature_synthesis,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            max_workers=inner_workers,
        )

    with cf.ThreadPoolExecutor(max_workers=group_workers) as ex:
        fut_to_idx = {ex.submit(_run_one_group, i): i for i in range(num_groups)}
        for fut in cf.as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            group_results[i] = fut.result()

    # Ensure we have strings for all groups
    synthesized_groups = [g if isinstance(g, str) else "" for g in group_results]

    # Final cross-group synthesis
    final_synth_messages = _build_synthesis_messages(synthesized_groups, messages[0]["content"])
    final_temp = temperature_final_synthesis if temperature_final_synthesis is not None else temperature_synthesis
    final_answer = _one_completion(
        client,
        final_synth_messages,
        model=model,
        temperature=final_temp,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
    )
    return final_answer


def deeper_pro_mode(
    prompt: str,
    *,
    num_groups: int = 4,
    runs_per_group: int = 8,
    model: str = DEFAULT_MODEL,
    temperature_candidates: float = 0.9,
    temperature_synthesis: float = 0.2,
    temperature_final_synthesis: Optional[float] = None,
    top_p: float = 1.0,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    groq_api_key: Optional[str] = None,
    max_workers_per_group: Optional[int] = None,
    max_group_workers: Optional[int] = None,
) -> str:
    """
    Convenience wrapper for deeper pro mode with a simple (prompt: str) signature.
    """
    messages = [{"role": "user", "content": prompt}]
    return gpt_5_pro_mode_deeper(
        messages,
        num_groups=num_groups,
        runs_per_group=runs_per_group,
        model=model,
        temperature_candidates=temperature_candidates,
        temperature_synthesis=temperature_synthesis,
        temperature_final_synthesis=temperature_final_synthesis,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        groq_api_key=groq_api_key,
        max_workers_per_group=max_workers_per_group,
        max_group_workers=max_group_workers,
    )


# ---------------------------------------
# CoSTAR: Consensus Skeleton + Targeted Agreement & Repair
# ---------------------------------------

# ---- JSON helpers ----

def _extract_json_block(text: str) -> Optional[str]:
    """
    Try to robustly extract a JSON object/array from the model's text.
    Supports code fences and extra prose; returns the first plausible JSON block.
    """
    if not text:
        return None
    # Prefer fenced ```json ... ```
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        if candidate.startswith("{") or candidate.startswith("["):
            return candidate
    # Fallback: find first '{' and last '}' / or '[' and ']'
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if 0 <= first_brace < last_brace:
        return text[first_brace : last_brace + 1]
    first_sq = text.find("[")
    last_sq = text.rfind("]")
    if 0 <= first_sq < last_sq:
        return text[first_sq : last_sq + 1]
    return None


def _parse_json(text: str) -> Optional[Any]:
    block = _extract_json_block(text)
    if not block:
        return None
    try:
        return json.loads(block)
    except Exception:
        # Try to fix common trailing commas
        block2 = re.sub(r",\s*([}\]])", r"\1", block)
        try:
            return json.loads(block2)
        except Exception:
            return None


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        k = x.strip()
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


# ---- CoSTAR prompt builders ----

def _rubric_messages(user_prompt: str) -> List[Dict[str, str]]:
    system = (
        "You are a meticulous task analyst. Your job is to derive a concise evaluation rubric "
        "and checklist for answering the user's request. Output STRICT JSON only. Do not include any prose."
    )
    user = f"""
From the following request, derive a rubric and checklist that an excellent answer must satisfy.

REQUEST:
{user_prompt}

Return STRICT JSON with this schema (no extra fields):
{{
  "criteria": [  // 3-7 crisp criteria used to judge correctness, coverage, clarity, and instruction adherence
    "..."
  ],
  "sections": [  // minimal set of sections the answer should contain, in the intended order
    {{
      "id": "S1",             // stable id like S1,S2,...
      "title": "Section Title",
      "must_include": [ "...", "..." ],   // atomic points that must appear (facts, steps, sub-answers)
      "optional": [ "...", "..." ],       // nice-to-have details
      "notes": "Short guidance (style or boundaries) for this section"
    }}
  ],
  "numbers_needed": [        // any numerics or units the answer should pin down; can be empty
    {{ "name":"", "units":"", "description":"" }}
  ],
  "answer_style": "One short sentence describing tone and target (e.g., 'Executive, concise, stepwise')."
}}

Rules:
- Keep must_include items atomic and verifiable.
- Keep it compact. Maximum 6 sections unless obviously needed.
- No commentary outside JSON.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _skeleton_messages(user_prompt: str, rubric_json: Dict[str, Any]) -> List[Dict[str, str]]:
    system = (
        "You are an outline planner. Build a minimal yet complete section skeleton from the rubric. "
        "Output STRICT JSON only. Do not include any prose."
    )
    user = f"""
Build a section skeleton that operationalizes the rubric into concrete section IDs and what each section must contain.

REQUEST:
{user_prompt}

RUBRIC (JSON):
```json
{json.dumps(rubric_json, ensure_ascii=False)}
```

Return STRICT JSON with this schema:
{{
  "sections": [
    {{
      "id": "S1",
      "title": "Section Title",
      "requirements": [  // transform must_include into atomic requirements for this section
        "..."
      ]
    }}
  ],
  "final_answer_guidance": "≤25 words on the direct final answer summary to produce."
}}

Rules:
- Use exactly the provided ids (S1..Sn) in order, matching rubric.sections length.
- Ensure every rubric.must_include is represented in 'requirements' (can distribute across sections if appropriate).
- No new ids, no commentary outside JSON.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _candidate_struct_messages(
    user_prompt: str, rubric_json: Dict[str, Any], skeleton_json: Dict[str, Any]
) -> List[Dict[str, str]]:
    system = (
        "You are an expert solver. Produce a structured content plan as STRICT JSON—no prose outside JSON. "
        "Claims must be atomic, factual, and minimal. DO NOT invent new sections or ids."
    )
    user = f"""
Using the request, rubric, and section skeleton, produce a structured candidate with atomic claims per section.
Write STRICT JSON ONLY.

REQUEST:
{user_prompt}

RUBRIC:
```json
{json.dumps(rubric_json, ensure_ascii=False)}
```

SKELETON:
```json
{json.dumps(skeleton_json, ensure_ascii=False)}
```

Return STRICT JSON with this schema:
{{
  "sections": [
    {{
      "id": "S#",                  // one of the skeleton ids
      "title": "Section Title",    // same as skeleton
      "claims": [
        {{
          "id": "C#",              // local per-section claim id like C1,C2,...
          "text": "Atomic, verifiable claim for this section (≤40 words).",
          "confidence": 0.0        // 0.0-1.0 self-estimated confidence
        }}
      ]
    }}
  ],
  "numbers": [                     // any numbers relevant to rubric.numbers_needed (if any)
    {{ "name":"", "value": "", "units":"", "confidence": 0.0 }}
  ],
  "assumptions": [ "...", "..." ], // explicit assumptions if the request has ambiguity; keep minimal
  "final_answer": "≤50 words direct answer consistent with the above"
}}

Rules:
- Use ONLY the section ids given in the skeleton and keep their order.
- Every skeleton.requirements item should be covered by at least one claim in its section.
- Keep each claim atomic (one fact/step). Avoid overlapping claims.
- If unsure, add a short assumption rather than guessing a fact.
- No commentary outside JSON.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _cluster_messages(
    user_prompt: str,
    rubric_json: Dict[str, Any],
    section: Dict[str, Any],
    indexed_claims: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    system = (
        "You are a semantic clustering assistant. Cluster near-duplicate claims by MEANING (paraphrases), "
        "ignoring wording differences. Output STRICT JSON only."
    )
    # Prepare a compact list for the LLM to cluster
    claims_for_prompt = [
        {"index": c["index"], "text": c["text"]} for c in indexed_claims
    ]
    user = f"""
Cluster the following claims for a single section into canonical claims.

REQUEST:
{user_prompt}

RUBRIC (for context):
```json
{json.dumps(rubric_json, ensure_ascii=False)}
```

SECTION:
```json
{json.dumps({"id": section["id"], "title": section["title"]}, ensure_ascii=False)}
```

CLAIMS (each has a stable integer index):
```json
{json.dumps(claims_for_prompt, ensure_ascii=False)}
```

Return STRICT JSON with this schema:
{{
  "canonical_claims": [
    {{
      "id": "K#",                  // K1,K2,...
      "text": "Canonical claim text representing a cluster (≤40 words).",
      "members": [                 // the integer indices of claims in this cluster
        0,1,2
      ]
    }}
  ]
}}

Hard rules:
- Every input claim index MUST appear in exactly one cluster's 'members' (no omissions, no duplicates).
- Canonical 'text' should best represent the shared meaning; do NOT add new facts.
- Keep the number of clusters as small as reasonable while preserving distinct meanings.
- No commentary outside JSON.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _adjudicate_messages(
    user_prompt: str,
    rubric_json: Dict[str, Any],
    section: Dict[str, Any],
    options: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    system = (
        "You are a strict adjudicator. Choose the SINGLE best canonical claim for this section among options, "
        "or return NONE if all are weak/duplicative. Output STRICT JSON only."
    )
    user = f"""
Pick the best canonical claim option for the section, using the rubric and request.

REQUEST:
{user_prompt}

RUBRIC:
```json
{json.dumps(rubric_json, ensure_ascii=False)}
```

SECTION:
```json
{json.dumps({"id": section["id"], "title": section["title"]}, ensure_ascii=False)}
```

OPTIONS:
Each option has: id (K#), text, support (count of members), support_ratio (0..1), avg_conf (0..1).
```json
{json.dumps(options, ensure_ascii=False)}
```

Return STRICT JSON:
{{
  "winner_id": "K#" | "NONE",
  "reason": "≤50 words concise justification"
}}

Guidance:
- Prefer higher support_ratio and higher avg_conf IF consistent with rubric.requirements for this section.
- Penalize options that introduce content outside the rubric or contradict other options.
- If options are equivalent or redundant, select the most precise one; else return "NONE" to drop this slot.
- No commentary outside JSON.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _verify_messages(
    user_prompt: str,
    rubric_json: Dict[str, Any],
    selected_payload: Dict[str, Any],
) -> List[Dict[str, str]]:
    system = (
        "You are a rigorous verifier. Check the selected claims against the rubric for contradictions, "
        "missing required items, and numeric/units issues. Output STRICT JSON only."
    )
    user = f"""
Verify and flag issues. Do NOT invent new facts.

REQUEST:
{user_prompt}

RUBRIC:
```json
{json.dumps(rubric_json, ensure_ascii=False)}
```

SELECTED (claims/numbers/assumptions/final_answer_short):
```json
{json.dumps(selected_payload, ensure_ascii=False)}
```

Return STRICT JSON with this schema:
{{
  "contradictions": [  // contradictions across selected claims or with final_answer_short
    {{ "section_id":"", "claim_ids":[ "S1:K1", "S1:K3" ], "issue":"Describe the contradiction briefly." }}
  ],
  "missing": [         // rubric.must_include or skeleton requirements not covered by any selected claim
    {{ "section_id":"", "requirement":"..." }}
  ],
  "number_issues": [   // units mismatch, scale errors, or inconsistent numerics
    {{ "name":"", "issue":"..." }}
  ],
  "edits": [           // only rephrase-for-clarity or removal suggestions; NO new facts
    {{ "type":"rephrase"|"remove", "target":"S#:K#"|"final_answer_short"|"assumptions", "text":"new wording if rephrase" }}
  ]
}}

Rules:
- You may recommend REMOVING a claim to resolve contradictions.
- You may REPHRASE a claim strictly to improve clarity (do not add content).
- You may NOT add new claims or numbers that aren't present.
- No commentary outside JSON.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _realize_messages(
    user_prompt: str,
    rubric_json: Dict[str, Any],
    consensus_payload: Dict[str, Any],
) -> List[Dict[str, str]]:
    system = (
        "You are an expert editor. Convert the consensus skeleton into polished prose. "
        "STRICT rule: do NOT introduce any new factual claims not present in the selected claims or numbers."
    )
    user = f"""
Produce the final answer from the consensus payload.

REQUEST:
{user_prompt}

RUBRIC (style guidance included):
```json
{json.dumps(rubric_json, ensure_ascii=False)}
```

CONSENSUS PAYLOAD:
```json
{json.dumps(consensus_payload, ensure_ascii=False)}
```

Output a clean, user-ready answer with:
- A brief executive summary (2–3 sentences).
- The main content organized by the given 'sections' order and titles.
- Use the provided 'assumptions' explicitly if they influence the answer.
- Match 'answer_style' from the rubric.
- DO NOT introduce any new facts or numbers beyond those in 'sections.claims' and 'numbers'.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---- CoSTAR core helpers ----

def _derive_rubric(client: Groq, model: str, user_prompt: str, *, lowT: float, top_p: float, max_tokens: int) -> Dict[str, Any]:
    resp = _one_completion(
        client,
        _rubric_messages(user_prompt),
        model=model,
        temperature=lowT,
        top_p=top_p,
        max_completion_tokens=max_tokens,
    )
    parsed = _parse_json(resp)
    if isinstance(parsed, dict) and "sections" in parsed and "criteria" in parsed:
        return parsed  # type: ignore
    # Fallback minimal rubric
    return {
        "criteria": ["Correctness", "Coverage", "Clarity", "Instruction adherence"],
        "sections": [
            {"id": "S1", "title": "Answer", "must_include": [], "optional": [], "notes": "Directly answer the question."}
        ],
        "numbers_needed": [],
        "answer_style": "Concise and direct",
    }


def _derive_skeleton(client: Groq, model: str, user_prompt: str, rubric_json: Dict[str, Any], *, lowT: float, top_p: float, max_tokens: int) -> Dict[str, Any]:
    resp = _one_completion(
        client,
        _skeleton_messages(user_prompt, rubric_json),
        model=model,
        temperature=lowT,
        top_p=top_p,
        max_completion_tokens=max_tokens,
    )
    parsed = _parse_json(resp)
    if isinstance(parsed, dict) and "sections" in parsed:
        # Ensure ids/titles exist
        sections = parsed.get("sections", [])
        cleaned = []
        for i, s in enumerate(sections, start=1):
            sid = s.get("id") or f"S{i}"
            title = s.get("title") or f"Section {i}"
            reqs = s.get("requirements") or []
            cleaned.append({"id": sid, "title": title, "requirements": reqs})
        return {
            "sections": cleaned if cleaned else [{"id": "S1", "title": "Answer", "requirements": []}],
            "final_answer_guidance": parsed.get("final_answer_guidance", "Provide a direct answer."),
        }
    # Fallback skeleton from rubric
    sections = rubric_json.get("sections", [{"id": "S1", "title": "Answer", "must_include": []}])
    out_sections = []
    for s in sections:
        out_sections.append({
            "id": s.get("id", "S1"),
            "title": s.get("title", "Answer"),
            "requirements": s.get("must_include", []),
        })
    return {"sections": out_sections, "final_answer_guidance": "Provide a direct answer."}


def _candidate_struct_once(
    client: Groq,
    model: str,
    user_prompt: str,
    rubric_json: Dict[str, Any],
    skeleton_json: Dict[str, Any],
    *,
    highT: float,
    top_p: float,
    max_tokens: int,
) -> Optional[Dict[str, Any]]:
    resp = _one_completion(
        client,
        _candidate_struct_messages(user_prompt, rubric_json, skeleton_json),
        model=model,
        temperature=highT,
        top_p=top_p,
        max_completion_tokens=max_tokens,
    )
    parsed = _parse_json(resp)
    if isinstance(parsed, dict) and "sections" in parsed:
        return parsed  # type: ignore
    return None


def _collect_struct_candidates_parallel(
    client: Groq,
    model: str,
    user_prompt: str,
    rubric_json: Dict[str, Any],
    skeleton_json: Dict[str, Any],
    *,
    num_groups: int,
    runs_per_group: int,
    max_group_workers: int,
    max_workers_per_group: int,
    highT: float,
    top_p: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Launch groups × runs parallel candidate struct generations.
    Returns a list of candidate dicts; invalid JSON candidates are skipped.
    """
    total_runs = num_groups * runs_per_group
    results: List[Optional[Dict[str, Any]]] = [None] * total_runs

    def _run(idx: int) -> Optional[Dict[str, Any]]:
        return _candidate_struct_once(
            client, model, user_prompt, rubric_json, skeleton_json,
            highT=highT, top_p=top_p, max_tokens=max_tokens
        )

    # We use a single pool sized to min(total_runs, max_group_workers * max_workers_per_group),
    # but to keep things simple and robust across environments, just use min(total_runs, 32).
    pool_size = min(total_runs, max(max_group_workers, 1) * max(max_workers_per_group, 1), 32)
    pool_size = max(1, pool_size)

    with cf.ThreadPoolExecutor(max_workers=pool_size) as ex:
        fut_to_idx = {ex.submit(_run, i): i for i in range(total_runs)}
        for fut in cf.as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            try:
                results[i] = fut.result()
            except Exception:
                results[i] = None

    # Filter out Nones
    return [r for r in results if isinstance(r, dict)]


def _gather_section_claims(
    candidates: List[Dict[str, Any]],
    skeleton_json: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a mapping section_id -> list of claims with indexes and meta.
    Each claim dict: {index:int, section_id:str, text:str, confidence:float}
    """
    section_ids = [s["id"] for s in skeleton_json.get("sections", [])]
    by_section: Dict[str, List[Dict[str, Any]]] = {sid: [] for sid in section_ids}
    idx_counter = 0
    for cand in candidates:
        for sec in cand.get("sections", []):
            sid = sec.get("id")
            if sid not in by_section:
                continue
            for claim in sec.get("claims", []) or []:
                text = (claim.get("text") or "").strip()
                if not text:
                    continue
                conf = claim.get("confidence", 0.0)
                by_section[sid].append(
                    {"index": idx_counter, "section_id": sid, "text": text, "confidence": float(conf)}
                )
                idx_counter += 1
    return by_section


def _cluster_section(
    client: Groq,
    model: str,
    user_prompt: str,
    rubric_json: Dict[str, Any],
    section: Dict[str, Any],
    indexed_claims: List[Dict[str, Any]],
    *,
    lowT: float,
    top_p: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Return list of canonical claims for the section:
    Each canonical claim: {"id":"K#", "text":"...", "members":[int,...]}
    """
    if not indexed_claims:
        return []
    resp = _one_completion(
        client,
        _cluster_messages(user_prompt, rubric_json, section, indexed_claims),
        model=model,
        temperature=lowT,
        top_p=top_p,
        max_completion_tokens=max_tokens,
    )
    parsed = _parse_json(resp)
    if not (isinstance(parsed, dict) and "canonical_claims" in parsed):
        # Fallback: treat each as its own cluster
        return [{"id": f"K{i+1}", "text": c["text"], "members": [c["index"]]} for i, c in enumerate(indexed_claims)]
    clusters = parsed.get("canonical_claims", [])
    # Basic sanity: ensure coverage; if invalid, fallback as above
    mentioned = set()
    for cl in clusters:
        for m in cl.get("members", []):
            mentioned.add(m)
    all_idx = {c["index"] for c in indexed_claims}
    if mentioned != all_idx:
        # Bad clustering; fallback to singletons
        return [{"id": f"K{i+1}", "text": c["text"], "members": [c["index"]]} for i, c in enumerate(indexed_claims)]
    return clusters  # type: ignore


def _resolve_disagreements_for_section(
    client: Groq,
    model: str,
    user_prompt: str,
    rubric_json: Dict[str, Any],
    section: Dict[str, Any],
    clusters: List[Dict[str, Any]],
    claims: List[Dict[str, Any]],
    *,
    total_candidates: int,
    micro_judges: int,
    lowT: float,
    top_p: float,
    max_tokens: int,
    majority_threshold: float,
    minority_threshold: float,
    max_kept: int,
) -> List[Dict[str, Any]]:
    """
    Select which canonical claims to keep for a section.
    Strategy:
      - Compute support and avg_conf for each cluster.
      - Keep clusters with support_ratio >= majority_threshold.
      - Drop clusters with support_ratio <= minority_threshold unless adjudication chooses them.
      - For middle zone, run micro adjudication multiple times; keep the winner if consistent.
      - Cap kept claims to max_kept (highest support first).
    Returns: [{"id":"S#:K#","text":"..."}...]
    """
    if not clusters:
        return []

    # Build quick lookup from index -> confidence
    conf_map = {c["index"]: float(c.get("confidence", 0.0)) for c in claims}

    enriched: List[Dict[str, Any]] = []
    for cl in clusters:
        members = cl.get("members", []) or []
        supp = len(members)
        avg_conf = 0.0
        if supp > 0:
            avg_conf = sum(conf_map.get(int(m), 0.0) for m in members) / supp
        ratio = supp / max(1, total_candidates)
        enriched.append({
            "id": cl.get("id", ""),
            "text": cl.get("text", ""),
            "members": members,
            "support": supp,
            "support_ratio": ratio,
            "avg_conf": avg_conf,
        })

    # Sort by support desc then avg_conf desc
    enriched.sort(key=lambda x: (x["support"], x["avg_conf"]), reverse=True)

    kept: List[Dict[str, Any]] = []

    # First, auto-keep majority clusters
    for opt in enriched:
        if opt["support_ratio"] >= majority_threshold:
            kept.append({"id": f"{section['id']}:{opt['id']}", "text": opt["text"]})

    # Then adjudicate middle zone if we still have capacity
    def adjudicate(options: List[Dict[str, Any]]) -> Optional[str]:
        # Run multiple micro judges and take majority winner (not NONE)
        wins: List[str] = []
        messages = _adjudicate_messages(user_prompt, rubric_json, section, options)
        for _ in range(micro_judges):
            try:
                resp = _one_completion(
                    client, messages, model=model, temperature=lowT, top_p=top_p, max_completion_tokens=max_tokens
                )
                parsed = _parse_json(resp)
                if isinstance(parsed, dict):
                    wid = parsed.get("winner_id")
                    if isinstance(wid, str):
                        wins.append(wid)
            except Exception:
                continue
        # Count non-NONE winners
        counts: Dict[str, int] = {}
        for w in wins:
            if w and w != "NONE":
                counts[w] = counts.get(w, 0) + 1
        if not counts:
            return None
        # Return most common winner
        return max(counts.items(), key=lambda kv: kv[1])[0]

    # Middle zone
    middle = [o for o in enriched if minority_threshold < o["support_ratio"] < majority_threshold]
    if middle and len(kept) < max_kept:
        wid = adjudicate(middle)
        if wid:
            best = next((o for o in middle if o["id"] == wid), None)
            if best:
                kept.append({"id": f"{section['id']}:{best['id']}", "text": best["text"]})

    # Optionally salvage a minority outlier if room remains and it's very high confidence
    if len(kept) < max_kept:
        minorities = [o for o in enriched if o["support_ratio"] <= minority_threshold]
        minorities.sort(key=lambda x: (x["avg_conf"], x["support"]), reverse=True)
        for m in minorities:
            if m["avg_conf"] >= 0.75:
                kept.append({"id": f"{section['id']}:{m['id']}", "text": m["text"]})
                break

    # Cap to max_kept
    kept = kept[:max_kept]
    return kept


def _consolidate_numbers(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Combine candidate 'numbers' by (name, units). Choose median numeric value if possible,
    otherwise the most common string. Returns list of dicts.
    """
    buckets: Dict[Tuple[str, str], List[str]] = {}
    for cand in candidates:
        for num in cand.get("numbers", []) or []:
            name = str(num.get("name", "")).strip()
            units = str(num.get("units", "")).strip()
            val = str(num.get("value", "")).strip()
            if not name:
                continue
            buckets.setdefault((name, units), []).append(val)

    out: List[Dict[str, Any]] = []
    for (name, units), vals in buckets.items():
        # Try numeric
        floats: List[float] = []
        for v in vals:
            try:
                # support forms like "123.4" or "1,234"
                v2 = float(re.sub(r"[,\s]", "", v))
                floats.append(v2)
            except Exception:
                pass
        if floats:
            chosen = median(floats)
            out.append({"name": name, "value": chosen, "units": units})
        else:
            # Choose most frequent string
            freq: Dict[str, int] = {}
            for v in vals:
                freq[v] = freq.get(v, 0) + 1
            chosen_str = max(freq.items(), key=lambda kv: kv[1])[0]
            out.append({"name": name, "value": chosen_str, "units": units})
    return out


def _merge_assumptions(candidates: List[Dict[str, Any]], limit: int = 8) -> List[str]:
    all_assumps: List[str] = []
    for c in candidates:
        all_assumps.extend([str(a).strip() for a in (c.get("assumptions") or []) if str(a).strip()])
    return _dedupe_preserve_order(all_assumps)[:limit]


def _short_final_answer(candidates: List[Dict[str, Any]], fallback: str = "") -> str:
    finals = [str(c.get("final_answer") or "").strip() for c in candidates]
    finals = [f for f in finals if f]
    if not finals:
        return fallback
    # Choose the most common; tie-break by length (shorter)
    freq: Dict[str, int] = {}
    for f in finals:
        freq[f] = freq.get(f, 0) + 1
    best, _ = max(freq.items(), key=lambda kv: (kv[1], -len(kv[0])))
    return best


def _apply_verifier_edits(
    selected_payload: Dict[str, Any],
    verifier_json: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply 'remove' and 'rephrase' edits conservatively.
    """
    sections = selected_payload.get("sections", [])
    edits = verifier_json.get("edits", []) or []

    # Build index for quick access
    sect_map = {s["id"]: s for s in sections}
    claim_map: Dict[str, Dict[str, Any]] = {}
    for s in sections:
        for cl in s.get("claims", []):
            claim_map[f"{s['id']}:{cl['id'].split(':')[-1]}"] = cl  # cl.id already includes S#:K#; we only use K# part for key
            claim_map[f"{cl['id']}"] = cl  # full id

    for e in edits:
        et = e.get("type")
        tgt = e.get("target")
        if et == "remove" and isinstance(tgt, str):
            # target can be "S#:K#"
            if tgt in claim_map:
                # remove it from its section
                for s in sections:
                    s["claims"] = [c for c in s.get("claims", []) if c.get("id") != tgt]
        elif et == "rephrase" and isinstance(tgt, str):
            new_text = str(e.get("text") or "").strip()
            if tgt == "final_answer_short":
                if new_text:
                    selected_payload["final_answer_short"] = new_text
            elif tgt in claim_map and new_text:
                claim_map[tgt]["text"] = new_text

    # Remove empty sections' claims list if needed (but keep section order)
    for s in sections:
        s["claims"] = [c for c in s.get("claims", []) if c.get("text")]
    selected_payload["sections"] = sections
    return selected_payload


# ---- CoSTAR orchestrator ----

def gpt_5_pro_mode_costar(
    messages: Any,
    *,
    # compute layout
    num_groups: int = 6,
    runs_per_group: int = 5,
    # model & temps
    model: str = DEFAULT_MODEL,
    temperature_candidates: float = 0.9,
    temperature_coordination: float = 0.2,   # low temperature for rubric/cluster/verify/realize
    temperature_final: Optional[float] = None,
    top_p: float = 1.0,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    groq_api_key: Optional[str] = None,
    # concurrency
    max_workers_per_group: Optional[int] = None,
    max_group_workers: Optional[int] = None,
    # selection knobs
    majority_threshold: float = 0.6,
    minority_threshold: float = 0.15,
    max_claims_per_section: int = 12,
    micro_judges_per_issue: int = 3,
) -> str:
    """
    CoSTAR pipeline:
      0) Derive rubric & skeleton.
      1) Parallel candidate STRUCTURE generations (groups×runs), JSON only.
      2) For each section: cluster claims -> compute support/confidence -> select via rules + micro adjudication.
      3) Consolidate numbers, merge assumptions, choose a short final answer.
      4) Verifier pass finds contradictions/missing/number issues -> apply safe edits (remove/rephrase only).
      5) Realize final prose from consensus payload (no new facts).

    Returns:
      Final realized answer (string).
    """
    # Normalize inputs from callers (e.g., OpenBench may pass a string or List[str])
    messages = _coerce_messages(messages)

    if not messages or not isinstance(messages, list) or not isinstance(messages[0], dict) or "content" not in messages[0]:
        raise ValueError("messages must resolve to a non-empty List[{'role','content'}].")

    if num_groups < 1 or runs_per_group < 1:
        raise ValueError("num_groups and runs_per_group must be >= 1")

    client = _get_client(groq_api_key)
    user_prompt = messages[0]["content"]
    print(f"question: {user_prompt}")

    # Temperatures
    lowT = temperature_final if temperature_final is not None else temperature_coordination
    lowT = float(lowT)
    highT = float(temperature_candidates)

    # Concurrency defaults
    inner_workers = min(runs_per_group, max_workers_per_group if max_workers_per_group is not None else 16)
    group_workers = min(num_groups, max_group_workers if max_group_workers is not None else 16)

    # ---- Phase 0: Rubric & Skeleton ----
    rubric_json = _derive_rubric(
        client, model, user_prompt, lowT=lowT, top_p=top_p, max_tokens=max_completion_tokens
    )
    skeleton_json = _derive_skeleton(
        client, model, user_prompt, rubric_json, lowT=lowT, top_p=top_p, max_tokens=max_completion_tokens
    )

    # ---- Phase 1: Parallel candidates (STRUCT JSON) ----
    candidates = _collect_struct_candidates_parallel(
        client, model, user_prompt, rubric_json, skeleton_json,
        num_groups=num_groups,
        runs_per_group=runs_per_group,
        max_group_workers=group_workers,
        max_workers_per_group=inner_workers,
        highT=highT,
        top_p=top_p,
        max_tokens=max_completion_tokens,
    )
    if not candidates:
        # Degenerate fallback: run one low-temp prose answer via classic mode
        return _gpt_5_pro_mode_impl(
            client, messages, n_runs=1, model=model,
            temperature_candidates=0.3, temperature_synthesis=0.2,
            top_p=top_p, max_completion_tokens=max_completion_tokens, max_workers=1
        )

    total_candidates = num_groups * runs_per_group

    # ---- Phase 2: Claim clustering by section ----
    section_claims = _gather_section_claims(candidates, skeleton_json)

    clustered_by_section: Dict[str, List[Dict[str, Any]]] = {}
    for section in skeleton_json.get("sections", []):
        sid = section["id"]
        claims = section_claims.get(sid, [])
        clusters = _cluster_section(
            client, model, user_prompt, rubric_json, section, claims,
            lowT=lowT, top_p=top_p, max_tokens=max_completion_tokens
        )
        clustered_by_section[sid] = clusters

    # ---- Phase 3: Disagreement-guided selection ----
    selected_sections: List[Dict[str, Any]] = []
    for section in skeleton_json.get("sections", []):
        sid = section["id"]
        claims = section_claims.get(sid, [])
        clusters = clustered_by_section.get(sid, [])
        kept = _resolve_disagreements_for_section(
            client, model, user_prompt, rubric_json, section, clusters, claims,
            total_candidates=total_candidates,
            micro_judges=micro_judges_per_issue,
            lowT=lowT, top_p=top_p, max_tokens=max_completion_tokens,
            majority_threshold=majority_threshold,
            minority_threshold=minority_threshold,
            max_kept=max_claims_per_section,
        )
        selected_sections.append({"id": sid, "title": section["title"], "claims": kept})

    numbers = _consolidate_numbers(candidates)
    assumptions = _merge_assumptions(candidates)
    final_short = _short_final_answer(candidates, fallback=skeleton_json.get("final_answer_guidance", ""))

    selected_payload = {
        "sections": selected_sections,
        "numbers": numbers,
        "assumptions": assumptions,
        "final_answer_short": final_short,
    }

    # ---- Phase 4: Verifier (line-item checks) ----
    verifier_resp = _one_completion(
        client,
        _verify_messages(user_prompt, rubric_json, selected_payload),
        model=model,
        temperature=lowT,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
    )
    # Be robust to non-object JSON (e.g., a string or array) from the verifier.
    _vj_raw = _parse_json(verifier_resp)
    _vj_base = {"contradictions": [], "missing": [], "number_issues": [], "edits": []}
    verifier_json = _vj_base if not isinstance(_vj_raw, dict) else {**_vj_base, **_vj_raw}

    # Apply allowed edits (remove/rephrase)
    selected_payload = _apply_verifier_edits(selected_payload, verifier_json)

    # ---- Phase 5: Realization (no new facts) ----
    final_answer = _one_completion(
        client,
        _realize_messages(user_prompt, rubric_json, selected_payload),
        model=model,
        temperature=lowT,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
    )
    return final_answer


def costar_mode(
    prompt: str,
    *,
    num_groups: int = 6,
    runs_per_group: int = 5,
    model: str = DEFAULT_MODEL,
    temperature_candidates: float = 0.9,
    temperature_coordination: float = 0.2,
    temperature_final: Optional[float] = None,
    top_p: float = 1.0,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    groq_api_key: Optional[str] = None,
    max_workers_per_group: Optional[int] = None,
    max_group_workers: Optional[int] = None,
    majority_threshold: float = 0.6,
    minority_threshold: float = 0.15,
    max_claims_per_section: int = 12,
    micro_judges_per_issue: int = 3,
) -> str:
    """
    Convenience wrapper for CoSTAR with a simple (prompt: str) signature.
    """
    messages = [{"role": "user", "content": prompt}]
    return gpt_5_pro_mode_costar(
        messages,
        num_groups=num_groups,
        runs_per_group=runs_per_group,
        model=model,
        temperature_candidates=temperature_candidates,
        temperature_coordination=temperature_coordination,
        temperature_final=temperature_final,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        groq_api_key=groq_api_key,
        max_workers_per_group=max_workers_per_group,
        max_group_workers=max_group_workers,
        majority_threshold=majority_threshold,
        minority_threshold=minority_threshold,
        max_claims_per_section=max_claims_per_section,
        micro_judges_per_issue=micro_judges_per_issue,
    )


__all__ = [
    "gpt_5_pro_mode",
    "pro_mode",
    "gpt_5_pro_mode_deeper",
    "deeper_pro_mode",
    "gpt_5_pro_mode_costar",
    "costar_mode",
]
