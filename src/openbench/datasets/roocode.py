"""
Dataset implementation for Roo-Code-Evals tasks.

This module fetches coding tasks from the Roo-Code-Evals GitHub repository
and uses the standardized language-specific prompts from the prompts/ folder.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
from functools import lru_cache
import requests  # type: ignore
import time
import hashlib

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

# TODO(Lucas): create HF dataset for roocode

# GitHub API configuration
ROOCODE_REPO_URL = "https://api.github.com/repos/RooCodeInc/Roo-Code-Evals/contents"
ROOCODE_RAW_URL = "https://raw.githubusercontent.com/RooCodeInc/Roo-Code-Evals/main"

# Cache configuration
CACHE_DIR = Path.home() / ".cache" / "openbench" / "roocode"
CACHE_VERSION = "v1"  # Increment when cache format changes

# Supported programming languages and their test commands
LANGUAGE_CONFIG = {
    "python": {
        "test_command": "uv run python3 -m pytest -o markers=task {task_name}_test.py",
        "setup_commands": [],
    },
    "go": {
        "test_command": "go test",
        "setup_commands": [],
    },
    "javascript": {
        "test_command": "pnpm test",
        "setup_commands": ["pnpm install --frozen-lockfile"],
    },
    "java": {
        "test_command": "./gradlew test",
        "setup_commands": [],
    },
    "rust": {
        "test_command": "cargo test",
        "setup_commands": [],
    },
}


def _get_cache_key(cache_type: str, identifier: str) -> str:
    """Generate a cache key for the given type and identifier."""
    key_data = f"{CACHE_VERSION}_{cache_type}_{identifier}"
    return hashlib.md5(key_data.encode()).hexdigest()


def _ensure_cache_dir():
    """Ensure the cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_from_cache(cache_key: str) -> Optional[Any]:
    """Load data from cache if it exists and is valid."""
    try:
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                # Check if cache is less than 24 hours old
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < 24 * 3600:  # 24 hours
                    return cached_data
                else:
                    cache_file.unlink()  # Remove expired cache
    except Exception as e:
        print(f"Failed to load cache for {cache_key}: {e}")
    return None


def _save_to_cache(cache_key: str, data: Any):
    """Save data to cache."""
    try:
        _ensure_cache_dir()
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Cached data for {cache_key}")
    except Exception as e:
        print(f"Failed to save cache for {cache_key}: {e}")


class RooCodeTaskLoader:
    """Loads and caches Roo-Code-Evals tasks and prompts from GitHub."""

    def __init__(self):
        self._prompts_cache: Dict[str, str] = {}
        self._tasks_cache: Dict[str, List[str]] = {}
        self._request_count = 0
        self._last_request_time = 0.0

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for GitHub API requests, including auth if available."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "OpenBench-RooCode-Loader/1.0",
        }

        # Add GitHub token if available
        github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get(
            "GITHUB_API_TOKEN"
        )
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        return headers

    def _rate_limit_delay(self):
        """Add delay between requests to avoid rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        # Minimum 0.1 seconds between requests
        min_delay = 0.1
        if time_since_last < min_delay:
            time.sleep(min_delay - time_since_last)

        self._last_request_time = time.time()
        self._request_count += 1

        # Log progress every 50 requests
        if self._request_count % 50 == 0:
            print(f"GitHub API requests made: {self._request_count}")

    @lru_cache(maxsize=None)
    def _fetch_github_content(self, path: str) -> List[Dict[str, Any]]:
        """Fetch directory contents from GitHub API with caching and rate limiting."""
        url = f"{ROOCODE_REPO_URL}/{path}" if path else ROOCODE_REPO_URL

        self._rate_limit_delay()

        try:
            headers = self._get_headers()
            response = requests.get(url, headers=headers, timeout=30)

            # Handle rate limiting
            if response.status_code == 403 and "rate limit" in response.text.lower():
                print("Rate limit hit. Waiting 60 seconds before retry...")
                time.sleep(60)
                response = requests.get(url, headers=headers, timeout=30)

            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}")

    @lru_cache(maxsize=None)
    def _fetch_file_content(self, path: str) -> str:
        """Fetch file content from GitHub raw URL with caching and rate limiting."""
        url = f"{ROOCODE_RAW_URL}/{path}"

        self._rate_limit_delay()

        try:
            headers = self._get_headers()
            response = requests.get(url, headers=headers, timeout=30)

            # Handle rate limiting
            if response.status_code == 403 and "rate limit" in response.text.lower():
                print(
                    "Rate limit hit on file fetch. Waiting 60 seconds before retry..."
                )
                time.sleep(60)
                response = requests.get(url, headers=headers, timeout=30)

            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch file {url}: {e}")

    def get_language_prompt(self, language: str) -> str:
        """Get the standardized prompt for a programming language."""
        if language not in LANGUAGE_CONFIG:
            raise ValueError(f"Unsupported language: {language}")

        if language not in self._prompts_cache:
            # Try cache first
            cache_key = _get_cache_key("prompt", language)
            cached_prompt = _load_from_cache(cache_key)

            if cached_prompt:
                self._prompts_cache[language] = cached_prompt
            else:
                prompt_path = f"prompts/{language}.md"
                try:
                    prompt = self._fetch_file_content(prompt_path)
                    self._prompts_cache[language] = prompt
                    _save_to_cache(cache_key, prompt)
                except RuntimeError as e:
                    raise RuntimeError(f"Failed to fetch prompt for {language}: {e}")

        return self._prompts_cache[language]

    def get_available_languages(self) -> List[str]:
        """Get list of available programming languages."""
        cache_key = _get_cache_key("languages", "all")
        cached_languages = _load_from_cache(cache_key)

        if cached_languages:
            return cached_languages

        root_contents = self._fetch_github_content("")
        languages = [
            item["name"]
            for item in root_contents
            if item["type"] == "dir" and item["name"] in LANGUAGE_CONFIG
        ]
        languages = sorted(languages)
        _save_to_cache(cache_key, languages)
        return languages

    def get_tasks_for_language(self, language: str) -> List[str]:
        """Get list of tasks for a specific programming language."""
        if language not in LANGUAGE_CONFIG:
            raise ValueError(f"Unsupported language: {language}")

        if language not in self._tasks_cache:
            try:
                lang_contents = self._fetch_github_content(language)
                tasks = [
                    item["name"] for item in lang_contents if item["type"] == "dir"
                ]
                self._tasks_cache[language] = sorted(tasks)
            except RuntimeError:
                # Language directory might not exist
                self._tasks_cache[language] = []

        return self._tasks_cache[language]

    def get_task_docs(self, language: str, task_name: str) -> Dict[str, str]:
        """Get documentation files from the docs/ subdirectory of a task."""
        cache_key = _get_cache_key("task_docs", f"{language}_{task_name}")
        cached_docs = _load_from_cache(cache_key)

        if cached_docs:
            return cached_docs

        docs = {}
        try:
            docs_contents = self._fetch_github_content(f"{language}/{task_name}/docs")
            for item in docs_contents:
                if item["type"] == "file" and item["name"].endswith(".md"):
                    file_path = f"{language}/{task_name}/docs/{item['name']}"
                    try:
                        docs[item["name"]] = self._fetch_file_content(file_path)
                    except RuntimeError:
                        continue
        except RuntimeError:
            # docs directory might not exist
            pass

        _save_to_cache(cache_key, docs)
        return docs


def create_roocode_sample(
    language: str, task_name: str, loader: RooCodeTaskLoader
) -> Sample:
    """Convert a Roo-Code task to an inspect_ai Sample."""

    # Get the language-specific prompt
    prompt = loader.get_language_prompt(language)

    task_docs = loader.get_task_docs(language, task_name)

    # Create sample ID
    sample_id = f"{language}/{task_name}"

    # Get language configuration
    lang_config = LANGUAGE_CONFIG[language]

    # Ensure test_command is treated as string for mypy
    test_command: str = cast(str, lang_config["test_command"])

    return Sample(
        id=sample_id,
        input=prompt,
        metadata={
            "language": language,
            "task_name": task_name,
            "test_command": test_command.format(task_name=task_name),
            "setup_commands": lang_config["setup_commands"],
            "task_docs": task_docs,
            "repo_path": f"{language}/{task_name}",
        },
    )


def get_roocode_dataset(
    languages: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Dataset:
    """
    Load Roo-Code-Evals dataset.

    Args:
        languages: List of programming languages to include. If None, includes all.
        tasks: List of specific task names to include. If None, includes all.
        limit: Maximum number of samples to include.

    Returns:
        Dataset: The Roo-Code dataset.
    """
    loader = RooCodeTaskLoader()
    samples = []

    # Determine which languages to process
    available_languages = loader.get_available_languages()
    if languages is None:
        target_languages = available_languages
    else:
        target_languages = [lang for lang in languages if lang in available_languages]
        if not target_languages:
            raise ValueError(
                f"None of the specified languages {languages} are available"
            )

    # Load tasks for each language
    for language in target_languages:
        available_tasks = loader.get_tasks_for_language(language)

        # Filter tasks if specified
        if tasks is not None:
            target_tasks = [task for task in tasks if task in available_tasks]
        else:
            target_tasks = available_tasks

        # Load each task with retry logic
        for task_name in target_tasks:
            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    sample = create_roocode_sample(language, task_name, loader)
                    samples.append(sample)
                    print(f"Loaded {language}/{task_name}")
                    break  # Success, exit retry loop

                except RuntimeError as e:
                    if "rate limit" in str(e).lower() and retry_count < max_retries - 1:
                        wait_time = (retry_count + 1) * 30  # 30, 60, 90 seconds
                        print(
                            f"Rate limit hit for {language}/{task_name}. Waiting {wait_time}s before retry {retry_count + 1}/{max_retries}"
                        )
                        time.sleep(wait_time)
                        retry_count += 1
                    else:
                        break  # Give up on this task

                except Exception as e:
                    print(f"Error loading {language}/{task_name}: {e}")
                    break

            # Check limit
            if limit is not None and len(samples) >= limit:
                break

        # Check limit
        if limit is not None and len(samples) >= limit:
            break

    if not samples:
        raise ValueError("No valid tasks found with the specified criteria")

    return MemoryDataset(samples=samples, name=f"roocode_{len(samples)}_samples")


# Convenience functions for specific languages
def get_roocode_python_dataset(**kwargs) -> Dataset:
    """Get Roo-Code dataset with Python tasks only."""
    return get_roocode_dataset(languages=["python"], **kwargs)


def get_roocode_javascript_dataset(**kwargs) -> Dataset:
    """Get Roo-Code dataset with JavaScript tasks only."""
    return get_roocode_dataset(languages=["javascript"], **kwargs)


def get_roocode_go_dataset(**kwargs) -> Dataset:
    """Get Roo-Code dataset with Go tasks only."""
    return get_roocode_dataset(languages=["go"], **kwargs)


def get_roocode_java_dataset(**kwargs) -> Dataset:
    """Get Roo-Code dataset with Java tasks only."""
    return get_roocode_dataset(languages=["java"], **kwargs)


def get_roocode_rust_dataset(**kwargs) -> Dataset:
    """Get Roo-Code dataset with Rust tasks only."""
    return get_roocode_dataset(languages=["rust"], **kwargs)


# Cache management utilities
def clear_roocode_cache():
    """Clear all cached Roo-Code data."""
    import shutil

    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache directory: {CACHE_DIR}")
    else:
        print("Cache directory does not exist")


def get_cache_info() -> Dict[str, Any]:
    """Get information about the current cache."""
    if not CACHE_DIR.exists():
        return {"cache_dir": str(CACHE_DIR), "exists": False, "files": []}

    cache_files = list(CACHE_DIR.glob("*.pkl"))
    total_size = sum(f.stat().st_size for f in cache_files)

    return {
        "cache_dir": str(CACHE_DIR),
        "exists": True,
        "files": len(cache_files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "files_list": [f.name for f in cache_files],
    }


def print_cache_info():
    """Print cache information."""
    info = get_cache_info()
    print(f"Cache directory: {info['cache_dir']}")
    if info["exists"]:
        print(f"Cache files: {info['files']}")
        print(f"Total size: {info['total_size_mb']} MB")
        if info["files"] > 0:
            print("Cached items:")
            for file in info["files_list"]:
                print(f"  - {file}")
    else:
        print("Cache does not exist yet")
