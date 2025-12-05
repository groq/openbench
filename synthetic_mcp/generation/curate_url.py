#!/usr/bin/env python3
"""
URL Curation Script for Synthetic MCP Web Corpus

This script helps curate URLs for the synthetic web corpus by:
1. Crawling a real URL and saving the HTML snapshot
2. Extracting/validating the answer for a task
3. Updating metadata.json with the new URL
4. Updating search_index.json with discoverable search terms

Usage:
    python curate_url.py --url "https://example.com/page" \
        --task-id "uuid-of-task" \
        --task-prompt "What is the answer from this page?" \
        [--answer "Known answer"]

Or for interactive mode:
    python curate_url.py --interactive
"""

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
SYNTHETIC_MCP_DIR = SCRIPT_DIR.parent
DATA_DIR = SYNTHETIC_MCP_DIR / "data"
WEB_DIR = DATA_DIR / "web"
HTML_DIR = WEB_DIR / "html"
METADATA_PATH = WEB_DIR / "metadata.json"
SEARCH_INDEX_PATH = WEB_DIR / "search_index.json"

# Path to progressivemcpbench.json
BENCHMARK_DATA_PATH = (
    SYNTHETIC_MCP_DIR.parent / "src" / "openbench" / "datasets" / "data" / "progressivemcpbench.json"
)


def load_json(path: Path) -> Any:
    """Load JSON from file."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """Save JSON to file with pretty printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: {path}")


def slugify_url(url: str) -> str:
    """Generate a filesystem-safe slug from a URL."""
    parsed = urlparse(url)

    # Get domain without www
    domain = parsed.netloc
    if domain.startswith("www."):
        domain = domain[4:]

    # Get path, replacing / with _
    path = parsed.path.strip("/")
    if path:
        path = re.sub(r"[^\w\-]", "_", path)
        path = re.sub(r"_+", "_", path)  # Collapse multiple underscores

    # Combine
    if path:
        slug = f"{domain}_{path}"
    else:
        slug = domain

    # Truncate if too long
    if len(slug) > 100:
        slug = slug[:100]

    return slug


def fetch_html(url: str) -> str:
    """Fetch HTML content from a URL."""
    try:
        import requests
    except ImportError:
        print("Error: requests library not installed. Run: pip install requests")
        sys.exit(1)

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    print(f"  Fetching: {url}")
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    # Try to detect encoding
    if response.encoding:
        html = response.text
    else:
        html = response.content.decode("utf-8", errors="replace")

    print(f"  ✓ Fetched {len(html):,} bytes")
    return html


def extract_title_from_html(html: str) -> str:
    """Extract title from HTML content."""
    # Simple regex extraction
    match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Untitled"


def generate_search_terms(title: str, task_prompt: str, url: str) -> dict:
    """Generate search index entry from available information."""
    parsed = urlparse(url)
    domain = parsed.netloc

    # Extract keywords from title and prompt
    all_text = f"{title} {task_prompt}".lower()

    # Remove common words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did", "will", "would", "could",
                  "should", "may", "might", "must", "shall", "can", "need", "dare",
                  "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
                  "into", "through", "during", "before", "after", "above", "below",
                  "between", "under", "again", "further", "then", "once", "here",
                  "there", "when", "where", "why", "how", "all", "each", "few",
                  "more", "most", "other", "some", "such", "no", "nor", "not",
                  "only", "own", "same", "so", "than", "too", "very", "just",
                  "and", "but", "if", "or", "because", "until", "while", "what",
                  "which", "who", "whom", "this", "that", "these", "those", "am",
                  "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
                  "your", "yours", "yourself", "yourselves", "he", "him", "his",
                  "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                  "they", "them", "their", "theirs", "themselves", "i", "help",
                  "find", "tell", "return", "give", "please"}

    # Extract words
    words = re.findall(r"\b[a-z]{3,}\b", all_text)
    tags = list(set(w for w in words if w not in stop_words))[:10]

    return {
        "title": title,
        "short_description": f"Web page from {domain}",
        "tags": tags,
        "example_queries": [task_prompt] if task_prompt else [],
    }


def add_url_to_corpus(
    url: str,
    html_content: str,
    task_id: str | None = None,
    title: str | None = None,
    tags: list[str] | None = None,
    example_queries: list[str] | None = None,
    short_description: str | None = None,
) -> str:
    """Add a URL to the web corpus.

    Returns the generated page ID.
    """
    # Generate slug and paths
    slug = slugify_url(url)
    html_filename = f"{slug}.html"
    html_path = HTML_DIR / html_filename

    # Save HTML
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  ✓ Saved HTML: {html_path}")

    # Extract title if not provided
    if not title:
        title = extract_title_from_html(html_content)

    # Generate page ID
    page_id = slug.replace(".", "_").replace("-", "_")

    # Update metadata.json
    metadata = load_json(METADATA_PATH)
    metadata[url] = {
        "id": page_id,
        "html_path": f"web/html/{html_filename}",
        "title": title,
        "domain": urlparse(url).netloc,
        "tags": tags or [],
        "task_ids": [task_id] if task_id else [],
    }
    save_json(METADATA_PATH, metadata)

    # Update search_index.json
    search_index = load_json(SEARCH_INDEX_PATH)
    if "entries" not in search_index:
        search_index["entries"] = []

    # Remove existing entry for this URL if present
    search_index["entries"] = [
        e for e in search_index["entries"] if e.get("url") != url
    ]

    # Add new entry
    search_index["entries"].append({
        "id": page_id,
        "url": url,
        "title": title,
        "short_description": short_description or f"Web page from {urlparse(url).netloc}",
        "tags": tags or [],
        "example_queries": example_queries or [],
    })
    save_json(SEARCH_INDEX_PATH, search_index)

    return page_id


def update_task_answer(task_id: str, answer: str) -> bool:
    """Update a task's answer in progressivemcpbench.json.

    Moves _answer to answer if present, or sets answer directly.
    """
    if not BENCHMARK_DATA_PATH.exists():
        print(f"  ⚠ Benchmark file not found: {BENCHMARK_DATA_PATH}")
        return False

    tasks = load_json(BENCHMARK_DATA_PATH)
    if not isinstance(tasks, list):
        print("  ⚠ Benchmark file is not a list")
        return False

    # Find the task
    task_found = False
    for task in tasks:
        if task.get("task_id") == task_id:
            task_found = True

            # Move _answer to answer if present
            if "_answer" in task:
                task["answer"] = task.pop("_answer")
                print(f"  ✓ Moved _answer to answer for task {task_id}")
            elif answer:
                task["answer"] = answer
                print(f"  ✓ Set answer for task {task_id}")

            # Ensure playwright is in required_servers
            if "playwright" not in task.get("required_servers", []):
                task.setdefault("required_servers", []).append("playwright")

            # Ensure playwright tool is in required_tools
            if "playwright_get_visible_html" not in task.get("required_tools", []):
                task.setdefault("required_tools", []).append(
                    "playwright_get_visible_html"
                )

            break

    if not task_found:
        print(f"  ⚠ Task not found: {task_id}")
        return False

    save_json(BENCHMARK_DATA_PATH, tasks)
    return True


def create_new_task(
    task_prompt: str,
    answer: str,
    category: str = "Lifestyle",
) -> str:
    """Create a new task in progressivemcpbench.json.

    Returns the new task ID.
    """
    task_id = str(uuid.uuid4())

    new_task = {
        "task_id": task_id,
        "Question": task_prompt,
        "category": category,
        "file_name": "",
        "Annotator Metadata": {
            "Number of steps": "2",
            "Number of tools": "2",
            "Steps": "1. Look up the relevant URL\n2. Fetch the page and extract the answer",
            "Tools": "domain-specific lookup, playwright_get_visible_html",
        },
        "answer": answer,
        "scorer_instructions": None,
        "required_servers": ["playwright"],
        "required_tools": ["playwright_get_visible_html"],
    }

    tasks = load_json(BENCHMARK_DATA_PATH)
    if not isinstance(tasks, list):
        tasks = []

    tasks.append(new_task)
    save_json(BENCHMARK_DATA_PATH, tasks)

    print(f"  ✓ Created new task: {task_id}")
    return task_id


def curate_url(
    url: str,
    task_id: str | None = None,
    task_prompt: str | None = None,
    answer: str | None = None,
    create_task: bool = False,
    category: str = "Lifestyle",
) -> None:
    """Main curation workflow."""
    print(f"\n{'='*60}")
    print(f"Curating URL: {url}")
    print(f"{'='*60}")

    # Step 1: Fetch HTML
    print("\n📥 Step 1: Fetching HTML...")
    html_content = fetch_html(url)

    # Step 2: Extract metadata
    print("\n📝 Step 2: Extracting metadata...")
    title = extract_title_from_html(html_content)
    print(f"  Title: {title}")

    # Generate search terms
    search_info = generate_search_terms(title, task_prompt or "", url)

    # Step 3: Add to corpus
    print("\n📦 Step 3: Adding to web corpus...")
    page_id = add_url_to_corpus(
        url=url,
        html_content=html_content,
        task_id=task_id,
        title=title,
        tags=search_info["tags"],
        example_queries=search_info["example_queries"],
        short_description=search_info["short_description"],
    )
    print(f"  Page ID: {page_id}")

    # Step 4: Update/create task
    if task_id:
        print("\n📋 Step 4: Updating task...")
        update_task_answer(task_id, answer or "")
    elif create_task and task_prompt and answer:
        print("\n📋 Step 4: Creating new task...")
        task_id = create_new_task(task_prompt, answer, category)

    print(f"\n{'='*60}")
    print("✅ Curation complete!")
    print(f"{'='*60}")
    print(f"\nURL: {url}")
    print(f"Page ID: {page_id}")
    if task_id:
        print(f"Task ID: {task_id}")
    print(f"\nHTML saved to: {HTML_DIR / slugify_url(url)}.html")


def main():
    parser = argparse.ArgumentParser(
        description="Curate URLs for the synthetic MCP web corpus"
    )
    parser.add_argument("--url", type=str, help="URL to curate")
    parser.add_argument("--task-id", type=str, help="Task ID to associate with this URL")
    parser.add_argument("--task-prompt", type=str, help="Task prompt/question")
    parser.add_argument("--answer", type=str, help="Known answer for the task")
    parser.add_argument("--create-task", action="store_true", help="Create a new task")
    parser.add_argument("--category", type=str, default="Lifestyle", help="Task category")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    if args.interactive:
        print("Interactive mode not yet implemented")
        return

    if not args.url:
        parser.print_help()
        sys.exit(1)

    curate_url(
        url=args.url,
        task_id=args.task_id,
        task_prompt=args.task_prompt,
        answer=args.answer,
        create_task=args.create_task,
        category=args.category,
    )


if __name__ == "__main__":
    main()
