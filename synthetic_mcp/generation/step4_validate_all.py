#!/usr/bin/env python3
"""
Step 4: Validate All & Integration Tests

This script validates the entire synthetic MCP pipeline:
1. Checks all data files exist
2. Starts the HTTP MCP server
3. Runs integration tests against each tool
4. Validates expected answers match tool outputs

This ensures the synthetic benchmark is ready for use.
"""

import json
import subprocess
import sys
import time
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
SYNTHETIC_MCP_DIR = SCRIPT_DIR.parent
CONFIG_DIR = SYNTHETIC_MCP_DIR / "config"
DATA_DIR = SYNTHETIC_MCP_DIR / "data"
TASKS_DIR = SYNTHETIC_MCP_DIR / "tasks"
SERVER_DIR = SYNTHETIC_MCP_DIR / "server"

SERVER_PORT = 8765


def load_json(path: Path) -> Any:
    """Load JSON from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_data_files() -> tuple[bool, list[str]]:
    """Check all required data files exist."""
    print("\n📂 Checking data files...")
    issues = []

    # Check filesystem data
    files_root = DATA_DIR / "files" / "root"
    required_files = [
        "txt/Android.txt",
        "txt/log_today.txt",
        "txt/log_yesterday.txt",
        "txt/paper_list.bib",
        "csv/customers-100.csv",
        "excel/people_data.xlsx",
        "word/exchange.docx",
        "music/mixkit-retro-game-emergency-alarm-1000.wav",
    ]

    for rel_path in required_files:
        full_path = files_root / rel_path
        if not full_path.exists():
            issues.append(f"Missing file: {rel_path}")
        else:
            print(f"  ✓ {rel_path}")

    # Check API data
    api_dir = DATA_DIR / "api"
    api_files = ["trials.json", "arxiv_papers.json", "maven_versions.json"]

    for api_file in api_files:
        full_path = api_dir / api_file
        if not full_path.exists():
            issues.append(f"Missing API data: {api_file}")
        else:
            print(f"  ✓ api/{api_file}")

    return len(issues) == 0, issues


def start_server() -> subprocess.Popen | None:
    """Start the HTTP MCP server in a subprocess."""
    print(f"\n🚀 Starting HTTP MCP server on port {SERVER_PORT}...")

    server_script = SERVER_DIR / "http_mcp_server.py"
    if not server_script.exists():
        print(f"  ✗ Server script not found: {server_script}")
        return None

    proc = subprocess.Popen(
        [sys.executable, str(server_script), "--port", str(SERVER_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    max_retries = 10
    for i in range(max_retries):
        time.sleep(0.5)
        try:
            conn = HTTPConnection("localhost", SERVER_PORT, timeout=2)
            conn.request("GET", "/")
            response = conn.getresponse()
            if response.status == 200:
                print(f"  ✓ Server started successfully")
                return proc
        except (ConnectionRefusedError, OSError):
            continue

    print(f"  ✗ Server failed to start after {max_retries} attempts")
    proc.kill()
    return None


def call_tool(server: str, tool: str, params: dict) -> dict:
    """Call a tool on the synthetic MCP server."""
    conn = HTTPConnection("localhost", SERVER_PORT, timeout=10)
    body = json.dumps(params)
    conn.request("POST", f"/mcp/{server}/tools/{tool}", body, {"Content-Type": "application/json"})
    response = conn.getresponse()
    data = json.loads(response.read().decode("utf-8"))
    return data


def run_integration_tests() -> tuple[int, int, list[str]]:
    """Run integration tests for all working tasks."""
    print("\n🧪 Running integration tests...")

    tasks = load_json(TASKS_DIR / "progressivemcpbench.json")
    passed = 0
    failed = 0
    failures = []

    test_cases = [
        # (task_id, server, tool, params, expected_check_fn)
        (
            "9a18862e",
            "filesystem",
            "read_file",
            {"path": "/root/txt/Android.txt", "head": 10},
            lambda r: "mVisiblity.getValue is false" in r.get("result", ""),
        ),
        (
            "6b7fc66e",
            "biomcp",
            "trial_protocol_getter",
            {"nct_id": "NCT04280705"},
            lambda r: "Adaptive COVID-19" in str(r.get("result", {})),
        ),
        (
            "31086c2d",
            "filesystem",
            "list_directory",
            {"path": "/root/music"},
            lambda r: "mixkit-retro-game" in r.get("result", ""),
        ),
        (
            "de75630e",
            "maven-deps-server",
            "get_latest_release",
            {"dependency": "com.fasterxml.jackson.core:jackson-databind"},
            lambda r: "2.17.0" in str(r.get("result", {})),
        ),
        (
            "507f592a",
            "filesystem",
            "read_file",
            {"path": "/root/csv/customers-100.csv"},
            lambda r: "mariokhan@ryan-pope.org" in r.get("result", ""),
        ),
        (
            "2e190c2a",
            "filesystem",
            "read_multiple_files",
            {"paths": ["/root/txt/log_today.txt", "/root/txt/log_yesterday.txt"]},
            lambda r: "ERROR" in str(r.get("result", {})),
        ),
        (
            "71a7a42a",
            "filesystem",
            "read_file",
            {"path": "/root/txt/paper_list.bib"},
            lambda r: "WebSRC" in r.get("result", ""),
        ),
        (
            "4aa079dd",
            "arxiv-mcp-server",
            "search_papers",
            {"query": "webarena mind2web"},
            lambda r: "Deng" in str(r.get("result", [])) and "Gou" in str(r.get("result", [])),
        ),
        (
            "99d7541a",
            "music-analysis",
            "tempo",
            {"path_audio_time_series_y": "/root/music/mixkit-retro-game-emergency-alarm-1000.wav"},
            lambda r: r.get("result") == 60.09,
        ),
        (
            "086e2393",
            "word-document-server",
            "get_document_text",
            {"filename": "/root/word/exchange.docx"},
            lambda r: "Everyone gave a gift" in str(r.get("result", "")),
        ),
    ]

    for task_id, server, tool, params, check_fn in test_cases:
        try:
            result = call_tool(server, tool, params)

            if "error" in result and result["error"]:
                failed += 1
                failures.append(f"{task_id}: {server}/{tool} returned error: {result['error']}")
                print(f"  ✗ {task_id}: {server}/{tool} - ERROR: {result['error']}")
            elif check_fn(result):
                passed += 1
                print(f"  ✓ {task_id}: {server}/{tool}")
            else:
                failed += 1
                failures.append(f"{task_id}: {server}/{tool} check failed. Got: {str(result)[:100]}")
                print(f"  ✗ {task_id}: {server}/{tool} - CHECK FAILED")
        except Exception as e:
            failed += 1
            failures.append(f"{task_id}: {server}/{tool} exception: {e!s}")
            print(f"  ✗ {task_id}: {server}/{tool} - EXCEPTION: {e}")

    return passed, failed, failures


def validate_excel_handler() -> tuple[bool, str]:
    """Test the Excel handler specifically."""
    print("\n📊 Testing Excel handler...")

    try:
        # Test describing sheets
        result = call_tool("excel", "excel_describe_sheets", {"fileAbsolutePath": "/root/excel/people_data.xlsx"})

        if "error" in result:
            return False, f"Error: {result['error']}"

        sheets = result.get("result", {}).get("sheets", [])
        if not sheets:
            return False, "No sheets returned"

        print(f"  ✓ excel_describe_sheets: {len(sheets)} sheets found")

        # Test reading sheet
        result = call_tool(
            "excel", "excel_read_sheet", {"fileAbsolutePath": "/root/excel/people_data.xlsx", "sheetName": sheets[0]["name"]}
        )

        if "error" in result:
            return False, f"Error reading sheet: {result['error']}"

        data = result.get("result", {}).get("data", [])

        # Find Sophia Moore
        sophia_data = None
        for row in data:
            if "Sophia Moore" in str(row.get("Name", "")):
                sophia_data = row
                break

        if sophia_data:
            # Calculate BMI
            height_cm = sophia_data.get("Height (cm)", 0)
            weight_kg = sophia_data.get("Weight (kg)", 0)
            if height_cm and weight_kg:
                height_m = height_cm / 100
                bmi = weight_kg / (height_m * height_m)
                print(f"  ✓ Found Sophia Moore: BMI = {bmi:.2f}")
                if abs(bmi - 20.76) < 0.1:
                    return True, "BMI calculation correct"
                return False, f"BMI mismatch: expected 20.76, got {bmi:.2f}"

        return False, "Sophia Moore not found in data"

    except Exception as e:
        return False, f"Exception: {e!s}"


def main():
    print("=" * 60)
    print("Step 4: Validate All & Integration Tests")
    print("=" * 60)

    # Check data files
    data_ok, data_issues = check_data_files()
    if not data_ok:
        print(f"\n⚠ Data file issues:")
        for issue in data_issues:
            print(f"  - {issue}")
        print("\nRun step2_generate_data.py first!")
        return 1

    # Start server
    server_proc = start_server()
    if not server_proc:
        print("\n⚠ Could not start server")
        return 1

    try:
        # Run integration tests
        passed, failed, failures = run_integration_tests()

        # Test Excel handler
        excel_ok, excel_msg = validate_excel_handler()
        if excel_ok:
            passed += 1
            print(f"  ✓ Excel BMI test: {excel_msg}")
        else:
            failed += 1
            failures.append(f"Excel BMI test: {excel_msg}")
            print(f"  ✗ Excel BMI test: {excel_msg}")

        # Print summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Tests passed: {passed}")
        print(f"  Tests failed: {failed}")

        if failures:
            print(f"\n  Failures:")
            for f in failures:
                print(f"    - {f}")

        if failed == 0:
            print("\n✅ All tests passed! Synthetic MCP benchmark is ready.")
            return 0
        else:
            print(f"\n⚠ {failed} tests failed. Review the failures above.")
            return 1

    finally:
        # Stop server
        print("\n🛑 Stopping server...")
        server_proc.terminate()
        server_proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
