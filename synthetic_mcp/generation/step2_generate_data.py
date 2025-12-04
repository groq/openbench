#!/usr/bin/env python3
"""
Step 2: Generate Synthetic Test Data

This script generates all the test data files needed for the synthetic MCP benchmark.
It copies/adapts real data from LiveMCPBench where available and generates synthetic
data for API-style lookups.

Data sources:
- Filesystem files: Copy from ~/.openbench/progressivemcpbench/copilot/raw/annotated_data/
- API lookups: Generate JSON tables for clinical trials, arxiv papers, maven versions
- Excel files: Copy from annotated_data

Output directories:
- synthetic_mcp/data/files/root/  - Filesystem data (txt, csv, pdf, word, music, excel)
- synthetic_mcp/data/api/         - JSON lookup tables
"""

import json
import shutil
from pathlib import Path
from typing import Any

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
SYNTHETIC_MCP_DIR = SCRIPT_DIR.parent
DATA_DIR = SYNTHETIC_MCP_DIR / "data"
CONFIG_DIR = SYNTHETIC_MCP_DIR / "config"

# LiveMCPBench annotated data
ANNOTATED_DATA_DIR = Path.home() / ".openbench/progressivemcpbench/copilot/raw/annotated_data"


def load_json(path: Path) -> Any:
    """Load JSON from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """Save JSON to file with pretty printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: {path}")


def copy_filesystem_data() -> None:
    """Copy filesystem test data from LiveMCPBench annotated_data."""
    print("\n📁 Copying filesystem data from LiveMCPBench...")

    files_root = DATA_DIR / "files" / "root"

    # Mapping of source directories to destination
    copy_mappings = [
        ("txt", "txt"),
        ("csv", "csv"),
        ("music", "music"),
        ("word", "word"),
        ("excel", "excel"),
        ("pdf", "pdf"),
    ]

    for src_subdir, dst_subdir in copy_mappings:
        src_dir = ANNOTATED_DATA_DIR / src_subdir
        dst_dir = files_root / dst_subdir

        if not src_dir.exists():
            print(f"  ⚠ Source not found: {src_dir}")
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)

        # Copy files recursively
        for src_file in src_dir.rglob("*"):
            if src_file.is_file():
                # Compute relative path to preserve subdirectories
                rel_path = src_file.relative_to(src_dir)
                dst_file = dst_dir / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                print(f"    ✓ {dst_subdir}/{rel_path}")


def generate_clinical_trials_data() -> None:
    """Generate JSON lookup table for clinical trials (biomcp)."""
    print("\n🏥 Generating clinical trials data...")

    trials = {
        "NCT04280705": {
            "nct_id": "NCT04280705",
            "brief_title": "Adaptive COVID-19 Treatment Trial (ACTT)",
            "official_title": "A Multicenter, Adaptive, Randomized Blinded Controlled Trial of the Safety and Efficacy of Investigational Therapeutics for the Treatment of COVID-19 in Hospitalized Adults",
            "status": "COMPLETED",
            "phase": "PHASE3",
            "sponsor": "National Institute of Allergy and Infectious Diseases (NIAID)",
            "conditions": ["COVID-19"],
            "interventions": ["Remdesivir", "Placebo"],
            "enrollment": 1062,
            "start_date": "2020-02-21",
            "completion_date": "2020-04-19",
            "eligibility": {
                "minimum_age": "18 Years",
                "sex": "ALL",
                "healthy_volunteers": "No",
            },
        },
        "NCT06524388": {
            "nct_id": "NCT06524388",
            "brief_title": "Study of Novel Cancer Immunotherapy",
            "official_title": "A Phase 2 Randomized Study of Novel Checkpoint Inhibitor Combination in Advanced Solid Tumors",
            "status": "RECRUITING",
            "phase": "PHASE2",
            "sponsor": "Example Pharmaceuticals",
            "conditions": ["Solid Tumors", "Advanced Cancer"],
            "interventions": ["Drug A", "Drug B", "Placebo"],
            "enrollment": 450,
            "start_date": "2024-01-15",
            "eligibility": {
                "minimum_age": "18 Years",
                "maximum_age": "75 Years",
                "sex": "ALL",
                "healthy_volunteers": "No",
            },
        },
    }

    save_json(DATA_DIR / "api" / "trials.json", trials)


def generate_arxiv_papers_data() -> None:
    """Generate JSON lookup table for arxiv papers (mcp-simple-arxiv)."""
    print("\n📚 Generating arxiv papers data...")

    papers = {
        "2203.12277": {
            "paper_id": "2203.12277",
            "title": "Example Paper on Machine Learning",
            "authors": ["John Smith", "Jane Doe"],
            "abstract": "This paper presents a novel approach to machine learning...",
            "published": "2022-03-23",
            "updated": "2022-03-25",
            "categories": ["cs.LG", "cs.AI"],
            "pdf_url": "https://arxiv.org/pdf/2203.12277",
        },
        "2301.00001": {
            "paper_id": "2301.00001",
            "title": "WebArena: A Realistic Web Environment for Building Autonomous Agents",
            "authors": ["Xiang Deng", "Yu Gu", "Boyuan Zheng", "Shijie Chen"],
            "abstract": "WebArena is a realistic web environment for building autonomous agents...",
            "published": "2023-01-15",
            "updated": "2023-01-20",
            "categories": ["cs.CL", "cs.AI"],
            "pdf_url": "https://arxiv.org/pdf/2301.00001",
        },
        "2305.00002": {
            "paper_id": "2305.00002",
            "title": "Mind2Web: Towards a Generalist Agent for the Web",
            "authors": ["Yuan Gou", "Boyuan Zheng", "Zhuosheng Zhang"],
            "abstract": "Mind2Web is a dataset for developing generalist web agents...",
            "published": "2023-05-20",
            "updated": "2023-05-25",
            "categories": ["cs.CL", "cs.AI"],
            "pdf_url": "https://arxiv.org/pdf/2305.00002",
        },
        "2402.00003": {
            "paper_id": "2402.00003",
            "title": "Mind2Web2: Scaling Web Agent Development",
            "authors": ["Wei Zhou", "Alice Wang", "Bob Chen"],
            "abstract": "Mind2Web2 extends the original Mind2Web dataset...",
            "published": "2024-02-10",
            "updated": "2024-02-15",
            "categories": ["cs.CL", "cs.AI"],
            "pdf_url": "https://arxiv.org/pdf/2402.00003",
        },
    }

    # Also create paper_metadata.json for the get_paper_data tool
    save_json(DATA_DIR / "api" / "paper_metadata.json", papers)
    save_json(DATA_DIR / "api" / "arxiv_papers.json", papers)


def generate_maven_versions_data() -> None:
    """Generate JSON lookup table for maven dependency versions."""
    print("\n📦 Generating maven versions data...")

    # The task asks about these specific dependencies being outdated
    maven_versions = {
        "com.fasterxml.jackson.core:jackson-databind": {
            "dependency": "com.fasterxml.jackson.core:jackson-databind",
            "latest_version": "2.17.0",
            "release_date": "2024-03-15",
            "description": "General data-binding functionality for Jackson",
        },
        "org.apache.poi:poi-ooxml": {
            "dependency": "org.apache.poi:poi-ooxml",
            "latest_version": "5.2.5",
            "release_date": "2024-01-20",
            "description": "Apache POI - Java API for Microsoft Documents",
        },
        "com.google.guava:guava": {
            "dependency": "com.google.guava:guava",
            "latest_version": "33.0.0-jre",
            "release_date": "2024-02-01",
            "description": "Google Core Libraries for Java",
        },
        # Also support full coordinate format
        "com.fasterxml.jackson.core:jackson-databind:2.13.4": {
            "dependency": "com.fasterxml.jackson.core:jackson-databind:2.13.4",
            "latest_version": "2.17.0",
            "release_date": "2024-03-15",
            "description": "General data-binding functionality for Jackson",
            "queried_version": "2.13.4",
            "is_outdated": True,
        },
        "org.apache.poi:poi-ooxml:5.2.0": {
            "dependency": "org.apache.poi:poi-ooxml:5.2.0",
            "latest_version": "5.2.5",
            "release_date": "2024-01-20",
            "description": "Apache POI - Java API for Microsoft Documents",
            "queried_version": "5.2.0",
            "is_outdated": True,
        },
        "com.google.guava:guava:30.1-jre": {
            "dependency": "com.google.guava:guava:30.1-jre",
            "latest_version": "33.0.0-jre",
            "release_date": "2024-02-01",
            "description": "Google Core Libraries for Java",
            "queried_version": "30.1-jre",
            "is_outdated": True,
        },
    }

    save_json(DATA_DIR / "api" / "maven_versions.json", maven_versions)


def generate_audio_analysis_data() -> None:
    """Generate lookup data for music analysis results."""
    print("\n🎵 Generating audio analysis data...")

    audio_files = {
        "/root/music/mixkit-retro-game-emergency-alarm-1000.wav": {
            "file_path": "/root/music/mixkit-retro-game-emergency-alarm-1000.wav",
            "duration_seconds": 120.0,
            "sample_rate": 44100,
            "tempo_bpm": 60.09,
            "beats": [],
        },
    }

    save_json(DATA_DIR / "api" / "audio_files.json", audio_files)


def create_pdf_metadata() -> None:
    """Create metadata for PDF files that will be served by the synthetic server."""
    print("\n📄 Creating PDF metadata...")

    # Metadata for the embodied AI papers (based on filenames)
    pdf_metadata = {
        "/root/pdf/embodied_ai_papers/RT-2, Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, Anthony Brohan et al., 2023, v1_compressed.pdf": {
            "title": "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control",
            "authors": ["Anthony Brohan", "Noah Brown", "Justice Carbajal"],
            "first_author_last_name": "Brohan",
            "year": "2023",
            "page_count": 22,
        },
        "/root/pdf/embodied_ai_papers/PaLM-E, An Embodied Multimodal Language Model, Danny Driess et al., 2023, v1_compressed.pdf": {
            "title": "PaLM-E: An Embodied Multimodal Language Model",
            "authors": ["Danny Driess", "Fei Xia", "Mehdi S. M. Sajjadi"],
            "first_author_last_name": "Driess",
            "year": "2023",
            "page_count": 27,
        },
        "/root/pdf/embodied_ai_papers/Voyager, An Open-Ended Embodied Agent with Large Language Models, Guanzhi Wang et al., 2023, v2_compressed.pdf": {
            "title": "Voyager: An Open-Ended Embodied Agent with Large Language Models",
            "authors": ["Guanzhi Wang", "Yuqi Xie", "Yunfan Jiang"],
            "first_author_last_name": "Wang",
            "year": "2023",
            "page_count": 18,
        },
    }

    save_json(DATA_DIR / "api" / "pdf_metadata.json", pdf_metadata)


def verify_data_integrity() -> None:
    """Verify all required data files exist and have expected content."""
    print("\n🔍 Verifying data integrity...")

    # Check filesystem files
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

    missing = []
    for rel_path in required_files:
        full_path = files_root / rel_path
        if not full_path.exists():
            missing.append(rel_path)
        else:
            print(f"    ✓ {rel_path}")

    if missing:
        print(f"\n  ⚠ Missing files: {missing}")
    else:
        print("\n  ✓ All required filesystem files present")

    # Check API data files
    api_dir = DATA_DIR / "api"
    api_files = [
        "trials.json",
        "arxiv_papers.json",
        "paper_metadata.json",
        "maven_versions.json",
        "audio_files.json",
        "pdf_metadata.json",
    ]

    for api_file in api_files:
        full_path = api_dir / api_file
        if full_path.exists():
            data = load_json(full_path)
            print(f"    ✓ {api_file}: {len(data)} entries")
        else:
            print(f"    ✗ {api_file}: NOT FOUND")


def main():
    print("=" * 60)
    print("Step 2: Generate Synthetic Test Data")
    print("=" * 60)

    # Check for LiveMCPBench data
    if not ANNOTATED_DATA_DIR.exists():
        print(f"\n⚠ LiveMCPBench annotated data not found at: {ANNOTATED_DATA_DIR}")
        print("Run the ProgressiveMCPBench eval first to populate the cache.")
        return

    # Create data directories
    print("\n📂 Creating data directories...")
    (DATA_DIR / "files" / "root").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "api").mkdir(parents=True, exist_ok=True)

    # Copy filesystem data
    copy_filesystem_data()

    # Generate API lookup tables
    generate_clinical_trials_data()
    generate_arxiv_papers_data()
    generate_maven_versions_data()
    generate_audio_analysis_data()
    create_pdf_metadata()

    # Verify everything
    verify_data_integrity()

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Data directory: {DATA_DIR}")
    print("  Filesystem data: data/files/root/")
    print("  API data: data/api/")

    print("\n✅ Step 2 complete!")
    print("   Next: Run step3_generate_tasks.py")


if __name__ == "__main__":
    main()
