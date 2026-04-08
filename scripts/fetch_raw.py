#!/usr/bin/env python3
"""
fetch_raw.py — Pull source documents from m9h repos into raw/projects/.

Fetches READMEs, research docs, manuscript files, and CLAUDE.md from each
repo into raw/projects/{repo_name}/, preserving directory structure.
These become the immutable source documents for wiki distillation.

Usage:
    cd ~/dev/neuro-kb
    python3 scripts/fetch_raw.py                    # all repos
    python3 scripts/fetch_raw.py --repos neurojax sbi4dwi
    python3 scripts/fetch_raw.py --local            # copy from ~/dev/
    python3 scripts/fetch_raw.py --refresh           # overwrite existing
"""

import argparse
import base64
import fnmatch
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

GITHUB_ORG = "m9h"
SINCE_DATE = "2025-01-01"
RAW_DIR = Path(__file__).resolve().parent.parent / "raw" / "projects"
LOCAL_DEV = Path.home() / "dev"

# Files to fetch from each repo
FETCH_PATTERNS = [
    "README.md",
    "CLAUDE.md",
    "CITATION.bib",
    "CITATION.cff",
    "docs/*.md",
    "docs/**/*.md",
    "research/*.md",
    "paper/*.md",
    "paper/*.bib",
    "paper/*.tex",
    "manuscript/*.md",
    "manuscript/*.bib",
    "autoresearch/*.md",
    "examples/README.md",
]

# Skip large/irrelevant files
SKIP_PATTERNS = [
    "docs/reference/*.rst",  # API reference stubs
    "*/node_modules/*",
    "*/.git/*",
    "*/cache/*",
]


def gh_api(endpoint: str) -> str | None:
    try:
        result = subprocess.run(
            ["gh", "api", endpoint],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def list_repos(org: str, since: str) -> list[dict]:
    raw = gh_api(f"users/{org}/repos?per_page=100&sort=created&direction=desc")
    if not raw:
        return []
    repos = json.loads(raw)
    return [r for r in repos if r.get("created_at", "") >= since]


def list_repo_files(repo: str) -> list[str]:
    for branch in ["main", "master"]:
        raw = gh_api(f"repos/{repo}/git/trees/{branch}?recursive=1")
        if raw:
            try:
                data = json.loads(raw)
                return [item["path"] for item in data.get("tree", [])
                        if item.get("type") == "blob"]
            except (json.JSONDecodeError, KeyError):
                pass
    return []


def match_patterns(files: list[str]) -> list[str]:
    matched = set()
    for pattern in FETCH_PATTERNS:
        for f in files:
            if fnmatch.fnmatch(f, pattern):
                skip = False
                for sp in SKIP_PATTERNS:
                    if fnmatch.fnmatch(f, sp):
                        skip = True
                        break
                if not skip:
                    matched.add(f)
    return sorted(matched)


def fetch_file_from_github(repo: str, path: str) -> bytes | None:
    raw = gh_api(f"repos/{repo}/contents/{path}")
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "content" in data:
            return base64.b64decode(data["content"])
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def fetch_from_github(repos: list[str] | None, refresh: bool):
    if repos:
        repo_list = [{"name": r} for r in repos]
    else:
        print(f"Fetching {GITHUB_ORG} repo list...")
        repo_list = list_repos(GITHUB_ORG, SINCE_DATE)

    print(f"Processing {len(repo_list)} repos...\n")

    total_files = 0
    for repo_info in repo_list:
        name = repo_info["name"]
        full_name = f"{GITHUB_ORG}/{name}"
        dest = RAW_DIR / name

        if dest.exists() and not refresh:
            print(f"  {name}: already fetched (use --refresh to overwrite)")
            continue

        all_files = list_repo_files(full_name)
        to_fetch = match_patterns(all_files)

        if not to_fetch:
            continue

        dest.mkdir(parents=True, exist_ok=True)
        fetched = 0
        for filepath in to_fetch:
            content = fetch_file_from_github(full_name, filepath)
            if content:
                out = dest / filepath
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(content)
                fetched += 1

        if fetched > 0:
            print(f"  {name}: {fetched} files")
            total_files += fetched

        time.sleep(0.3)

    print(f"\nTotal: {total_files} files fetched to {RAW_DIR}/")


def fetch_from_local(repos: list[str] | None, refresh: bool):
    projects = sorted(
        p.name for p in LOCAL_DEV.iterdir()
        if p.is_dir() and (p / ".git").exists()
    )
    if repos:
        projects = [p for p in projects if p in repos]

    print(f"Copying from {len(projects)} local projects...\n")

    total_files = 0
    for name in projects:
        src_dir = LOCAL_DEV / name
        dest = RAW_DIR / name

        if dest.exists() and not refresh:
            continue

        # Find matching files locally
        all_files = []
        for f in src_dir.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(src_dir))
                all_files.append(rel)

        to_copy = match_patterns(all_files)
        if not to_copy:
            continue

        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for filepath in to_copy:
            src = src_dir / filepath
            out = dest / filepath
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            copied += 1

        if copied > 0:
            print(f"  {name}: {copied} files")
            total_files += copied

    print(f"\nTotal: {total_files} files copied to {RAW_DIR}/")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch source documents from repos into raw/projects/"
    )
    parser.add_argument("--repos", nargs="+", help="Specific repos")
    parser.add_argument("--local", action="store_true",
                        help="Copy from ~/dev/ instead of GitHub")
    parser.add_argument("--refresh", action="store_true",
                        help="Overwrite existing fetched content")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.local:
        fetch_from_local(args.repos, args.refresh)
    else:
        fetch_from_github(args.repos, args.refresh)


if __name__ == "__main__":
    main()
