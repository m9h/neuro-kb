#!/usr/bin/env python3
"""
ingest_refs.py — Crawl m9h GitHub repos and extract references into neuro-kb.

Scans READMEs, research docs, and .bib files across all m9h repos.
Extracts DOIs, arXiv IDs, and existing BibTeX entries.
Resolves DOIs via CrossRef and arXiv IDs via arXiv API into BibTeX.
Deduplicates against the master references.bib and appends new entries.

Usage:
    cd ~/dev/neuro-kb
    uv run scripts/ingest_refs.py                  # all repos since 2025-01-01
    uv run scripts/ingest_refs.py --repos neurojax sbi4dwi  # specific repos
    uv run scripts/ingest_refs.py --dry-run         # show what would be added
    uv run scripts/ingest_refs.py --local           # scan ~/dev/ instead of GitHub
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GITHUB_ORG = "m9h"
SINCE_DATE = "2025-01-01"
MASTER_BIB = Path(__file__).resolve().parent.parent / "references.bib"
LOCAL_DEV = Path.home() / "dev"

# File patterns to scan for references
SCAN_GLOBS = [
    "README.md",
    "docs/*.md",
    "docs/**/*.md",
    "research/*.md",
    "paper/*.bib",
    "paper/references.bib",
    "manuscript/*.bib",
    "manuscript/references.bib",
    "docs/references.bib",
    "CITATION.bib",
    "CITATION.cff",
]

# Regex patterns for extracting references
DOI_PATTERN = re.compile(
    r'(?:doi(?:\.org)?[:/]\s*|DOI[:/]\s*|https?://doi\.org/)'
    r'(10\.\d{4,9}/[^\s,;)\]}"\']+)',
    re.IGNORECASE,
)
ARXIV_PATTERN = re.compile(
    r'(?:arxiv(?:\.org/abs)?[:/]\s*|arXiv:\s*)'
    r'(\d{4}\.\d{4,5}(?:v\d+)?)',
    re.IGNORECASE,
)
BIBTEX_ENTRY_PATTERN = re.compile(
    r'(@\w+\{([^,]+),.*?\n\})', re.DOTALL
)


# ---------------------------------------------------------------------------
# BibTeX parsing
# ---------------------------------------------------------------------------
def parse_bib_keys(text: str) -> set[str]:
    """Extract all BibTeX keys from text."""
    return set(re.findall(r'@\w+\{([^,]+),', text))


def parse_bib_entries(text: str) -> dict[str, str]:
    """Extract bib entries as {key: full_text}."""
    entries = {}
    for match in BIBTEX_ENTRY_PATTERN.finditer(text):
        full = match.group(1)
        key = match.group(2).strip()
        if key not in entries:
            entries[key] = full
    return entries


def extract_dois_from_bib(text: str) -> dict[str, str]:
    """Map DOIs to their bib keys from existing entries."""
    doi_to_key = {}
    for match in BIBTEX_ENTRY_PATTERN.finditer(text):
        entry = match.group(1)
        key = match.group(2).strip()
        doi_match = re.search(r'doi\s*=\s*\{([^}]+)\}', entry, re.IGNORECASE)
        if doi_match:
            doi = doi_match.group(1).strip().lower()
            doi = re.sub(r'^https?://doi\.org/', '', doi)
            doi_to_key[doi] = key
    return doi_to_key


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------
def gh_api(endpoint: str) -> str | None:
    """Call gh CLI API and return stdout, or None on failure."""
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
    """List repos created since a date."""
    raw = gh_api(
        f"users/{org}/repos?per_page=100&sort=created&direction=desc"
    )
    if not raw:
        return []
    repos = json.loads(raw)
    return [r for r in repos if r.get("created_at", "") >= since]


def fetch_file_from_github(repo: str, path: str) -> str | None:
    """Fetch a single file's content from GitHub."""
    raw = gh_api(f"repos/{repo}/contents/{path}")
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "content" in data:
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def list_repo_files(repo: str) -> list[str]:
    """List all file paths in a repo via the git tree API."""
    # Try common default branches
    for branch in ["main", "master"]:
        raw = gh_api(f"repos/{repo}/git/trees/{branch}?recursive=1")
        if raw:
            try:
                data = json.loads(raw)
                return [item["path"] for item in data.get("tree", [])]
            except (json.JSONDecodeError, KeyError):
                pass
    return []


def match_scan_patterns(files: list[str]) -> list[str]:
    """Filter file list to those matching our scan patterns."""
    import fnmatch
    matched = set()
    for pattern in SCAN_GLOBS:
        for f in files:
            if fnmatch.fnmatch(f, pattern):
                matched.add(f)
    return sorted(matched)


# ---------------------------------------------------------------------------
# Local filesystem helpers
# ---------------------------------------------------------------------------
def scan_local_project(project_dir: Path) -> list[Path]:
    """Find scannable files in a local project directory."""
    import fnmatch
    results = []
    for pattern in SCAN_GLOBS:
        for f in project_dir.rglob("*"):
            if f.is_file() and fnmatch.fnmatch(str(f.relative_to(project_dir)), pattern):
                results.append(f)
    return sorted(set(results))


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------
def extract_refs_from_text(text: str) -> tuple[set[str], set[str], dict[str, str]]:
    """Extract DOIs, arXiv IDs, and BibTeX entries from text.

    Returns (dois, arxiv_ids, bib_entries).
    """
    dois = set()
    for m in DOI_PATTERN.finditer(text):
        doi = m.group(1).rstrip(".,;)")
        # Clean trailing markdown/URL artifacts
        doi = re.sub(r'[)\]}>]+$', '', doi)
        doi = doi.lower()
        dois.add(doi)

    arxiv_ids = set()
    for m in ARXIV_PATTERN.finditer(text):
        aid = m.group(1)
        # Strip version suffix for dedup
        arxiv_ids.add(re.sub(r'v\d+$', '', aid))

    bib_entries = parse_bib_entries(text)

    return dois, arxiv_ids, bib_entries


# ---------------------------------------------------------------------------
# DOI / arXiv resolution
# ---------------------------------------------------------------------------
def doi_to_bibtex(doi: str, retries: int = 2) -> str | None:
    """Resolve a DOI to BibTeX via CrossRef content negotiation."""
    url = f"https://doi.org/{doi}"
    req = Request(url, headers={"Accept": "application/x-bibtex"})
    for attempt in range(retries + 1):
        try:
            with urlopen(req, timeout=15) as resp:
                bib = resp.read().decode("utf-8", errors="replace").strip()
                if bib.startswith("@"):
                    return bib
        except HTTPError as e:
            if e.code == 404:
                return None
            if attempt < retries:
                time.sleep(1 + attempt)
        except Exception:
            if attempt < retries:
                time.sleep(1 + attempt)
    return None


def arxiv_to_bibtex(arxiv_id: str) -> str | None:
    """Resolve an arXiv ID to BibTeX via the arXiv API."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        with urlopen(url, timeout=15) as resp:
            xml = resp.read().decode("utf-8")
    except Exception:
        return None

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml)
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
        authors = [
            a.findtext("atom:name", "", ns).strip()
            for a in entry.findall("atom:author", ns)
        ]
        published = entry.findtext("atom:published", "", ns)[:4]  # year
        summary = entry.findtext("atom:summary", "", ns).strip()[:200]

        # Build a key: firstauthor_year_firstword
        first_author = authors[0].split()[-1].lower() if authors else "unknown"
        first_author = re.sub(r'[^a-z]', '', first_author)
        first_word = re.sub(r'[^a-z]', '', title.split()[0].lower()) if title else "untitled"
        key = f"{first_author}{published}{first_word}"

        author_str = " and ".join(authors)
        if len(authors) > 3:
            author_str = f"{authors[0]} and {authors[1]} and {authors[2]} and others"

        bib = (
            f"@misc{{{key},\n"
            f"  title={{{title}}},\n"
            f"  author={{{author_str}}},\n"
            f"  year={{{published}}},\n"
            f"  eprint={{{arxiv_id}}},\n"
            f"  archivePrefix={{arXiv}}\n"
            f"}}"
        )
        return bib
    except ET.ParseError:
        return None


def make_bib_key(bib_text: str) -> str | None:
    """Extract the key from a BibTeX entry."""
    m = re.match(r'@\w+\{([^,]+),', bib_text)
    return m.group(1).strip() if m else None


def normalize_crossref_key(bib_text: str, doi: str) -> str:
    """Replace CrossRef's auto-generated key with authorYYYYword format."""
    m = re.match(r'(@\w+\{)([^,]+)(,)', bib_text)
    if not m:
        return bib_text

    # Try to extract author and year from the entry
    author_m = re.search(r'author\s*=\s*\{([^}]+)\}', bib_text, re.IGNORECASE)
    year_m = re.search(r'year\s*=\s*\{?(\d{4})\}?', bib_text, re.IGNORECASE)
    title_m = re.search(r'title\s*=\s*\{([^}]+)\}', bib_text, re.IGNORECASE)

    if author_m and year_m and title_m:
        first_author = author_m.group(1).split(",")[0].split(" and ")[0].strip()
        last_name = first_author.split()[-1].lower()
        last_name = re.sub(r'[^a-z]', '', last_name)
        year = year_m.group(1)
        # First significant word of title (skip articles)
        title_words = title_m.group(1).split()
        skip = {"a", "an", "the", "on", "in", "of", "for", "and", "with", "to"}
        first_word = "untitled"
        for w in title_words:
            clean = re.sub(r'[^a-z]', '', w.lower())
            if clean and clean not in skip:
                first_word = clean
                break
        new_key = f"{last_name}{year}{first_word}"
    else:
        # Fallback: use sanitized DOI
        new_key = re.sub(r'[^a-zA-Z0-9]', '_', doi)

    return m.group(1) + new_key + m.group(3)


def add_keywords_to_entry(bib_text: str, project: str) -> str:
    """Add keywords={project} to a bib entry if not already present."""
    if re.search(r'keywords\s*=', bib_text, re.IGNORECASE):
        return bib_text
    # Insert before closing brace
    return bib_text.rstrip().rstrip("}") + f",\n  keywords={{{project}}}\n}}"


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------
def ingest(
    repos: list[str] | None = None,
    dry_run: bool = False,
    local: bool = False,
    verbose: bool = False,
):
    # Load existing bib
    if MASTER_BIB.exists():
        existing_text = MASTER_BIB.read_text()
    else:
        existing_text = ""

    existing_keys = parse_bib_keys(existing_text)
    existing_dois = extract_dois_from_bib(existing_text)
    print(f"Master bib: {len(existing_keys)} entries, {len(existing_dois)} with DOIs")

    # Collect all DOIs and arXiv IDs across repos
    all_dois: dict[str, str] = {}       # doi → source project
    all_arxiv: dict[str, str] = {}      # arxiv_id → source project
    all_bib_entries: dict[str, tuple[str, str]] = {}  # key → (entry, project)

    if local:
        # Scan local ~/dev/ projects
        projects = sorted(p.name for p in LOCAL_DEV.iterdir() if p.is_dir() and (p / ".git").exists())
        if repos:
            projects = [p for p in projects if p in repos]
        print(f"\nScanning {len(projects)} local projects...")

        for project in projects:
            project_dir = LOCAL_DEV / project
            files = scan_local_project(project_dir)
            if not files:
                continue

            n_refs = 0
            for filepath in files:
                text = filepath.read_text(errors="replace")
                dois, arxiv_ids, bib_entries = extract_refs_from_text(text)

                for doi in dois:
                    if doi not in existing_dois and doi not in all_dois:
                        all_dois[doi] = project
                        n_refs += 1
                for aid in arxiv_ids:
                    if aid not in all_arxiv:
                        all_arxiv[aid] = project
                        n_refs += 1
                for key, entry in bib_entries.items():
                    if key not in existing_keys and key not in all_bib_entries:
                        all_bib_entries[key] = (entry, project)
                        n_refs += 1

            if n_refs > 0 or verbose:
                print(f"  {project}: {len(files)} files, {n_refs} new refs")

    else:
        # Scan GitHub repos
        if repos:
            repo_list = [{"name": r} for r in repos]
        else:
            print(f"\nFetching {GITHUB_ORG} repos since {SINCE_DATE}...")
            repo_list = list_repos(GITHUB_ORG, SINCE_DATE)
        print(f"Scanning {len(repo_list)} repos...")

        for repo_info in repo_list:
            name = repo_info["name"]
            full_name = f"{GITHUB_ORG}/{name}"

            # List files and find matches
            all_files = list_repo_files(full_name)
            scan_files = match_scan_patterns(all_files)

            if not scan_files:
                if verbose:
                    print(f"  {name}: no scannable files")
                continue

            n_refs = 0
            for filepath in scan_files:
                content = fetch_file_from_github(full_name, filepath)
                if not content:
                    continue

                dois, arxiv_ids, bib_entries = extract_refs_from_text(content)

                for doi in dois:
                    if doi not in existing_dois and doi not in all_dois:
                        all_dois[doi] = name
                        n_refs += 1
                for aid in arxiv_ids:
                    if aid not in all_arxiv:
                        all_arxiv[aid] = name
                        n_refs += 1
                for key, entry in bib_entries.items():
                    if key not in existing_keys and key not in all_bib_entries:
                        all_bib_entries[key] = (entry, name)
                        n_refs += 1

            if n_refs > 0 or verbose:
                print(f"  {name}: {len(scan_files)} files, {n_refs} new refs")

            # Rate limit: be gentle with GitHub API
            time.sleep(0.3)

    # Summary before resolution
    print(f"\nFound: {len(all_dois)} new DOIs, {len(all_arxiv)} new arXiv IDs, "
          f"{len(all_bib_entries)} new BibTeX entries")

    if dry_run:
        print("\n--- DRY RUN ---")
        if all_dois:
            print("\nNew DOIs:")
            for doi, proj in sorted(all_dois.items()):
                print(f"  [{proj}] {doi}")
        if all_arxiv:
            print("\nNew arXiv IDs:")
            for aid, proj in sorted(all_arxiv.items()):
                print(f"  [{proj}] {aid}")
        if all_bib_entries:
            print("\nNew BibTeX keys:")
            for key, (_, proj) in sorted(all_bib_entries.items()):
                print(f"  [{proj}] {key}")
        return

    # Resolve DOIs → BibTeX
    resolved = []
    if all_dois:
        print(f"\nResolving {len(all_dois)} DOIs via CrossRef...")
        for i, (doi, project) in enumerate(sorted(all_dois.items())):
            bib = doi_to_bibtex(doi)
            if bib:
                bib = normalize_crossref_key(bib, doi)
                bib = add_keywords_to_entry(bib, project)
                key = make_bib_key(bib)
                if key and key not in existing_keys:
                    resolved.append((bib, project))
                    existing_keys.add(key)
                    if verbose:
                        print(f"  [{i+1}/{len(all_dois)}] {doi} → {key}")
            else:
                print(f"  FAILED: {doi}")
            # Rate limit CrossRef
            time.sleep(0.5)

    # Resolve arXiv IDs → BibTeX
    if all_arxiv:
        # Filter out arXiv IDs that correspond to DOIs we already resolved
        print(f"\nResolving {len(all_arxiv)} arXiv IDs...")
        for aid, project in sorted(all_arxiv.items()):
            bib = arxiv_to_bibtex(aid)
            if bib:
                bib = add_keywords_to_entry(bib, project)
                key = make_bib_key(bib)
                if key and key not in existing_keys:
                    resolved.append((bib, project))
                    existing_keys.add(key)
                    if verbose:
                        print(f"  {aid} → {key}")
            time.sleep(0.3)

    # Add pre-existing BibTeX entries from .bib files
    for key, (entry, project) in sorted(all_bib_entries.items()):
        if key not in existing_keys:
            entry = add_keywords_to_entry(entry, project)
            resolved.append((entry, project))
            existing_keys.add(key)

    if not resolved:
        print("\nNo new entries to add.")
        return

    # Group by source project and append
    print(f"\nAppending {len(resolved)} new entries to {MASTER_BIB.name}...")
    by_project: dict[str, list[str]] = {}
    for bib, project in resolved:
        by_project.setdefault(project, []).append(bib)

    with open(MASTER_BIB, "a") as f:
        for project in sorted(by_project):
            entries = by_project[project]
            f.write(f"\n% --- auto-ingested from {project} "
                    f"({len(entries)} entries) ---\n")
            for entry in entries:
                f.write(f"\n{entry}\n")

    # Final stats
    final_text = MASTER_BIB.read_text()
    final_keys = parse_bib_keys(final_text)
    dupes = [k for k in set(final_keys) if list(final_keys).count(k) > 1]
    print(f"\nDone. Master bib now has {len(set(final_keys))} unique entries.")
    if dupes:
        print(f"WARNING: duplicate keys found: {dupes}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ingest references from m9h repos into neuro-kb master bibliography"
    )
    parser.add_argument(
        "--repos", nargs="+",
        help="Specific repo names to scan (default: all since 2025-01-01)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be added without resolving or writing"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Scan ~/dev/ locally instead of GitHub API"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed progress"
    )
    args = parser.parse_args()
    ingest(repos=args.repos, dry_run=args.dry_run, local=args.local, verbose=args.verbose)


if __name__ == "__main__":
    main()
