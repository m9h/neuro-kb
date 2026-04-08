#!/usr/bin/env python3
"""
distill.py — Distill raw project sources into interlinked wiki pages.

Reads raw/projects/ and references.bib, then uses Claude API to generate
wiki pages following the CLAUDE.md schema. Outputs to wiki/.

Usage:
    cd ~/dev/neuro-kb
    python3 scripts/distill.py --plan                # show what pages would be created
    python3 scripts/distill.py --topic tissues        # distill one topic cluster
    python3 scripts/distill.py --topic modalities
    python3 scripts/distill.py --topic methods
    python3 scripts/distill.py --topic physics
    python3 scripts/distill.py --all                  # distill everything
    python3 scripts/distill.py --project neurojax     # distill from one project
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Install anthropic SDK: uv pip install anthropic")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw" / "projects"
WIKI_DIR = ROOT / "wiki"
BIB_FILE = ROOT / "references.bib"
SCHEMA_FILE = ROOT / "CLAUDE.md"

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# Topic clusters — which projects contribute to which wiki pages
TOPIC_CLUSTERS = {
    "modalities": {
        "description": "Imaging and stimulation modality pages",
        "pages": [
            ("eeg.md", "Electroencephalography", ["neurojax", "coffeine", "vbjax"]),
            ("meg.md", "Magnetoencephalography", ["neurojax", "coffeine", "hippy-feat"]),
            ("fnirs.md", "Functional Near-Infrared Spectroscopy", ["dot-jax", "sbi4dwi"]),
            ("structural-mri.md", "Structural MRI", ["neurojax", "sbi4dwi", "hippy-feat"]),
            ("diffusion-mri.md", "Diffusion MRI", ["sbi4dwi", "SpinDoctor.jl", "MCMRSimulator.jl"]),
            ("fmri.md", "Functional MRI", ["neurojax", "hippy-feat", "vbjax"]),
            ("mrs.md", "MR Spectroscopy", ["mrs-jax", "neurojax"]),
            ("tus.md", "Transcranial Ultrasound", ["sbi4dwi", "jwave", "brain-fwi", "openlifu-python"]),
            ("tms.md", "Transcranial Magnetic Stimulation", ["neurojax", "vbjax"]),
        ],
    },
    "tissues": {
        "description": "Tissue property pages with multi-modal values",
        "pages": [
            ("tissue-scalp.md", "Scalp", ["dot-jax", "neurojax", "sbi4dwi"]),
            ("tissue-skull.md", "Skull / Cortical Bone", ["sbi4dwi", "brain-fwi", "jwave"]),
            ("tissue-csf.md", "Cerebrospinal Fluid", ["dot-jax", "neurojax", "sbi4dwi"]),
            ("tissue-gray-matter.md", "Gray Matter", ["dot-jax", "neurojax", "sbi4dwi", "SpinDoctor.jl"]),
            ("tissue-white-matter.md", "White Matter", ["sbi4dwi", "SpinDoctor.jl", "MCMRSimulator.jl"]),
            ("tissue-optical-properties.md", "Tissue Optical Properties (cross-tissue)", ["dot-jax", "sbi4dwi"]),
            ("tissue-electrical-properties.md", "Tissue Electrical Conductivity (cross-tissue)", ["neurojax", "vbjax"]),
            ("tissue-acoustic-properties.md", "Tissue Acoustic Properties (cross-tissue)", ["sbi4dwi", "jwave", "brain-fwi"]),
        ],
    },
    "methods": {
        "description": "Computational method pages",
        "pages": [
            ("method-fem.md", "Finite Element Method", ["neurojax", "sbi4dwi", "SpinDoctor.jl"]),
            ("method-bem.md", "Boundary Element Method", ["neurojax"]),
            ("method-monte-carlo.md", "Monte Carlo Simulation", ["dot-jax", "MCMRSimulator.jl", "sbi4dwi"]),
            ("method-sbi.md", "Simulation-Based Inference", ["sbi4dwi"]),
            ("method-neural-ode.md", "Neural ODEs / Differentiable Simulation", ["sbi4dwi", "jaxctrl", "vbjax"]),
            ("method-source-imaging.md", "Source Imaging / Inverse Problems", ["neurojax", "brain-fwi"]),
            ("method-spectral-analysis.md", "Spectral Analysis (multitaper, wavelets)", ["neurojax", "coffeine"]),
            ("method-hmm-dynamics.md", "Hidden Markov Models for Brain Dynamics", ["neurojax", "vbjax"]),
            ("method-active-inference.md", "Active Inference", ["alf", "spinning-up-alf"]),
            ("method-hypergraph.md", "Hypergraph Methods", ["hgx", "jaxctrl"]),
        ],
    },
    "physics": {
        "description": "Physical principles pages",
        "pages": [
            ("physics-electromagnetic.md", "Electromagnetic Forward Problem", ["neurojax", "vbjax"]),
            ("physics-diffusion-equation.md", "Diffusion Equation (water, photon)", ["sbi4dwi", "SpinDoctor.jl", "dot-jax"]),
            ("physics-acoustic.md", "Acoustic Wave Propagation", ["jwave", "brain-fwi", "sbi4dwi"]),
            ("physics-bloch.md", "Bloch Equations / MR Physics", ["sbi4dwi", "mrs-jax", "MCMRSimulator.jl"]),
            ("physics-hemodynamic.md", "Hemodynamic Response / Neurovascular Coupling", ["neurojax", "vbjax", "hippy-feat"]),
        ],
    },
    "infrastructure": {
        "description": "Shared JAX infrastructure and tooling",
        "pages": [
            ("jax-ecosystem.md", "JAX Ecosystem for Neuroimaging", ["sbi4dwi", "neurojax", "vbjax", "hgx", "alf", "jaxctrl", "setae"]),
            ("data-formats.md", "Neuroimaging Data Formats", ["neurojax", "sbi4dwi", "dot-jax"]),
        ],
    },
}


def load_project_content(project: str, max_chars: int = 50000) -> str:
    """Load all raw content for a project, truncated to max_chars."""
    project_dir = RAW_DIR / project
    if not project_dir.exists():
        return ""

    content_parts = []
    # Prioritize README, CLAUDE.md, then research docs, then manuscript
    priority = ["README.md", "CLAUDE.md"]
    other_files = []

    for f in sorted(project_dir.rglob("*")):
        if f.is_file() and f.suffix in (".md", ".bib", ".cff", ".tex"):
            rel = str(f.relative_to(project_dir))
            if f.name in priority:
                content_parts.insert(0, f"### {rel}\n\n{f.read_text(errors='replace')}")
            else:
                other_files.append((rel, f))

    for rel, f in other_files:
        content_parts.append(f"### {rel}\n\n{f.read_text(errors='replace')}")

    combined = "\n\n---\n\n".join(content_parts)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n\n[... truncated ...]"
    return combined


def load_bib_for_projects(projects: list[str]) -> str:
    """Extract bib entries relevant to given projects."""
    if not BIB_FILE.exists():
        return ""
    text = BIB_FILE.read_text()
    # Extract entries with matching project keywords
    relevant = []
    for match in re.finditer(r'(@\w+\{[^}]+\}.*?\n\})', text, re.DOTALL):
        entry = match.group(1)
        keywords = re.search(r'keywords\s*=\s*\{([^}]+)\}', entry, re.IGNORECASE)
        if keywords:
            kw = keywords.group(1).lower()
            for proj in projects:
                if proj.lower().replace("-", "").replace(".", "") in kw.replace("-", "").replace(".", ""):
                    relevant.append(entry)
                    break
    return "\n\n".join(relevant[:30])  # cap at 30 entries


def load_schema() -> str:
    if SCHEMA_FILE.exists():
        return SCHEMA_FILE.read_text()
    return ""


def generate_wiki_page(
    filename: str,
    title: str,
    projects: list[str],
    client: anthropic.Anthropic,
) -> str:
    """Use Claude to distill a wiki page from project sources."""

    # Gather source material
    source_parts = []
    for proj in projects:
        content = load_project_content(proj)
        if content:
            source_parts.append(f"## Project: {proj}\n\n{content}")

    source_material = "\n\n{'='*80}\n\n".join(source_parts)
    bib_entries = load_bib_for_projects(projects)
    schema = load_schema()

    # Check for existing page to update rather than overwrite
    existing = ""
    existing_path = WIKI_DIR / filename
    if existing_path.exists():
        existing = existing_path.read_text()

    prompt = f"""You are a knowledge engineer building a wiki page for a shared neuroimaging knowledge base.

## Schema (from CLAUDE.md)

{schema}

## Task

Create (or update) the wiki page `{filename}` with title "{title}".

Source projects: {', '.join(projects)}

## Requirements

1. Start with YAML frontmatter matching the appropriate entity type from the schema
2. Write clear, specific, quantitative content — prefer "CSF conductivity is 1.79 S/m at 10 Hz (Gabriel 1996)" over "CSF is conductive"
3. Include a Properties/Parameters table where applicable with actual numerical values from the source material
4. Cross-reference other wiki pages using markdown links: [page-name.md](page-name.md)
5. Cite bibliography entries using BibTeX keys: [@key]
6. Include a "Relevant Projects" section listing which ~/dev/ projects implement this concept
7. Include a "See Also" section with links to related wiki pages
8. Keep the page focused — one concept per page, aim for 100-250 lines

## Source Material

{source_material[:80000]}

## Relevant Bibliography Entries

{bib_entries[:10000]}

{"## Existing Page Content (update, don't start from scratch)" + chr(10) + existing if existing else ""}

Generate ONLY the wiki page content (frontmatter + markdown). No preamble or explanation."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def update_index():
    """Regenerate wiki/index.md from all wiki pages."""
    pages_by_type = {}
    for f in sorted(WIKI_DIR.glob("*.md")):
        if f.name in ("index.md", "log.md"):
            continue
        text = f.read_text()
        # Extract type from frontmatter
        type_match = re.search(r'^type:\s*(.+)$', text, re.MULTILINE)
        title_match = re.search(r'^title:\s*(.+)$', text, re.MULTILINE)
        page_type = type_match.group(1).strip() if type_match else "concept"
        title = title_match.group(1).strip() if title_match else f.stem

        pages_by_type.setdefault(page_type, []).append((f.name, title))

    lines = [
        "---",
        "title: Index",
        "description: Master catalog of all wiki pages",
        "---",
        "",
        "# Wiki Index",
        "",
    ]

    type_labels = {
        "head-model": "Head Models",
        "modality": "Modalities",
        "physics": "Physics",
        "tissue": "Tissues",
        "method": "Methods",
        "coordinate-system": "Coordinate Systems",
        "concept": "Concepts",
        "infrastructure": "Infrastructure",
    }

    for ptype in ["head-model", "modality", "physics", "tissue", "method",
                   "coordinate-system", "infrastructure", "concept"]:
        label = type_labels.get(ptype, ptype.title())
        pages = pages_by_type.get(ptype, [])
        lines.append(f"## {label}")
        if pages:
            for filename, title in sorted(pages, key=lambda x: x[1]):
                lines.append(f"- [{title}]({filename})")
        else:
            lines.append("*(none yet)*")
        lines.append("")

    (WIKI_DIR / "index.md").write_text("\n".join(lines))


def update_log(pages_created: list[str]):
    """Append to wiki/log.md."""
    from datetime import date
    log_path = WIKI_DIR / "log.md"
    existing = log_path.read_text() if log_path.exists() else ""
    today = date.today().isoformat()
    entry = f"\n## {today}\n- Distilled {len(pages_created)} wiki pages: {', '.join(pages_created)}\n"
    log_path.write_text(existing + entry)


def plan_pages(topics: list[str] | None):
    """Show what pages would be created."""
    if topics:
        clusters = {t: TOPIC_CLUSTERS[t] for t in topics if t in TOPIC_CLUSTERS}
    else:
        clusters = TOPIC_CLUSTERS

    total = 0
    for topic, info in clusters.items():
        print(f"\n{'='*60}")
        print(f"  {topic}: {info['description']}")
        print(f"{'='*60}")
        for filename, title, projects in info["pages"]:
            exists = (WIKI_DIR / filename).exists()
            status = "EXISTS" if exists else "NEW"
            sources_available = sum(1 for p in projects if (RAW_DIR / p).exists())
            print(f"  [{status}] {filename}")
            print(f"         {title}")
            print(f"         Sources: {', '.join(projects)} ({sources_available}/{len(projects)} available)")
            total += 1

    print(f"\n{total} pages planned across {len(clusters)} topics")


def distill(
    topics: list[str] | None = None,
    project: str | None = None,
    all_topics: bool = False,
    dry_run: bool = False,
):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    if all_topics:
        clusters = TOPIC_CLUSTERS
    elif topics:
        clusters = {t: TOPIC_CLUSTERS[t] for t in topics if t in TOPIC_CLUSTERS}
    elif project:
        # Find all pages that reference this project
        clusters = {}
        for topic, info in TOPIC_CLUSTERS.items():
            pages = [(f, t, p) for f, t, p in info["pages"] if project in p]
            if pages:
                clusters[topic] = {"description": info["description"], "pages": pages}
    else:
        print("Specify --topic, --project, or --all")
        return

    pages_created = []
    for topic, info in clusters.items():
        print(f"\n{'='*60}")
        print(f"  Distilling: {topic}")
        print(f"{'='*60}")

        for filename, title, projects in info["pages"]:
            wiki_path = WIKI_DIR / filename
            print(f"\n  → {filename} ({title})")
            print(f"    Sources: {', '.join(projects)}")

            if dry_run:
                print(f"    [DRY RUN] Would generate from {len(projects)} projects")
                continue

            # Check source availability
            available = [p for p in projects if (RAW_DIR / p).exists()]
            if not available:
                print(f"    SKIP: no source material available")
                continue

            try:
                content = generate_wiki_page(filename, title, available, client)
                wiki_path.write_text(content)
                pages_created.append(filename)
                print(f"    ✓ Written ({len(content)} chars)")
            except Exception as e:
                print(f"    ERROR: {e}")

            # Rate limit
            import time
            time.sleep(1)

    if pages_created:
        update_index()
        update_log(pages_created)
        print(f"\n{'='*60}")
        print(f"  Done: {len(pages_created)} pages written")
        print(f"  Updated: index.md, log.md")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Distill raw project sources into wiki pages"
    )
    parser.add_argument("--plan", action="store_true",
                        help="Show what pages would be created")
    parser.add_argument("--topic", nargs="+",
                        choices=list(TOPIC_CLUSTERS.keys()),
                        help="Distill specific topic clusters")
    parser.add_argument("--project", help="Distill pages relevant to one project")
    parser.add_argument("--all", action="store_true", dest="all_topics",
                        help="Distill all topic clusters")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without API calls")

    args = parser.parse_args()

    if args.plan:
        plan_pages(args.topic)
    else:
        distill(
            topics=args.topic,
            project=args.project,
            all_topics=args.all_topics,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
