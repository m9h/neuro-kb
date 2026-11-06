#!/usr/bin/env python3
"""
distill.py — Distill raw project sources into interlinked wiki pages.

Reads raw/projects/ and references.bib, then uses Claude API to generate
wiki pages following the CLAUDE.md schema. Outputs to wiki/.

Usage:
    cd ~/Workspace/neuro-kb
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

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000  # pages run 100-250 lines; the old 4096 cap truncated long tables
EFFORT = "high"

# Topic clusters — which projects contribute to which wiki pages
TOPIC_CLUSTERS = {
    "modalities": {
        "description": "Imaging and stimulation modality pages",
        "pages": [
            ("eeg.md", "Electroencephalography", ["neurojax", "coffeine", "vbjax", "emeg-fm", "ephys-tokenizer-jax", "hnn-jax", "nanoeeg", "zuna-eeg-pipeline"]),
            ("meg.md", "Magnetoencephalography", ["neurojax", "coffeine", "hippy-feat", "emeg-fm", "ephys-tokenizer-jax", "hnn-jax"]),
            ("fnirs.md", "Functional Near-Infrared Spectroscopy", ["dot-jax", "sbi4dwi"]),
            ("structural-mri.md", "Structural MRI", ["neurojax", "sbi4dwi", "hippy-feat", "smri-fm", "T1Prep", "insula"]),
            ("diffusion-mri.md", "Diffusion MRI", ["sbi4dwi", "SpinDoctor.jl", "MCMRSimulator.jl"]),
            ("fmri.md", "Functional MRI", ["neurojax", "hippy-feat", "vbjax", "brainmarks-hbn", "brainmarks-wand", "brainmarks-dlbs", "brainmarks-syn", "vpjax"]),
            ("mrs.md", "MR Spectroscopy", ["mrs-jax", "neurojax"]),
            ("tus.md", "Transcranial Ultrasound", ["sbi4dwi", "jwave", "brain-fwi", "openlifu-python", "open-fus-plan"]),
            ("fus.md", "Functional Ultrasound Imaging", ["open-fus-plan", "brain-fwi", "jwave"]),
            ("tms.md", "Transcranial Magnetic Stimulation", ["neurojax", "vbjax"]),
            ("temporal-interference.md", "Temporal Interference Stimulation", ["ti-sim", "fem-trials", "neurojax"]),
            ("acoustoelectric-imaging.md", "Acoustoelectric Neuroimaging", ["ae-sim", "fem-trials", "jwave", "brain-fwi"]),
            ("neuronal-current-mri.md", "Neuronal-Current MRI, MREIT and MRCDI", ["neuroelectric-mri", "mri-rf-stack", "neurojax"]),
            ("optical-microscopy.md", "Computational Optical Microscopy", ["bsccm-jax", "openconfocal-cm", "dot-jax"]),
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
            ("method-fem.md", "Finite Element Method", ["neurojax", "sbi4dwi", "SpinDoctor.jl", "ti-sim", "ae-sim", "fem-trials"]),
            ("method-bem.md", "Boundary Element Method", ["neurojax"]),
            ("method-monte-carlo.md", "Monte Carlo Simulation", ["dot-jax", "MCMRSimulator.jl", "sbi4dwi"]),
            ("method-sbi.md", "Simulation-Based Inference", ["sbi4dwi", "inferjax", "hnn-jax", "vpjax"]),
            ("method-neural-ode.md", "Neural ODEs / Differentiable Simulation", ["sbi4dwi", "jaxctrl", "vbjax", "vpjax"]),
            ("method-source-imaging.md", "Source Imaging / Inverse Problems", ["neurojax", "brain-fwi"]),
            ("method-spectral-analysis.md", "Spectral Analysis (multitaper, wavelets)", ["neurojax", "coffeine"]),
            ("method-hmm-dynamics.md", "Hidden Markov Models for Brain Dynamics", ["neurojax", "vbjax"]),
            ("method-active-inference.md", "Active Inference", ["alf", "spinning-up-alf"]),
            ("method-hypergraph.md", "Hypergraph Methods", ["hgx", "jaxctrl"]),
            ("htsr-weight-analysis.md", "HT-SR Weight-Spectrum Analysis", ["wwj", "WeightWatcher", "emeg-fm", "smri-fm"]),
            ("neural-mass-models.md", "Neural Mass Models", ["vbjax", "hnn-jax", "vc2008-netpyne", "gnf-jax", "neurojax"]),
            ("method-neural-fields.md", "Neural Field Models on Connectome Harmonics", ["gnf-jax", "vbjax"]),
            ("method-pharmaco-eeg.md", "Pharmaco-EEG: Spectral Parameterization and Drug Effects", ["neurojax", "vpjax"]),
            ("method-eeg-foundation-models.md", "EEG Foundation Models and Representation Learning", ["nanoeeg", "emeg-fm", "zuna-eeg-pipeline"]),
        ],
    },
    "physics": {
        "description": "Physical principles pages",
        "pages": [
            ("physics-electromagnetic.md", "Electromagnetic Forward Problem", ["neurojax", "vbjax"]),
            ("physics-diffusion-equation.md", "Diffusion Equation (water, photon)", ["sbi4dwi", "SpinDoctor.jl", "dot-jax"]),
            ("physics-acoustic.md", "Acoustic Wave Propagation", ["jwave", "brain-fwi", "sbi4dwi"]),
            ("physics-bloch.md", "Bloch Equations / MR Physics", ["sbi4dwi", "mrs-jax", "MCMRSimulator.jl"]),
            ("physics-hemodynamic.md", "Hemodynamic Response / Neurovascular Coupling", ["neurojax", "vbjax", "hippy-feat", "vpjax"]),
            ("physics-cerebrovascular.md", "Cerebrovascular, Metabolic and Pharmacokinetic Physiology", ["vpjax", "vbjax"]),
        ],
    },
    "infrastructure": {
        "description": "Shared JAX infrastructure and tooling",
        "pages": [
            ("jax-ecosystem.md", "JAX Ecosystem for Neuroimaging", ["sbi4dwi", "neurojax", "vbjax", "hgx", "alf", "jaxctrl", "setae", "inferjax", "vpjax", "hnn-jax", "gnf-jax", "mrs-jax"]),
            ("data-formats.md", "Neuroimaging Data Formats", ["neurojax", "sbi4dwi", "dot-jax", "hippy-feat"]),
            ("compute-infrastructure.md", "Compute: NSG, NeuroFedora, NeuroDebian and Containers", ["nsg-agent-kit", "neurofedora", "neurodebian", "neurocontainers", "research-infra"]),
        ],
    },
    "foundation-models": {
        "description": "Brain foundation models and weights-first benchmarking",
        "pages": [
            ("foundation-models.md", "Neuroimaging Foundation Models", ["smri-fm", "emeg-fm", "ephys-tokenizer-jax", "hippy-feat", "nanopath", "mars-fm"]),
            ("benchmark-datasets.md", "fMRI-FM Benchmark Datasets (Brainmarks)", ["brainmarks-hbn", "brainmarks-wand", "brainmarks-dlbs", "brainmarks-syn"]),
        ],
    },
}

# Pages added or whose sources changed in the 2026-11 refresh (for --pages shorthand)
REFRESH_2026_11 = [
    "fus.md", "temporal-interference.md", "acoustoelectric-imaging.md", "neuronal-current-mri.md",
    "optical-microscopy.md", "neural-mass-models.md", "method-neural-fields.md", "method-pharmaco-eeg.md",
    "method-eeg-foundation-models.md", "physics-cerebrovascular.md", "compute-infrastructure.md",
    "eeg.md", "meg.md", "fmri.md", "tus.md", "structural-mri.md", "method-fem.md", "method-sbi.md",
    "method-neural-ode.md", "physics-hemodynamic.md", "jax-ecosystem.md", "data-formats.md",
]


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

    # Every page that exists or is planned, so cross-links resolve
    planned = {f for info in TOPIC_CLUSTERS.values() for f, _, _ in info["pages"]}
    on_disk = {p.name for p in WIKI_DIR.glob("*.md")} - {"index.md", "log.md"}
    existing_pages = ", ".join(sorted(planned | on_disk))

    # Check for existing page to update rather than overwrite
    existing = ""
    existing_path = WIKI_DIR / filename
    if existing_path.exists():
        existing = existing_path.read_text()

    from datetime import datetime
    timestamp = datetime.now().astimezone().replace(microsecond=0).isoformat()
    prompt = f"""You are a knowledge engineer building a wiki page for a shared neuroimaging knowledge base.

## Schema (from CLAUDE.md)

{schema}

## Task

Create (or update) the wiki page `{filename}` with title "{title}".

Source projects: {', '.join(projects)}

## Requirements

1. Start with YAML frontmatter matching the appropriate entity type from the schema, plus the OKF fields: `description` (one sentence, quoted), `timestamp` (ISO 8601 with offset; use {timestamp}), and `tags` (3-6 lowercase keywords). Name the reference section `## Citations`
2. Write clear, specific, quantitative content — prefer "CSF conductivity is 1.79 S/m at 10 Hz (Gabriel 1996)" over "CSF is conductive"
3. Include a Properties/Parameters table where applicable with actual numerical values from the source material
4. Cross-reference other wiki pages using markdown links: [page-name.md](page-name.md). Link ONLY to pages in the "Existing wiki pages" list below (plus `{filename}` itself); never invent a filename. The `related:` frontmatter list must obey the same rule.
5. Cite bibliography entries using BibTeX keys: [@key]. Use only keys that appear in the bibliography excerpt below; if a fact has no matching key, cite the raw source file path instead.
6. Include a "Relevant Projects" section listing which ~/Workspace/ projects implement this concept
7. Include a "See Also" section with links to related wiki pages
8. Keep the page focused — one concept per page, aim for 100-250 lines

## Existing wiki pages (the only valid link targets)

{existing_pages}

## Source Material

{source_material[:80000]}

## Relevant Bibliography Entries

{bib_entries[:10000]}

{"## Existing Page Content (update, don't start from scratch)" + chr(10) + existing if existing else ""}

Generate ONLY the wiki page content (frontmatter + markdown). No preamble or explanation."""

    # Opus 5.5: thinking is always on; effort is the depth control. Server-side
    # fallbacks re-run a safety-declined request on another model automatically.
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        detail = response.stop_details.explanation if response.stop_details else ""
        raise RuntimeError(f"refused: {detail}")
    if response.stop_reason == "max_tokens":
        raise RuntimeError(f"truncated at {MAX_TOKENS} tokens; page too long")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    # Strip a stray ```markdown fence if the model wrapped the page
    text = re.sub(r"^```(?:markdown|md)?\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text + "\n"


def update_index():
    """Regenerate wiki/index.md in OKF form: root frontmatter carries only
    okf_version; every entry has an inline description from the page's
    `description:` field (falls back to the first body sentence)."""
    pages_by_type = {}
    for f in sorted(WIKI_DIR.glob("*.md")):
        if f.name in ("index.md", "log.md"):
            continue
        text = f.read_text()
        fm = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
        head, body = (fm.group(1), fm.group(2)) if fm else ("", text)
        def field(name):
            m = re.search(rf"^{name}:\s*(.+)$", head, re.M)
            return m.group(1).strip().strip("'\"") if m else ""
        page_type = field("type") or "concept"
        title = field("title") or f.stem
        desc = field("description")
        if not desc:
            para = next((x for x in re.split(r"\n\s*\n", body)
                         if x.strip() and not x.lstrip().startswith(("#", "|", "`", "-", "*", "$", "<"))), "")
            desc = re.split(r"(?<=[.!?])\s", re.sub(r"\*\*|`", "", para).replace("\n", " ").strip())[0]
        if len(desc) > 110:
            desc = desc[:109].rstrip() + "…"
        if desc:
            desc = desc[0].lower() + desc[1:] if not desc[:2].isupper() else desc
        pages_by_type.setdefault(page_type, []).append((f.name, title, desc))

    lines = ["---", 'okf_version: "0.1"', "---", "# Wiki Index", ""]
    type_labels = {
        "head-model": "Head Models", "modality": "Modalities", "physics": "Physics",
        "tissue": "Tissues", "method": "Methods", "coordinate-system": "Coordinate Systems",
        "infrastructure": "Infrastructure", "concept": "Concepts",
    }
    order = ["head-model", "modality", "physics", "tissue", "method",
             "coordinate-system", "infrastructure", "concept"]
    for ptype in order + sorted(set(pages_by_type) - set(order)):
        pages = pages_by_type.get(ptype, [])
        if not pages:
            continue
        lines.append(f"## {type_labels.get(ptype, ptype.title())}")
        for filename, title, desc in sorted(pages, key=lambda x: x[1].lower()):
            lines.append(f"- [{title}]({filename})" + (f" - {desc}" if desc else ""))
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
    pages: list[str] | None = None,
):
    # Credentials resolve from ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
    # `ant auth login` profile; the zero-arg client handles all three.
    client = anthropic.Anthropic()

    if pages:
        wanted = set(pages)
        clusters = {}
        for topic, info in TOPIC_CLUSTERS.items():
            sel = [(f, t, p) for f, t, p in info["pages"] if f in wanted]
            if sel:
                clusters[topic] = {"description": info["description"], "pages": sel}
        missing = wanted - {f for info in clusters.values() for f, _, _ in info["pages"]}
        if missing:
            print(f"WARNING: not in TOPIC_CLUSTERS: {', '.join(sorted(missing))}")
    elif all_topics:
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
    parser.add_argument("--pages", nargs="+",
                        help="Distill only these page filenames (use 'refresh' for the 2026-11 set)")

    args = parser.parse_args()

    pages = args.pages
    if pages == ["refresh"]:
        pages = REFRESH_2026_11

    if args.plan:
        plan_pages(args.topic)
    else:
        distill(
            topics=args.topic,
            project=args.project,
            all_topics=args.all_topics,
            dry_run=args.dry_run,
            pages=pages,
        )


if __name__ == "__main__":
    main()
