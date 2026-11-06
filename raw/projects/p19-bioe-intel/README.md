# P19 BioE Intel

An undergraduate research project investigating ASCL1/MASH1-induced P19 neurogenesis as a substrate for temporal-pattern and MNIST reservoir computing, with DishBrain-inspired adaptation as a later extension.

**Start with the [article](manuscript/main.pdf), [research evidence map](docs/research-foundations.md), and [project status](docs/project-status.md), and [benchmark plan](docs/benchmark-plan.md), and [student optical roadmap](docs/student-roadmap.md).** The article is a literature review and prospective experimental framework. It reports no new experimental results.

## What is established

Project mentor Gabriel McKinsey identifies the group's factor as MASH1/ASCL1. His account of undergraduate research in David Turner's laboratory is preserved in [project-context.txt](project-context.txt). Published studies support P19 neuronal differentiation, excitability, and functional networks under particular conditions. The current project's construct, delivery, maturation, network function, and task adaptation still require documentation.

The immediate milestone is collecting and characterizing baseline electrophysiology from the actual ASCL1/P19 preparation. The project lead reports little electrophysiology collected so far. Temporal-pattern and MNIST tasks depend on those results. The student roadmap explicitly includes calcium imaging, optogenetic input, and multimodal integration. Mind in Vitro recording/stimulation and bench-mounted UCLA Miniscope imaging are proposed components. See the [MiV reading map](docs/miv-reading-map.md) for source-review coverage. Faster preparation and better reliability across student cohorts are hypotheses to test.

## Read and contribute

1. Read the [research foundations](docs/research-foundations.md) for an annotated starting list with article links. Access PDFs through your institution or linked open-access versions.
2. Open an issue for a reading assignment, missing method, or experimental question.
3. Edit the article or reading notes on a branch and open a pull request. Record the source, page/figure, and what the evidence does and does not establish.
4. Have a project reviewer check scientific claims before merging. See [CONTRIBUTING.md](CONTRIBUTING.md).

PDF annotations can live in an external annotation service; link the annotated paper from its reading note. Keep experimental records and supporting measurements distinct from literature summaries.

## Files

| Path | Contents |
| --- | --- |
| `manuscript/main.tex` | Editable article |
| `manuscript/main.pdf` | Compiled review and prospective study framework |
| `manuscript/references.bib` | Generated article bibliography with DOI/URL links |
| `docs/` | Reading priorities and project status |
| `literature/` | Tagged BibTeX/RIS exports, screening audit, search methods, and source metadata |
| `literature/raw/` | Archived Europe PMC metadata responses |
| `student-materials/parameter-lock.json` | Unresolved method and analysis parameters |
| `student-materials/experiment-record.csv` | Empty experiment record template |
| `scripts/` | Offline library rebuild and reference validation |

The search is metadata triage with focused source inspection, not an exhaustive systematic review. Candidate counts are not verified experiment counts. See [methods](literature/methods.txt). No publisher PDFs or student experimental data are included.

## Build

Install [Tectonic](https://tectonic-typesetting.github.io/), then run from the repository root:

```sh
python3 scripts/check_manuscript.py
cd manuscript
tectonic --keep-logs main.tex
```

GitHub Actions checks references, compiles the article, and provides the PDF as the `p19-bioe-intel-article` artifact on each push and pull request. The checked-in PDF is a convenient snapshot; the successful build artifact corresponds to the selected workflow run. When editing the article, update the snapshot before committing.

To regenerate the bibliographic exports from archived sources:

```sh
python3 scripts/build_library.py
```

Add bibliographic records to the source metadata or `literature/manual-sources.json`; `manuscript/references.bib` is generated. Preserve human screening decisions outside the generated screening table before rebuilding. Python uses only its standard library; Tectonic can download TeX resources on its first run.

## Ownership and release status

Working repository for review by the project team. Authorship, affiliations, experimental methods, and release licenses remain to be agreed. The repository's presence does not imply endorsement by named institutions. The legacy [README.txt](README.txt) points to this current guide.
