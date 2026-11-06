# nanoeeg

Pretrain a small EEG foundation model from scratch on openly redistributable
EEG under a fixed 1e18-FLOP budget (~1 H100-hour), and maximise mean downstream
probe score across a frozen suite — while publishing, alongside it, an
**identity-free score** that reveals how much of the gain is subject identity
rather than the labelled construct.

Every EEG benchmark in existence ranks on a number that subject-identity leakage
inflates. nanoeeg publishes both scores and refuses to promote a run that climbs
one while regressing the other.

**Status: M1 scaffold.** The frozen evaluation surface is implemented and
tested. No data loaders, no training, no GPU work yet. See
[`docs/DECISIONS.md`](docs/DECISIONS.md) for the choices this encodes and where
they differ from the originating handoff.

---

## What is here

```
nanoeeg/frozen/       the evaluation contract — editing any of it invalidates a run
  probes.py             suite definition: 11 slots / 14 sub-probes
  splits_oeb.py         8 OpenEEGBench splits, transcribed verbatim (BSD-3)
  splits_nemar.py       3 OpenNeuro replacement probes + split recipes
  scoring.py            κ + balanced accuracy, both scalars, trap-climbing gate
  accounting.py         FLOP / window-presentation budget
  seal.py               content hash of the frozen surface
nanoeeg/loaders/      data access (materialiser implemented; probe loaders not)
nanoeeg/baselines/    B1 and B2 specs + the M1 variance table
scripts/              one-shot split materialisation
tests/                35 tests, no EEG data required
```

```bash
uv run python -m nanoeeg.frozen     # print the suite, its seal, and the budget
uv run --with pytest python -m pytest tests/ -q
```

The frozen surface is verifiable before a single window is downloaded — that is
deliberate. Anyone can audit the evaluation contract without the data.

## The probe suite

**8 inherited from OpenEEGBench** — splits and HF revision hashes copied
verbatim so numbers stay comparable to the published Arena rows: `bcic2a`,
`bcic2020_3`, `physionet`, `chbmit`, `isruc_sleep`, `faced`, `mdd_mumtaz2016`,
`arithmetic_zyma2019`.

**3 from OpenNeuro/NEMAR**, replacing the four DUA-gated Arena datasets
(`tuab`, `tuev`, `seed-v`, `seed-vig`). Verified against the OpenNeuro GraphQL
API on 2026-07-26:

| replaces | accession | N | licence | axis gained |
|---|---|---|---|---|
| TUAB | `ds004504` | 88 | CC0 | neurodegeneration — 3-way AD/FTD/HC |
| TUEV | `ds006018` | 127 | CC0 | **evoked/ERP time-locked decoding**, 4 paradigms |
| SEED-V | `ds005505` | 136 | CC-BY-SA 4.0 | **developmental/pediatric** (HBN R1) |

Everything in the suite is CC0 or CC-BY, so windowed shards can actually be
redistributed — the exact property the gated datasets lack, and the reason an
open community-runnable EEG benchmark has to be built this way.

## The two scalars

- **`mean_probe_score`** — mean Cohen's κ over the 11 slots. Primary rank.
- **`identity_free_score`** — the same mean after subject-identity erasure.
  Published as a co-equal column.

Both metrics (κ *and* balanced accuracy) are computed for every probe and the
metric name travels with every published value. This is not fussiness: OEB's
leaderboard mixes κ and balanced accuracy in identically-named columns, making
its rows non-comparable, and that defect is the direct reason `ProbeResult`
cannot be constructed without both.

**Trap-climbing.** A run whose `mean_probe_score` rises while its
`identity_free_score` falls got better at the leaderboard and worse at the thing
the leaderboard is a proxy for. Such runs are ranked and displayed but **cannot
take the lead**. Identity-free is a constraint on the incumbent, not the ranking
objective — see [D5](docs/DECISIONS.md#d5).

**The leader threshold is measured, not assumed.** nanopath's `+0.006` constant
presumes low run-to-run variance; cross-subject EEG splits on 88-subject cohorts
do not have it. The threshold is `2 × pooled seed SD`, measured by baseline B1,
and runs must submit ≥3 seeds (≥5 on `ds004504`).

## Where this came from

Borrows nanopath's *mechanism* (fixed compute cap, frozen/editable seam, one
scalar, content-sealed provenance) and OpenEEGBench's *structure* (dataset +
split + metric as config, separate from architecture + objective + training),
while dropping OEB's unobtainable datasets. It is not a bridge between them —
the trust models and score semantics are incompatible.

- [labless.dev](https://labless.dev) / [MedARC-AI/nanopath](https://github.com/MedARC-AI/nanopath)
- [braindecode/OpenEEGBench](https://github.com/braindecode/OpenEEGBench) — BSD-3;
  split definitions in `splits_oeb.py` are transcribed from it with attribution.

## Next

1. **B1 — random-init linear probe**, ≥3 seeds across the suite. Produces the
   variance table and therefore the leader threshold. Nothing downstream is
   meaningful without it.
2. **B2 — frozen public checkpoint + linear probe.** The premise test, moved up
   from M2 ([D7](docs/DECISIONS.md#d7)): if a frozen public checkpoint beats
   anything trainable inside the cap, the project re-scopes toward adaptation
   before the GPU budget is spent rather than after.
3. Probe loaders for the three NEMAR datasets, then
   `scripts/materialise_splits.py` — which pins those splits permanently.

Open and unresolved: the HBN data use agreement ([D4](docs/DECISIONS.md#d4)) and
the missing regression axis ([D2](docs/DECISIONS.md#d2)).
