# Session state — resume here

Last worked: 2026-07-28. Written so a fresh session (or a fresh agent) can pick
up cold. Read this, then `README.md`, then `DECISIONS.md`.

Origin document: `~/Downloads/nanoeeg_handoff.md` (2026-07-26). Everything in
this repo is downstream of it. Where this repo contradicts that document, this
repo is the corrected version and `DECISIONS.md` says why.

---

## Where things stand

**M1 scaffold complete and green.** The frozen evaluation surface is implemented
and tested; 35 tests pass with no EEG data required.

```bash
cd ~/Workspace/nanoeeg
uv run --with pytest python -m pytest tests/ -q     # 35 passed
uv run python -m nanoeeg.frozen                     # suite + seal + budget
```

Nothing is committed — `git init` was run, no commits made. That was deliberate
(commits weren't asked for), not an oversight.

## Environment, as measured on this laptop

| | |
|---|---|
| Python | 3.14.6 (`/usr/bin/python3`), `uv` at `~/.local/bin/uv` |
| GPU | **none** — `nvidia-smi` absent, `torch.cuda.is_available()` False |
| Disk | 648 GB free on `/home` |
| `braindecode` / `mne` | 1.6.1 / 1.12.1, install cleanly under `uv run --with`; `BaseConcatDataset.pull_from_hub` exists |
| 8 public probe datasets | ~25 GB total (largest: `isruc-sleep` 13.1 GB, `chbmit` 4.7 GB) |

**Paths from the handoff's §9 do not exist on this machine.** `~/dev/emeg-fm`,
`~/dev/nsg-agent-kit`, `/mnt/t9/oeb_libs` are all absent — they are on a
different box. This matters most for `emeg_fm/fmscope_bridge.py`; see Blocker 1.

## Provenance — do not lose this

The eight inherited splits in `nanoeeg/frozen/splits_oeb.py` were transcribed
from:

    https://github.com/braindecode/OpenEEGBench
    commit 3e4d034ae3e009deeda9c987d05a37e18fd87e15  (2026-07-09)
    file:   open_eeg_bench/default_configs/datasets.py
    licence: BSD 3-Clause

That clone lived in the session scratchpad under `/tmp` and is gone after the
reboot. Re-clone if a transcription needs re-checking:

```bash
git clone https://github.com/braindecode/OpenEEGBench /tmp/oeb
git -C /tmp/oeb checkout 3e4d034ae3e009deeda9c987d05a37e18fd87e15
diff <(...) # compare against nanoeeg/frozen/splits_oeb.py
```

Note that HEAD at the time was *"Fix degenerate MSE loss on regression datasets
(issue #42)"* — upstream is actively changing. `test_bcic2a_split_matches_
upstream_verbatim` guards the one split most likely to drift.

## Verifications already done — don't redo these

- **HF availability, unauthenticated, 2026-07-26.** 8 × HTTP 200, 4 × HTTP 401.
  Public: `arithmetic_zyma2019 bcic2a bcic2020-3 physionet chbmit faced
  isruc-sleep mdd_mumtaz2016`. Gated: `tuab tuev seed-v seed-vig`.
  This is what produced decisions D1 and D2 — the handoff's "7 obtainable / 3
  gated" missed public `chbmit` and gated `seed-vig`.
- **OpenNeuro GraphQL, 2026-07-26.** All three NEMAR replacements confirmed:
  `ds004504` (1.0.9, CC0, N=88), `ds006018` (1.2.2, CC0, N=127), `ds005505`
  (1.0.1, CC-BY-SA 4.0, N=136). N, licence and task lists match the handoff
  exactly. `loaders/openneuro.check_pinned()` re-runs this on demand.

## Blockers, in priority order

1. **Identity-erasure implementation.** `emeg_fm/fmscope_bridge.py` is not on
   this machine and `identity_free_score` is the project's differentiator.
   LEACE can be reimplemented from the published closed form, but then
   nanoeeg's identity-free numbers stop being commensurable with the existing
   FMScope / meeg-brain-age results. The handoff also flags a per-trial vs
   pooled distinction that would otherwise be guesswork. **Needs: the repo, or
   a decision to reimplement and accept the discontinuity.**
2. **Ratify or overturn D1 and D5.** Adding `chbmit` (11 slots, not 10) and
   gating promotion on identity-free are contract changes — free now,
   comparability-breaking once a run is published.
3. **Compute.** B1 is CPU-feasible but slow; B2 means forward passes over
   ~25 GB through a frozen checkpoint and wants a GPU. Is the NSG/Expanse
   offer live? If NSG: V100s, so pin `torch==2.4.1` with the cu121 index-url
   *before* installing peft/transformers.
4. **Which public checkpoint for B2** (LaBraM / EEGPT / REVE) — unknown which
   are obtainable without a DUA. Given what the `braindecode/*` check turned
   up, worth verifying rather than assuming.
5. **HBN data use agreement** — needs a human reader. Blocks publishing shards,
   not M1 code.

## Next actions, none of which need the blockers cleared

1. Write the 8 OEB probe loaders against `BaseConcatDataset.pull_from_hub`,
   honouring the frozen splits exactly. Start with `arithmetic_zyma2019`
   (0.12 GB) and `bcic2a` (0.34 GB) — smallest, fastest to iterate.
2. Write the 3 NEMAR loaders (anonymous S3, `s3://openneuro.org/<accession>`).
3. Run B1 (random-init linear probe) on the CPU-feasible probes to get a real
   variance table. That number sets `leader_threshold` and nothing downstream
   is meaningful without it.
4. **Before** `scripts/materialise_splits.py` runs, read the real `events.tsv`
   for `ds006018` — the four paradigm contrasts in `splits_nemar.py` are
   *intent, not event codes*. Materialising pins those splits permanently.

## Known gotchas, inherited and paid for

- `open_eeg_bench`'s exca cache silently replays a stale result if `infra.folder`
  is reused across a broken and a fixed run. Use a fresh folder whenever a
  number looks suspicious.
- Never `pip install --user` torch or jax on the other machine — it shadows the
  container stack. On this laptop, `uv run --with` keeps everything isolated;
  the `.venv/` here is disposable and gitignored.

## Outreach plan (not started)

Sequencing matters more than the pitch; see the memory note
`nanoeeg-delorme-outreach` for the full reasoning. Short version:

1. Consume the standing NEMAR + NSG offer from 2026-07-19 with a small concrete
   ask — don't stack a new request on an unconsumed one.
2. File the three OEB defects as issues/PRs **before** proposing anything.
   Contribution first, proposal second; same content, opposite reception.
3. Then propose nanoeeg with M1 already built.
4. Bring in Scotti/MedARC last — M0–M2 don't depend on Labless.

Do **not** pitch this as a merge of OpenEEGBench and nanopath. Handoff §7 is
right that a merge fails on trust model and score semantics, and the technical
objections to it are correct.
