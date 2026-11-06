# Decisions

Choices the scaffold encodes that the handoff left open or got wrong. Each is a
**proposal implemented as the default**, not a ratified decision — every one is
a single-file change if you disagree. Recorded here so the reasoning is
reviewable rather than buried in code.

Written 2026-07-26 against `nanoeeg_handoff.md` of the same date.

---

## D1 — chbmit is included, making the suite 11 slots, not 10

**The handoff is wrong about the arithmetic.** It says 7 of OpenEEGBench's
datasets are obtainable and 3 are not. OEB actually ships **12** datasets, and
an unauthenticated check of the HF datasets API on 2026-07-26 gives:

| status | datasets |
|---|---|
| HTTP 200 (public) | arithmetic_zyma2019, bcic2a, bcic2020-3, physionet, **chbmit**, faced, isruc-sleep, mdd_mumtaz2016 |
| HTTP 401 (gated) | tuab, tuev, seed-v, **seed-vig** |

So it is **8 obtainable and 4 gated**, not 7 and 3. Two corrections:

- **`chbmit` was missed.** It is public, pulls fine, and is free comparability
  with the published Arena rows. Included as an eighth inherited probe. Its test
  set is two subjects, so its per-probe number is low-confidence — but excluding
  a public probe costs comparability for nothing.
- **`seed-vig` was missed.** It is gated too, which is why 4/12 = 33% matches
  the handoff's "30% of their benchmark" better than 3/12 = 25% does.

Suite is therefore **8 inherited + 3 NEMAR = 11 slots / 14 sub-probes**.

**To revert:** drop `chbmit` from `splits_oeb.OBTAINABLE`.

---

## D2 — the regression axis is gone, and nothing replaces it yet

`seed-vig` was the only regression target in OEB (continuous perclos vigilance).
Losing it makes nanoeeg an all-classification suite, which narrows what a
pretrained representation is tested for — a model can be good at categorical
readout and bad at continuous state estimation, and this suite would not notice.

Not fixed here, because fixing it means picking a NEMAR dataset with a
continuous target and that needs the same verification pass the other three got.
`scoring` assumes classification throughout (κ and balanced accuracy), so adding
regression is a real change, not a config flip.

**Flagged as open.** Candidates worth checking: any NEMAR set with continuous
behavioural covariates (reaction time, drowsiness ratings, age).

---

## D3 — ds006018 is 4 sub-probes but 1 slot

The handoff asks: 4 separate probes or 1 averaged? It recommends 4, on the
grounds that it "costs nothing."

It costs something. Four probes drawn from a single 127-subject cohort would
carry **4/14 ≈ 29%** of `mean_probe_score`. A run that overfits the acquisition
characteristics of one lab's recording setup would climb the leaderboard on that
alone.

Implemented as: report all four separately (the flanker/oddball contrast *is*
informative, exactly as the handoff argues), average them into **one slot** for
ranking. Best of both. `test_score_run_aggregates_slots_not_subprobes` pins this.

The four paradigms also **share one subject partition**, enforced in
`materialise.py` — otherwise a subject held out of `flanker` reappears in
`auditory_oddball`'s training set and the slot average is contaminated.

---

## D4 — CC-BY-SA is sidestepped, not resolved

The handoff asks whether CC-BY-SA permits redistributing windowed shards of HBN
(`ds005505`). That question is not answered here, because it does not need to be
if shards are **published per-dataset and never merged**. Share-alike then
propagates only to the HBN shards, which can carry CC-BY-SA, and never touches
the CC0 sets.

**The real risk is elsewhere.** HBN carries a data use agreement *in addition to*
its CC licence, and a DUA can bind harder than the licence it sits next to —
that is precisely the mechanism that makes TUAB/TUEV/SEED-V unobtainable. Check
the HBN DUA before the CC-BY-SA question; it is the one that can actually stop
redistribution.

**Not verified here.** Needs a human reading the actual agreement.

---

## D5 — rank on `mean_probe_score`, gate on `identity_free_score`

The handoff's open question 1 offers two options: rank on the raw score with
identity-free as a co-equal published column, or rank primarily on identity-free.

Implemented as a third thing: **rank on `mean_probe_score`, but block promotion
on any run that regresses `identity_free_score`.** Trap-climbing runs are
ranked and displayed, never promoted to leader.

Why not rank on identity-free directly: it would make the leaderboard's ordering
a function of the erasure method, which is itself a contested research artifact.
Every dispute about a rank becomes a dispute about LEACE. As a *gate* rather
than an *objective*, trap-climbing still cannot win, and the erasure method
never has to be defended as uniquely correct.

Two consequences worth knowing:

- A run that omits the identity-free score **cannot take the lead** — not
  because the score is required for ranking, but because without it
  trap-climbing cannot be ruled out.
- On identity-sensitive probes (`ds005505`, and `bcic2020_3` because it splits on
  `run` rather than subject, sharing subjects across train and test by
  construction) a missing identity-free score makes the whole run **ineligible**.
  A raw number on HBN with no identity-free counterpart is not interpretable as
  construct decoding, and the scorer refuses to publish it as if it were.

---

## D6 — the leader threshold is measured; the FLOP cap includes attention

Two arithmetic changes from the handoff.

**Threshold.** nanopath's `+0.006` constant is not adopted. `leader_threshold`
returns `2 × pooled_seed_SD`, pooled as **RMS** across sub-probes (SDs combine
in quadrature; a plain mean of SDs understates the aggregate spread). The input
comes from baseline B1, which is why B1 is the first thing M1 runs. Until B1 has
run, there is no threshold and the leaderboard cannot distinguish a real lead
from a lucky seed.

Note this is a per-slot SD, so it is **conservative** for the mean across 11
slots — if slot noise were independent, the mean's SD would be smaller by
√11. It is not independent (shared seed, shared pretrained checkpoint, shared
curation), so the conservative reading is the honest one until M1 measures the
actual correlation structure.

**FLOP accounting.** The handoff budgets with `6ND`, which omits attention
because attention does not scale with parameter count. At nanoeeg's shapes that
omission is not small: a 4-second, 64-channel window at 64-sample patches is
~512 tokens, and at 10M params / 12 layers / d_model 256 the attention term adds
**~30%**. Budgeting with `6ND` alone would hand every run roughly 30% more
compute than the cap allows. `flops_per_token` uses
`6N + 12·n_layers·d_model·seq_len`.

Net effect on the handoff's planning table: 10M params buys **1.27e10** tokens,
not 1.7e10 — about **2.8 epochs** over a 10k-hour corpus rather than 3.6. The
conclusion is unchanged and is the important part: **at 10–20M parameters the
corpus binds before the FLOPs do**, which is what promotes data curation from a
formality to the highest-leverage editable surface.

---

## D7 — the premise test moves from M2 to M1

The handoff's open question 4 — is a from-scratch 10–20M model in one H100-hour
actually competitive with a frozen public checkpoint plus a linear probe? — is
deferred to M2, i.e. answered *after* the reference pretrain is spent.

Moved to M1 as baseline **B2**. A frozen public checkpoint with a linear head
runs on the harness M1 is building anyway; it is one extra config. Answering it
before the GPU budget is committed rather than after is strictly better, and if
the answer is negative the project re-scopes toward adaptation having spent
nothing.

A negative answer is a **finding**, as the handoff itself says: "here is the
compute frontier at which from-scratch EEG pretraining stops paying" is a
publishable claim nobody has made with an open, reproducible probe suite behind
it.

---

## D8 — montage harmonisation defaults to the 10-20 subset

Open question 5 calls montage harmonisation across NEMAR's heterogeneous channel
sets "the hardest unglamorous problem here" and expects it to dominate M3.

Proposed default: **the 19-channel standard 10-20 subset**, present in nearly
every dataset; drop everything else. Lossy, and honest about being lossy.

The reason this is the right default rather than a compromise: it turns the
hardest problem into **leaderboard headroom**. Channel-agnostic architectures are
an editable surface, so a contributor who builds one that exploits the full
montage beats the default and the benchmark records exactly how much the extra
channels were worth. A hard problem that blocks M3 becomes a hard problem
somebody gets credit for solving.

**Not implemented** — M3 work, recorded here so M1 does not accidentally
foreclose it.
