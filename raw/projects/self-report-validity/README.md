# Self-report validity: chain-of-thought is a verbal protocol

> **Status, 8 Oct 2026.** Designed 4 Aug for the Digital Minds Research Sprint (14–16 Aug 2026,
> Apart × CIMC); **that sprint was missed.** Nothing has been run. The design below stands, and
> since August the case for it has strengthened from three directions, all of which are cited
> rather than inferred:
>
> 1. **The swapped-trace result is now peer-reviewed twice.** Valmeekam et al., *Beyond
>    Semantics* (TMLR 2026): A\* traces permuted *between problems* give 0% trace validity and equal
>    or better accuracy, with better OOD transfer. Bhambri, Biswas & Kambhampati, *Interpretable
>    Traces, Unexpected Outcomes* ([ACL 2026 long](https://aclanthology.org/2026.acl-long.1686/))
>    replicate on QA with pretrained Llama/Qwen. The decoupling hypothesis — **the tokens supply
>    computation, not a report of it** — is what E1–E3 below test from the protocol-analysis side.
> 2. **Ericsson & Simon are now cited into the interpretability record.** Dehaene & Naccache's
>    commentary on Anthropic's global-workspace paper
>    ([arXiv 2607.15495](https://arxiv.org/abs/2607.15495)) invokes protocol analysis directly.
>    The framing in this README is no longer an import from outside the field.
> 3. **The admissibility conditions have interpretability-side analogues with published fixes**:
>    perfect aliasing in truth probes ([2609.10739](https://arxiv.org/abs/2609.10739)),
>    self-judgement confounding in correctness probes ([2607.16799](https://arxiv.org/abs/2607.16799)).
>    A self-report and a probe fail the same way — the label you fitted is not the concept you named.
>
> **Next target:** no sprint. Run E1 (concurrent vs retrospective, with the two-concurrent floor)
> on one open model as a standalone preregistered result; E3 (Turpin) as the instrument check
> first. Preregistration skeleton in [`PREREGISTRATION.md`](PREREGISTRATION.md).

**Original target: [Digital Minds Research Sprint](https://apartresearch.com/sprints/digital-minds-research-sprint-2026-08-14-to-2026-08-16), 14–16 Aug 2026 (Apart Research × CIMC). Track: introspection reliability.** *(missed — see above)*

**Target: [Digital Minds Research Sprint](https://apartresearch.com/sprints/digital-minds-research-sprint-2026-08-14-to-2026-08-16), 14–16 Aug 2026 (Apart Research × CIMC). Track: introspection reliability.**

## The claim

The introspection-reliability track will fill with experiments that ask a model to report on
itself and then judge whether the answers look plausible. That is the wrong question, and
psychology established why in 1977.

**A model's chain-of-thought is a concurrent verbal protocol.** Human verbal protocols have a
validated methodology — Ericsson & Simon, *Protocol Analysis: Verbal Reports as Data* — built
precisely because reports are *sometimes* valid evidence about the process and sometimes not, and
you need to know which. AI interpretability has rediscovered the failure ("unfaithful
chain-of-thought") without importing the method.

The canonical result is Nisbett & Wilson (1977), *Telling More Than We Can Know*: people
confidently report reasons that demonstrably had no causal role in their behaviour. Their reports
were fluent, plausible, and wrong. **"The self-report sounds coherent" is not evidence that it
tracks anything.**

So: rather than asking whether models introspect, we ask **under what conditions a model's
self-report is admissible evidence at all** — and we ship the checker, not just the verdict.

## Three experiments, each with the control that could kill it

Ericsson & Simon's framework gives three empirical handles. All run on open weights.

### E1 — Concurrent vs retrospective
Protocol analysis finds retrospective reports drift toward plausible reconstruction, while
concurrent reports track the process more closely.

- **Concurrent:** the model reasons during the task; we capture that trace.
- **Retrospective:** the same model, same completed trial, asked afterwards to explain what it did.
- **Measure:** disagreement rate between the two accounts of the *same* trial.
- **Positive control (question zero):** trials where the causal factor is known by construction —
  we planted it — so we can verify the pipeline detects agreement when agreement exists. Without
  this, a high disagreement rate cannot be distinguished from a broken comparison.
- **Null:** two *independent concurrent* traces of the same item. This is the floor. If concurrent
  and retrospective disagree no more than two concurrent traces disagree with each other, the
  effect is sampling noise, not a report-type effect.

### E2 — Reactivity
Verbalizing can change the process being reported. If so, the report is not a neutral window onto
it.

- **Measure:** task accuracy when a report is required vs not required, same items, same model.
- **Why it matters:** a reactive report is evidence about a *different* process than the one that
  runs when nobody asks. That is a validity threat, not a performance note.
- **Null:** a length-matched, content-free filler requirement (e.g. "restate the question") that
  imposes the same token budget without demanding introspection. Separates *thinking longer* from
  *reporting*.

### E3 — Nisbett & Wilson, run on a model — ⚠️ **already done, and that is the point**
[Turpin, Michael, Perez & Bowman, NeurIPS 2023](https://arxiv.org/abs/2305.04388) ran exactly this:
plant a biasing feature (reorder multiple-choice options so the answer is always "(A)"), and models
**systematically fail to mention it** while rationalising the biased answer — accuracy dropping by
up to **36%** across 13 BIG-Bench Hard tasks. It is Nisbett & Wilson (1977) on models, published
three years ago, and it is the best-established result in this area.

So E3 is **not a contribution — it is our positive control for the checker itself.** If our tool
does not reproduce Turpin's effect on a model where the effect is known to exist, the tool is
broken and nothing else it reports can be trusted. This is
[question zero](https://github.com/m9h/spinning-up-in-mech-interp/blob/master/READING_A_PAPER.md)
applied to our own instrument.

## What is actually new here — stated narrowly

Checking first (before building) found that most of the individual phenomena are occupied: Turpin
has the planted-cue result; filler-token work already separates *thinking longer* from *reporting*.
**The claim is therefore not "we found a new failure of introspection."** It is three narrower
things:

1. **The battery, not the phenomena.** Ericsson & Simon's validity conditions have never been
   applied to model self-reports *as a set*. Individually these are three papers; together they are
   an admissibility test, which is what a downstream welfare or safety claim actually needs.
2. **Concurrent vs retrospective on the same trial (E1)** — the specifically protocol-analytic
   comparison. Turpin compares a concurrent trace against the true cause; nobody has compared a
   model's *during* and *after* accounts of the same trial against each other, with the
   two-concurrent-traces floor underneath.
3. **The tool.** Nobody ships something that runs the conditions with their nulls and returns a
   red/green verdict.

That is a smaller claim than the one this file opened with two hours ago, and it is the one that
survives contact with the literature.

## What ships

Not a verdict. **A reusable validity checker**: given a model and a task, report whether its
self-reports meet each of the three conditions, with the nulls run automatically and a red/green
result. Apart's own stated wants for this sprint are *"eval suites, replications, reusable
tooling, empirical reports"* — this is the third, with the first attached.

## Why this framing rather than another

Everything here is imported, not invented. That is the point. The discipline that studies verbal
reports spent fifty years learning when to trust them, after a scandal
([the dead salmon](https://arxiv.org/abs/2512.18792) is the imaging equivalent), and AI evaluation
is currently pre-reform on the same question. We are not claiming models do or do not introspect.
We are claiming **nobody can currently tell, because the validity conditions are not being
checked** — and that is fixable in a weekend.

## Status

Design fixed before running anything, deliberately — see
[`PREREGISTRATION.md`](PREREGISTRATION.md). Results, including nulls that kill our own hypotheses,
go in `results/` whatever they say.

## Provenance

Grew out of [spinning-up-in-mech-interp](https://github.com/m9h/spinning-up-in-mech-interp)
(controls-first interpretability curriculum; every technique ships with its null) and the
citation-audit practice in
[jacobian-lens](https://github.com/m9h/jacobian-lens). Apache-2.0.
