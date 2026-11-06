# Spinning Up in Mechanistic Interpretability

**From vision circuits to the global workspace — a hands-on, controls-first curriculum for reading the inside of neural networks, and getting to where the 2025–2026 papers begin.**

<p align="center">
  <img src="https://img.shields.io/badge/rungs-8-blue" alt="8 rungs"/>
  <img src="https://img.shields.io/badge/models-open--weight_only-green" alt="Open weights"/>
  <img src="https://img.shields.io/badge/every_technique-shipped_with_a_control-8a2be2" alt="Controls first"/>
  <img src="https://img.shields.io/badge/license-Apache_2.0-orange" alt="License"/>
</p>

---

In 2020, a group at OpenAI opened up an image classifier and found **curve detectors** —
individual neurons that fire for curves, wired together into legible **circuits**. They
argued that neural networks are not inscrutable: they are built from **features** connected
by **weights**, and you can reverse-engineer them. In 2021 that program moved to
transformers, and over the next five years produced a chain of ideas — the residual stream,
superposition, induction heads, sparse autoencoders, attribution graphs — that leads, in
2025–2026, to claims that language models contain an **introspectable** subspace, a
**global workspace**, and a **society of thought**.

Those recent papers are exciting and easy to misread. They are also mostly run on models no
outsider can open. This curriculum walks the chain from the beginning, on **open-weight
models you can run yourself**, and it teaches one discipline the source papers often skip:
**every technique is paired with the control that could kill it.** By the last rung you can
not only read the 2025–2026 papers — you can reproduce their methods on open weights and
check whether the claims survive a null.

> Modeled on OpenAI's [Spinning Up in Deep RL](https://spinningup.openai.com/) and the
> [`spinning-up-in-*`](https://github.com/m9h) family. Companion to the
> [Consciousness-Indicator Scorecard](https://github.com/m9h/jacobian-lens) — the last rung
> *is* the first cells of that benchmark.

## How this fits with other resources

There is good mech-interp material already. This one exists for three reasons none of the
others combine.

| | [ARENA](https://arena.education/) | [Neel Nanda's TransformerLens tutorials](https://www.neelnanda.io/mechanistic-interpretability) | [Distill Circuits](https://distill.pub/2020/circuits/) + [Transformer Circuits](https://transformer-circuits.pub/) | **This curriculum** |
|---|---|---|---|---|
| **Format** | engineering bootcamp (weeks) | explainer + notebooks | primary research articles | short curriculum, 8 project rungs |
| **Vision-circuits origin** | — | — | ✅ (the origin) | ✅ rung 1, taught where it's cleanest |
| **Transformer foundations** | ✅ deep | ✅ deep | ✅ (the papers) | ✅ rungs 2–5, on the shoulders of both |
| **A control with every technique** | named once, in a bonus¹ | — | rarely | ✅ **the through-line** — each rung ends by running its null |
| **Open-weights-only, laptop-first** | mostly | ✅ | mixed (recent work is closed) | ✅ **hard requirement** — no frontier access anywhere |
| **On-ramp to the 2025–26 property papers** | — | — | they *are* the papers | ✅ **the destination** (rung 8): J-space, introspection, societies-of-thought, metacognition |
| **Adjudication mindset** | — | — | — | ✅ reproduce-and-check; negatives are first-class |

¹ ARENA §1.3.3 identifies the problem precisely — that autointerp on *randomized* SAE latents
scores better than expected, because top-k over a large dataset finds patterns in anything — but
leaves it as an optional bonus bullet with no starter code or solution. It is the best statement
of the issue in any existing course, and it is an aside. See [rung 5](rung5_sparse_autoencoders/).

**The gap we fill:** existing courses teach the *techniques*; the source threads *are* the
recent papers but run them on closed models with few controls. Nobody teaches the chain as
a **controls-first, open-weights on-ramp whose explicit endpoint is the ability to
adjudicate the 2025–2026 claims yourself.** That endpoint is the
[Scorecard](https://github.com/m9h/jacobian-lens) program; this repo is how you learn to
read its cells.

## Who is this for?

- Anyone who wants to *understand* the recent interpretability papers (global workspace,
  introspection, societies of thought) rather than take them on faith.
- Students and researchers entering mechanistic interpretability who want to build, not just
  read.
- People who have seen a striking interpretability result and want to learn the reflex of
  asking "what is the control?"

**Prerequisites:** Python, PyTorch or JAX basics, and comfort with linear algebra and
softmax. No prior interpretability experience. **No GPU required** — every runnable rung was
verified end-to-end on a laptop CPU in under 25 seconds each.

### Start here

| | |
|---|---|
| [**GETTING_STARTED.md**](GETTING_STARTED.md) | install, run, and the exact output each rung should produce |
| [**GLOSSARY.md**](GLOSSARY.md) | the vocabulary of the papers — residual stream, QK/OV, superposition, SAE, patching |
| [**READING_A_PAPER.md**](READING_A_PAPER.md) | six questions for reading an interpretability result critically |
| [**PITFALLS.md**](PITFALLS.md) | every trap that cost us a wrong result while building this |
| [**ECOSYSTEM.md**](ECOSYSTEM.md) | what is live, what died (Microscope, Distill, ARENA's old URLs), the benchmarks you can enter, and the communities that will take a contribution |
| [**reading-group/**](reading-group/) | **the Cognitive Hexagon reading group** — read the source construct before the AI paper that borrows it |

```bash
pip install -r requirements.txt && ./verify_all.sh
```

---

## The curriculum

Eight rungs. Every rung has the same shape: **read the paper → run the technique on an open
model → run the control that could falsify it.** Each `rungN_*/` directory is a
self-contained project with its own README (the reading, the build, the null).

### Part 0 — Build the model first (prerequisite)

| rung | project | what you learn | primary reading |
|---|---|---|---|
| [0](rung0_build_the_model/) | **Build the model** | construct a transformer (and an RL loop) by hand, so the rest is concrete | Karpathy *nanoGPT*; Raschka *Build an LLM / a Reasoning Model from Scratch*; Mike X Cohen |

### Part I — Where the paradigm came from (vision)

| rung | project | what you learn | primary reading |
|---|---|---|---|
| [1](rung1_features_and_circuits/) | **Features & circuits** | that a network is features wired by weights; find one circuit end-to-end | Olah et al., *Zoom In* (2020); *Curve Circuits* |

### Part II — Transformers, from the ground up

| rung | project | what you learn | primary reading |
|---|---|---|---|
| [2](rung2_residual_stream/) | **The residual stream** | QK/OV, the residual stream as a channel | Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021) |
| [3](rung3_superposition/) | **Superposition** | why features hide; the toy autoencoder, on a laptop | Elhage et al., *Toy Models of Superposition* (2022) |
| [4](rung4_induction_heads/) | **Induction heads** | the circuit behind in-context learning; ablate it | Olsson et al., *In-Context Learning and Induction Heads* (2022) |
| [5](rung5_sparse_autoencoders/) | **Sparse autoencoders** | pulling monosemantic features out of superposition; steering one | Bricken et al., *Towards Monosemanticity* (2023); *Scaling Monosemanticity* |

### Part III — Reading and tracing computation

| rung | project | what you learn | primary reading |
|---|---|---|---|
| [6](rung6_lenses/) | **Lenses** | reading the residual stream into vocabulary: logit → tuned → **Jacobian** lens | Gurnee et al., *Verbalizable Representations…* (2026) + the lens lineage |
| [7](rung7_attribution_graphs/) | **Attribution graphs** | following a computation across layers | Ameisen et al., *Circuit Tracing*; Lindsey et al., *On the Biology of an LLM* (2025) |

### Part IV — The recent claims, adjudicated (capstone)

| rung | project | what you learn | primary reading |
|---|---|---|---|
| [8](rung8_property_claims/) | **Property claims** | reproduce a 2025–26 claim on open weights *with its null*: workspace, introspection, society-of-thought, metacognition | the four recent papers + [our results](https://github.com/m9h/jacobian-lens) |

## How each rung is built

Every `rungN_*/README.md` follows the same structure:

1. **The idea** — the concept in a paragraph, and why it is a rung on this ladder.
2. **Read** — the primary paper (and the one prerequisite reading), with links.
3. **Build** — the software project: a specific technique to implement on a named open model.
4. **The control** — the null, baseline, or ablation that could falsify what you just built,
   and what passing/failing it means. *This is the point of the rung.*
5. **Toward the recent papers** — the sentence of the 2025–26 papers this rung unlocks.

Rungs **1–5 and 7 ship a tested, self-contained `starter.py`** you can run today — each
finds a real result on an open model and then runs its own null, using only
`torch` / `transformers` / `torchvision` (canonical tools like TransformerLens, SAELens,
lucent and `circuit-tracer` are named as scale-ups, not required). Rung 3 needs only a laptop
CPU; rungs 1, 2, 4, 5, 7 download a small open model (GPT-2 or InceptionV1) once. Rungs 6 and
8 draw directly on our working research code
([`jacobian-lens`](https://github.com/m9h/jacobian-lens),
[`jlens-lab`](https://github.com/m9h/jlens-lab),
[lenses on the Hub](https://huggingface.co/mhough/olmo3-jacobian-lenses)).

What each runnable starter finds, and the null that guards it:

| rung | finds (on an open model) | the null it runs |
|---|---|---|
| 1 | an orientation/edge detector in InceptionV1 (channel #48, selectivity 1.000) | **two** nulls: random-init (Adebayo) *and* the stronger weight-shuffle |
| 2 | GPT-2's copying heads (OV z ≈ +5.8) and previous-token head **L4H11** (1.00) | **random matrix** OV ≈ 0; **uniform-attention** QK baseline |
| 3 | superposition — >m features packed into m dims | **random dictionary** recovery floor |
| 4 | GPT-2's induction heads (L5H5 …) | **ablate random heads** → in-context loss barely moves |
| 5 | an SAE feature you can interpret *and* steer | **random-direction** steer nets ~0; **negation** reverses |
| 7 | the IOI computation moving subject-token → final-token across layers | **median (layer,pos) patch** restores ~0% |

## Resources

- **Build-it-from-scratch (rung 0):** Karpathy,
  [`nanoGPT`](https://github.com/karpathy/nanoGPT) +
  [Zero to Hero](https://karpathy.ai/zero-to-hero.html) · Raschka,
  [*Build an LLM from Scratch*](https://github.com/rasbt/LLMs-from-scratch) and
  [*Build a Reasoning Model from Scratch*](https://github.com/rasbt/reasoning-from-scratch) ·
  Mike X Cohen, [*50 ML projects to understand LLMs*](https://github.com/mikexcohen/ML4LLM_book) (investigate transformer internals as data)
- **Post-training, hands-on:** [TRL](https://github.com/huggingface/trl) v1.0 (SFT + reward
  modelling + DPO + GRPO in one stack) and the
  [smol-course](https://huggingface.co/learn/smol-course/unit0/1).
  The three layers fit together: Lambert's course is the **theory** of post-training, TRL is the
  **implementation**, and this repo is the **measurement** — what those stages do to a model's
  representations, with a control attached.
- **Post-training (the training side of rung 8):** Nathan Lambert,
  [*RLHF Book*](https://rlhfbook.com) + [course](https://rlhfbook.com/course) — free, 11
  lectures, and Conversation 2 is a case study in DPO **for OLMo 3**, the model rung 8 measures.
  Lectures and theory, no exercises; complementary to this repo's runnable projects.
- **Foundational threads:** [Distill Circuits](https://distill.pub/2020/circuits/) ·
  [Transformer Circuits Thread](https://transformer-circuits.pub/)
- **Tooling:** [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) ·
  [SAELens](https://github.com/decoderesearch/SAELens) ·
  [Gemma Scope](https://huggingface.co/google/gemma-scope) ·
  [circuit-tracer](https://github.com/decoderesearch/circuit-tracer) ·
  [Neuronpedia](https://neuronpedia.org)
- **Our companion work:** [jacobian-lens](https://github.com/m9h/jacobian-lens) ·
  [jlens-lab](https://github.com/m9h/jlens-lab) · the Consciousness-Indicator Scorecard
- **Further:** Nanda, [200 Concrete Open Problems in Mechanistic Interpretability](https://www.alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability)

## License

Apache-2.0. Course text CC BY 4.0. An
[Orthogonal Research and Education Lab (OREL)](https://orthogonal-research-lab.github.io/) project.

## Related

Part of one program — a controls-first, open-weights attempt to make the 2025–26 interpretability
claims checkable:

| repo | what |
|---|---|
| [spinning-up-in-mech-interp](https://github.com/m9h/spinning-up-in-mech-interp) | the **curriculum** — 8 rungs, 6 runnable on a laptop, each ending in its own null |
| [jacobian-lens](https://github.com/m9h/jacobian-lens) | the **research** — OLMo post-training ladder, metacognition, the Consciousness-Indicator Scorecard |
| [tri-lens](https://github.com/m9h/tri-lens) | do **three instruments agree** about the same activation? |
| [societies-of-thought](https://github.com/m9h/societies-of-thought) | the **adversarial replication** — rebuild a no-code/no-data paper, then try to break it |
| [controls-and-trajectories](https://github.com/m9h/controls-and-trajectories) | the **published datasets** — nulls and developmental trajectories |

Datasets: [induction-emergence-pythia](https://huggingface.co/datasets/mhough/induction-emergence-pythia) ·
[inceptionv1-tuning-atlas](https://huggingface.co/datasets/mhough/inceptionv1-tuning-atlas) ·
[trilens-instrument-agreement](https://huggingface.co/datasets/mhough/trilens-instrument-agreement) ·
[olmo3-jacobian-lenses](https://huggingface.co/mhough/olmo3-jacobian-lenses)
