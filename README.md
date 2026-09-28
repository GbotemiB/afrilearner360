# AfriLearner360

Culturally-grounded assessment item generation for Rwandan primary education (P1–P5), built to
run entirely on free-tier language models at zero API cost.

This repository is the artifact for **"Culturally-Grounded Assessment Item Generation for Primary
Education Under Zero-Budget Constraints: A Rwandan Case Study"** (GlobalSouthAI @ NeurIPS 2026).

> **No classroom validation has been done.** No pilot has run, and no teacher or learner has used
> this system. Everything measured here is a property of the generation and scoring pipeline,
> measurable without human subjects. Nothing in this repository is evidence about learning
> outcomes.

## What this is

Three constraints shaped the design: no per-learner device, no budget for paid model access, and
a teacher between generated content and learners rather than direct delivery.

The pipeline has four stages:

1. **Item generation.** A language model writes forced-choice assessment items grounded in a
   locale-specific cultural knowledge base, under a strict JSON schema.
2. **Response capture.** A learner splits a fixed point budget across four options per item.
3. **Scoring.** Deterministic, non-AI code turns allocations into a continuous engagement-trait
   vector over four traits (visual, game, structured, story).
4. **Rendering.** A language model turns the numeric profile into a teacher-readable summary.

The model is used only at steps 1 and 4. **No model output influences a score.**

### What the system deliberately does not do

It does not match learners to a preferred modality. The meshing hypothesis, that instruction
works best when matched to an assessed learning style, is not supported by the evidence, and
validating it would require a crossover interaction that the literature has repeatedly failed to
produce. Trait vectors are consumed **only in aggregate**, as a generation-budget signal deciding
which representation of a lesson to generate first under rate limits. Every learner then
encounters every generated representation.

## Repository layout

```
llm/
  src/afrilearner360_llm/
    item_generation/     prompts, JSON schema, OpenRouter client, item-bank persistence
    profiling/           scorer.py — deterministic scoring (no model calls)
    recommendations/     teacher-facing summary generation
    clustering/          k-means over trait vectors — see "Not used by the paper" below
    common/              config and locale loading
  data/
    cultural_knowledge_base/rwanda.md      the grounding corpus
    locales/rwanda.yaml                    trait definitions, point budget, delivery languages
    item_banks/rwanda/approved/P5.json     example generated items (reviewed)
    item_banks/rwanda/drafts/              raw generation output, pre-review
    responses/P5A.json                     34 synthetic response sets
    recommendations/                       example generated teacher summaries
  scripts/eval/          the evaluation harnesses behind every number in the paper
  tests/                 118 tests
AfriLearner360.zip       React teacher-facing prototype (mocked data, no backend)
backend/, frontend/      placeholders, not implemented
```

## Mapping paper claims to code

| Paper claim | Produced by | Recorded in |
|---|---|---|
| Table 1, schema adherence across free models | `scripts/eval/exp1_schema_reliability.py` | `results/exp1_*.json`, `results/summary.md` |
| Failure taxonomy (missing fields, wrong root container) | `exp1_schema_reliability.py:classify_error_list` | `results/raw.json` |
| Determinism of the scoring layer | `scripts/eval/exp2_determinism.py`, `fixtures.py` | `results/exp2_determinism.json` |
| Free-tier model availability survey | `scripts/eval/catalogue_snapshot.py` | `results/catalogue/catalogue_*.json` |
| Cost and latency projection | `scripts/eval/cost_projection.py` | `results/cost_projection.json` |
| Deterministic scoring, no model in the loop | `src/afrilearner360_llm/profiling/scorer.py` | `tests/test_profiling.py` |
| Cultural grounding, retrieval-style not parametric | `item_generation/prompts.py` + `data/cultural_knowledge_base/rwanda.md` | `data/item_banks/rwanda/approved/P5.json` |
| Anti-deficit constraints in teacher-facing text | `recommendations/prompts.py` | `data/recommendations/` |

The evaluation harnesses **import the shipping code paths** rather than reimplementing them
(`SYSTEM_PROMPT`, `build_user_prompt`, `ItemGenerationResponse`, `generator._parse_response`,
`profiling.scorer.score_student`). A failure they record is a failure the pipeline would hit.
Nothing under `src/` was modified to make the experiments run.

## Key results

**Structured-output reliability.** Every free model tested emitted syntactically valid JSON on
every attempt. Schema adherence separated them completely, along the declared capability rather
than model identity:

| Model | Capability declared | N | JSON parse | Schema adherence | First attempt | p50 latency | Compl. tokens |
|---|---|---|---|---|---|---|---|
| `nvidia/nemotron-3-super-120b-a12b:free` | `structured_outputs` | 10 | 1.00 | **1.00** | 1.00 | 40.7 s | 2889 |
| `dots-studio/dots-3-note-preview:free` | `structured_outputs` | 9 | 1.00 | **1.00** | 1.00 | 96.5 s | 9072 |
| `liquid/lfm-2.5-2.6b:free` | `structured_outputs` | 10 | 1.00 | **1.00** | 1.00 | 31.9 s | 4631 |
| `minimax/minimax-m3:free` | `response_format` | 10 | 1.00 | **0.00** | 0.00 | 15.6 s | 1183 |

Valid JSON is not valid output. A pipeline that checks only parseability will accept unusable
items. Note also that 17 of 20 replies from the non-conforming model arrived wrapped in markdown
code fences despite strict schema enforcement being requested, so fence-stripping is load-bearing
rather than defensive.

Conformance is not obviously free, but the cost is harder to attribute than a single pair
suggests: conforming models ran at 2.0× to 6.2× the median latency of the non-conforming one, yet
the spread *within* the conforming group is wider than the gap between groups. With one model in
the non-conforming class, capability and model identity cannot be separated.

**Determinism.** 25 fixtures (17 scoring to a profile, 8 expected to raise), covering single-trait
dominance, exact four-way ties reached two ways, both classifier thresholds at their boundary, and
every rejected input. Scored profiles were byte-identical across 250 within-process scorings and
across three additional interpreters under `PYTHONHASHSEED` 0, 1 and 42.

One non-determinism was found and is **reported rather than repaired**: a validation error path
interpolates an unordered `set` of enum members into its message, so the message text varies by
process. It affects diagnostics only, never a scored profile. It is recorded because the
within-process check passed completely and would have concealed it.

**Model availability.** Of 458 models on OpenRouter on 2026-09-27, 17 were free and only 4
declared server-side JSON-schema enforcement. Availability is also non-stationary: the free
variants of `minimax/minimax-m3` and `z-ai/glm-5.2` were withdrawn between evaluation and
camera-ready while their **paid** entries remained. The capability did not disappear, it stopped
being free. Three further free models returned HTTP 429 from a provider-shared pool with our own
quota untouched.

**Cost.** A full term for P1–P5, 10 topics in 4 representations, is 200 generation calls: 2.26
hours of model time, roughly $0.81 at paid rates, but **4.0 days** under the free tier's 50
requests/day cap. Throughput, not price, is the binding constraint. Total API expenditure for this
project was zero.

## Not used by the paper

`src/afrilearner360_llm/clustering/` implements k-means over learner trait vectors. **The paper
does not use it and does not endorse it.** Grouping learners by engagement trait is the same
unsupported premise as modality matching, one level up. Where the system supports differentiated
grouping at all, the grouping variable is demonstrated competency with reassessment, which has an
evidence base that trait-based assignment does not. The module is retained because it is tested
and may be useful for competency-based grouping later, not because trait clustering is
recommended.

## Running it

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
cd llm
uv sync
cp .env.example .env     # add a free OpenRouter API key
```

**Tests** (no API calls, no key needed):

```bash
uv run pytest -q          # 118 tests
```

**Reproduce the evaluations:**

```bash
# offline, no quota consumed
uv run python scripts/eval/exp2_determinism.py
uv run python scripts/eval/cost_projection.py
uv run python scripts/eval/catalogue_snapshot.py      # one unauthenticated GET

# consumes free-tier quota (50 requests/day)
uv run python scripts/eval/exp1_schema_reliability.py --model <model-id> --n 10

# collate both experiments into results/summary.md
uv run python scripts/eval/make_summary.py
```

**Generate items:**

```bash
uv run python scripts/generate_items.py --topic "fractions" --grade-band P3 --num-items 10
```

Generated items land in `data/item_banks/<locale>/drafts/`. Promotion to `approved/` is a
deliberate human act; nothing automates it. `load_approved()` refuses a set with duplicate ids,
items still flagged `needs_translation`, a language mismatch, or the wrong item count.

### Reproducibility caveats

Free-tier model availability changes. Models named above may be unreachable, may no longer be
free, or may no longer exist as free variants. `exp1` records HTTP 429 and blocked tasks rather
than failing, so a run against an unavailable model produces an honest empty result rather than a
crash. This is a property of the substrate, not a defect in the harness, and it is one of the
paper's findings.

## Limitations

- **No classroom validation.** No pilot, no teacher or learner feedback.
- **Cultural grounding is unvalidated.** `rwanda.md` was assembled from public online sources, not
  elicited from local teachers. A local validation pass is a prerequisite for deployment.
- **Response data is synthetic.** `data/responses/P5A.json` holds 34 generated response sets
  seeded by intended trait. It demonstrates pipeline self-consistency, not measurement validity.
- **The traits are not validated as a construct.** Determinism shows identical input gives
  identical output. It does not show the traits measure anything about learners. Establishing that
  would require, at minimum, showing trait distributions from real responses differ from random
  responding.
- **Evaluation scale is small.** 9–10 distinct requests per model across four models, bounded by
  free-tier caps and latency. The figures establish a qualitative separation between capability
  classes, not precise adherence rates.
- **Text only.** The free models available cannot generate images, so items are text-only despite
  the design calling for visual items.
- **Not offline-capable.** Generation requires a live API call.

## License

MIT. See [LICENSE](LICENSE).
