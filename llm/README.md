# AfriLearner360 — LLM/AI Component

This is the LLM/model/AI piece of **AfriLearner360**, an adaptive learning assistant for African
primary education (CMU Africa SIP project). This repo owns the assessment item generation,
student profiling, classroom clustering, and teaching recommendation pieces of the system —
it's designed to be dropped into (or integrated alongside) the rest of the team's Flask/Postgres
app, which teammates are building separately.

## MVP scope

Assessment intake → profiling → clustering → teacher recommendations. Content transformation,
formative mini-assessments, and mode-switching are explicitly Phase 2, not in this MVP.

## How students get profiled (the four traits)

Students respond to short scenario items by splitting a fixed point budget (default: 10) across
four options, one per engagement trait:

- **visual** — learning through pictures, diagrams, colors, and visual demonstration
- **game** — learning through play, competition, and interactive challenges
- **structured** — learning through clear steps, rules, and organized procedures
- **story** — learning through narrative, characters, and storytelling

This is a forced-choice (ipsative) format, chosen because it's more resistant to acquiescence /
"rate everything high" bias than independent Likert ratings. **Known limitation:** ipsative
scoring produces trait proportions relative to each student's own total rather than a fully
comparable absolute scale (see Hicks, 1970, and the modern Thurstonian IRT literature for the
proper statistical fix). This MVP uses simple sum-and-normalize scoring and documents this as a
known limitation rather than solving it — proper IRT-based scoring is future work.

## Cultural adaptation

The system is explicitly **not** one-size-fits-all. Each deployment locale gets its own:

1. **Assessment content pack** — the scenario/option content shown to students, grounded in real
   local material (see `data/cultural_knowledge_base/`), not the LLM's own general knowledge.
2. **Recommendation context profile** — resource constraints, pedagogical norms, and language
   preferences that condition how teaching recommendations are phrased.

Both live together in `data/locales/<locale>.yaml`, keyed to the same locale so the two stages
stay consistent. The Rwanda pack (`data/locales/rwanda.yaml` +
`data/cultural_knowledge_base/rwanda.md`) is the pilot; it's built from public web research for
MVP speed, not live local teacher/student interviews — a local-teacher review pass is
recommended before real deployment.

## Item generation pipeline

1. `data/cultural_knowledge_base/<locale>.md` — grounding material, organized by which trait it
   best informs.
2. `item_generation/prompts.py` — the system prompt (rules an item must follow: use only the
   supplied knowledge base, keep all four options equally appealing, no gendering, age-
   appropriate language, etc.) and the per-request user prompt template.
3. `item_generation/generator.py` — calls the LLM (via OpenRouter's OpenAI-compatible API,
   default model `google/gemma-4-26b-a4b-it:free`) with structured-output enforcement against
   `item_generation/schema.py`, so a malformed item fails loudly instead of reaching students.
   The default is a **free-tier** model, so no API credits are needed; free models are capped at
   20 requests/minute and 50 requests/day, so pass a larger `--num-items` to batch rather than
   making many small calls. Swap models with the `AFRILEARNER_LLM_MODEL` env var.
4. **Human review is required before any generated item goes live.** Each item's
   `cultural_anchors_used` and `reviewer_notes` fields exist specifically to make that review
   fast — a reviewer can see exactly which cultural references were used and what the model
   itself flagged as uncertain.

### Items are always drafted in English

The LLM writes every item in English, regardless of grade band. A live test showed the free model
produces structurally valid but linguistically broken Kinyarwanda — it repurposed
`amashirahamwe` (associations/cooperatives) to mean "patterns" and admitted the substitution in
its own reviewer note. So **Kinyarwanda is produced by a human translator during review, not by
the model.**

Items destined for a different delivery language come back flagged, so an untranslated draft
can't be mistaken for a finished one:

```json
{ "language": "english", "delivery_language": "swahili", "needs_translation": true }
```

The locale YAML separates two concepts: `generation_language` (what the model writes) and
`delivery_language_by_grade_band` (what students must receive). The system prompt also tells the
model to keep sentences short and avoid idioms and wordplay that wouldn't survive translation.

**For Rwanda, both bands now deliver in English**, so nothing currently needs translating. Lower
primary was originally Kinyarwanda, but since no free model writes usable Kinyarwanda, that made
every P1-P3 item wait on a human translator and blocked the band entirely.

⚠️ **This only holds if P1-P3 items are administered orally** — the teacher reads each scenario
and its options aloud, rendering them in Kinyarwanda, and records the child's allocation. Only
38% of lower-primary students read grade-level English. Handed to a 7-year-old as text, the
assessment measures English reading ability rather than engagement preference. For lower primary
this is a deployment requirement, not a suggestion. See DESIGN.md §4.1 and §6.

## The item bank

Generated items are stored as JSON on disk, split into two directories that encode the review
gate structurally rather than by convention:

```
data/item_banks/rwanda/
  drafts/    P3-fractions-20260813T104500.json   raw LLM output, NOT usable with students
  approved/  P3.json                             the fixed assessment: reviewed + translated
```

Generation only ever writes to `drafts/`. Nothing in this codebase promotes a draft
automatically — a human reviews it for relatability and stereotyping, supplies the translation
where `needs_translation` is set, and saves the result to `approved/`.

`item_bank.load_approved()` then refuses to serve a set that isn't fit for students. It rejects
duplicate item ids (one item scored twice skews a profile), items still flagged
`needs_translation`, items whose `language` doesn't match their `delivery_language`, items from
the wrong locale or grade band, and — when given a locale config — the wrong item count or point
budget.

**Every student in a class answers the same items.** Sampling different items per student would
make k-means cluster partly on *which items a student got* rather than on their traits — the
differential-item-functioning risk from DESIGN.md §4.3 arriving through the back door.

**Assessment length is 10 items** (`items_per_assessment` in the locale YAML), chosen for
resolution. Each item allocates 10 points, so N items give 10N total and one point moves a trait
proportion by `10/N` percentage points: 1.0pp at 10 items, but 3.3pp at 3. The profiler's 12pp
balanced and 15pp secondary thresholds only mean something if the underlying proportions are
finer-grained than they are.

JSON rather than a database: the review gate is a human editing this content, JSON diffs readably
in a pull request, and the operational database belongs to the team's Flask app. That app reads
`approved/`; this repo produces it.

## Profiling

`profiling/scorer.py` turns a student's item responses into a trait profile. It is deterministic
plain code, no LLM call:

1. Sum the points the student gave to each trait across every item they answered.
2. Divide by their total, giving four proportions that sum to 100.
3. **Primary trait** = the highest proportion. **Secondary trait** = the runner-up, but only when
   it's within 15 percentage points of the primary — beyond that, the student has one clear
   dominant mode and naming a runner-up would overstate it. **Balanced** = when the spread
   between the highest and lowest trait is under 12 points, no mode really dominates, so the
   profile is flagged `is_balanced` and no secondary is reported.

Both thresholds are arguments to `score_student()`, not hardcoded, so they can be tuned against
real pilot data.

```python
from afrilearner360_llm.profiling import ItemResponse, score_student

profile = score_student(
    student_id="s-001",
    responses=[ItemResponse(item_id="rwanda-P3-fractions-01", allocations={...})],
    expected_point_budget=10,   # optional: raises if an item's points don't add up
)
profile.proportions      # {visual: 50.0, game: 25.0, structured: 15.0, story: 10.0}
profile.primary_trait    # Trait.VISUAL
profile.vector()         # [50.0, 25.0, 15.0, 10.0] -- fixed TRAIT_ORDER, feeds clustering
```

Every trait must have an allocation, including explicit zeros — a missing key raises rather than
being treated as 0, since it's far more likely a serialization bug than a real answer.
`vector()` always returns proportions in the same `TRAIT_ORDER`, so clustering feature positions
stay stable across runs.

## Clustering

`clustering/cluster.py` groups a class by their trait vectors using k-means with a **fixed** k
(default 4). Fixed rather than auto-selected per class: silhouette-based k is more statistically
defensible but gives an unpredictable number of groups per classroom, and a teacher planning
station activities needs the same number of groups every term. k-means rather than simply
bucketing by dominant trait, because bucketing throws away the blend — two students both "primary
visual", one at 80% and one at 35%, don't need the same teaching.

A fixed `random_state` makes it deterministic. The same roster always produces the same grouping,
which matters the moment a teacher asks why a particular child is in a particular group.

```python
from afrilearner360_llm.clustering import fit_classroom, assign_student, is_stale

result = fit_classroom(profiles, k=4)          # cold start over the whole roster
for summary in result.summaries:
    print(summary.cluster_id, summary.size, summary.descriptor())
    # 2 10 strongly visual
    # 3 8  balanced across modes

assign_student(result.model, new_profile)      # mid-term joiner, nearest centroid
is_stale(result.model, max_age_days=90)        # is a scheduled recompute due?
```

The three lifecycle paths:

1. **Cold start** — `fit_classroom()` over the whole initial roster.
2. **Mid-term joiner** — `assign_student()` places them at the nearest existing centroid. It
   deliberately does *not* refit: refitting on every arrival would reshuffle students the teacher
   has already grouped and invalidate recommendations in use. The cost is centroid drift, which
   the scheduled recompute corrects.
3. **Scheduled recompute** — `is_stale()` reports whether the model has aged past a term. It's a
   predicate, not a trigger; the scheduling belongs to whatever owns the school calendar. The
   design chose scheduled over threshold-based reclustering so groups reset at moments teachers
   can anticipate.

`ClusterModel` serializes to JSON so a class is clustered once and later joiners are placed
against the stored model. Loading refuses a model whose `trait_order` doesn't match the current
one — otherwise feature positions would shift silently and students would land in wrong groups.
Empty clusters are reported with size 0 rather than dropped, so a teacher promised k groups is
told when one came out empty.

## Recommendations

`recommendations/` turns cluster summaries into teaching-method guidance for the teacher — how to
approach each group, not the lesson content itself (content generation is Phase 2).

```bash
uv run python scripts/generate_recommendations.py \
    --responses data/responses/P5A.json --topic "fractions" --k 4
```

That command runs the whole chain: score every student, cluster the class, generate per-group
recommendations, save them. The responses file is the contract with the team's Flask app —
`class_id`, `grade_band`, and each student's per-item point allocations.

**Two modes**, since class sections vary in how they're processed:

| Mode | Requests | Trade-off |
|---|---|---|
| `whole_class` (default) | 1 | Model sees all groups, so it differentiates them against each other and plans how they coexist. One bad response loses everything. |
| `per_cluster` | k | Failure-isolated and retryable per group, but each group is planned blind to the others — so class-level advice is assembled in code rather than authored by the model. |

**Stored and retrievable**, keyed by class section so P5A and P5B stay separate:

```
data/recommendations/rwanda/P5A/P5-fractions-20260813T144220.json
```

```python
from afrilearner360_llm.recommendations import latest_recommendations
latest_recommendations("rwanda", "P5A", topic="fractions")   # None if nothing stored yet
```

No drafts/approved gate here, unlike the item bank. Assessment items are an instrument
administered to children, so they get a blocking human review; recommendations are advisory
guidance to a professional who can disregard a bad suggestion.

### The guardrails

This is the one artifact a teacher actually reads, and it's where an LLM most easily undoes the
project's framing. Asked for per-group teaching advice, models drift into deficit language —
"this group struggles", "slower learners" — which reintroduces exactly the fixed-ability labeling
the project moved away from (DESIGN.md §1). The system prompt therefore forbids ability or
deficit language, forbids implying a trait is fixed, requires all groups be framed as equally
capable, requires pitfalls be phrased about the *teaching* rather than the children, and forbids
permanently separating students.

Sample of real output against those rules, from the live run above:

> **Watch:** This approach can lose pace if the sequence of steps is overly long or if
> instructions are not clearly numbered.

Phrased about the method, not the children — which is the intent.

`constraints_addressed` and `cultural_anchors_used` mirror the item schema's traceability fields,
so a reviewer can check grounding at a glance rather than re-reading everything.

Why an LLM for item generation but not for profiling/clustering: item generation is a language-generation task
(interpreting cultural context into a well-formed scenario) — a good fit for an LLM. Profiling
(turning point-allocations into trait proportions) and clustering (k-means over those vectors)
are pure computation with no ambiguity to resolve, so they're plain deterministic code instead —
faster, free, auditable, and reproducible. This split is a general design principle for the
whole project, not just this stage.

## Setup

Dependencies live in `pyproject.toml` (there is no `requirements.txt`). The project uses
[uv](https://docs.astral.sh/uv/); `uv.lock` is committed so everyone gets identical versions.

```bash
uv sync                # creates .venv and installs everything, including dev deps
cp .env.example .env   # then fill in OPENROUTER_API_KEY from https://openrouter.ai/
```

Not using uv? `python -m venv .venv && source .venv/bin/activate && pip install -e .` works for
the runtime deps (add `--group dev` on pip ≥ 25.1, or `pip install pytest`, to run the tests),
but won't respect the lockfile.

## Usage

```bash
uv run python scripts/generate_items.py --topic "fractions" --grade-band P3
```

Writes draft items to `data/item_banks/<locale>/drafts/`, timestamped so repeated runs accumulate
instead of overwriting each other. Defaults to the locale's `items_per_assessment` (10); override
with `--num-items`, redirect with `--out`, or add `--print` to also dump JSON to stdout.

**Drafts are never served to students.** Promoting a draft to `approved/` is a deliberate human
act — see the item bank section below.

```bash
uv run pytest -q
```

## Project layout

```
data/
  cultural_knowledge_base/   # locale grounding material (markdown)
  locales/                   # locale config: languages, point budget, recommendation context
src/afrilearner360_llm/
  item_generation/           # prompts, schema, LLM calls -- implemented
  profiling/                 # deterministic scoring -- implemented
  clustering/                # k-means classroom clustering -- not yet implemented
  recommendations/           # per-cluster LLM recommendations -- not yet implemented
  common/                    # config + locale loading shared across modules
scripts/                     # CLI entry points
tests/
```

## Status

| Piece | Status |
|---|---|
| Item generation (prompts, schema, API call) | Implemented, tested |
| Rwanda locale pack (knowledge base + config) | Implemented |
| Profiling (deterministic scoring) | Implemented, tested |
| Clustering (k-means + lifecycle) | Designed, not coded |
| Recommendation generation | Designed, not coded |
| Human review/approval tooling for generated items | Not designed yet — manual step |
| Integration with the team's Flask/DB app | Out of scope for this repo |

Design rationale for every decision above — including the ones deliberately *not* taken — is in
[`DESIGN.md`](DESIGN.md).
