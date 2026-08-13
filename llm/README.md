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
   default model `minimax/minimax-m3`) with structured-output enforcement against
   `item_generation/schema.py`, so a malformed item fails loudly instead of reaching students.
4. **Human review is required before any generated item goes live.** Each item's
   `cultural_anchors_used` and `reviewer_notes` fields exist specifically to make that review
   fast — a reviewer can see exactly which cultural references were used and what the model
   itself flagged as uncertain.

Why an LLM here but not for profiling/clustering: item generation is a language-generation task
(interpreting cultural context into a well-formed scenario) — a good fit for an LLM. Profiling
(turning point-allocations into trait proportions) and clustering (k-means over those vectors)
are pure computation with no ambiguity to resolve, so they're plain deterministic code instead —
faster, free, auditable, and reproducible. This split is a general design principle for the
whole project, not just this stage.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY from https://openrouter.ai/
```

## Usage

```bash
python scripts/generate_items.py --topic "fractions" --grade-band P3 --num-items 3
```

Prints draft items as JSON to stdout. Nothing here writes to a "live" item bank automatically —
that's an intentional gate until a human review step exists.

## Project layout

```
data/
  cultural_knowledge_base/   # locale grounding material (markdown)
  locales/                   # locale config: languages, point budget, recommendation context
src/afrilearner360_llm/
  item_generation/           # prompts, schema, LLM calls -- implemented
  profiling/                 # deterministic scoring -- not yet implemented
  clustering/                # k-means classroom clustering -- not yet implemented
  recommendations/           # per-cluster LLM recommendations -- not yet implemented
  common/                    # config + locale loading shared across modules
scripts/                     # CLI entry points
tests/
```

## Status

Item generation (prompts + schema + API call) is implemented. Profiling, clustering, and
recommendation generation are scoped and designed (see design notes / project memory) but not
yet implemented in code.
