# AfriLearner360 — LLM/AI Component: Design Notes

This file is the handoff record of every design decision made for this component so far, with
the reasoning behind each one. It exists so work can continue in a different tool (e.g. Claude
Code) without losing the "why" behind the code. If you're picking this up fresh: read this end
to end before changing the pipeline shape.

## 1. Project context

AfriLearner360 is an AI-powered adaptive learning assistant for African primary education, built
as a CMU Africa SIP (Summer Internship/Project). It's a team project — this repo covers only the
**LLM/model/AI component** (assessment generation, student profiling, classroom clustering,
teaching recommendations). Teammates are building the rest of the stack separately (Flask app,
PostgreSQL, frontend, deployment to Vercel), so this repo is scoped narrowly on purpose and is
meant to be dropped into or integrated alongside that work, not to duplicate it.

The project's framing shifted, prior to this repo existing, away from classifying students into
fixed/permanent "learning style" labels (a framing that drew research critique) and toward
identifying **evolving engagement traits** and empowering teachers, rather than labeling
students. That shift is why everything downstream is built around proportions and dominance
rather than hard categories.

## 2. MVP scope

**In scope:** assessment → profiling → clustering → teacher recommendations.

**Explicitly out of scope for MVP (Phase 2):** automated curriculum content transformation
(taking a topic and generating full lesson materials in each mode), the continuous
formative-assessment / mode-switching loop, and full culturally-contextualized content
generation beyond the assessment items themselves.

Reasoning: the team has a 10-day build window. Stopping at "diagnose and recommend" rather than
also "generate the adapted materials" keeps the MVP's core claim — that this saves teachers time
even for large classes — testable without needing to also solve content generation quality and
review at the same time.

## 3. The four engagement traits

Every student gets scored across exactly four modes, not sorted into one:

- **Visual** — learning through pictures, diagrams, colors, patterns, visual demonstration
- **Game** — learning through play, competition, challenges, interactive activities
- **Structured** — learning through clear steps, rules, sequences, organized procedures
- **Story** — learning through narrative, characters, storytelling

A student's profile is a proportion across all four (e.g. 42% Visual / 28% Game / 18% Structured
/ 12% Story), not a single label. This directly reflects the project's non-binary framing: a
student can show up in all four traits, with one or two having real dominance.

## 4. Pipeline stage by stage

### 4.1 Assessment intake

**Format decision: forced-choice point allocation, not independent Likert ratings.** Each
scenario gives the student a fixed point budget (default 10) to split across four options, one
per trait, rather than rating each option separately on its own scale.

*Why:* Forced-choice naturally produces proportions that sum to a constant, which is exactly the
shape the profile needs — no extra normalization logic required. More importantly, independent
ratings are vulnerable to response bias: an agreeable/enthusiastic child can rate every option
highly, a shy or literal-minded child can rate everything low, and neither pattern tells you which
trait is *relatively* dominant — only how positively that child responds to questions in general.
Forced-choice makes trade-offs unavoidable, so the resulting numbers reflect relative preference.

**Research check performed (see Known Limitations, §6):** literature does support forced-choice
being more resistant to acquiescence bias than Likert formats (near-zero correlation with
acquiescence bias in at least one comparative study), with modest resistance to social
desirability bias too. But Likert scales — especially psychometrically corrected ones — showed
notably higher reliability, and forced-choice/ipsative data carries a real, well-documented
comparability caveat. Decision: keep the 4-way point-split for MVP, document the limitation
rather than switching to pairwise comparisons (the format more directly validated with young
children in the preference-assessment literature) or building full IRT-based scoring (the
statistically correct fix, but too much infrastructure for a 10-day MVP).

### 4.2 Profiling engine — deterministic code, NOT an LLM call

Raw point allocations are summed per trait across all items a student answered, then divided by
the total points allocated, producing a proportion vector that sums to 100%. From that:
- **Primary trait** = highest proportion.
- **Secondary trait** = next-highest, but only if within a closeness threshold of the primary
  (e.g. ~15 points) — otherwise the student has one clear dominant trait.
- **Balanced profile** = if the gap between highest and lowest trait is small (e.g. <10-15
  points), label the student as balanced across modes rather than forcing an arbitrary primary.

**Why this is plain code, not an LLM call:** this is pure arithmetic with no ambiguity to
resolve — there's no free text to interpret, since responses are numeric point allocations. Using
an LLM here would add cost, latency, and non-determinism for zero benefit, and would actively hurt
auditability for a system profiling children (you want to be able to say "42% because of this
math," not "the LLM said so"). This is the general design principle for the whole project: **use
an LLM where the task is generating or interpreting language** (item scenarios, cultural
grounding, teacher-facing narrative/recommendations); **use plain code where the task is
computation** (scoring, normalization, k-means distance). An LLM can still turn the *numeric*
profile into a readable narrative for teachers afterward — that's presentation layered on top of
deterministic math, not a replacement for it, and probably belongs inside the Stage 4 call.

Not yet implemented in code (see `src/afrilearner360_llm/profiling/`, currently a stub).

### 4.3 Classroom clustering

**Fixed small k (k=3-4) via k-means** over student trait-proportion vectors, chosen over two
alternatives:
- *Dynamic cluster count* (auto-picking k per class via silhouette score) — more statistically
  accurate but produces an unpredictable number of groups per class, which is harder to build a
  stable teacher-facing UI/template around for an MVP.
- *Rule-based binning by dominant trait* — simplest to implement (bucket by primary trait,
  max 4 buckets) but throws away the blended-profile nuance that's central to this project's
  whole non-binary framing.

K-means itself needs no cultural awareness (it's pure distance in feature space). The risk to
guard against is **differential item functioning**: if one assessment item bank were reused
across meaningfully different cultural contexts without adaptation, clusters could end up
reflecting item familiarity rather than genuine trait differences. Mitigated by keeping item
banks locale-specific (§5).

**Clustering lifecycle (handles new students joining mid-term):**
1. *Cold start* — k-means runs on the whole initial class roster.
2. *New student joins mid-term* — assigned via nearest-centroid to an existing cluster. Cheap,
   instant, and doesn't reshuffle other students or invalidate recommendations the teacher is
   already using.
3. *Full re-clustering* — happens on a **schedule** (e.g. start of each term), not
   threshold-based (e.g. after N new students) or purely manual (teacher-triggered refresh) —
   chosen for predictability, so teachers know groupings reset at known moments rather than
   unpredictably. This is also the natural point to fold in a student's own periodic profile
   re-assessment (the ~2-year cognitive-development refresh from the original project vision).

Not yet implemented in code (see `src/afrilearner360_llm/clustering/`, currently a stub).

### 4.4 Recommendation generation — LLM

Per-cluster, topic-aware **teaching-method** recommendations for the teacher — not generated
lesson content itself, just how-to-teach guidance (e.g. "for this group, favor visual aids and
step-by-step structure"). This is an LLM task because it's language generation conditioned on
context, the same category as item generation (§5).

The prompt for this stage should be conditioned on a **locale context profile** (see
`data/locales/rwanda.yaml`, `recommendation_context` section) — realistic classroom resource
constraints, local pedagogical norms, local analogies, and language preference — keyed to the
same locale as the class's assessment pack so both stages stay consistent for a given deployment.

Not yet implemented in code (see `src/afrilearner360_llm/recommendations/`, currently a stub).

### 4.5 Delivery

Teacher sees a Diversity Report (cluster breakdown of the class) plus per-cluster
recommendations. Not yet built (owned by the team's Flask/frontend layer, outside this repo).

## 5. Cultural adaptation layer

**This is explicitly not a one-size-fits-all system.** Rwanda (specifically a public primary
school) is the pilot locale, but the architecture is: generic item templates and generic
pipeline logic, plus a swappable **locale pack** supplying the surface content and context. A new
country/region means writing a new locale pack, not redesigning the pipeline.

### Where the cultural layer plugs in — and where it doesn't

Researched via web search on culturally-responsive-assessment literature: validity risk is
highest wherever content or scenarios are interpreted by humans (item wording, LLM-generated
text), and near-zero wherever the pipeline is pure math. So:
- **Assessment intake (high leverage):** item scenarios need locally relatable content, or a
  student's response reflects unfamiliarity with the scenario rather than genuine trait
  preference. This is a real *validity* concern, not just a nice-to-have.
- **Profiling math (low leverage):** the normalization/dominance arithmetic is culture-agnostic;
  accuracy depends entirely on upstream item validity (garbage in, garbage out). Worth validating
  with local teachers only that the 4 trait *category definitions* map onto local pedagogical
  concepts.
- **Clustering (low leverage, one risk):** k-means doesn't care about semantics; the risk is
  differential item functioning if item banks aren't locale-specific (see §4.3).
- **Recommendation generation (high leverage):** the natural place to inject resource-constraint
  awareness, local pedagogical norms, and local analogies into generated text.

One explicit caution carried through the whole design: don't treat "cultural awareness" as a
single generic layer — that risks essentializing/stereotyping, since there's huge diversity
within and across African countries and regions. The design uses locale-specific packs validated
(or at least reviewed) with people from that specific context, not a generic "African" layer.

### Rwanda locale pack

`data/cultural_knowledge_base/rwanda.md` + `data/locales/rwanda.yaml`. Content areas: games &
play (Igisoro, Guhana Ubute, rope jumping, etc. — Game-based items), oral storytelling (Imigani
folktales, Ibisakuzo riddles — Story-based items), visual arts & crafts (Imigongo geometric art,
Agaseke baskets — Visual items), community/structured activity (Umuganda — Structured items), and
public-school context (pupil-teacher ratios, double-shift schooling, English-literacy data —
feeds the *recommendation* context, not the items themselves).

**Grounding source decision:** the original preferred approach was live local teacher/student
interviews (leveraging CMU Africa's Kigali presence for direct access) to build this knowledge
base. For MVP speed, this was revised to internet-sourced research instead — faster, but lower
authenticity, since sources are a mix of institutional (UNICEF, education-policy outlets) and
informal/tourism sites. **A local-teacher validation/spot-check pass is still recommended before
real deployment** — this was explicitly not dropped, just deferred past MVP.

### Cultural item generation workflow (why an LLM is used here, and how)

General LLMs have shallow, skewed knowledge of specific local culture — left ungrounded, they
tend to produce generic or stereotyped content. The fix used here: **retrieval-style grounding**,
not parametric generation. The LLM is instructed (see `item_generation/prompts.py`
`SYSTEM_PROMPT`) to draw *only* from the supplied knowledge base file, never to invent cultural
details. Pipeline:
1. Knowledge base (§5, Rwanda pack) organized by which trait it best informs.
2. Knowledge base + generic item template + trait definitions → LLM, which drafts candidate
   forced-choice items.
3. **Human (ideally local teacher) validation gate** before any item goes live — checks
   relatability, absence of stereotyping, reading level, language. This gate is not yet built as
   tooling; right now it's a manual step reviewers do against the JSON output.
4. Approved items enter the live locale item bank (not yet implemented — currently
   `generate_items.py` just prints drafts to stdout).

Two schema fields exist specifically to make step 3 fast: `cultural_anchors_used` (which specific
knowledge-base references the model drew on, for traceability) and `reviewer_notes` (the model's
own flagged uncertainties).

## 6. Known limitations (carry into any project write-up)

- **Ipsativity of forced-choice scoring.** The simple sum-and-normalize scoring produces ipsative
  data — a student's trait proportions are relative to their own total, not on a shared absolute
  scale. This is a well-documented psychometric limitation (Hicks, 1970, and ongoing research) for
  comparing profiles *across* students. The statistically correct fix is Thurstonian IRT modeling
  (Brown & Maydeu-Olivares), which converts forced-choice responses into properly comparable
  scores — but that requires real statistical modeling infrastructure beyond MVP scope. This is a
  conscious, documented trade-off, not an oversight.
- **Forced-choice item complexity for young children.** The literature that most directly
  validates forced-choice with young children (used in special-education/behavioral-intervention
  preference assessment) uses simpler *pairwise* comparisons (two options at a time), not
  simultaneous 4-way point-splitting. The 4-way format was kept for MVP speed and because it maps
  more directly onto the proportional-profile design; pairwise is a documented alternative worth
  reconsidering if pilot data shows children struggling with the task.
- **Cultural content is internet-sourced, not locally validated**, for MVP speed. A lighter local
  teacher review/validation pass is recommended before real classroom deployment.
- **"MiniMax H3" (from the original project brief) could not be verified** as a real model name.
  The current MiniMax lineup (mid-2026) is the M-series (e.g. M3). Worth confirming with whoever
  wrote the original brief whether "H3" was a placeholder, typo, or planned future release.

## 7. LLM/API choice

**Decision: MiniMax M3, accessed via OpenRouter** (OpenAI-compatible API), not called directly
against MiniMax's own API.

Why MiniMax M3: 1M token context window (comfortably holds the full locale knowledge base as
grounding without needing a retrieval/chunking step), very low cost (~$0.23/$0.96 per million
input/output tokens as of mid-2026), multimodal (text/image/video in), and it matches what the
original project brief was pointing at.

Why OpenRouter as the access layer rather than MiniMax's API directly: exposes many providers
(MiniMax, DeepSeek, Gemini, Llama, Claude, GPT) behind one API key and one OpenAI-compatible
interface, so a different model is a one-line config change (`AFRILEARNER_LLM_MODEL` env var) if
MiniMax has quality or availability issues. Also gives free-tier models for early pipeline
prototyping before spending on real generation. At MVP scale, actual generation cost is trivial
regardless of model choice (a few hundred items, not high-volume traffic) — the real drivers were
context length and easy API integration, not price.

## 8. What's implemented vs. not

| Piece | Status |
|---|---|
| Item generation: system prompt, schema, API call | Implemented, tested (`tests/test_schema.py`) |
| Rwanda locale pack (knowledge base + config) | Implemented |
| Profiling (scoring function) | Designed (§4.2), not coded |
| Clustering (k-means + lifecycle) | Designed (§4.3), not coded |
| Recommendation generation | Designed (§4.4), not coded |
| Human review/approval tooling for generated items | Not designed yet — currently a manual step |
| Integration with the team's Flask/DB app | Out of scope for this repo |

## 9. Suggested next steps

1. Implement `profiling/scorer.py` per §4.2 (deterministic, unit-testable).
2. Implement `clustering/cluster.py` per §4.3, including the nearest-centroid assignment path for
   new joiners and the scheduled-recompute path.
3. Implement `recommendations/` generation, reusing the same prompt-design pattern as
   `item_generation/` (system prompt + locale-conditioned user prompt + structured schema).
4. Decide on and build the human-review step for generated assessment items (even a simple
   CLI/spreadsheet-based approve/reject flow would close the loop described in §5).
5. `git init` this repo and push to GitHub; coordinate with teammates on how it plugs into the
   shared Flask app.
6. When time allows: a lightweight local-teacher review pass on the Rwanda cultural knowledge
   base (§5), to convert it from "internet-sourced first draft" to "spot-checked."
