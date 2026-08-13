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

**Assessment language: English for all grade bands, with lower primary administered orally.**

Originally P1-P3 items were to be delivered in Kinyarwanda, on the evidence that only 38% of
lower-primary students read grade-level English (knowledge base, `rwanda.md`). That was reversed
for MVP: no available free model writes usable Kinyarwanda (§7), so a Kinyarwanda delivery
language meant every P1-P3 item required a human translation pass, and none had been produced —
lower primary was blocked entirely.

*What makes this defensible:* **teacher-mediated oral administration.** The teacher reads each
scenario and its four options aloud, rendering them in Kinyarwanda as they go, and records the
child's point allocation. Under that model the English text is a teacher-facing script, not
student-facing reading material. This is also closer to how an assessment of 7-year-olds would
realistically be run in a 44:1 (often worse) classroom regardless of language.

*What it costs, stated plainly:* if a P1-P3 assessment is ever handed to children as text, it
measures English reading ability rather than engagement preference, and the forced-choice design
of this section — which assumes the child understands all four options well enough to trade off
between them — stops holding. The oral-administration assumption is therefore a **deployment
requirement for lower primary, not a nice-to-have**, and it shifts the translation burden from a
per-item authoring task to a live, per-session task carried by the teacher. That is a real
trade-off: it removes a blocking dependency but introduces variability, since two teachers may
render the same option differently.

Upper primary (P4-P6) is unaffected — those students are the nominal English-instruction cohort
and the items are delivered as written.

The translation machinery is configured, not deleted: `delivery_language_by_grade_band` still
supports a non-English delivery language per band, and items still come back flagged
`needs_translation` when it differs. Rwanda simply no longer exercises that path.

**Assessment length: 10 items, the same 10 for every student in a class.**

*Why a fixed set rather than sampling per student:* if students answer different items, k-means
downstream clusters partly on *which items a student happened to get* rather than on their traits
— the differential-item-functioning risk of §4.3 reintroduced through the back door. A fixed set
also keeps the ipsative comparability caveat (§6) from getting any worse than it already is.

*Why 10:* this is resolution arithmetic. Each item allocates `point_budget` (10) points, so N
items give 10N total and a single point moves a trait proportion by `10/N` percentage points — 1.0pp
at 10 items, 1.25pp at 8, 3.3pp at 3. The profiler's thresholds are 12pp (balanced) and 15pp
(secondary); at 3 items the balanced threshold is only ~3.6 points of allocation difference, so
profiles come out chunky and small differences flip the classification. 10 keeps the thresholds
meaningful without making the sitting unreasonably long. Configured per locale
(`items_per_assessment`), because attention span for P1-P2 may yet argue for fewer — that is a
pedagogical call for the team, and lowering it means revisiting the thresholds too.

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

Implemented in `clustering/cluster.py`. Notes from building it:

- **Determinism was made explicit.** `random_state` is fixed (0) and `n_init=10`, so the same
  roster always produces the same grouping. Unstable groupings would be indefensible the first
  time a teacher asks why a child is in a particular group, and it is the same auditability
  argument that keeps profiling out of the LLM (§4.2).
- **`ClusterModel` is serializable**, because the mid-term joiner path requires it: a class is
  clustered once at term start, the model is stored, and each later joiner is placed against it
  without the original roster being present.
- **Loading validates `trait_order`** against the current `TRAIT_ORDER` and refuses a mismatch.
  A model saved under a different ordering would silently shift feature positions and put
  students in wrong groups with no visible error — the worst failure mode available here.
- **Empty clusters are reported, not dropped** (size 0, described by their own centroid). A
  teacher promised k groups should be told one came out empty rather than quietly handed fewer.
- **Rosters smaller than k raise** rather than silently reducing k, as do duplicate student ids.
  Both would otherwise produce a meaningless grouping.
- **`is_stale` is a predicate, not a trigger.** The scheduled-recompute decision needs a school
  calendar, which lives in the team's Flask app; this module answers the question when asked.
- Cluster descriptors ("visual-leaning, game-secondary", "balanced across modes") are generated
  by deterministic string formatting over the same dominance rules used for individual students —
  `classify_proportions`, extracted from the profiler so the two cannot drift apart. Naming a
  group is not an LLM task; turning the group into teaching guidance (§4.4) is.

### 4.4 Recommendation generation — LLM

Per-cluster, topic-aware **teaching-method** recommendations for the teacher — not generated
lesson content itself, just how-to-teach guidance (e.g. "for this group, favor visual aids and
step-by-step structure"). This is an LLM task because it's language generation conditioned on
context, the same category as item generation (§5).

The prompt for this stage should be conditioned on a **locale context profile** (see
`data/locales/rwanda.yaml`, `recommendation_context` section) — realistic classroom resource
constraints, local pedagogical norms, local analogies, and language preference — keyed to the
same locale as the class's assessment pack so both stages stay consistent for a given deployment.

Implemented in `recommendations/`, same four-part shape as `item_generation/` (prompts, schema,
generator, plus a store). Decisions made while building it:

- **Anti-deficit guardrails are the core of the system prompt, not a footnote.** This is the only
  artifact a teacher reads end to end, and an LLM asked for per-group teaching advice drifts
  naturally into "this group struggles" / "slower learners". That would reintroduce the fixed
  ability labeling this project was explicitly reframed away from (§1) — in the most visible
  place possible. The prompt forbids ability/deficit language, forbids implying a trait is
  permanent, requires every group be framed as equally capable, requires `watch_out_for` be about
  the teaching rather than the children, and forbids permanently separating students. Verified
  against a live run: output was clean, and pitfalls came back as "this approach can lose pace
  if the sequence of steps is overly long" rather than anything about the children.
- **Two call modes, selectable** (`RecommendationMode`). `whole_class` sends every cluster in one
  request, so the model differentiates groups against each other and plans how they share a room;
  it also costs 1 request rather than k against a 50/day free tier. `per_cluster` sends one
  request per group — failure-isolated and retryable — but no call ever sees the whole class, so
  the class-level advice (`class_overview`, `running_the_class`) is generated deterministically in
  code for that mode rather than attributed to a model that lacked the information to write it.
  Both modes return the same artifact so callers do not branch.
- **Grounded in two sources**, not just the constraints block: the locale's
  `recommendation_context` (class size, double-shift hours, no-device assumption) stops the model
  recommending resources that do not exist, and the cultural knowledge base is what produces
  "explain dividing seeds using Igisoro mechanics" instead of generic "use manipulatives" (§5
  flags this stage as high-leverage for local grounding).
- **Coverage is validated, not assumed.** `ClassroomRecommendations` rejects a response that
  misses a cluster, duplicates one, or invents one. A missing cluster means a group of real
  children silently has no guidance.
- **Persisted, no approval gate.** Stored under `data/recommendations/<locale>/<class_id>/`,
  timestamped, retrievable via `latest_recommendations()`. Keyed by class section because a school
  runs several. Deliberately lighter than the item bank's drafts/approved split: items are an
  instrument administered to children and warrant a blocking gate, while recommendations are
  advice to a professional who can disregard them. `reviewer_notes` still carries model flags.
- **The caller's cluster id wins.** In per-cluster mode the model is told which cluster it is
  advising on but can echo the wrong id; the generator overrides it, since advice silently
  attached to the wrong group is worse than a visible error.

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
4. **Human translation** for any item whose delivery language differs from English (i.e. all
   lower-primary items in the Rwanda pack). Added after the first live run showed the model
   cannot write usable Kinyarwanda — see §7. Items arrive flagged `needs_translation`, so an
   untranslated draft can't be confused for a finished one.
5. Approved items enter the live locale item bank (`item_generation/item_bank.py`). Storage is
   plain JSON under `data/item_banks/<locale>/`, split into `drafts/` (what generation writes,
   timestamped so runs accumulate) and `approved/` (what may be served, one file per grade band).
   Promotion between the two is a deliberate human act — nothing in the code does it
   automatically, which is what makes steps 3 and 4 a real gate rather than a convention.
   `load_approved()` refuses a set with duplicate item ids, items still flagged
   `needs_translation`, a `language`/`delivery_language` mismatch, a wrong locale or grade band,
   or (given a locale config) the wrong item count or point budget.

   JSON rather than a database: the review gate is a human editing this content, JSON diffs
   readably in a pull request, and the operational database belongs to the team's Flask app
   (out of scope, §1). That app reads `approved/`; this repo produces it.

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
- **Lower primary depends on oral administration.** Because no available free model writes usable
  Kinyarwanda (§7), P1-P3 items are now generated and stored in English and expected to be read
  aloud by the teacher in Kinyarwanda at assessment time (§4.1). This unblocked lower primary, but
  it means the instrument's validity for P1-P3 rests on an assumption about *how* it is
  administered, which the code cannot enforce. Two risks follow: handing the text directly to
  children would silently turn it into an English reading test, and live teacher translation
  introduces session-to-session variability that authored translations would not have. Worth
  revisiting if a Kinyarwanda-capable model becomes available or a standing translation
  arrangement exists.
- **"MiniMax H3" (from the original project brief) could not be verified** as a real model name.
  The current MiniMax lineup (mid-2026) is the M-series (e.g. M3). Worth confirming with whoever
  wrote the original brief whether "H3" was a placeholder, typo, or planned future release.

## 7. LLM/API choice

**Decision: `google/gemma-4-26b-a4b-it:free`, accessed via OpenRouter** (OpenAI-compatible API).

### Why OpenRouter as the access layer

Exposes many providers (Google, MiniMax, DeepSeek, Llama, Claude, GPT) behind one API key and one
OpenAI-compatible interface, so switching models is a one-line config change
(`AFRILEARNER_LLM_MODEL` env var) rather than a code change. It also aggregates free-tier models
from many providers under a single account, which is what makes the zero-credit constraint below
survivable.

### Why a free model, and why this one

**Hard constraint: the project has no API credits.** That, not context length or benchmark
quality, is the binding constraint on model choice.

Of 409 models on OpenRouter, 18 are free, and only 6 of those support **`structured_outputs`** —
server-side JSON-schema enforcement. That capability is not optional here: `generator.py` sends
`response_format: {type: "json_schema", strict: true}` and relies on the model returning
schema-conforming output. Without it, schema conformance degrades to best-effort and malformed
items surface as pydantic `ValidationError`s instead. That single filter eliminates two thirds of
the free tier, including the *larger* free Gemma (`gemma-4-31b-it:free` supports only loose
`response_format`, while the smaller MoE `gemma-4-26b-a4b-it:free` supports full structured
outputs — a genuine trap, since the bigger model is the worse fit).

Among the 6 survivors, the choice is driven by **Kinyarwanda quality**, since lower-primary items
must be written in Kinyarwanda (§5). Google's Gemma line advertises far broader language coverage
than the alternatives (NVIDIA Nemotron and GPT-OSS are English/reasoning-focused; Llama 3.3's
official multilingual list is ~8 languages and excludes Kinyarwanda). `openrouter/free` was
rejected outright despite meeting the schema requirement: it routes to a *randomly selected* free
model per request, making it impossible to record which model produced a given item — that breaks
the reviewer traceability the §5 validation gate depends on.

### Result of the first live run: the model cannot write Kinyarwanda

This was flagged as the pipeline's highest-risk untested assumption, then tested. **It failed.**

Two live calls were made against `google/gemma-4-26b-a4b-it:free` on the same topic (fractions),
one at P3 (Kinyarwanda delivery) and one at P5 (English delivery):

- **English output was good.** Imigongo and Agaseke used correctly and idiomatically, all four
  options balanced in length and equally appealing, correct grounding, `reviewer_notes: "no
  concerns"`. Usable as a draft.
- **Kinyarwanda output was structurally perfect but linguistically broken.** Schema, traits, and
  cultural anchors were all correct; the language itself was not. Concrete failures:
  `amashirahamwe` (associations/cooperatives) used repeatedly to mean "patterns" — with the model
  conceding the substitution in its own `reviewer_notes`; `umunyamakuru` (journalist) appearing in
  `"inkuru y'agaseke k'umunyamakuru w'amabara"`; `"amapaji y'agaseke yatsindwe"` reading as "pages
  of the basket that were defeated". The same broken phrase was recycled across both items,
  suggesting vocabulary collapse rather than isolated slips.

Caveat on that assessment: it was made by identifying word-level meanings against dictionary
definitions, not by a Kinyarwanda speaker, and still warrants confirmation from one. But the
model's own admission that it repurposed a word is difficult to explain away.

**Decision: the LLM now drafts in English only, always** (`generation_language` in the locale
YAML). Kinyarwanda is produced by a human translator inside the §5 review gate, not by the model.
Items whose delivery language differs from English are returned flagged `needs_translation` so no
untranslated item can be mistaken for a ready one. The system prompt additionally instructs the
model to keep sentences short and avoid idioms, puns, and wordplay that would not survive
translation, and to leave cultural terms in their original form.

Options considered and rejected: testing other free models (the alternatives are more
English-centric than Gemma, so likely worse); keeping Kinyarwanda generation as a rough draft for
a reviewer to rewrite (the observed drafts may be worse than starting from nothing); paying for a
stronger model (no credits, and no guarantee any model handles Kinyarwanda well).

Cost of this decision: the "fully automated pipeline" claim no longer holds for lower primary —
those items require a human in the loop before they can reach students. That is a real reduction
in scope, and an honest one.

### Rate limits shape how the pipeline is used

Free (`:free`) models on OpenRouter are capped at **20 requests/minute and 50 requests/day** on an
account that has never purchased credits (1,000/day after a one-time $10 purchase). The cap is on
*requests*, not tokens, so generation should batch — one call with `--num-items 10` costs the same
quota as one call with `--num-items 1`. At 50 calls/day that is ~500 draft items/day, far beyond
MVP need.

This request-based capping is also why OpenRouter beat **Groq's free tier**, which was evaluated
as an alternative. Groq is faster and its strict-schema models (`openai/gpt-oss-120b`/`20b`) are
capable, but its free tier is **token**-capped (8K TPM / 200K TPD for gpt-oss-120b). This
pipeline re-sends the entire locale knowledge base (~1,340 tokens) as grounding on every single
call — that is the whole retrieval-style grounding strategy from §5, not incidental overhead.
Token-based caps charge for that grounding on every request; request-based caps do not. Groq
remains a reasonable fallback provider (also OpenAI-compatible, so only the base URL and key
would change), but wiring it was deliberately deferred to keep MVP config simple.

### Paid upgrade path, if free-tier quality disappoints

**MiniMax M3** (`minimax/minimax-m3`) was the original choice before the no-credits constraint was
confirmed, and remains the documented upgrade: 1M token context, multimodal, strict structured
outputs, at **$0.30/$1.20 per million input/output tokens** (as of the check that produced this
section — an earlier draft of this doc quoted $0.23/$0.96, which was stale). At this pipeline's
size a call costs roughly **$0.0026**, so a 200-item bank is about **$0.50** — cost was never the
real obstacle, only the absence of any billing setup. Note that M3's 1M context window, cited in
the original rationale, is not actually load-bearing: a full item-generation call is only ~2,200
tokens, so any model above ~32K context is sufficient. If locale knowledge bases grow by orders of
magnitude, that changes.

### Known gap

`generator.py` has no retry or backoff. Hitting the 20 requests/minute ceiling raises straight out
of the CLI. Worth adding before any bulk generation run.

## 8. What's implemented vs. not

| Piece | Status |
|---|---|
| Item generation: system prompt, schema, API call | Implemented, tested, verified against the live API |
| Rwanda locale pack (knowledge base + config) | Implemented |
| Profiling (scoring function) | Implemented, tested (`tests/test_profiling.py`) |
| Clustering (k-means + lifecycle) | Implemented, tested (`tests/test_clustering.py`) |
| Recommendation generation | Implemented, tested, verified against the live API |
| Item bank persistence (drafts/approved on disk) | Implemented, tested (`tests/test_item_bank.py`) |
| Human review/approval tooling for generated items | Storage + validation gate exist; the review UI/flow itself is still manual |
| Integration with the team's Flask/DB app | Out of scope for this repo |

## 9. Suggested next steps

1. ~~Implement `profiling/scorer.py` per §4.2~~ — done (`profiling/scorer.py`, 15 tests).
2. ~~Implement `clustering/cluster.py` per §4.3~~ — done (`clustering/cluster.py`, 23 tests),
   including the nearest-centroid joiner path and the scheduled-recompute predicate.
3. ~~Implement `recommendations/` generation~~ — done (`recommendations/`, 22 tests), with both
   call modes, persistence keyed by class section, and the anti-deficit guardrails described in
   §4.4. Future expansion the team flagged: accept a curriculum object as input rather than a
   free-text topic, and recommend against it. The schema was shaped so that slots in without
   reshaping the artifact.
4. Build the human-review *flow* on top of the storage layer (§5 step 5 now exists: drafts land on
   disk and `load_approved` validates the gate). What's still missing is the reviewer-facing
   part — a simple CLI or spreadsheet approve/reject pass that walks `needs_translation` items,
   collects the Kinyarwanda text, and writes the approved file. Today a reviewer hand-edits JSON.
5. **Get a Kinyarwanda speaker to confirm the §7 language assessment.** The English-only decision
   rests on a non-speaker's word-level analysis. If a speaker judges the output salvageable, the
   trade-off is worth revisiting; CMU Africa's Kigali presence makes this cheap to check.
6. Add retry/backoff to `generator.py`. Free models are capped at 20 requests/minute and hitting
   that ceiling currently raises straight out of the CLI, which will bite during bulk generation.
7. `git init` this repo and push to GitHub; coordinate with teammates on how it plugs into the
   shared Flask app.
8. When time allows: a lightweight local-teacher review pass on the Rwanda cultural knowledge
   base (§5), to convert it from "internet-sourced first draft" to "spot-checked."
