# Evaluation results

Date run: **2026-09-06** (UTC). Locale: `rwanda`. All raw per-attempt records in `raw.json`; raw model replies in `raw/`.

Both experiments call the shipping code paths rather than reimplementations: Experiment 1 imports `SYSTEM_PROMPT`, `build_user_prompt`, `ItemGenerationResponse` and `generator._parse_response`; Experiment 2 imports `profiling.scorer.score_student`. No file under `src/` was modified.

## Experiment 1 -- structured-output reliability

Each task is one (grade band, topic) pair requesting 3 items; tasks are drawn from a 5 grade band x 6 topic grid so N distinct requests are made, not one request repeated N times. The engine itself does not retry, so retries (max 2 per task) are added by the eval harness to make attempts-to-success measurable.

| Model | JSON schema mode | Tasks (N) | Requests | `json_parse_rate` | `schema_adherence_rate` | `first_attempt_success_rate` | Mean attempts to success | Latency p50 (s) | Latency p95 (s) | Mean prompt tok | Mean completion tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `dots-studio/dots-3-note-preview:free` | `structured_outputs` | 9 | 9 | 1.00 | 1.00 | 1.00 | 1 | 96.5 | 130.2 | 1975 | 9072 |
| `liquid/lfm-2.5-2.6b:free` | `structured_outputs` | 10 | 10 | 1.00 | 1.00 | 1.00 | 1 | 31.9 | 70.3 | 2000 | 4631 |
| `minimax/minimax-m3:free` | `response_format` | 10 | 20 | 1.00 | 0.00 | 0.00 | n/a | 15.6 | 27.6 | 2065 | 1183 |
| `nvidia/nemotron-3-super-120b-a12b:free` | `structured_outputs` | 10 | 10 | 1.00 | 1.00 | 1.00 | 1 | 40.7 | 112.8 | 2012 | 2889 |

### Failure taxonomy (count of failing requests)

| Failure mode | `dots-studio/dots-3-note-preview:free` | `liquid/lfm-2.5-2.6b:free` | `minimax/minimax-m3:free` | `nvidia/nemotron-3-super-120b-a12b:free` |
|---|---|---|---|---|
| `missing_required_field` | 0 | 0 | 15 | 0 |
| `wrong_root_container` | 0 | 0 | 5 | 0 |
| **Total failed requests** | 0 | 0 | 20 | 0 |

## Experiment 2 -- determinism of the scoring layer

25 hand-built fixtures (17 score to a profile, 8 are expected to raise), covering single-trait dominance, an exact four-way tie reached two different ways, two- and three-way ties, both classifier thresholds at their exact boundary, minimum and maximum response-set lengths, and every input the scorer rejects. Outputs are canonicalised (sorted keys, `{:.10f}` floats) and SHA-256 hashed.

| Check | Fixtures | Scorings | `exact_match_rate` (scored profile) | `exact_match_rate` (incl. error text) |
|---|---|---|---|---|
| Within one process (10 repeats) | 25 | 250 | 1.0000 | 1.0000 |
| Across processes (`PYTHONHASHSEED` 0/1/42) | 25 | 100 | 1.0000 | 0.9600 |

## Findings

Two free-tier models on OpenRouter were evaluated against the item-generation schema on
2026-09-06, and the deterministic scorer was evaluated offline the same day. Every reply from
both models parsed as JSON (`json_parse_rate` = 1.00 in both cases), but schema adherence
separated them completely: `nvidia/nemotron-3-super-120b-a12b:free`, which advertises
server-side `structured_outputs`, validated on 10 of 10 first attempts, while
`minimax/minimax-m3:free`, which offers only `response_format` JSON mode, validated on 0 of 20
requests across 10 tasks and 2 attempts each. MiniMax M3's failures were concentrated in omitted
required fields (most often `scenario_text` and each option's `trait`) and, in 5 of 20 requests,
emission of a bare JSON array in place of the `{"items": [...]}` root object; 17 of its 20 replies
were additionally wrapped in a markdown code fence despite the request specifying a strict JSON
schema, so the pipeline's fence-stripping fallback was load-bearing rather than defensive.
Schema adherence carried a latency cost: the compliant model was roughly 2.7x slower at the
median (40.7s versus 15.6s) and emitted roughly 2.4x more completion tokens per call. The
scoring layer produced byte-identical output for identical input in every condition tested --
250 within-process scorings and 25 fixtures re-scored across three additional interpreters under
different `PYTHONHASHSEED` values all agreed on hash -- with one exception affecting error
message text only, described below.

## Anomalies and failures

- **Non-determinism found (error message text only).** Fixture `err_missing_trait_key` produced a
  different SHA-256 in all four processes (4 distinct message hashes from 4 processes). The
  scored profile is unaffected and `exact_match_rate` over scored output remains 1.0000. Cause:
  `ItemResponse.all_traits_present_and_non_negative` in `profiling/scorer.py` interpolates two raw
  `set`s of `Trait` enum members into its error message; `Enum.__hash__` hashes the member name,
  so string-hash randomisation reorders the set repr per process. Reported, not fixed, per the
  task brief. It is a cosmetic defect in a message, but it would make any test asserting on that
  message string flaky, and it would break byte-comparison of logs across runs.
- **The engine's configured default model could not be measured.** `AFRILEARNER_LLM_MODEL` is
  `google/gemma-4-26b-a4b-it:free`, which returned HTTP 429 on every attempt with
  `limit_source: upstream_provider_shared_pool` -- a Google AI Studio shared-pool limit,
  independent of this account's quota (`usage: 0`, daily cap untouched). `google/gemma-4-31b-it:free`
  and `z-ai/glm-5.2:free` were unavailable for the same reason. The two models tabulated above are
  substitutes, so the table does not describe the configuration the project currently ships.
- **N is below the 30 originally planned.** Experiment 1 ran 10 tasks per model. Nemotron's
  median latency of 40.7s (p95 112.8s) made N=30 infeasible inside the time budget, and MiniMax M3
  consumed 2 requests per task on retries against a 50-requests/day free-tier cap. Both runs
  stopped on their configured request budget, not on an error.
- **Zero-success rows carry no attempts-to-success figure.** `mean_attempts_to_success` is
  computed only over tasks that eventually validated, so it is `n/a` for MiniMax M3 rather than
  silently reported as the retry ceiling.
- **The scorer accepts none of the empty or missing-response cases** the task brief anticipated:
  empty response lists, all-zero allocations, duplicate items, unknown items, budget mismatches
  and missing trait keys all raise. These are recorded as expected-raise fixtures and their error
  paths are hashed alongside the scored ones, which is how the defect above surfaced.

- `google/gemma-4-26b-a4b-it:free`: no content reply obtained; 2 request(s) all returned HTTP 429 from the provider-shared pool. Not tabulated above, since this measures provider availability rather than the model's schema behaviour.
- `google/gemma-4-31b-it:free`: no content reply obtained; 3 request(s) all returned HTTP 429 from the provider-shared pool. Not tabulated above, since this measures provider availability rather than the model's schema behaviour.
- `qwen/qwen3.8-27b:free`: no content reply obtained; 12 request(s) all returned HTTP 429 from the provider-shared pool. Not tabulated above, since this measures provider availability rather than the model's schema behaviour.
