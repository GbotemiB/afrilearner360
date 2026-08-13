# Approved assessments

Reviewed, translated, ready-to-serve assessments. One file per grade band, e.g. `P3.json`.
Every student in a class answers the same fixed set from here — see the `items_per_assessment`
note in `data/locales/rwanda.yaml` for why. Loaded by `item_generation.item_bank.load_approved`.

## Current contents

| File | Status |
|---|---|
| `P5.json` | **PROVISIONAL — not human-reviewed.** Placeholder to unblock downstream work. |

### About `P5.json`

Generated in a single model call (`--topic "everyday learning at school" --grade-band P5`) and
promoted without a human review pass, so that profiling and clustering had a real assessment to
develop against. It passes `load_approved`'s structural gate honestly — P5 delivers in English,
so no translation was owed and none was faked — but it has **not** been through the
relatability/stereotyping/reading-level review that DESIGN.md §5 requires.

Known weaknesses spotted in a quick read, all of which add noise to trait scoring:

- Several options are weak fits for the trait they are labelled with. Item 05's `structured`
  option ("Work together on a task") reads communal rather than step-based; item 07's `story`
  option ("Listen to how the game is played") is instruction, not narrative; item 08's
  `structured` option ("Read short proverbs") has no procedure in it.
- Options are not always equally appealing, which the system prompt requires. Item 04 pits
  "Race to see who weaves fastest" against "Complete the steps of weaving" — the game option
  simply sounds more fun, so it will attract points regardless of genuine preference.

**Do not use this with real students.** Replace it with a reviewed set before any pilot.
No P1-P3 assessment exists yet; those need Kinyarwanda translation first (DESIGN.md §7).
