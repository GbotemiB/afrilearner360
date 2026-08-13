"""System and user prompts for per-cluster teaching recommendations.

Design notes:
- **The guardrails matter more here than anywhere else in the pipeline.** This project was
  reframed away from sorting children into fixed "learning style" labels toward evolving
  engagement traits (DESIGN.md §1). An LLM asked for per-group teaching advice drifts naturally
  into deficit language -- "this group struggles", "slower learners", "less able" -- which would
  quietly reintroduce exactly the framing the project rejected, in the one artifact a teacher
  actually reads. Rules 1-4 below exist for that.
- Grounded in two sources: the locale's `recommendation_context` (class size, double-shift hours,
  device assumptions, pedagogical norms) and the cultural knowledge base. The context stops the
  model recommending resources that do not exist; the knowledge base is what lets it suggest an
  Igisoro counting game rather than generic "use manipulatives" (DESIGN.md §5 flags this stage as
  high-leverage for local grounding).
- Methods, not materials: recommending *how to teach* is in MVP scope, generating the lesson
  content itself is explicitly Phase 2 (DESIGN.md §2).
"""
from __future__ import annotations

from typing import Sequence

from ..clustering.cluster import ClusterSummary

SYSTEM_PROMPT = """\
You are advising a primary school teacher in a real, resource-constrained classroom. Students \
have been grouped by how they prefer to engage with learning, measured across four modes: \
visual, game, structured, and story. Your job is to recommend TEACHING METHODS for each group -- \
how the teacher should approach them -- not to write lesson content, worksheets, or materials.

Follow these rules exactly:

1. Never describe any group of children as slow, weak, struggling, behind, less able, or as \
   having difficulties. These groups reflect PREFERENCE, not ability. Every group contains \
   children of every ability level.
2. Never imply a child's engagement mode is fixed or permanent. These traits shift as children \
   develop. Write about what works for a group "at the moment", not what they "are".
3. Frame all four groups as equally capable and equally worth teaching well. No group is the \
   advanced one and no group is the remedial one. If one group's advice sounds more ambitious \
   than another's, rewrite it.
4. Phrase pitfalls in `watch_out_for` as things about the TEACHING to watch, never as deficits \
   in the children. Write "this approach can lose pace if the sequence is too long", not "these \
   children lose focus easily".
5. Do not recommend permanently separating children or seating them apart for good. Grouping is \
   a flexible teaching tool used for some activities, not a fixed classroom hierarchy.
6. Respect the stated classroom constraints absolutely. Do not suggest anything requiring \
   projectors, printing, tablets, per-student devices, internet, or materials the constraints \
   say are unavailable. Assume a chalkboard, the teacher's voice, shared textbooks, and things \
   found locally at no cost.
7. Account for the stated class size. Advice that only works with 15 children is useless in a \
   room of 60. Say explicitly how the teacher manages the other groups while working with one.
8. Ground any cultural reference, game, craft, or analogy ONLY in the supplied cultural \
   knowledge base. Do not invent local details. If unsure a detail is accurate, leave it out.
9. Be concrete. "Use visual aids" is not useful. "Draw the fraction as segments of an Imigongo \
   pattern on the chalkboard, then have pairs copy it into their exercise books" is.
10. In `constraints_addressed`, list which of the stated classroom constraints your advice \
    accounts for, and in `cultural_anchors_used`, list the specific knowledge base references \
    you drew on. Both exist so a reviewer can check your work quickly.
11. Return ONLY the structured output matching the given schema. No prose outside the schema.
"""


def _format_constraints(recommendation_context: dict) -> str:
    """Render the locale's recommendation_context block as labelled lines for the prompt."""
    lines = []
    for key in recommendation_context:
        label = key.replace("_", " ")
        lines.append(f"- {label}: {recommendation_context[key]}")
    return "\n".join(lines)


def _format_cluster(summary: ClusterSummary, *, total_students: int) -> str:
    proportions = ", ".join(
        f"{trait.value} {summary.mean_proportions[trait]:.0f}%" for trait in summary.mean_proportions
    )
    share = (100.0 * summary.size / total_students) if total_students else 0.0
    return (
        f"- Cluster {summary.cluster_id}: {summary.size} students ({share:.0f}% of the class), "
        f"described as {summary.descriptor()}. Average engagement mix: {proportions}."
    )


def _class_picture(summaries: Sequence[ClusterSummary], *, total_students: int) -> str:
    return "\n".join(_format_cluster(s, total_students=total_students) for s in summaries)


def build_whole_class_prompt(
    *,
    summaries: Sequence[ClusterSummary],
    topic: str,
    grade_band: str,
    locale: str,
    recommendation_context: dict,
    knowledge_base_excerpt: str,
) -> str:
    """One call covering every cluster, so groups can be differentiated against each other."""
    total_students = sum(summary.size for summary in summaries)

    return f"""\
Recommend teaching methods for one class, broken down by group.

Topic: {topic}
Grade band: {grade_band}
Locale: {locale}
Class size: {total_students} students in {len(summaries)} groups

THE CLASS, GROUP BY GROUP:
{_class_picture(summaries, total_students=total_students)}

CLASSROOM CONSTRAINTS (your advice must work within these):
{_format_constraints(recommendation_context)}

CULTURAL KNOWLEDGE BASE (the only source for local references, games, crafts, and analogies):
---
{knowledge_base_excerpt}
---

Give one recommendation per group, using the cluster_id values above. Also give a short
`class_overview` describing the spread of this class in plain language a teacher can read at a
glance, and `running_the_class`: concrete steps for running these groups together in one room of
{total_students} students, including what the other groups are doing while the teacher works with
one of them.
"""


def build_single_cluster_prompt(
    *,
    summary: ClusterSummary,
    all_summaries: Sequence[ClusterSummary],
    topic: str,
    grade_band: str,
    locale: str,
    recommendation_context: dict,
    knowledge_base_excerpt: str,
) -> str:
    """One call for one group.

    The other groups are still described, briefly: without them the model cannot judge what makes
    this group distinctive, and tends to produce advice that would suit any group equally.
    """
    total_students = sum(s.size for s in all_summaries)
    others = [s for s in all_summaries if s.cluster_id != summary.cluster_id]
    others_text = (
        _class_picture(others, total_students=total_students)
        if others
        else "- (this is the only group in the class)"
    )

    return f"""\
Recommend teaching methods for ONE group within a class.

Topic: {topic}
Grade band: {grade_band}
Locale: {locale}
Class size: {total_students} students in {len(all_summaries)} groups

THE GROUP YOU ARE ADVISING ON:
{_format_cluster(summary, total_students=total_students)}

THE OTHER GROUPS IN THIS CLASS (context only -- do not advise on them, but use them to judge what
makes your group distinctive):
{others_text}

CLASSROOM CONSTRAINTS (your advice must work within these):
{_format_constraints(recommendation_context)}

CULTURAL KNOWLEDGE BASE (the only source for local references, games, crafts, and analogies):
---
{knowledge_base_excerpt}
---

Return a single recommendation for cluster {summary.cluster_id} only. Say what the rest of the
class is doing while the teacher works with this group.
"""
