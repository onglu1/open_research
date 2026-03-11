"""Stage 3 — Idea Discovery 的 LLM Prompt 模板。"""

from __future__ import annotations


SYSTEM_PROMPT = """\
You are a creative research scientist generating novel research ideas \
by combining insights from multiple research directions and their frontiers. \
Prioritize cross-direction combination ideas. Return strict JSON matching the schema below. \
Do NOT wrap in markdown code fences.

Output JSON schema:
{
  "ideas": [
    {
      "id": str (e.g. "IDEA-001"),
      "title_cn": str,
      "title_en": str,
      "one_liner": str,
      "description": str (2-3 paragraphs),
      "motivation": str,
      "method_sketch": str,
      "novelty_points": [str],
      "expected_contributions": [str],
      "validation_plan": str,
      "source_directions": [str],
      "discovery_strategy": str ("gap_filling"|"method_combination"|"cross_domain_transfer"|...),
      "related_gaps": [str],
      "related_papers": [str (paper titles)],
      "estimated_effort": str ("low"|"medium"|"high"),
      "risk_factors": [str],
      "preliminary_score": float (0-100)
    }
  ],
  "generation_strategies_used": [str]
}
"""


def build_user_prompt(
    *,
    direction_scans_summary: str,
    research_gaps: list[str],
    trend_signals: list[str],
    hot_topics: list[str],
    strategies: list[str],
    max_ideas: int,
    encourage_cross: bool,
    resource_constraints: str,
    knowledge_context: str,
) -> str:
    parts = [
        "Based on the following frontier scan results, generate research ideas.",
        f"\n--- Direction scan summaries ---\n{direction_scans_summary}",
        f"\n--- Key research gaps ---\n" + "\n".join(f"- {g}" for g in research_gaps[:30]),
        f"\n--- Trend signals ---\n" + "\n".join(f"- {s}" for s in trend_signals[:20]),
        f"\n--- Hot topics ---\n" + "\n".join(f"- {t}" for t in hot_topics[:20]),
        f"\nGeneration strategies to apply: {', '.join(strategies)}",
        f"Maximum number of ideas: {max_ideas}",
    ]
    if encourage_cross:
        parts.append("IMPORTANT: Prioritize ideas that combine insights from 2+ directions.")
    if resource_constraints:
        parts.append(f"\nResource constraints: {resource_constraints}")
    if knowledge_context:
        parts.append(f"\n--- Existing knowledge (avoid duplicating these) ---\n{knowledge_context}")

    parts.append(
        "\nEnsure each idea has all required fields. IDs should be IDEA-001, IDEA-002, etc."
    )
    return "\n".join(parts)
