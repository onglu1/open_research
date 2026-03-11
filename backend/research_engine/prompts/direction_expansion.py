"""Stage 1 — Direction Expansion 的 LLM Prompt 模板。"""

from __future__ import annotations


SYSTEM_PROMPT = """\
You are a senior research strategist. Given a core research direction, \
generate a set of parallel and cross-disciplinary directions. Return \
strict JSON matching the schema below. Do NOT wrap in markdown code fences.

Output JSON schema:
{
  "core_direction": {
    "name": str, "name_en": str, "description": str,
    "keywords_en": [str], "keywords_cn": [str], "categories": [str],
    "key_researchers": [str], "relevant_venues": [str]
  },
  "parallel_directions": [
    {
      "name": str, "name_en": str, "description": str,
      "relationship": "parallel"|"cross"|"external",
      "strategies": [str],
      "keywords_en": [str], "keywords_cn": [str], "categories": [str],
      "cross_value": str, "key_researchers": [str],
      "relevance_score": float (0-1), "relevant_venues": [str]
    }
  ],
  "cross_pollination_opportunities": [
    {"direction_a": str, "direction_b": str, "description": str, "potential_value": str}
  ],
  "relevant_venues": [str]
}
"""


def build_user_prompt(
    *,
    direction_name: str,
    direction_description: str,
    keywords_en: list[str],
    categories: list[str],
    strategies: list[str],
    external_directions: list[dict],
    parallel_directions: list[dict],
    forced_crosses: list[list[str]],
    knowledge_context: str,
) -> str:
    parts = [
        f"Core research direction: {direction_name}",
        f"Description: {direction_description}",
        f"English keywords: {', '.join(keywords_en)}",
        f"arXiv categories: {', '.join(categories)}",
        f"Expansion strategies: {', '.join(strategies)}",
    ]
    if external_directions:
        items = "; ".join(f"{d.get('name', '')} ({d.get('name_en', '')})" for d in external_directions)
        parts.append(f"User-provided external directions (must include): {items}")
    if parallel_directions:
        items = "; ".join(f"{d.get('name', '')} ({d.get('name_en', '')})" for d in parallel_directions)
        parts.append(f"User-provided parallel directions (must include): {items}")
    if forced_crosses:
        items = "; ".join(f"{a} × {b}" for a, b in forced_crosses)
        parts.append(f"Forced cross-direction pairs: {items}")
    if knowledge_context:
        parts.append(f"\n--- Existing knowledge context ---\n{knowledge_context}")

    parts.append(
        "\nGenerate at least 5 parallel/cross directions (including all user-provided ones). "
        "Ensure each has complete fields. Prioritize directions with high cross-pollination potential."
    )
    return "\n".join(parts)
