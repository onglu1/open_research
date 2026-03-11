"""Stage 4 — Feasibility Ranking 的 LLM Prompt 模板。"""

from __future__ import annotations


# ── 单条 idea 逐一评估模式的 prompt ──

SINGLE_IDEA_SYSTEM_PROMPT = """\
You are a research evaluation expert. You will be given ONE research idea to \
evaluate. Score it on multiple dimensions using absolute criteria (0-100). \
Be rigorous, critical, and provide concrete rationale for each score. \
Return strict JSON matching the schema below. Do NOT wrap in markdown code fences.

Output JSON schema:
{
  "idea_id": str,
  "title_cn": str,
  "title_en": str,
  "one_liner": str,
  "dimension_scores": {
    "<dimension_name>": {"score": float (0-100), "rationale": str}
  },
  "weighted_total": float (0-100),
  "tier": "S"|"A"|"B"|"C",
  "recommendation": str,
  "next_steps": [str],
  "target_venues": [str],
  "time_estimate": str,
  "key_dependencies": [str]
}
"""


def build_single_idea_prompt(
    *,
    idea_json: str,
    idea_index: int,
    total_ideas: int,
    dimensions: dict[str, dict],
    tier_thresholds: dict[str, int],
    resource_constraints: str,
    knowledge_context: str,
) -> str:
    """为单条 idea 构建评估 prompt。"""
    dim_desc = "\n".join(
        f"- {name}: weight={d.get('weight', 0.2)}, {d.get('description', '')}"
        for name, d in dimensions.items()
    )
    tier_desc = ", ".join(
        f"{k}>={v}" for k, v in sorted(tier_thresholds.items(), key=lambda x: -x[1])
    )

    parts = [
        f"Evaluate the following research idea ({idea_index}/{total_ideas}).",
        "Score it using ABSOLUTE criteria — do not compare with other ideas.",
        f"\n--- Evaluation dimensions ---\n{dim_desc}",
        f"\nTier thresholds (by weighted_total): {tier_desc}",
        f"\n--- Research idea to evaluate ---\n{idea_json}",
    ]
    if resource_constraints:
        parts.append(f"\nResource constraints to consider: {resource_constraints}")
    if knowledge_context:
        parts.append(f"\n--- Additional context ---\n{knowledge_context}")

    parts.append(
        "\nScore this idea on EVERY dimension with detailed rationale. "
        "Compute weighted_total using the weights above and assign the appropriate tier."
    )
    return "\n".join(parts)


# ── 兼容旧调用的批量模式 prompt（保留但不再作为主路径）──

SYSTEM_PROMPT = """\
You are a research evaluation expert. Score each candidate idea on multiple \
dimensions and produce a ranked list. Return strict JSON matching the schema below. \
Do NOT wrap in markdown code fences.

Output JSON schema:
{
  "ranked_ideas": [
    {
      "idea_id": str,
      "title_cn": str,
      "title_en": str,
      "one_liner": str,
      "dimension_scores": {
        "<dimension_name>": {"score": float (0-100), "rationale": str}
      },
      "weighted_total": float (0-100),
      "tier": "S"|"A"|"B"|"C",
      "recommendation": str,
      "next_steps": [str],
      "target_venues": [str],
      "time_estimate": str,
      "key_dependencies": [str]
    }
  ],
  "ranking_rationale": str
}
"""


def build_user_prompt(
    *,
    ideas_json: str,
    dimensions: dict[str, dict],
    tier_thresholds: dict[str, int],
    resource_constraints: str,
    knowledge_context: str,
) -> str:
    dim_desc = "\n".join(
        f"- {name}: weight={d.get('weight', 0.2)}, {d.get('description', '')}"
        for name, d in dimensions.items()
    )
    tier_desc = ", ".join(f"{k}>={v}" for k, v in sorted(tier_thresholds.items(), key=lambda x: -x[1]))

    parts = [
        "Evaluate and rank the following research ideas.",
        f"\n--- Evaluation dimensions ---\n{dim_desc}",
        f"\nTier thresholds (by weighted_total): {tier_desc}",
        f"\n--- Candidate ideas ---\n{ideas_json}",
    ]
    if resource_constraints:
        parts.append(f"\nResource constraints to consider: {resource_constraints}")
    if knowledge_context:
        parts.append(f"\n--- Additional context ---\n{knowledge_context}")

    parts.append(
        "\nScore each idea on EVERY dimension. Sort ranked_ideas by weighted_total descending."
    )
    return "\n".join(parts)
