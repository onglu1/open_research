"""Stage 5 — Deep Analysis 的 LLM Prompt 模板。"""

from __future__ import annotations


SYSTEM_PROMPT = """\
You are a senior research advisor producing a detailed research proposal / \
deep analysis for a high-priority research idea. The output should be \
comprehensive enough to guide actual research execution. Return strict JSON \
matching the schema below. Do NOT wrap in markdown code fences.

Output JSON schema:
{
  "idea_id": str,
  "title_cn": str,
  "title_en": str,
  "executive_summary": str (2-3 paragraphs),
  "related_work_review": str (structured review, 3-5 paragraphs),
  "key_paper_analysis": [
    {"paper_title": str, "paper_url": str, "relevance": str, "key_findings": str, "limitations": str}
  ],
  "technical_approach": str (detailed, 3-5 paragraphs),
  "system_architecture": str,
  "key_components": [str],
  "experiment_design": {
    "research_questions": [str],
    "datasets": [str],
    "baselines": [str],
    "metrics": [str],
    "ablations": [str],
    "expected_results": str
  },
  "risk_assessment": str,
  "resource_risks": str,
  "timeline": [
    {"name": str, "duration": str, "deliverables": [str]}
  ],
  "target_venues": [str],
  "go_no_go_verdict": str ("GO"|"CONDITIONAL_GO"|"NO_GO" with explanation)
}
"""


def build_user_prompt(
    *,
    idea_json: str,
    frontier_context: str,
    resource_constraints: str,
    knowledge_context: str,
) -> str:
    parts = [
        "Produce a deep analysis / research proposal for the following idea.",
        f"\n--- Idea details ---\n{idea_json}",
        f"\n--- Relevant frontier context ---\n{frontier_context}",
    ]
    if resource_constraints:
        parts.append(f"\nResource constraints: {resource_constraints}")
    if knowledge_context:
        parts.append(f"\n--- Additional knowledge context ---\n{knowledge_context}")

    parts.append(
        "\nProvide a thorough analysis covering ALL fields in the schema. "
        "The go_no_go_verdict should be a clear recommendation with justification."
    )
    return "\n".join(parts)
