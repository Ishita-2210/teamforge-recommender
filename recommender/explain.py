
def explain_recommendation(rec):
    '''
    rec: one recommendation dict
    returns: human-readable explanation string
    '''

    reasons = []

    # ---- Skill-based explanation ----
    skill = rec.get("skill_score", 0.0)
    if skill > 0:
        reasons.append(
            f"Strong skill match with team requirements (score: {skill:.2f})"
        )

    # ---- Semantic / NLP explanation ----
    semantic = rec.get("semantic_score", 0.0)
    if semantic > 0:
        reasons.append(
            f"Similar interests and goals based on profile text (score: {semantic:.2f})"
        )

    # ---- Graph / collaboration explanation (FIXED KEY) ----
    graph = rec.get("graph_sim", 0.0)
    if graph > 0:
        reasons.append(
            f"High collaboration potential from past activity or network proximity (score: {graph:.2f})"
        )

    # ---- Online learning / bandit explanation ----
    bandit = rec.get("bandit_boost", 0.0)
    if bandit > 0:
        reasons.append(
            f"Boosted due to positive recent feedback (boost: {bandit:.2f})"
        )

    # ---- Fallback ----
    if not reasons:
        return (
            "Recommended as a potential good fit based on overall profile "
            "and system confidence."
        )

    # ---- Final explanation text ----
    explanation = "Recommended because:\n"
    for r in reasons:
        explanation += f"• {r}\n"

    return explanation.strip()
