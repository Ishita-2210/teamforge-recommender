
"""
compute_graph_score: convert multi-edge weights into a 0..1 score.
Edges considered: collab, skill, domain, feedback (if present)
"""
def compute_graph_score(G, anchor_user, candidate_user, caps=(5,5,5,5)):
    # caps to limit single-edge dominance
    if anchor_user is None:
        return 0.0
    try:
        if not G.has_edge(anchor_user, candidate_user):
            return 0.0
        e = G[anchor_user][candidate_user]
        collab = min(e.get('collab',0), caps[0])
        skill = min(e.get('skill',0), caps[1])
        domain = min(e.get('domain',0), caps[2])
        feedback = min(e.get('feedback',0), caps[3])
        # weighted combination (tweakable)
        total = 0.5 * collab + 0.3 * skill + 0.15 * domain + 0.05 * feedback
        # scale down by sum of caps to get approx 0..1, using first cap as base
        norm = total / (caps[0] * 0.5 + caps[1] * 0.3 + caps[2] * 0.15 + caps[3] * 0.05)
        return max(0.0, min(1.0, norm))
    except Exception:
        return 0.0
