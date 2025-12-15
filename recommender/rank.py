
from recommender.skill_match import compute_skill_fit
from recommender.graph_score import compute_graph_score
# try import embedding cache
try:
    from recommender.embedding_cache import semantic_scores_for_team
    _HAS_EMBED_CACHE = True
except Exception:
    _HAS_EMBED_CACHE = False

import random

EXPOSURE_CAP_PER_DAY = 50
POPULARITY_PENALTY = 0.0005

def rank_candidates(team_id, team_skills_df, team_project_text, users_df, user_skills_df, G,
                    anchor_user=None, impressions_map=None, epsilon=0.05, top_k=50):
    impressions_map = impressions_map or {}
    candidates = []

    semantic_map = {}
    if _HAS_EMBED_CACHE:
        try:
            semantic_map = semantic_scores_for_team(team_id)
        except Exception:
            semantic_map = {}

    for _, u in users_df.iterrows():
        uid = int(u['id'])
        user_sk = user_skills_df[user_skills_df['user_id'] == uid]
        skill_score = compute_skill_fit(user_sk, team_skills_df)
        graph_score = compute_graph_score(G, anchor_user, uid) if anchor_user is not None else 0.0
        if uid in semantic_map:
            semantic = semantic_map[uid]
        else:
            # fallback: simple 0.0 when cache missing
            semantic = 0.0
        base_score = 0.5 * skill_score + 0.3 * semantic + 0.2 * graph_score
        impressions = impressions_map.get(uid, 0)
        penalty = max(0.0, (impressions - EXPOSURE_CAP_PER_DAY) * POPULARITY_PENALTY)
        final = max(0.0, base_score - penalty)
        candidates.append((uid, round(final,4)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    if random.random() < epsilon and len(candidates) > 0:
        sample = candidates[:min(100, len(candidates))]
        pick = random.choice(sample)
        return [pick] + candidates[:top_k-1]
    return candidates[:top_k]
