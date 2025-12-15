
"""
compute_skill_fit: handles multiple user skills vs multiple needed skills.
Uses level and priority; supports normalization and missing-skill penalty.
"""
LEVEL_MAP = {"Beginner": 1, "Intermediate": 2, "Pro": 3}
PRIORITY_MAP = {"Low": 1, "Medium": 2, "High": 3}

def compute_skill_fit(user_skills_df, team_needed_skills_df):
    # user_skills_df rows: user_id, skill, level
    # team_needed_skills_df rows: team_id, skill, min_level, priority
    # Build user skill -> best level
    user_map = {}
    for _, r in user_skills_df.iterrows():
        s = r['skill']
        lvl = LEVEL_MAP.get(r.get('level','Beginner'), 1)
        user_map[s] = max(user_map.get(s, 0), lvl)

    score = 0.0
    max_possible = 0.0
    for _, need in team_needed_skills_df.iterrows():
        req = LEVEL_MAP.get(need.get('min_level','Intermediate'), 2)
        pr = PRIORITY_MAP.get(need.get('priority','Medium'), 2)
        weight = pr
        max_possible += weight * 3
        if need['skill'] in user_map:
            # match quality: 1 if exactly match, more if overshoot
            diff = user_map[need['skill']] - req
            bonus = 1 + max(0, diff)  # Pro overshoot helpful
            score += weight * bonus
        else:
            # missing skill: negative scaled by priority
            score -= weight * 1.5

    if max_possible <= 0:
        return 0.0
    # normalize to 0..1 by mapping score from [-max_possible, +max_possible] -> [0,1]
    normalized = (score + max_possible) / (2 * max_possible)
    return max(0.0, min(1.0, normalized))
