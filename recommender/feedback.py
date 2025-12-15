
from recommender.bandit import bandit

# map actions to rewards
REWARD_MAP = {
    "swipe_right": 1.0,
    "accept": 2.0,
    "team_formed": 3.0,
    "swipe_left": 0.0,
    "reject": 0.0,
    "spam": -1.0
}

def record_feedback(user_id, action):
    reward = REWARD_MAP.get(action, 0.0)
    bandit.update(user_id, reward)
    return reward
