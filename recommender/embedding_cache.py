
import os
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(os.path.dirname(__file__)) if os.path.basename(os.path.dirname(__file__))=="recommender" else "/content/drive/MyDrive/teamforge"
USER_EMB_PATH = os.path.join(BASE, "user_embs.npy")
USER_IDS_PATH = os.path.join(BASE, "user_ids.npy")
TEAM_EMB_PATH = os.path.join(BASE, "team_embs.npy")
TEAM_IDS_PATH = os.path.join(BASE, "team_ids.npy")

_user_embs = None
_user_ids = None
_team_embs = None
_team_ids = None

def load_cache():
    global _user_embs, _user_ids, _team_embs, _team_ids
    if _user_embs is None:
        _user_embs = np.load(USER_EMB_PATH)
        _user_ids = np.load(USER_IDS_PATH)
    if _team_embs is None:
        _team_embs = np.load(TEAM_EMB_PATH)
        _team_ids = np.load(TEAM_IDS_PATH)

def get_user_embedding_map():
    load_cache()
    # returns dict {user_id: emb_index}
    return {int(uid): idx for idx, uid in enumerate(_user_ids)}

def get_team_embedding_map():
    load_cache()
    return {int(tid): idx for idx, tid in enumerate(_team_ids)}

def semantic_scores_for_team(team_id):
    """
    Returns dict: {user_id: cosine_sim(team_emb, user_emb)}
    """
    load_cache()
    team_map = get_team_embedding_map()
    user_map = get_user_embedding_map()
    if int(team_id) not in team_map:
        return {}
    tidx = team_map[int(team_id)]
    t_emb = _team_embs[tidx].reshape(1, -1)
    # compute cosine similarity to all users (fast)
    sims = cosine_similarity(t_emb, _user_embs)[0]  # shape (n_users,)
    return {int(_user_ids[i]): float(sims[i]) for i in range(len(sims))}
