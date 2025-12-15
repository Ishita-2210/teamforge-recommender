
"""
recommender/graph_embedding.py

Provides:
 - get_user_embedding(user_id)
 - graph_similarity(anchor_user_id, candidate_user_id, method="cosine")
 - batch_graph_similarity(anchor_user_id, user_id_list, method="cosine")

Uses NumPy arrays saved as:
 - user_graph_embs.npy
 - user_graph_ids.npy
in the folder /content/drive/MyDrive/teamforge
"""
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE = "/content/drive/MyDrive/teamforge"

_user_graph_embs = None
_user_graph_ids = None
_user_index_map = None

def load_graph_embs():
    global _user_graph_embs, _user_graph_ids, _user_index_map
    if _user_graph_embs is None:
        emb_path = os.path.join(BASE, "user_graph_embs.npy")
        ids_path = os.path.join(BASE, "user_graph_ids.npy")
        if not os.path.exists(emb_path) or not os.path.exists(ids_path):
            raise RuntimeError(f"Graph embeddings not found at {emb_path} or {ids_path}. Run Node2Vec first.")
        _user_graph_embs = np.load(emb_path)
        _user_graph_ids = np.load(ids_path).astype(int)
        _user_index_map = {int(uid): idx for idx, uid in enumerate(_user_graph_ids)}
    return _user_graph_embs, _user_graph_ids, _user_index_map

def get_user_embedding(user_id):
    """
    Return numpy vector for user_id or None if missing.
    """
    embs, ids, mp = load_graph_embs()
    uid = int(user_id)
    if uid not in mp:
        return None
    return embs[mp[uid]]

def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    return float(cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0][0])

def graph_similarity(anchor_user_id, candidate_user_id, method="cosine"):
    """
    Single pair similarity.
    """
    a = get_user_embedding(anchor_user_id)
    b = get_user_embedding(candidate_user_id)
    if method == "cosine":
        return cosine_sim(a, b)
    elif method == "dot":
        if a is None or b is None: return 0.0
        return float(np.dot(a, b))
    elif method == "euclidean":
        if a is None or b is None: return 0.0
        d = np.linalg.norm(a - b)
        return float(1.0 / (1.0 + d))
    else:
        return cosine_sim(a, b)

def batch_graph_similarity(anchor_user_id, user_id_list, method="cosine"):
    """
    Compute similarity between one anchor and many candidates.
    Returns dict {user_id: score}
    """
    embs, ids, mp = load_graph_embs()
    anchor_vec = get_user_embedding(anchor_user_id)
    if anchor_vec is None:
        # return zeros for all
        return {int(uid): 0.0 for uid in user_id_list}

    # build candidate embedding matrix (missing -> zeros)
    cand_embs = []
    gids = []
    for uid in user_id_list:
        gids.append(int(uid))
        if int(uid) in mp:
            cand_embs.append(embs[mp[int(uid)]])
        else:
            cand_embs.append(np.zeros(embs.shape[1]))

    cand_matrix = np.stack(cand_embs)  # (n, dim)

    if method == "cosine":
        sims = cosine_similarity(anchor_vec.reshape(1,-1), cand_matrix)[0]
    elif method == "dot":
        sims = np.dot(cand_matrix, anchor_vec.reshape(-1))
    elif method == "euclidean":
        dists = np.linalg.norm(cand_matrix - anchor_vec.reshape(1,-1), axis=1)
        sims = 1.0 / (1.0 + dists)
    else:
        sims = cosine_similarity(anchor_vec.reshape(1,-1), cand_matrix)[0]

    return {int(uid): float(score) for uid, score in zip(gids, sims)}
