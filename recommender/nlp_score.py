
"""
Compute semantic similarity using sentence-transformers.
Provides a small cache mechanism (in-memory) to avoid repeated encoding in a session.
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

_model = None
_cache = {}

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_texts(texts):
    """
    texts: list of strings
    returns: numpy array (n, dim)
    """
    model = _get_model()
    emb = model.encode(texts, show_progress_bar=False)
    return np.array(emb)

def semantic_score_single(team_text, user_text):
    # quick per-pair compute (for small batches)
    k = (team_text, user_text)
    if k in _cache:
        return _cache[k]
    model = _get_model()
    emb = model.encode([team_text, user_text])
    sim = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
    _cache[k] = sim
    return sim
