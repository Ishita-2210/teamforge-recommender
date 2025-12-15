
import os, sys, json, numpy as np, pandas as pd
BASE = "/content/drive/MyDrive/teamforge"
sys.path.append(BASE)

# safe imports
try:
    from recommender.skill_match import compute_skill_fit
except Exception as e:
    raise RuntimeError("Missing recommender.skill_match: " + str(e))
try:
    from recommender.embedding_cache import semantic_scores_for_team
except Exception:
    def semantic_scores_for_team(tid): return {}
try:
    from recommender.graph_embedding import batch_graph_similarity
except Exception:
    def batch_graph_similarity(a, b_list, method="cosine"): return {int(u): 0.0 for u in b_list}

# optional fairness helper
fairness_rerank = None
try:
    from recommender.fairness_helper import fairness_rerank as _fr
    fairness_rerank = _fr
except Exception:
    fairness_rerank = None

# models & scalers
import joblib
import xgboost as xgb

conf_path = os.path.join(BASE, "ensemble_conf.json")
if os.path.exists(conf_path):
    with open(conf_path, "r") as f:
        ENSEMBLE_CONF = json.load(f)
else:
    ENSEMBLE_CONF = {"xgb_weight": 0.45, "lgbm_weight": 0.55, "features_order":["skill","semantic","graph_sim"]}

# load XGB
_xgb = None
_xgb_scaler = None
XGB_PATHS = [os.path.join(BASE,"xgb_hybrid_tuned.json"), os.path.join(BASE,"xgb_hybrid_safe.model")]
XGB_SCALERS = [os.path.join(BASE,"xgb_tuned_scaler.pkl"), os.path.join(BASE,"xgb_hybrid_safe.scaler.npy")]
for p in XGB_PATHS:
    if os.path.exists(p):
        try:
            _xgb = xgb.Booster(); _xgb.load_model(p); break
        except Exception:
            _xgb = None
for s in XGB_SCALERS:
    if os.path.exists(s):
        try:
            _xgb_scaler = joblib.load(s); break
        except:
            try:
                import numpy as _np
                vec = _np.load(s)
                class DummySc:
                    def __init__(self, scale): self.scale_ = scale
                _xgb_scaler = DummySc(vec); break
            except:
                _xgb_scaler = None

# load LGBM ranker + scaler if present
_lgbm = None
_lgbm_scaler = None
lgbm_path = os.path.join(BASE, "lgbm_ranker.pkl")
lgbm_scaler_path = os.path.join(BASE, "lgbm_ranker_scaler.pkl")
if os.path.exists(lgbm_path):
    try:
        _lgbm = joblib.load(lgbm_path)
    except Exception:
        _lgbm = None
if os.path.exists(lgbm_scaler_path):
    try:
        _lgbm_scaler = joblib.load(lgbm_scaler_path)
    except Exception:
        _lgbm_scaler = None

# in-memory impressions
_impressions = {}
EXPOSURE_CAP = 50
POPULARITY_PENALTY = 0.0005

def _xgb_predict(feats):
    # feats: numpy array shape (n,3)
    if _xgb is None:
        return np.zeros((feats.shape[0],))
    # scaling
    if _xgb_scaler is not None and hasattr(_xgb_scaler, "transform"):
        Xs = _xgb_scaler.transform(feats)
    elif _xgb_scaler is not None and hasattr(_xgb_scaler, "scale_"):
        Xs = feats * _xgb_scaler.scale_
    else:
        Xmin = feats.min(axis=0); Xmax = feats.max(axis=0); denom = (Xmax-Xmin); denom[denom==0]=1
        Xs = (feats - Xmin)/denom
    dmat = xgb.DMatrix(Xs)
    preds = _xgb.predict(dmat)
    return preds

def _lgbm_predict(feats):
    # Pass DataFrame with exact column names used during training to avoid warning
    if _lgbm is None:
        return np.zeros((feats.shape[0],))
    cols = ['skill','semantic','graph_sim']
    df = pd.DataFrame(feats, columns=cols)
    if _lgbm_scaler is not None:
        Xs = _lgbm_scaler.transform(df.values)
    else:
        Xmin = df.values.min(axis=0); Xmax = df.values.max(axis=0); denom = (Xmax-Xmin); denom[denom==0]=1
        Xs = (df.values - Xmin)/denom
    preds = _lgbm.predict(Xs)
    return preds

def _minmax_normalize(arr):
    a = np.array(arr, dtype=float)
    if a.size == 0:
        return a
    mn = a.min(); mx = a.max()
    if mx == mn:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)

def recommend(team_id, top_k=20, filter_roles=None, require_skills=None):
    users = pd.read_csv(os.path.join(BASE,"users.csv"))
    user_skills = pd.read_csv(os.path.join(BASE,"user_skills.csv"))
    team_skills = pd.read_csv(os.path.join(BASE,"team_needed_skills.csv"))
    teams = pd.read_csv(os.path.join(BASE,"teams.csv"))
    participation = pd.read_csv(os.path.join(BASE,"participation.csv"))

    # normalize team id
    team_id_col = next((c for c in ['team_id','id','teamId','team'] if c in teams.columns), 'team_id')
    if team_id_col != 'team_id':
        teams = teams.rename(columns={team_id_col:'team_id'})

    tre = teams[teams['team_id'] == int(team_id)]
    if tre.shape[0] == 0:
        return []
    tre = tre.iloc[0]
    owner = int(tre['owner_id']) if 'owner_id' in tre.index and not pd.isna(tre['owner_id']) else None
    event_id = int(tre['hackathon_id']) if 'hackathon_id' in tre.index and not pd.isna(tre['hackathon_id']) else (int(tre['event_id']) if 'event_id' in tre.index and not pd.isna(tre['event_id']) else None)

    # candidate pool
    cand = users.copy()
    if filter_roles and 'primary_role' in cand.columns:
        cand = cand[cand['primary_role'].isin(filter_roles)]
    if require_skills:
        u_map = user_skills.groupby('user_id')['skill'].apply(set).to_dict()
        keep=[]
        for uid in cand['id'].astype(int).tolist():
            if any(sk in u_map.get(uid,set()) for sk in require_skills):
                keep.append(uid)
        cand = cand[cand['id'].isin(keep)]

    # exclude double-booked
    pid_col = 'profile_id' if 'profile_id' in participation.columns else ('user_id' if 'user_id' in participation.columns else None)
    if pid_col and event_id is not None:
        if 'event_id' in participation.columns:
            taken = participation[participation['event_id'] == event_id]
        elif 'hackathon_id' in participation.columns:
            taken = participation[participation['hackathon_id'] == event_id]
        else:
            taken = pd.DataFrame()
        if not taken.empty and pid_col in taken.columns:
            taken_ids = taken[pid_col].astype(int).tolist()
            cand = cand[~cand['id'].isin(taken_ids)]

    if cand.shape[0] == 0:
        return []

    user_ids = cand['id'].astype(int).tolist()
    needs_df = team_skills[team_skills['team_id'] == int(team_id)]

    # semantic + graph maps
    try:
        sem_map = semantic_scores_for_team(int(team_id))
    except:
        sem_map = {}
    try:
        graph_map = batch_graph_similarity(owner, user_ids, method="cosine") if owner is not None else {u:0.0 for u in user_ids}
    except:
        graph_map = {u:0.0 for u in user_ids}

    ks=[]; sems=[]; gs=[]
    for uid in user_ids:
        u_skill_df = user_skills[user_skills['user_id'] == uid]
        try:
            s = float(compute_skill_fit(u_skill_df, needs_df))
        except:
            s = 0.0
        ks.append(s)
        sems.append(sem_map.get(uid, 0.0))
        gs.append(graph_map.get(uid, 0.0))

    feats = np.vstack([ks, sems, gs]).T

    # get raw model scores
    xgb_raw = _xgb_predict(feats)    # may be in [0,1]
    lgbm_raw = _lgbm_predict(feats)  # possibly unbounded

    # normalize both to 0..1 per-call to align scales
    xgb_norm = _minmax_normalize(xgb_raw)
    lgbm_norm = _minmax_normalize(lgbm_raw)

    # ensemble weighted sum (weights from config)
    w_x = ENSEMBLE_CONF.get("xgb_weight", 0.45)
    w_l = ENSEMBLE_CONF.get("lgbm_weight", 0.55)
    final = w_x * xgb_norm + w_l * lgbm_norm

    out=[]
    for uid, f, xr, lr, skl, sem_v, g_v in zip(user_ids, final, xgb_raw, lgbm_raw, ks, sems, gs):
        out.append({
            "user_id": int(uid),
            "score": float(f),
            "xgb_raw": float(xr),
            "lgbm_raw": float(lr),
            "skill_score": float(skl),
            "semantic_score": float(sem_v),
            "graph_sim": float(g_v)
        })

    out_sorted = sorted(out, key=lambda x: x['score'], reverse=True)

    # fairness reranker if available
    if fairness_rerank is not None:
        try:
            users_df = users
            out_sorted = fairness_rerank(out_sorted, users_df, top_k=top_k)
        except:
            pass

    # update impressions
    for r in out_sorted[:top_k]:
        _impressions[r['user_id']] = _impressions.get(r['user_id'], 0) + 1

    return out_sorted[:top_k]
