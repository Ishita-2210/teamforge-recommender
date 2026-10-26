# TeamForge — explainable teammate recommendations

A Python prototype that ranks potential teammates using skill compatibility, cached text similarity and collaboration-graph features. Optional trained XGBoost/LightGBM rankers combine those signals; without model artifacts, an explicitly labelled weighted baseline keeps the core usable.

## Setup

Python 3.11+ in an activated virtual environment:

```sh
python -m pip install -r requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

`GET /health` checks the service. `GET /recommend?team_id=1&top_k=10` requests candidates. `role` filters an exact primary role and comma-separated `skills` requires all listed skills. Invalid IDs return 422; missing CSVs return 503 rather than exposing local paths.

Set `TEAMFORGE_DATA_DIR` to your data/artifact directory; the default is the repository root. Google Drive and Colab are not required. Required files are not bundled:

| CSV | Required fields |
| --- | --- |
| `users.csv` | `id`; optional `primary_role` |
| `user_skills.csv` | `user_id`, `skill`, `level` |
| `team_needed_skills.csv` | `team_id`, `skill`, `min_level`, `priority` |
| `teams.csv` | `team_id` or `id`; optional `owner_id`, `event_id`/`hackathon_id` |
| `participation.csv` | `profile_id` or `user_id`, `team_id`; optional `event_id`/`hackathon_id` |

Levels are `Beginner`, `Intermediate`, `Pro`; priorities are `Low`, `Medium`, `High`. The historical `main.py` graph demonstration additionally reads `hackathons.csv` with `event_id` and optional `domain`. It is separate from the HTTP pipeline.

## Ranking and optional artifacts

`recommender/skill_match.py` scores weighted requirements. `pipeline.py` excludes the owner and already participating candidates, applies filters, computes the three signals and returns scores with `scoring_mode`. The no-model baseline is `0.5 × skill + 0.3 × semantic + 0.2 × graph`; absent embedding files contribute zero. This fallback is not a trained-model result.

Install `requirements-models.txt` only for model-backed work. The pipeline looks for `xgb_hybrid_tuned.json` or `xgb_hybrid_safe.model`, optional XGBoost scalers, and `lgbm_ranker.pkl` with its optional scaler. Semantic caches use `user_embs.npy`, `user_ids.npy`, `team_embs.npy`, `team_ids.npy`; graph caches use `user_graph_embs.npy`, `user_graph_ids.npy`. Only load trusted pickle/joblib artifacts. Training data, model generation scripts and validated ranking metrics are not supplied.

`rank.py` is a separate weighted ranker with exposure penalties and optional random exploration. `bandit.py`/`feedback.py` provide persistent feedback utilities; the main ensemble does not currently apply bandit boosts. `explain.py` formats signal explanations, but the API returns signal scores rather than automatically adding explanation text. These distinctions avoid overstating integration.

## SQLite constraint study

`teamforge_edgecases.py` contains profile versioning, unique users, event participation, spam counters and atomic matching. Set `TEAMFORGE_DB_PATH` for isolated storage; the default is a local ignored database. Connections enforce foreign keys. It is a separate development module and not authentication or production concurrency infrastructure.

## Verify

```sh
python -m pip install pytest httpx
python -m pytest -q
```

Tests create synthetic CSVs and temporary SQLite databases. They exercise baseline ranking, all-skill filtering, owner exclusion, unique exploration results, event-domain edges, profile updates, foreign keys and HTTP errors. No external dataset or pretrained weight is required. Real-data recommendation quality and optional artifact compatibility remain separate validation tasks.
