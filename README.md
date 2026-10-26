# TeamForge — explainable teammate recommendations

A Python prototype that ranks potential teammates using skill compatibility, cached text similarity and collaboration-graph features. Optional trained XGBoost/LightGBM rankers combine those signals; without model artifacts, an explicitly labelled weighted baseline keeps the core usable.

Also described as **Talent-to-Team Recommendation Engine** on the author's resume.

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
| `user_skills.csv` | `user_id`, `skill`, `level`; graph builders also accept `profile_id` for `user_id` |
| `team_needed_skills.csv` | `team_id`, `skill`, `min_level`, `priority` |
| `teams.csv` | `team_id` or `id`; optional `owner_id`, `event_id`/`hackathon_id` |
| `participation.csv` | `profile_id` or `user_id`, `team_id`; optional `event_id`/`hackathon_id` |

Levels are `Beginner`, `Intermediate`, `Pro`; priorities are `Low`, `Medium`, `High`. The historical `main.py` graph demonstration additionally reads `hackathons.csv`. Both graph builders accept its event identifier as `event_id`, `id` or `hackathon_id`, and participation identifiers as `user_id`/`profile_id` and `event_id`/`hackathon_id`. Canonical `user_id` and `event_id` take precedence when aliases coexist; input data frames are not modified. Event metadata is optional for collaboration/skill graphs. `domain` (or `event_type` in the user-to-user builder) supplies the domain signal. The graph demonstration is separate from the HTTP pipeline.

The original CSV header shapes are supported; unused descriptive columns may be omitted:

```text
users.csv: id,name,primary_role,timezone,language,bio_text,looking_for_text
user_skills.csv: id,user_id,skill,level
teams.csv: id,hackathon_id,owner_id,name,project_text,looking_for_text
team_needed_skills.csv: id,team_id,skill,min_level,priority
participation.csv: id,user_id,hackathon_id,team_id,status
hackathons.csv: id,name,event_type,domain,difficulty,mode,location,start_date,end_date,description
```

## Ranking and optional artifacts

`recommender/skill_match.py` scores weighted requirements. `pipeline.py` excludes the owner and already participating candidates, applies filters, computes the three signals and returns scores with `scoring_mode`. The no-model baseline is `0.5 × skill + 0.3 × semantic + 0.2 × graph`; absent embedding files contribute zero. This fallback is not a trained-model result.

Skill matching uses each user's best reported level per skill. Exact qualification earns a contribution of 1; each level above or below the requirement changes that contribution by 1. Missing skills contribute -1.5. Priority weights and normalization produce a bounded score. An underqualified candidate therefore ranks below an otherwise identical qualified candidate. Duplicate user-skill rows contribute only their best level to the graph, without self-loops or repeated pair weights.

Install `requirements-models.txt` only for model-backed work. The pipeline looks for `xgb_hybrid_tuned.json` or `xgb_hybrid_safe.model`, optional XGBoost scalers, and `lgbm_ranker.pkl` with its optional scaler. Semantic caches use `user_embs.npy`, `user_ids.npy`, `team_embs.npy`, `team_ids.npy`; graph caches use `user_graph_embs.npy`, `user_graph_ids.npy`. Only load trusted pickle/joblib artifacts. Training data, model generation scripts and validated ranking metrics are not supplied.

`rank.py` is a separate weighted ranker with exposure penalties and optional random exploration. `bandit.py`/`feedback.py` provide persistent feedback utilities; the main ensemble does not currently apply bandit boosts. `explain.py` formats signal explanations, but the API returns signal scores rather than automatically adding explanation text. These distinctions avoid overstating integration.

## Historical evaluation and current evidence

The resume describes a historical synthetic-data ranking experiment. Its training/evaluation scripts, labelled training pairs, model artifacts and split/metric manifests are not included here, so its reported benchmark scores and improvements cannot be reproduced from this checkout. Current tests establish ranking behavior, schema compatibility and application contracts using small synthetic inputs; they do not reproduce that historical experiment or establish recommendation quality. Source CSVs and private records are not redistributed.

## SQLite constraint study

`teamforge_edgecases.py` contains profile versioning, unique users, event participation, spam counters and atomic matching. Set `TEAMFORGE_DB_PATH` for isolated storage; the default is a local ignored database. Connections enforce foreign keys. It is a separate development module and not authentication or production concurrency infrastructure.

## Verify

```sh
python -m pip install pytest httpx
python -m pytest -q
```

Tests create synthetic CSVs and temporary SQLite databases. They exercise qualification ordering, original/canonical graph headers, duplicate-skill handling, baseline ranking, all-skill filtering, owner exclusion, unique exploration results, event-domain edges, profile updates, foreign keys and HTTP errors. No external dataset or pretrained weight is required. Real-data recommendation quality and optional artifact compatibility remain separate validation tasks.
