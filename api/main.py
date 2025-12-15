
from fastapi import FastAPI, Query
import sys, os

BASE = "/content/drive/MyDrive/teamforge"
sys.path.append(BASE)

from recommender.pipeline import recommend

app = FastAPI(
    title="TeamForge Recommender API",
    description="Hybrid AI teammate recommendation engine",
    version="1.0"
)

@app.get("/health")
def health():
    return {"status": "ok", "model": "hybrid-ensemble"}

@app.get("/recommend")
def recommend_api(
    team_id: int = Query(...),
    top_k: int = Query(10, ge=1, le=50),
    role: str | None = Query(None),
    skills: str | None = Query(None)
):
    role_filter = [role] if role else None
    skill_filter = skills.split(",") if skills else None

    results = recommend(
        team_id=team_id,
        top_k=top_k,
        filter_roles=role_filter,
        require_skills=skill_filter
    )

    return [
        {
            "user_id": r["user_id"],
            "score": round(r["score"], 4),
            "skill_score": round(r["skill_score"], 4),
            "semantic_score": round(r["semantic_score"], 4),
            "graph_sim": round(r["graph_sim"], 4),
        }
        for r in results
    ]
