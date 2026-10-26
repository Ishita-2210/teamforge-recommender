import importlib
import sqlite3
from unittest.mock import patch
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from graph.build_graph import build_user_graph
from recommender.rank import rank_candidates
from recommender import pipeline

def test_baseline_ranking_filters_owner_and_requires_all_skills(tmp_path,monkeypatch):
    pd.DataFrame({"id":[1,2,3]}).to_csv(tmp_path/"users.csv",index=False)
    pd.DataFrame({"user_id":[2,2,3],"skill":["Python","SQL","Python"],"level":["Pro"]*3}).to_csv(tmp_path/"user_skills.csv",index=False)
    pd.DataFrame({"team_id":[10],"skill":["SQL"],"min_level":["Intermediate"],"priority":["High"]}).to_csv(tmp_path/"team_needed_skills.csv",index=False)
    pd.DataFrame({"id":[10],"owner_id":[1]}).to_csv(tmp_path/"teams.csv",index=False)
    pd.DataFrame(columns=["profile_id","event_id","team_id"]).to_csv(tmp_path/"participation.csv",index=False)
    monkeypatch.setattr(pipeline,"BASE",str(tmp_path))
    monkeypatch.setattr(pipeline,"_xgb",None);monkeypatch.setattr(pipeline,"_lgbm",None)
    result=pipeline.recommend(10)
    assert [r["user_id"] for r in result]==[2,3]
    assert result[0]["score"]>result[1]["score"]
    assert result[0]["scoring_mode"]=="weighted-baseline"
    assert [r["user_id"] for r in pipeline.recommend(10,require_skills=["Python","SQL"])]==[2]
    with pytest.raises(ValueError): pipeline.recommend(10,top_k=0)

def test_exploration_does_not_duplicate_candidates():
    users=pd.DataFrame({"id":[1,2,3]})
    skills=pd.DataFrame(columns=["user_id","skill"])
    needs=pd.DataFrame(columns=["skill"])
    with patch("recommender.rank.random.random",return_value=0), patch("recommender.rank.random.choice",side_effect=lambda rows:rows[0]):
        result=rank_candidates(1,needs,"",users,skills,None,top_k=3)
    assert len(result)==3
    assert len({uid for uid,score in result})==3

def test_event_domain_edges_use_event_id():
    users=pd.DataFrame({"id":[1,2]})
    skills=pd.DataFrame(columns=["user_id","skill"])
    participation=pd.DataFrame({"profile_id":[1,2],"event_id":[10,20],"team_id":[1,2]})
    events=pd.DataFrame({"event_id":[10,20],"domain":["AI","AI"]})
    graph=build_user_graph(users,skills,participation,events)
    assert graph[1][2]["domain"]==1

def test_sqlite_constraints_and_profile_update(tmp_path,monkeypatch):
    monkeypatch.setenv("TEAMFORGE_DB_PATH",str(tmp_path/"team.db"))
    import teamforge_edgecases as edge
    edge=importlib.reload(edge)
    uid=edge.create_user("demo")
    assert edge.create_user("demo")==uid
    with pytest.raises(sqlite3.IntegrityError): edge.create_profile(999,"Developer")
    pid=edge.create_profile(uid,"Developer",bio="old",skills=[("Python","Pro")])
    updated=edge.update_profile_atomic(pid,new_bio="",new_skills=[])
    with edge.get_conn() as conn:
        assert conn.execute("SELECT bio_text FROM profiles WHERE profile_id=?",(updated,)).fetchone()[0]==""
        assert conn.execute("SELECT count(*) FROM profile_skills WHERE profile_id=?",(updated,)).fetchone()[0]==0
    assert edge.safe_join_team(updated,1,10)
    assert not edge.safe_join_team(updated,1,20)

def test_api_reports_missing_data_without_leaking_path(monkeypatch,tmp_path):
    monkeypatch.setattr(pipeline,"BASE",str(tmp_path))
    from api.main import app
    with TestClient(app) as client:
        assert client.get("/health").status_code==200
        assert client.get("/recommend?team_id=0").status_code==422
        response=client.get("/recommend?team_id=1")
        assert response.status_code==503
        assert str(tmp_path) not in response.text
