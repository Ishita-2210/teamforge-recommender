import networkx as nx
import pandas as pd
import pytest

from graph.build_graph import build_user_graph
from graph.build_graph_fast import build_user_graph_fast
from recommender import pipeline
from recommender.skill_match import compute_skill_fit


@pytest.mark.parametrize("required", ["Beginner", "Intermediate", "Pro"])
def test_skill_qualification_improves_with_each_level(required):
    needs = pd.DataFrame({"skill": ["Python"], "min_level": [required], "priority": ["High"]})
    levels = [None, "Beginner", "Intermediate", "Pro"]
    scores = []
    for level in levels:
        rows = [] if level is None else [{"skill": "Python", "level": level}]
        scores.append(compute_skill_fit(pd.DataFrame(rows), needs))
    assert all(0 <= score <= 1 for score in scores)
    assert all(lower < higher for lower, higher in zip(scores, scores[1:]))


def test_baseline_ranks_qualified_candidates_above_level_shortfalls(tmp_path, monkeypatch):
    tables = {
        "users": pd.DataFrame({"id": [1, 2, 3, 4]}),
        "user_skills": pd.DataFrame({
            "user_id": [2, 3, 4], "skill": ["Python"] * 3,
            "level": ["Beginner", "Pro", "Intermediate"],
        }),
        "team_needed_skills": pd.DataFrame({
            "team_id": [10], "skill": ["Python"], "min_level": ["Pro"], "priority": ["High"],
        }),
        "teams": pd.DataFrame({"id": [10], "hackathon_id": [100], "owner_id": [1]}),
        "participation": pd.DataFrame(columns=["id", "user_id", "hackathon_id", "team_id", "status"]),
    }
    for name, frame in tables.items():
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    monkeypatch.setattr(pipeline, "BASE", str(tmp_path))
    monkeypatch.setattr(pipeline, "_xgb", None)
    monkeypatch.setattr(pipeline, "_lgbm", None)
    monkeypatch.setattr(pipeline, "semantic_scores_for_team", lambda _: {})
    monkeypatch.setattr(pipeline, "batch_graph_similarity", lambda *args, **kwargs: {})
    results = pipeline.recommend(10)
    assert [row["user_id"] for row in results] == [3, 4, 2]
    assert all(row["scoring_mode"] == "weighted-baseline" for row in results)
    assert results[0]["skill_score"] > results[1]["skill_score"] > results[2]["skill_score"]


@pytest.mark.parametrize("event_id", ["id", "event_id", "hackathon_id"])
@pytest.mark.parametrize("person_id", ["user_id", "profile_id"])
@pytest.mark.parametrize("participation_event", ["event_id", "hackathon_id"])
def test_graph_builders_accept_original_and_canonical_headers(event_id, person_id, participation_event):
    users = pd.DataFrame({"id": [1, 2, 3]})
    skills = pd.DataFrame({person_id: [1, 2, 3], "skill": ["Python", "SQL", "C++"]})
    participation = pd.DataFrame({
        person_id: [1, 2, 3], participation_event: [10, 20, 30], "team_id": [100, 200, 300],
    })
    events = pd.DataFrame({event_id: [10, 20, 30], "domain": ["AI", "AI", "CV"]})
    originals = [frame.copy(deep=True) for frame in (skills, participation, events)]

    graph = build_user_graph(users, skills, participation, events)
    assert graph[1][2]["domain"] == 1
    assert not graph.has_edge(1, 3)

    heterogeneous = build_user_graph_fast(users, skills, participation, events)
    assert heterogeneous.has_edge("user_1", "event_10")
    assert heterogeneous.has_edge("user_1", "skill_Python")
    assert heterogeneous.has_edge("event_10", "domain_AI")
    assert heterogeneous.has_edge("event_20", "domain_AI")
    assert not heterogeneous.has_edge("event_30", "domain_AI")
    for frame, original in zip((skills, participation, events), originals):
        pd.testing.assert_frame_equal(frame, original)


def test_duplicate_members_and_skills_do_not_create_self_loops_or_extra_weight():
    users = pd.DataFrame({"id": [1, 2]})
    skills = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2], "skill": ["Python"] * 5,
        "level": ["Beginner", "Pro", "Pro", "Intermediate", "Intermediate"],
    })
    participation = pd.DataFrame({
        "user_id": [1, 1, 2], "team_id": [100] * 3, "hackathon_id": [10] * 3,
    })
    events = pd.DataFrame({"id": [10], "domain": ["AI"]})
    graph = build_user_graph(users, skills, participation, events)
    assert nx.number_of_selfloops(graph) == 0
    assert graph.number_of_edges() == 1
    assert graph[1][2] == {"collab": 1, "skill": 2.5, "domain": 1, "feedback": 0}


def test_event_metadata_is_optional_for_collaboration_and_skill_graphs():
    users = pd.DataFrame({"id": [1, 2]})
    skills = pd.DataFrame({"user_id": [1, 2], "skill": ["Python", "SQL"]})
    participation = pd.DataFrame({"profile_id": [1, 2], "team_id": [100, 100]})
    graph = build_user_graph(users, skills, participation)
    assert graph[1][2]["collab"] == 1
    heterogeneous = build_user_graph_fast(users, skills, participation)
    assert heterogeneous.has_edge("user_1", "skill_Python")
    assert not any(str(node).startswith("event_") for node in heterogeneous)


@pytest.mark.parametrize("label", [None, "domain", "event_type"])
def test_missing_domain_metadata_does_not_invent_shared_interests(label):
    users = pd.DataFrame({"id": [1, 2, 3]})
    skills = pd.DataFrame(columns=["user_id", "skill"])
    participation = pd.DataFrame({
        "user_id": [1, 2, 3], "team_id": [100, 200, 300], "hackathon_id": [10, 20, None],
    })
    events = pd.DataFrame({"id": [10, 20]})
    if label is not None:
        events[label] = [None, None]
    graph = build_user_graph(users, skills, participation, events)
    assert graph.number_of_edges() == 0
    heterogeneous = build_user_graph_fast(users, skills, participation, events)
    assert not any(str(node).startswith("domain_") for node in heterogeneous)
    assert heterogeneous.degree("user_3") == 0
