
"""
main.py - runner to load CSVs, build graph, and call ranker.
Also shows how to call edge-case functions for demo.
"""
import pandas as pd
from graph.build_graph import build_user_graph
from recommender.rank import rank_candidates
import teamforge_edgecases as te
import os

BASE = "/content/drive/MyDrive/teamforge"

# load CSVs
users = pd.read_csv(os.path.join(BASE, "users.csv"))
user_skills = pd.read_csv(os.path.join(BASE, "user_skills.csv"))
events = pd.read_csv(os.path.join(BASE, "hackathons.csv"))
teams = pd.read_csv(os.path.join(BASE, "teams.csv"))
team_skills = pd.read_csv(os.path.join(BASE, "team_needed_skills.csv"))
participation = pd.read_csv(os.path.join(BASE, "participation.csv"))

print("Loaded counts:", len(users), len(user_skills), len(events), len(teams), len(team_skills), len(participation))

# Build graph
G = build_user_graph(users, user_skills, participation, events)

# choose a team
team_row = teams.iloc[0]
team_id = team_row['id']
team_skill_subset = team_skills[team_skills['team_id']==team_id]
project_text = team_row.get('project_text','')

# choose anchor_user as owner_id if present and exists in users
anchor = None
if 'owner_id' in team_row and team_row['owner_id'] in users['id'].values:
    anchor = int(team_row['owner_id'])

# simple impressions map (for demo all zero)
impressions = {}

top = rank_candidates(team_id, team_skill_subset, project_text, users, user_skills, G, anchor_user=anchor, impressions_map=impressions, top_k=10)
print("Top recommendations (user_id, score):")
for u,s in top:
    print(u,s)

# Demonstrate edge-case module usage (comment/uncomment as needed)
# uid = te.create_user("alice@example.com")
# pid = te.create_profile(uid, "Frontend", event_id=None, bio="Frontend dev", looking_for="Backend", skills=[("React","Pro"),("HTML","Pro"),("CSS","Intermediate")])
# uid2 = te.create_user("bob@example.com")
# pid2 = te.create_profile(uid2, "Backend", event_id=None, bio="Backend dev", skills=[("Node.js","Pro"),("PostgreSQL","Intermediate")])
# print("Created demo profiles", pid, pid2)
