
"""
build_graph.py
Multi-edge user graph builder:
 - collaboration edges (shared teams) with recency/weight support (placeholder)
 - skill overlap edges (weighted by level)
 - domain/event edges (users in same event types)
Designed to be safe to call repeatedly.
"""
import networkx as nx
from collections import defaultdict
import math

def build_user_graph(users_df, user_skills_df, participation_df, events_df=None):
    G = nx.Graph()
    # add nodes from users
    for uid in users_df['id'].tolist():
        G.add_node(int(uid))

    # 1) collaboration edges: users in same team
    team_col = "profile_id" if "profile_id" in participation_df.columns else "user_id"
    grouped = participation_df.groupby("team_id")[team_col].apply(list)
    for members in grouped:
        members = [int(m) for m in members]
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                u, v = members[i], members[j]
                if G.has_edge(u, v):
                    G[u][v]["collab"] = G[u][v].get("collab", 0) + 1
                else:
                    G.add_edge(u, v, collab=1, skill=0, domain=0, feedback=0)

    # 2) skill overlap edges: count shared skills and consider level weight
    # We assign numeric level: Beginner=1, Intermediate=2, Pro=3
    level_map = {"Beginner":1, "Intermediate":2, "Pro":3}
    skill_users = defaultdict(list)
    for _, r in user_skills_df.iterrows():
        try:
            skill_users[r['skill']].append((int(r['user_id']), level_map.get(r.get('level','Beginner'),1)))
        except:
            pass

    for skill, entries in skill_users.items():
        # for each pair increment 'skill' by product of levels (captures stronger overlap)
        for i in range(len(entries)):
            for j in range(i+1, len(entries)):
                u, lu = entries[i]
                v, lv = entries[j]
                weight = (lu + lv) / 2.0  # average level contribution
                if G.has_edge(u, v):
                    G[u][v]['skill'] = G[u][v].get('skill', 0) + weight
                else:
                    G.add_edge(u, v, collab=0, skill=weight, domain=0, feedback=0)

    # 3) domain/event overlap (optional): if events_df provided
    if events_df is not None and 'id' in events_df.columns:
        # build user -> set of domains they participated in (from participation -> events)
        event_map = {}
        for _, ev in events_df.iterrows():
            event_map[int(ev['event_id'])] = ev.get('domain', ev.get('event_type', None))
        user_domains = defaultdict(set)
        for _, p in participation_df.iterrows():
            uid = int(p[team_col])
            eid = int(p['event_id']) if 'event_id' in p else p.get('hackathon_id', None)
            if eid and eid in event_map:
                user_domains[uid].add(event_map[eid])
        # connect users sharing domains
        domain_buckets = defaultdict(list)
        for uid, doms in user_domains.items():
            for d in doms:
                domain_buckets[d].append(uid)
        for d, users in domain_buckets.items():
            for i in range(len(users)):
                for j in range(i+1, len(users)):
                    u, v = users[i], users[j]
                    if G.has_edge(u, v):
                        G[u][v]['domain'] = G[u][v].get('domain', 0) + 1
                    else:
                        G.add_edge(u, v, collab=0, skill=0, domain=1, feedback=0)

    # Normalize or cap large values to prevent dominance - caller can adjust/scale as needed
    return G
