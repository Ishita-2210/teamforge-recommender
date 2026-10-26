
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
from pandas import isna, notna
from graph.schema import normalize_graph_inputs

def build_user_graph(users_df, user_skills_df, participation_df, events_df=None):
    user_skills_df, participation_df, events_df = normalize_graph_inputs(
        user_skills_df, participation_df, events_df
    )
    G = nx.Graph()
    # add nodes from users
    for uid in users_df['id'].tolist():
        G.add_node(int(uid))

    # 1) collaboration edges: users in same team
    grouped = participation_df.groupby("team_id")["user_id"].apply(list)
    for members in grouped:
        members = sorted({int(m) for m in members})
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
    skill_users = defaultdict(dict)
    for _, r in user_skills_df.iterrows():
        try:
            uid = int(r['user_id'])
            level = level_map.get(r.get('level','Beginner'), 1)
            skill_users[r['skill']][uid] = max(skill_users[r['skill']].get(uid, 0), level)
        except:
            pass

    for members in skill_users.values():
        # Each user contributes their best level once per skill, avoiding self-loops.
        entries = list(members.items())
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
    if events_df is not None and 'event_id' in participation_df.columns:
        # build user -> set of domains they participated in (from participation -> events)
        event_map = {}
        for _, ev in events_df.iterrows():
            domain = ev.get('domain', ev.get('event_type', None))
            if notna(domain):
                event_map[int(ev['event_id'])] = domain
        user_domains = defaultdict(set)
        for _, p in participation_df.iterrows():
            if isna(p['event_id']):
                continue
            uid = int(p['user_id'])
            eid = int(p['event_id'])
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
