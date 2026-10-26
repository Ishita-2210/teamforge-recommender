
import networkx as nx
from pandas import isna
from graph.schema import normalize_graph_inputs

def build_user_graph_fast(users_df, user_skills_df, participation_df, events_df=None):
    user_skills_df, participation_df, events_df = normalize_graph_inputs(
        user_skills_df, participation_df, events_df
    )
    G = nx.Graph()

    # ---- Add user nodes ----
    for uid in users_df["id"].astype(int).tolist():
        G.add_node(f"user_{uid}")

    # ---- Add skill nodes ----
    for sk in user_skills_df["skill"].unique():
        G.add_node(f"skill_{sk}")

    # User → Skill edges
    for _, row in user_skills_df.iterrows():
        uid = int(row["user_id"])
        sk = row["skill"]
        G.add_edge(f"user_{uid}", f"skill_{sk}", weight=1.0)

    # ---- Add event nodes ----
    if events_df is not None:
        for _, row in events_df.iterrows():
            eid = int(row["event_id"])
            G.add_node(f"event_{eid}")

    # User → Event edges
    if "event_id" in participation_df.columns:
        for _, row in participation_df.iterrows():
            if isna(row["event_id"]):
                continue
            uid = int(row["user_id"])
            eid = int(row["event_id"])
            G.add_edge(f"user_{uid}", f"event_{eid}", weight=2.0)

    # ---- Add domain nodes ----
    if events_df is not None and "domain" in events_df.columns:
        domain_events = events_df.dropna(subset=["domain"])
        for dom in domain_events["domain"].unique():
            G.add_node(f"domain_{dom}")

        # Event → Domain edges
        for _, row in domain_events.iterrows():
            eid = int(row["event_id"])
            dom = row["domain"]
            G.add_edge(f"event_{eid}", f"domain_{dom}", weight=1.0)

    return G
