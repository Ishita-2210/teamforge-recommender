"""Resolve CSV header aliases without mutating the caller's data frames."""


def _canonical_column(frame, column, aliases, required=True):
    if column in frame.columns:
        return frame
    for alias in aliases:
        if alias in frame.columns:
            return frame.rename(columns={alias: column})
    if required:
        choices = ", ".join((column, *aliases))
        raise ValueError(f"Graph input requires one of these columns: {choices}")
    return frame


def normalize_graph_inputs(user_skills, participation, events=None):
    """Use user_id/event_id internally; canonical names win if aliases coexist."""
    skills = _canonical_column(user_skills, "user_id", ("profile_id",))
    members = _canonical_column(participation, "user_id", ("profile_id",))
    members = _canonical_column(members, "event_id", ("hackathon_id",), required=False)
    if events is not None:
        events = _canonical_column(events, "event_id", ("id", "hackathon_id"))
    return skills, members, events
