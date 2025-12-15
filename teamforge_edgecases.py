
"""
teamforge_edgecases.py

SQLite-based module to manage profiles, skills, participation, interactions and edge-case logic.
Functions:
 - create_user(username)
 - create_profile(user_id, role, event_id, skills)
 - update_profile_atomic(profile_id, new_bio, new_skills)
 - handle_swipe(viewer_profile_id, target_profile_id, team_id, event_id, action)
 - invite_profile(inviter_profile_id, target_profile_id, team_id, event_id)
 - process_mutual_match_atomic(...)
 - safe_join_team(...)
Notes: for production use Postgres; this is a Colab/dev ready sqlite implementation.
"""
import os, sqlite3, time
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "teamforge_edgecases.db")

# spam config
SPAM_WINDOW_SECONDS = 60
SPAM_LIMIT = 30

def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    # users
    cur.execute(\"\"\"CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        created_at INTEGER DEFAULT (strftime('%s','now'))
    );\"\"\")
    # profiles (versioned)
    cur.execute(\"\"\"CREATE TABLE IF NOT EXISTS profiles (
        profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        event_id INTEGER,
        bio_text TEXT,
        looking_for_text TEXT,
        created_at INTEGER DEFAULT (strftime('%s','now')),
        active INTEGER DEFAULT 1,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );\"\"\")
    # profile_skills
    cur.execute(\"\"\"CREATE TABLE IF NOT EXISTS profile_skills (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER NOT NULL,
        skill TEXT,
        level TEXT,
        FOREIGN KEY(profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
    );\"\"\")
    # events minimal
    cur.execute(\"\"\"CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER PRIMARY KEY,
        name TEXT,
        event_type TEXT,
        domain TEXT
    );\"\"\")
    # teams
    cur.execute(\"\"\"CREATE TABLE IF NOT EXISTS teams (
        team_id INTEGER PRIMARY KEY,
        event_id INTEGER,
        owner_profile_id INTEGER,
        project_text TEXT,
        looking_for_text TEXT
    );\"\"\")
    # team_needed_skills
    cur.execute(\"\"\"CREATE TABLE IF NOT EXISTS team_needed_skills (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        team_id INTEGER,
        skill TEXT,
        min_level TEXT,
        priority TEXT
    );\"\"\")
    # participation: unique per (event, profile)
    cur.execute(\"\"\"CREATE TABLE IF NOT EXISTS participation (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        event_id INTEGER,
        team_id INTEGER,
        status TEXT,
        created_at INTEGER DEFAULT (strftime('%s','now')),
        UNIQUE(event_id, profile_id)
    );\"\"\")
    # interactions
    cur.execute(\"\"\"CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        viewer_profile_id INTEGER,
        target_profile_id INTEGER,
        team_id INTEGER,
        event_id INTEGER,
        action TEXT,
        created_at INTEGER DEFAULT (strftime('%s','now'))
    );\"\"\")
    # spam counters
    cur.execute(\"\"\"CREATE TABLE IF NOT EXISTS spam_counters (
        profile_id INTEGER PRIMARY KEY,
        window_start INTEGER,
        count INTEGER DEFAULT 0,
        flagged INTEGER DEFAULT 0
    );\"\"\")
    conn.commit()
    conn.close()

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    try:
        yield conn
    finally:
        conn.close()

def create_user(username):
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(\"BEGIN IMMEDIATE;\")
            cur.execute(\"INSERT INTO users (username) VALUES (?);\", (username,))
            uid = cur.lastrowid
            conn.commit()
            return uid
        except sqlite3.IntegrityError:
            conn.rollback()
            cur.execute(\"SELECT id FROM users WHERE username = ?;\", (username,))
            return cur.fetchone()[0]

def create_profile(user_id, role, event_id=None, bio=None, looking_for=None, skills=None):
    # create new profile row (scoped to event if provided)
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(\"BEGIN IMMEDIATE;\")
        cur.execute(\"INSERT INTO profiles (user_id, role, event_id, bio_text, looking_for_text) VALUES (?,?,?,?,?);\", (user_id, role, event_id, bio, looking_for))
        pid = cur.lastrowid
        if skills:
            for s,l in skills:
                cur.execute(\"INSERT INTO profile_skills (profile_id, skill, level) VALUES (?,?,?);\", (pid, s, l))
        conn.commit()
        return pid

def update_profile_atomic(profile_id, new_bio=None, new_skills=None):
    # clone profile: create new profile copy, deactivate old, attach new skills
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(\"BEGIN IMMEDIATE;\")
        cur.execute(\"SELECT user_id, role, event_id, bio_text, looking_for_text FROM profiles WHERE profile_id = ? AND active = 1;\", (profile_id,))
        r = cur.fetchone()
        if not r:
            conn.rollback()
            raise ValueError('profile not found or inactive')
        user_id, role, event_id, bio_text, looking_for_text = r
        # create clone
        cur.execute(\"INSERT INTO profiles (user_id, role, event_id, bio_text, looking_for_text) VALUES (?,?,?,?,?);\", (user_id, role, event_id, new_bio or bio_text, looking_for_text))
        new_pid = cur.lastrowid
        # copy skills
        cur.execute(\"SELECT skill, level FROM profile_skills WHERE profile_id = ?;\", (profile_id,))
        rows = cur.fetchall()
        for s,l in rows:
            cur.execute(\"INSERT INTO profile_skills (profile_id, skill, level) VALUES (?,?,?);\", (new_pid, s, l))
        if new_skills:
            cur.execute(\"DELETE FROM profile_skills WHERE profile_id = ?;\", (new_pid,))
            for s,l in new_skills:
                cur.execute(\"INSERT INTO profile_skills (profile_id, skill, level) VALUES (?,?,?);\", (new_pid, s, l))
        # deactivate old
        cur.execute(\"UPDATE profiles SET active = 0 WHERE profile_id = ?;\", (profile_id,))
        conn.commit()
        return new_pid

def increment_spam(profile_id):
    now = int(time.time())
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(\"BEGIN IMMEDIATE;\")
        cur.execute(\"SELECT window_start, count, flagged FROM spam_counters WHERE profile_id = ?;\", (profile_id,))
        r = cur.fetchone()
        if not r:
            cur.execute(\"INSERT INTO spam_counters (profile_id, window_start, count, flagged) VALUES (?,?,?,0);\", (profile_id, now, 1))
            conn.commit()
            return False
        ws, cnt, flagged = r
        if now - ws <= SPAM_WINDOW_SECONDS:
            cnt += 1
            if cnt >= SPAM_LIMIT:
                cur.execute(\"UPDATE spam_counters SET count = ?, flagged = 1 WHERE profile_id = ?;\", (cnt, profile_id))
                conn.commit()
                return True
            else:
                cur.execute(\"UPDATE spam_counters SET count = ? WHERE profile_id = ?;\", (cnt, profile_id))
                conn.commit()
                return False
        else:
            cur.execute(\"UPDATE spam_counters SET window_start = ?, count = 1, flagged = 0 WHERE profile_id = ?;\", (now, profile_id))
            conn.commit()
            return False

def log_interaction(viewer_profile_id, target_profile_id, team_id, event_id, action):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(\"BEGIN IMMEDIATE;\")
        cur.execute(\"INSERT INTO interactions (viewer_profile_id, target_profile_id, team_id, event_id, action) VALUES (?,?,?,?,?);\", (viewer_profile_id, target_profile_id, team_id, event_id, action))
        conn.commit()

def safe_join_team(profile_id, event_id, team_id, status='accepted'):
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(\"BEGIN IMMEDIATE;\")
            cur.execute(\"INSERT INTO participation (profile_id, event_id, team_id, status) VALUES (?,?,?,?);\", (profile_id, event_id, team_id, status))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            conn.rollback()
            return False

def process_mutual_match_atomic(a_profile, b_profile, team_id, event_id):
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(\"BEGIN IMMEDIATE;\")
            # ensure not already in a team
            cur.execute(\"SELECT 1 FROM participation WHERE profile_id = ? AND event_id = ?;\", (a_profile, event_id))
            if cur.fetchone():
                conn.rollback(); return False
            cur.execute(\"SELECT 1 FROM participation WHERE profile_id = ? AND event_id = ?;\", (b_profile, event_id))
            if cur.fetchone():
                conn.rollback(); return False
            # lock both into team
            cur.execute(\"INSERT INTO participation (profile_id, event_id, team_id, status) VALUES (?,?,?,'locked');\", (a_profile, event_id, team_id))
            cur.execute(\"INSERT INTO participation (profile_id, event_id, team_id, status) VALUES (?,?,?,'locked');\", (b_profile, event_id, team_id))
            # remove their pending swipe interactions for that team/event
            cur.execute(\"DELETE FROM interactions WHERE ((viewer_profile_id = ? AND target_profile_id = ?) OR (viewer_profile_id = ? AND target_profile_id = ?)) AND event_id = ? AND team_id = ?;\", (a_profile, b_profile, b_profile, a_profile, event_id, team_id))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            conn.rollback(); return False

def handle_swipe(viewer_profile_id, target_profile_id, team_id, event_id, action):
    # 1) spam guard
    if increment_spam(viewer_profile_id):
        return {'status':'spam_flagged'}
    # 2) log swipe
    log_interaction(viewer_profile_id, target_profile_id, team_id, event_id, action)
    # 3) if swipe_right, check mutual
    if action == 'swipe_right':
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(\"SELECT 1 FROM interactions WHERE viewer_profile_id = ? AND target_profile_id = ? AND action = 'swipe_right' AND team_id = ? AND event_id = ? LIMIT 1;\", (target_profile_id, viewer_profile_id, team_id, event_id))
            if cur.fetchone():
                ok = process_mutual_match_atomic(viewer_profile_id, target_profile_id, team_id, event_id)
                return {'status':'matched' if ok else 'match_failed'}
    return {'status':'ok'}

def invite_profile(inviter_profile_id, target_profile_id, team_id, event_id):
    # log invite
    log_interaction(inviter_profile_id, target_profile_id, team_id, event_id, 'invite')
    # try to lock the target into the team (first-wins)
    joined = safe_join_team(target_profile_id, event_id, team_id)
    return {'invited':True, 'joined': joined}

# initialize DB on import
if not os.path.exists(DB_PATH):
    init_db()
