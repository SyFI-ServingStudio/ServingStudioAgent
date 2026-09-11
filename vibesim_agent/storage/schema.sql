CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    naming_state TEXT NOT NULL DEFAULT 'manual',
    sandbox TEXT NOT NULL,
    autonomous INTEGER NOT NULL,
    agent_mode TEXT NOT NULL,
    interrupted_role TEXT NOT NULL DEFAULT '',
    prompt_fingerprint TEXT,
    peer_workspace TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE role_settings (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    session_scope TEXT NOT NULL,
    model_id TEXT NOT NULL,
    effort TEXT NOT NULL,
    service_tier TEXT NOT NULL,
    PRIMARY KEY (conversation_id, role)
);

CREATE TABLE agent_sessions (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    session_scope TEXT NOT NULL,
    session_id TEXT NOT NULL,
    PRIMARY KEY (conversation_id, role)
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts REAL NOT NULL,
    metadata_json TEXT NOT NULL,
    turn_id TEXT
);
CREATE INDEX messages_conversation_order ON messages(conversation_id, id);

CREATE TABLE turns (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE turn_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE (turn_id, sequence)
);

CREATE TABLE execution_jobs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_id TEXT NOT NULL,
    role TEXT NOT NULL,
    status TEXT NOT NULL,
    job_kind TEXT NOT NULL,
    artifact_path TEXT,
    resource_id TEXT,
    analyzer_resource_id TEXT,
    descriptor_json TEXT NOT NULL DEFAULT '{}',
    summary_json TEXT,
    experiment_id TEXT,
    experiment_path TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    origin_kind TEXT NOT NULL,
    job_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE conversation_experiments (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    turn_id TEXT,
    relation TEXT NOT NULL,
    PRIMARY KEY (conversation_id, experiment_id, relation)
);
