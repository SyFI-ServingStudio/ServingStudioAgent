-- Frozen from legacy Store sqlite_schema; see source.json.

CREATE TABLE codex_sessions (
                        conversation_id TEXT NOT NULL REFERENCES conversations(id)
                            ON DELETE CASCADE,
                        role TEXT NOT NULL,
                        family TEXT NOT NULL DEFAULT 'gpt',
                        session_id TEXT NOT NULL,
                        PRIMARY KEY (conversation_id, role)
                    );

CREATE TABLE conversation_experiments (
                        conversation_id TEXT NOT NULL REFERENCES conversations(id)
                            ON DELETE CASCADE,
                        experiment_id TEXT NOT NULL REFERENCES experiments(id)
                            ON DELETE CASCADE,
                        turn_id TEXT,
                        relation TEXT NOT NULL,
                        PRIMARY KEY (conversation_id, experiment_id, relation)
                    );

CREATE TABLE conversations (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        naming_state TEXT NOT NULL DEFAULT 'manual',
                        sandbox TEXT NOT NULL,
                        autonomous INTEGER NOT NULL,
                        agent_mode TEXT NOT NULL DEFAULT 'orchestrated',
                        interrupted_role TEXT NOT NULL DEFAULT '',
                        orchestrator_model TEXT NOT NULL DEFAULT 'gpt-5.6-sol',
                        orchestrator_effort TEXT NOT NULL DEFAULT 'xhigh',
                        orchestrator_service_tier TEXT NOT NULL DEFAULT 'default',
                        implementer_model TEXT NOT NULL DEFAULT 'gpt-5.6-sol',
                        implementer_effort TEXT NOT NULL DEFAULT 'xhigh',
                        implementer_service_tier TEXT NOT NULL DEFAULT 'default',
                        assistant_model TEXT NOT NULL DEFAULT 'gpt-5.6-sol',
                        assistant_effort TEXT NOT NULL DEFAULT 'xhigh',
                        assistant_service_tier TEXT NOT NULL DEFAULT 'default',
                        prompt_fingerprint TEXT,
                        peer_workspace TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );

CREATE TABLE execution_jobs (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL REFERENCES conversations(id)
                            ON DELETE CASCADE,
                        turn_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        status TEXT NOT NULL,
                        job_kind TEXT NOT NULL DEFAULT 'simulation',
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

CREATE TABLE messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL REFERENCES conversations(id)
                            ON DELETE CASCADE,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        ts REAL NOT NULL,
                        metadata_json TEXT NOT NULL,
                        -- The turn that produced this message. Null for rows
                        -- imported from the legacy JSON store, which predates
                        -- turns; a citation into one of those can name the
                        -- message but not a turn.
                        turn_id TEXT
                    );

CREATE TABLE schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at REAL NOT NULL
                    );

CREATE TABLE turn_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        turn_id TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
                        sequence INTEGER NOT NULL,
                        kind TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        UNIQUE(turn_id, sequence)
                    );

CREATE TABLE turns (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL REFERENCES conversations(id)
                            ON DELETE CASCADE,
                        status TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );

CREATE INDEX messages_conversation_order
                        ON messages(conversation_id, id);
