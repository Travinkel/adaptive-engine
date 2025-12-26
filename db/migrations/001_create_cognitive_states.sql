-- db/migrations/001_create_cognitive_states.sql

CREATE TABLE IF NOT EXISTS cognitive_states (
    id SERIAL PRIMARY KEY,
    learner_id VARCHAR(255) NOT NULL,
    state_vector FLOAT[] NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
