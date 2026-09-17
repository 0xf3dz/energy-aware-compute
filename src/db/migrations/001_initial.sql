CREATE TABLE jobs (
    id text PRIMARY KEY,
    workload text NOT NULL,
    status text NOT NULL CHECK (status IN ('QUEUED', 'WAITING_FOR_ENERGY', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),
    priority integer NOT NULL,
    created_at timestamptz NOT NULL,
    scheduled_at timestamptz NOT NULL,
    started_at timestamptz,
    completed_at timestamptz,
    deadline timestamptz,
    "deferrable" boolean NOT NULL,
    estimated_energy_wh double precision,
    actual_estimated_energy_wh double precision,
    payload jsonb NOT NULL DEFAULT '{}',
    result jsonb,
    error text,
    dedupe_key text UNIQUE,
    energy_estimate jsonb
);
CREATE INDEX jobs_pending_order ON jobs (priority DESC, deadline ASC NULLS LAST, created_at ASC, id ASC)
    WHERE status IN ('QUEUED', 'WAITING_FOR_ENERGY');
CREATE INDEX jobs_completed_at ON jobs (completed_at);

CREATE TABLE energy_samples (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id text REFERENCES jobs(id),
    recorded_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source text NOT NULL CHECK (source IN ('energy_state', 'compute')),
    sample jsonb NOT NULL
);
CREATE INDEX energy_samples_latest ON energy_samples (source, recorded_at DESC, id DESC);
CREATE INDEX energy_samples_job ON energy_samples (job_id);

CREATE TABLE inference_metrics (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id text NOT NULL REFERENCES jobs(id),
    recorded_at timestamptz NOT NULL,
    metrics jsonb NOT NULL
);
CREATE INDEX inference_metrics_latest ON inference_metrics (recorded_at DESC, id DESC);

CREATE TABLE weather_forecasts (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id text NOT NULL REFERENCES jobs(id),
    recorded_at timestamptz NOT NULL,
    forecast jsonb NOT NULL
);
CREATE TABLE weather_briefings (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id text NOT NULL REFERENCES jobs(id),
    recorded_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    text text NOT NULL
);
CREATE INDEX weather_briefings_latest ON weather_briefings (recorded_at DESC, id DESC);

CREATE TABLE scheduler_decisions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id text NOT NULL REFERENCES jobs(id),
    recorded_at timestamptz NOT NULL,
    decision text NOT NULL,
    reason text NOT NULL,
    energy_state jsonb NOT NULL
);
CREATE INDEX scheduler_decisions_latest ON scheduler_decisions (recorded_at DESC, id DESC);

CREATE TABLE provider_cache (
    key text PRIMARY KEY,
    value jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);
