-- news_documents holds raw crawls
CREATE TABLE IF NOT EXISTS news_documents (
    id           SERIAL PRIMARY KEY,
    source       TEXT          NOT NULL,
    url          TEXT UNIQUE   NOT NULL,
    pub_date     DATE          NOT NULL,
    raw_text     TEXT          NOT NULL,
    pulled_at    TIMESTAMPTZ   DEFAULT now()
);

-- events table (filled by process_events.py)
CREATE TABLE IF NOT EXISTS events (
    event_id      SERIAL PRIMARY KEY,
    company_id    INTEGER,
    pub_date      DATE,
    p_ai_causal   REAL,
    headcount_raw INTEGER,
    title_strings TEXT[],
    doc_id        INTEGER REFERENCES news_documents(id)
);

-- soc_events table (title→SOC fan-out)
CREATE TABLE IF NOT EXISTS soc_events (
    event_id          INTEGER REFERENCES events(event_id),
    soc               CHAR(7),
    weighted_headcount REAL,
    p_ai_causal        REAL,
    PRIMARY KEY (event_id, soc)
);
