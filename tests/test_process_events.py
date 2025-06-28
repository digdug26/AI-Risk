import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import process_events as pe


def fake_classify(sentence):
    return {
        "company": "Acme",
        "ai_causal": "yes",
        "headcount": 50,
        "job_titles": ["engineer"],
    }, 1.0


class DummyConn:
    def __init__(self):
        self.events = []
        self.soc_events = []

    def cursor(self):
        return self

    def execute(self, q, params=None):
        if q.strip().startswith("INSERT INTO events"):
            self.events.append(params)
            self._last_id = len(self.events)
        elif q.strip().startswith("INSERT INTO soc_events"):
            self.soc_events.append(params)
        elif q.strip().startswith("SELECT company_id"):
            self._fetch = (1,)
        elif q.strip().startswith("INSERT INTO companies"):
            self._fetch = (1,)
        elif q.strip().startswith("SELECT doc_id"):
            self._fetchall = []

    def fetchone(self):
        return getattr(self, "_fetch", (1,))

    def fetchall(self):
        return getattr(self, "_fetchall", [])

    def commit(self):
        pass

    def close(self):
        pass


def test_process_document(monkeypatch):
    monkeypatch.setattr(pe, "classify_sentence", fake_classify)
    conn = DummyConn()
    nlp = None
    cw = pe.load_crosswalk()
    idf, vecs = pe.prepare_tfidf(cw)
    doc = {"doc_id": 1, "raw_text": "Acme will layoff 50 engineers due to automation.", "pub_date": "2024-01-01"}
    prob = pe.process_document(conn, doc, nlp, cw, idf, vecs)
    assert prob == 1.0
    assert conn.events
    assert conn.soc_events

