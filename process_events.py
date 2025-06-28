import os
import re
import csv
import json
import sys
import math
import subprocess
from collections import defaultdict
from datetime import datetime

try:
    import psycopg2
except Exception:  # pragma: no cover - psycopg2 may be missing in test env
    psycopg2 = None

try:
    import spacy
except Exception:  # pragma: no cover - spacy may be missing in test env
    spacy = None

from utils.company import get_company_id

KEYWORDS = {"layoff", "eliminate", "attrition", "backfill", "automation"}
SYSTEM_PROMPT = open("system_prompt.txt", encoding="utf-8").read()
USER_TEMPLATE = open("user_prompt_template.txt", encoding="utf-8").read()
CROSSWALK_PATH = os.path.join("data", "title_soc_crosswalk.csv")
MODEL_PATH = os.path.join("models", "bert_ai_causal.pt")

_api_calls = 0
MAX_API_CALLS = int(os.getenv("MAX_API_CALLS", "1000000"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_nlp():
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_trf")
    except Exception:
        return spacy.blank("en")


def sent_tokenize(text, nlp):
    if not nlp:
        for s in re.split(r"(?<=[.!?])\s+", text):
            if s:
                yield s
        return
    doc = nlp(text)
    for sent in doc.sents:
        yield sent.text


def filter_sentences(text, nlp):
    for sent in sent_tokenize(text, nlp):
        lower = sent.lower()
        if any(k in lower for k in KEYWORDS):
            yield sent.strip()


def call_claude(sentence: str) -> str:
    global _api_calls
    import anthropic

    _api_calls += 1
    client = anthropic.Anthropic()
    user_prompt = USER_TEMPLATE.replace("{{SENTENCE_HERE}}", sentence)
    msg = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=256,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return msg.content[0].text.strip()


def validate_json(raw: str):
    p = subprocess.run(
        [sys.executable, "validate_return.py", raw], capture_output=True, text=True
    )
    if p.returncode != 0:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


# Local fallback model -------------------------------------------------------

def predict_local(sentence: str) -> float:
    try:
        import torch
    except Exception:
        return 0.5
    try:
        model = torch.load(MODEL_PATH)
        model.eval()
        with torch.no_grad():
            # placeholder: simple length-based prob
            t = torch.tensor([[len(sentence)]], dtype=torch.float)
            out = model(t)
            prob = float(torch.sigmoid(out).item())
            return prob
    except Exception:
        return 0.5


# Crosswalk + TFIDF ---------------------------------------------------------

def load_crosswalk(path=CROSSWALK_PATH):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"title": row["title"], "soc": row["soc"]})
    return rows


def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def _vector(tokens, idf):
    tf = defaultdict(int)
    for t in tokens:
        tf[t] += 1
    return {t: tf[t] * idf.get(t, 0.0) for t in tf}


def _cosine(v1, v2):
    if not v1 or not v2:
        return 0.0
    keys = set(v1) | set(v2)
    num = sum(v1.get(k, 0.0) * v2.get(k, 0.0) for k in keys)
    den1 = math.sqrt(sum(v * v for v in v1.values()))
    den2 = math.sqrt(sum(v * v for v in v2.values()))
    if den1 == 0 or den2 == 0:
        return 0.0
    return num / (den1 * den2)


def prepare_tfidf(crosswalk):
    docs = [_tokenize(r["title"]) for r in crosswalk]
    df = defaultdict(int)
    for doc in docs:
        for t in set(doc):
            df[t] += 1
    N = len(docs)
    idf = {t: math.log(N / (1 + df[t])) for t in df}
    vectors = [_vector(doc, idf) for doc in docs]
    return idf, vectors


def map_titles_to_socs(titles, crosswalk, idf, vectors):
    soc_weights = defaultdict(float)
    for title in titles:
        tokens = _tokenize(title)
        vec = _vector(tokens, idf)
        sims = []
        for cw_row, cw_vec in zip(crosswalk, vectors):
            sim = _cosine(vec, cw_vec)
            if sim > 0:
                sims.append((cw_row["soc"], sim))
        if not sims:
            continue
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:3]
        tot = sum(s for _, s in top)
        for soc, s in top:
            soc_weights[soc] += s / tot if tot else 0
    total = sum(soc_weights.values())
    if total > 0:
        for soc in list(soc_weights.keys()):
            soc_weights[soc] /= total
    return soc_weights


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def classify_sentence(sentence):
    if _api_calls >= MAX_API_CALLS:
        prob = predict_local(sentence)
        label = {
            "company": "",  # unknown
            "ai_causal": "yes" if prob > 0.5 else "no",
            "headcount": None,
            "job_titles": [],
        }
        return label, prob

    try:
        raw = call_claude(sentence)
    except Exception:
        prob = predict_local(sentence)
        label = {
            "company": "",
            "ai_causal": "yes" if prob > 0.5 else "no",
            "headcount": None,
            "job_titles": [],
        }
        return label, prob

    data = validate_json(raw)
    if not data:
        return None, None
    prob = 1.0 if data.get("ai_causal") == "yes" else 0.0
    return data, prob


def process_document(conn, doc, nlp, crosswalk, idf, vectors):
    sentences = list(filter_sentences(doc["raw_text"], nlp))
    if not sentences:
        return None
    combined_titles = []
    ai_probs = []
    company = None
    headcount = None
    for s in sentences:
        result, prob = classify_sentence(s)
        if not result:
            continue
        if result.get("company"):
            company = result["company"]
        if result.get("headcount") is not None:
            headcount = result["headcount"]
        combined_titles.extend(result.get("job_titles", []))
        ai_probs.append(prob)
    if not combined_titles:
        return None
    p_ai_causal = sum(ai_probs) / len(ai_probs) if ai_probs else 0.0
    company_id = get_company_id(conn, company or "Unknown")
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO events (company_id, pub_date, p_ai_causal, headcount_raw, title_strings, doc_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING event_id
        """,
        (company_id, doc["pub_date"], p_ai_causal, headcount, combined_titles, doc["doc_id"]),
    )
    event_id = cur.fetchone()[0]
    conn.commit()
    soc_map = map_titles_to_socs(combined_titles, crosswalk, idf, vectors)
    for soc, weight in soc_map.items():
        weighted_headcount = (headcount or 0) * weight
        cur.execute(
            """
            INSERT INTO soc_events (event_id, soc, weighted_headcount, p_ai_causal)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (event_id, soc)
            DO UPDATE SET weighted_headcount = EXCLUDED.weighted_headcount,
                          p_ai_causal = EXCLUDED.p_ai_causal
            """,
            (event_id, soc, weighted_headcount, p_ai_causal),
        )
    conn.commit()
    cur.close()
    return p_ai_causal


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    pg_uri = os.environ.get("PG_URI")
    if not pg_uri:
        print("PG_URI env var required", file=sys.stderr)
        return 1
    if psycopg2 is None:
        print("psycopg2 not available", file=sys.stderr)
        return 1
    conn = psycopg2.connect(pg_uri)
    nlp = load_nlp()
    crosswalk = load_crosswalk()
    idf, vectors = prepare_tfidf(crosswalk)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT doc_id, raw_text, pub_date
        FROM news_documents n
        WHERE NOT EXISTS (SELECT 1 FROM events e WHERE e.doc_id = n.doc_id)
        """
    )
    docs = cur.fetchall()
    cur.close()
    processed = 0
    ai_caused = 0
    for doc in docs:
        doc_data = {"doc_id": doc[0], "raw_text": doc[1], "pub_date": doc[2]}
        prob = process_document(conn, doc_data, nlp, crosswalk, idf, vectors)
        if prob is None:
            continue
        processed += 1
        if prob > 0.5:
            ai_caused += 1
    conn.close()
    print(f"Processed {processed} documents; AI causal={ai_caused}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
