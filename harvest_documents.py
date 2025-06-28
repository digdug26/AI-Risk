import os
import sys
import json
import argparse
from datetime import datetime, timedelta

import requests
import feedparser
import psycopg2
from ratelimit import limits, sleep_and_retry


EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
LAYOFFS_URL = "https://layoffs.fyi/wp-json/wp/v2/posts?per_page=100"
TECHCRUNCH_FEED = "https://techcrunch.com/tag/layoffs/feed/"

RATE = 10
PER_SECOND = 1


@sleep_and_retry
@limits(calls=RATE, period=PER_SECOND)
def edgar_request(method: str, url: str, **kwargs):
    headers = kwargs.pop("headers", {})
    ua = os.environ.get("EDGAR_USER_AGENT")
    if not ua:
        raise RuntimeError("EDGAR_USER_AGENT env var required")
    headers.setdefault("User-Agent", ua)
    return requests.request(method, url, headers=headers, **kwargs)


def parse_date(dt_str: str) -> datetime:
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return datetime.utcnow()


def fetch_layoffs_fyi(since: datetime):
    print("Fetching Layoffs.fyi...")
    try:
        resp = requests.get(LAYOFFS_URL, timeout=30)
        resp.raise_for_status()
        for item in resp.json():
            pub = parse_date(item.get("date", ""))
            if pub < since:
                continue
            text = item.get("content", {}).get("rendered", "")
            yield {
                "source": "layoffs.fyi",
                "url": item.get("link"),
                "pub_date": pub,
                "raw_text": text,
            }
    except Exception as exc:
        print(f"Layoffs.fyi fetch error: {exc}")


def fetch_edgar(since: datetime):
    print("Fetching SEC EDGAR...")
    payload = {
        "keys": "Item 2.05",
        "forms": ["8-K"],
        "dateRange": "3d",
        "from": 0,
        "size": 100,
        "sort": "filedAt",
        "order": "desc",
    }
    try:
        resp = edgar_request("post", EDGAR_SEARCH_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {})
            filed = parse_date(src.get("filedAt", ""))
            if filed < since:
                continue
            file_url = src.get("fileUrl")
            if not file_url:
                continue
            doc_url = f"https://www.sec.gov/Archives/{file_url}"
            try:
                doc_resp = edgar_request("get", doc_url, timeout=30)
                doc_resp.raise_for_status()
                text = doc_resp.text
            except Exception as e:
                print(f"EDGAR document fetch error: {e}")
                text = ""
            yield {
                "source": "edgar",
                "url": doc_url,
                "pub_date": filed,
                "raw_text": text,
            }
    except Exception as exc:
        print(f"EDGAR fetch error: {exc}")


def fetch_techcrunch(since: datetime):
    print("Fetching TechCrunch RSS...")
    try:
        feed = feedparser.parse(TECHCRUNCH_FEED)
        for entry in feed.entries:
            pub = parse_date(entry.get("published", ""))
            if pub < since:
                continue
            link = entry.get("link")
            text = ""
            try:
                resp = requests.get(link, timeout=30)
                resp.raise_for_status()
                text = resp.text
            except Exception as e:
                print(f"TechCrunch article fetch error: {e}")
            yield {
                "source": "techcrunch",
                "url": link,
                "pub_date": pub,
                "raw_text": text,
            }
    except Exception as exc:
        print(f"TechCrunch fetch error: {exc}")


FETCHERS = [fetch_layoffs_fyi, fetch_edgar, fetch_techcrunch]


def insert_documents(conn, docs):
    cur = conn.cursor()
    for doc in docs:
        try:
            cur.execute(
                """
                INSERT INTO news_documents (source, url, pub_date, raw_text, pulled_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (url) DO NOTHING
                """,
                (doc["source"], doc["url"], doc["pub_date"], doc["raw_text"]),
            )
        except Exception as exc:
            conn.rollback()
            print(f"DB insert error for {doc['url']}: {exc}")
        else:
            conn.commit()
            print(f"Inserted {doc['url']}")
    cur.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Harvest workforce news documents")
    parser.add_argument("--since", help="Only pull documents on/after this date (YYYY-MM-DD)")
    args = parser.parse_args(argv)

    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d")
    else:
        since = datetime.utcnow() - timedelta(days=1)

    pg_uri = os.environ.get("PG_URI")
    if not pg_uri:
        print("PG_URI env var required", file=sys.stderr)
        return 1

    conn = psycopg2.connect(pg_uri)

    for fetcher in FETCHERS:
        docs = list(fetcher(since))
        insert_documents(conn, docs)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
