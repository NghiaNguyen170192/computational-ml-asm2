#!/usr/bin/env python3
"""
Temporary script: fetch recent crypto news via public RSS feeds, perform a light
rule-based sentiment analysis, and upsert into PostgreSQL table `crypto_news`.

No external ML/NLP dependencies are required. This is intended for quick demos.

Environment variables (same as the app):
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

Run (inside the app container):
  docker-compose exec bitcoinpredictor python temp_fetch_news.py
"""

import os
import re
import sys
import json
import html
import time
from datetime import datetime, timedelta
from typing import Dict, List
import requests
import psycopg2
import xml.etree.ElementTree as ET


RSS_SOURCES = [
    {
        "name": "CoinDesk",
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"
    },
    {
        "name": "CoinTelegraph",
        "url": "https://cointelegraph.com/rss"
    },
    {
        "name": "Bitcoin Magazine",
        "url": "https://bitcoinmagazine.com/.rss/full/"
    }
]


POSITIVE_WORDS = {
    "surge", "rally", "bull", "bullish", "gain", "gains", "record", "all-time",
    "adoption", "approval", "etf", "upgrade", "support", "partnership",
    "breakthrough", "positive", "optimistic", "buy", "bought"
}

NEGATIVE_WORDS = {
    "crash", "bear", "bearish", "fall", "falls", "drop", "plunge", "selloff",
    "ban", "hack", "exploit", "outage", "lawsuit", "probe", "negative",
    "regulatory crackdown", "fear", "sell", "sold"
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_sentiment(title: str, summary: str) -> Dict:
    """Lightweight keyword-based sentiment scoring (polarity in [-1,1])."""
    text = f"{title} {summary}".lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    score = 0.0
    if pos or neg:
        score = (pos - neg) / float(pos + neg)
    sentiment_class = "neutral"
    if score > 0.15:
        sentiment_class = "positive"
    elif score < -0.15:
        sentiment_class = "negative"
    # Subjectivity heuristic: more absolute words → more subjective
    subjectivity = min(1.0, (pos + neg) / 10.0)
    return {
        "class": sentiment_class,
        "polarity": round(score, 3),
        "subjectivity": round(subjectivity, 3)
    }


def parse_rss(source: Dict) -> List[Dict]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/140.0.0.0 Safari/537.36"
        }
        resp = requests.get(source["url"], headers=headers, timeout=20, allow_redirects=True)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = []
        # RSS 2.0 structure: channel/item
        for item in root.findall('.//item'):
            title = clean_text((item.findtext('title') or ''))
            link = (item.findtext('link') or '').strip()
            desc = clean_text((item.findtext('description') or ''))
            pub = item.findtext('pubDate') or item.findtext('{http://purl.org/dc/elements/1.1/}date') or ''
            try:
                # Try multiple formats; fall back to now
                date = datetime.strptime(pub[:25], "%a, %d %b %Y %H:%M:%S") if pub else datetime.utcnow()
            except Exception:
                date = datetime.utcnow()
            items.append({
                "source": source["name"],
                "title": title,
                "url": link,
                "text": desc,
                "date": date
            })
        return items
    except Exception as e:
        print(f"[WARN] Failed to parse RSS from {source['name']}: {e}")
        return []


def db_connect():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'postgres'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'airflow'),
        user=os.getenv('DB_USER', 'airflow'),
        password=os.getenv('DB_PASSWORD', 'airflow'),
    )


def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS crypto_news (
                id SERIAL PRIMARY KEY,
                date TIMESTAMP NOT NULL,
                sentiment JSONB,
                source VARCHAR(255),
                subject TEXT,
                text TEXT,
                title TEXT,
                url TEXT
            );
            """
        )
    conn.commit()

    # Skip creating any unique index to avoid conflicts; you can index later manually.
    return


def upsert_news(conn, rows: List[Dict]) -> int:
    if not rows:
        return 0
    inserted = 0
    with conn.cursor() as cur:
        for r in rows:
            sentiment = simple_sentiment(r["title"], r["text"])
            cur.execute(
                """
                INSERT INTO crypto_news (date, sentiment, source, subject, text, title, url)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (r["date"], json.dumps(sentiment), r["source"], None, r["text"], r["title"], r["url"])
            )
            inserted += 1
    conn.commit()
    return inserted


def main():
    print("[INFO] Fetching RSS from sources...")
    all_items: List[Dict] = []
    for src in RSS_SOURCES:
        items = parse_rss(src)
        all_items.extend(items)
        time.sleep(0.5)

    # Keep only last 30 days
    cutoff = datetime.utcnow() - timedelta(days=30)
    recent = [it for it in all_items if it["date"] >= cutoff and it["title"]]
    print(f"[INFO] Collected {len(recent)} recent items")

    if not recent:
        print("[WARN] No recent items collected.")
        return

    conn = db_connect()
    try:
        ensure_table(conn)
        inserted = upsert_news(conn, recent)
        print(f"[INFO] Upserted {inserted} rows into crypto_news")
    finally:
        conn.close()


if __name__ == "__main__":
    main()


