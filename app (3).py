"""
TubePilot AI - YouTube keyword research & video discovery workspace (Flask).

Run:
    pip install -r requirements.txt
    cp .env.example .env   # or export the variables below
    python app.py          # http://127.0.0.1:5000
Production:
    gunicorn -w 2 -b 0.0.0.0:8000 app:app

Environment variables (all optional except SECRET_KEY in production):
    SECRET_KEY          Flask session signing key (REQUIRED in production)
    APP_NAME            Product name shown in the UI (default: TubePilot AI)
    DATABASE_PATH       SQLite file path (default: tubepilot.db)
    YOUTUBE_API_KEY     YouTube Data API v3 key. Without it the app runs in
                        clearly labeled SAMPLE-DATA mode.
    ANTHROPIC_API_KEY   Enables LLM keyword expansion / briefs. Without it,
                        template-based (non-AI) generation is used and labeled.
    ANTHROPIC_MODEL     Model id for the LLM provider (default: claude-sonnet-5-5)
    CACHE_TTL_SECONDS   API response cache lifetime (default: 21600)
    RATE_LIMIT_PER_MIN  Per-IP request limit for /api/* (default: 120)

Data honesty rules implemented here:
  * YouTube exposes no public search-volume figure; none is ever invented.
  * Every metric carries source, kind (measured / inferred / sample), date.
  * Missing signals are excluded from the score and lower its confidence.
  * Sample data is flagged and never mixed silently with real data.
"""
import csv
import io
import json
import math
import os
import re
import secrets
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from functools import wraps

import requests
from dotenv import load_dotenv
from flask import Flask, Response, g, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "TubePilot AI")
TAGLINE = "Find the demand. Discover the opportunity. Create what wins."
DB_PATH = os.getenv("DATABASE_PATH", "tubepilot.db")
YT_KEY = os.getenv("YOUTUBE_API_KEY", "")
LLM_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "21600"))
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))

DEFAULT_WEIGHTS = {"demand": 30, "competition": 25, "relevance": 20, "gap": 15, "trend": 10}
PLAN_LIMITS = {"free": {"searches_per_day": 20, "playlists": 10, "tracked": 10}}  # configurable

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY") or secrets.token_hex(32),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("FLASK_ENV") == "production",
    MAX_CONTENT_LENGTH=1024 * 1024,
)

# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
  weights TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS searches(
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  query TEXT NOT NULL, params TEXT, sample INTEGER DEFAULT 0, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_searches_user ON searches(user_id, created_at);
CREATE TABLE IF NOT EXISTS playlists(
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name TEXT NOT NULL, objective TEXT DEFAULT '', created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_playlists_user ON playlists(user_id);
CREATE TABLE IF NOT EXISTS playlist_items(
  id INTEGER PRIMARY KEY, playlist_id INTEGER NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
  video_id TEXT NOT NULL, video TEXT NOT NULL, position INTEGER NOT NULL DEFAULT 0,
  note TEXT DEFAULT '', tags TEXT DEFAULT '', status TEXT DEFAULT 'Not Reviewed',
  keyword TEXT DEFAULT '', added_at TEXT NOT NULL, UNIQUE(playlist_id, video_id));
CREATE INDEX IF NOT EXISTS idx_items_pl ON playlist_items(playlist_id, position);
CREATE TABLE IF NOT EXISTS tracked(
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  keyword TEXT NOT NULL, region TEXT DEFAULT 'US', created_at TEXT NOT NULL,
  UNIQUE(user_id, keyword, region));
CREATE TABLE IF NOT EXISTS snapshots(
  id INTEGER PRIMARY KEY, tracked_id INTEGER NOT NULL REFERENCES tracked(id) ON DELETE CASCADE,
  taken_at TEXT NOT NULL, sample INTEGER DEFAULT 0, video_count INTEGER,
  top_video_ids TEXT, metrics TEXT);
CREATE INDEX IF NOT EXISTS idx_snap ON snapshots(tracked_id, taken_at);
CREATE TABLE IF NOT EXISTS briefs(
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  keyword TEXT NOT NULL, content TEXT NOT NULL, method TEXT, scheduled_for TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_briefs_user ON briefs(user_id);
CREATE TABLE IF NOT EXISTS cache(
  key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS usage(
  user_id INTEGER NOT NULL, day TEXT NOT NULL, kind TEXT NOT NULL, n INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(user_id, day, kind));
"""


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Security helpers: auth, CSRF, rate limiting, error handling
# --------------------------------------------------------------------------- #
_hits = defaultdict(list)


@app.before_request
def guard():
    if request.path.startswith("/api/"):
        ip = request.remote_addr or "?"
        t = time.time()
        _hits[ip] = [x for x in _hits[ip] if t - x < 60]
        if len(_hits[ip]) >= RATE_LIMIT:
            return jsonify(error="Rate limit exceeded. Try again in a minute."), 429
        _hits[ip].append(t)
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            token = request.headers.get("X-CSRF-Token", "")
            if not session.get("csrf") or not secrets.compare_digest(token, session["csrf"]):
                return jsonify(error="Invalid or missing CSRF token. Reload the page."), 403


@app.after_request
def headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' https://i.ytimg.com https://yt3.ggpht.com data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")
    return resp


@app.errorhandler(Exception)
def on_error(e):
    code = getattr(e, "code", 500)
    if not isinstance(code, int):
        code = 500
    if code == 500:
        app.logger.exception("unhandled error")
        return jsonify(error="Internal error. Check server logs."), 500
    return jsonify(error=getattr(e, "description", "Error")), code


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not session.get("uid"):
            return jsonify(error="Authentication required."), 401
        return fn(*a, **kw)
    return wrapper


def uid():
    return session["uid"]


def own_playlist(pid):
    """Ownership check: returns the playlist row only if it belongs to the user."""
    row = db().execute("SELECT * FROM playlists WHERE id=? AND user_id=?", (pid, uid())).fetchone()
    if not row:
        from werkzeug.exceptions import NotFound
        raise NotFound("Playlist not found.")  # 404 (not 403) to avoid leaking existence
    return row


def own_tracked(tid):
    row = db().execute("SELECT * FROM tracked WHERE id=? AND user_id=?", (tid, uid())).fetchone()
    if not row:
        from werkzeug.exceptions import NotFound
        raise NotFound("Tracked keyword not found.")
    return row


def own_brief(bid):
    row = db().execute("SELECT * FROM briefs WHERE id=? AND user_id=?", (bid, uid())).fetchone()
    if not row:
        from werkzeug.exceptions import NotFound
        raise NotFound("Brief not found.")
    return row


def bump_usage(kind, limit_key=None):
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d = db()
    d.execute("INSERT OR IGNORE INTO usage(user_id,day,kind,n) VALUES(?,?,?,0)", (uid(), day, kind))
    n = d.execute("SELECT n FROM usage WHERE user_id=? AND day=? AND kind=?", (uid(), day, kind)).fetchone()["n"]
    if limit_key and n >= PLAN_LIMITS["free"][limit_key]:
        return False
    d.execute("UPDATE usage SET n=n+1 WHERE user_id=? AND day=? AND kind=?", (uid(), day, kind))
    d.commit()
    return True


# --------------------------------------------------------------------------- #
# Cache + provider adapters
# --------------------------------------------------------------------------- #
def cache_get(key):
    row = db().execute("SELECT value, created_at FROM cache WHERE key=?", (key,)).fetchone()
    if row and time.time() - row["created_at"] < CACHE_TTL:
        return json.loads(row["value"])
    return None


def cache_set(key, value):
    db().execute("INSERT OR REPLACE INTO cache(key,value,created_at) VALUES(?,?,?)",
                 (key, json.dumps(value), time.time()))
    db().commit()


class ProviderError(Exception):
    pass


def http_json(method, url, retries=2, **kw):
    """HTTP with timeout and retry on transient failures."""
    last = None
    for attempt in range(retries + 1):
        try:
            r = requests.request(method, url, timeout=20, **kw)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(0.6 * (attempt + 1))
                continue
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(0.6 * (attempt + 1))
    raise ProviderError(f"Network error: {last}")


class YouTubeAdapter:
    """YouTube Data API v3 (official, quota-limited). search.list costs 100 units."""
    BASE = "https://www.googleapis.com/youtube/v3/"

    @property
    def configured(self):
        return bool(YT_KEY)

    def _get(self, endpoint, params):
        key = "yt:" + endpoint + ":" + json.dumps(params, sort_keys=True)
        hit = cache_get(key)
        if hit is not None:
            return hit
        r = http_json("GET", self.BASE + endpoint, params={**params, "key": YT_KEY})
        if r.status_code != 200:
            try:
                reason = r.json()["error"]["errors"][0]["reason"]
            except Exception:
                reason = str(r.status_code)
            if reason in ("quotaExceeded", "rateLimitExceeded", "dailyLimitExceeded"):
                raise ProviderError("YouTube API quota exhausted. Quota resets daily (Pacific time).")
            if reason in ("keyInvalid", "badRequest") or r.status_code == 400:
                raise ProviderError("YouTube API rejected the request (check YOUTUBE_API_KEY).")
            raise ProviderError(f"YouTube API error: {reason}")
        data = r.json()
        cache_set(key, data)
        return data

    @staticmethod
    def parse_duration(iso):
        m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
        if not m:
            return None
        h, mi, s = (int(x or 0) for x in m.groups())
        return h * 3600 + mi * 60 + s

    def search_videos(self, q, region="US", lang="en", max_results=15, order="relevance",
                      published_after=None, duration="any"):
        params = {"part": "snippet", "type": "video", "q": q, "maxResults": min(max_results, 50),
                  "regionCode": region, "relevanceLanguage": lang, "order": order,
                  "safeSearch": "moderate"}
        if published_after:
            params["publishedAfter"] = published_after
        if duration in ("short", "medium", "long"):
            params["videoDuration"] = duration
        found = self._get("search", params)
        ids = [i["id"]["videoId"] for i in found.get("items", []) if i.get("id", {}).get("videoId")]
        return self.videos_by_ids(ids)

    def videos_by_ids(self, ids):
        if not ids:
            return []
        data = self._get("videos", {"part": "snippet,statistics,contentDetails", "id": ",".join(ids)})
        chan_ids = list({v["snippet"]["channelId"] for v in data.get("items", [])})
        subs = {}
        if chan_ids:
            cd = self._get("channels", {"part": "statistics", "id": ",".join(chan_ids[:50])})
            for c in cd.get("items", []):
                st = c.get("statistics", {})
                subs[c["id"]] = None if st.get("hiddenSubscriberCount") else int(st.get("subscriberCount", 0))
        out = []
        for v in data.get("items", []):
            sn, st = v["snippet"], v.get("statistics", {})
            th = sn.get("thumbnails", {})
            thumb = (th.get("medium") or th.get("high") or th.get("default") or {}).get("url")
            out.append({
                "id": v["id"], "title": sn.get("title", ""), "channel": sn.get("channelTitle", ""),
                "channel_id": sn.get("channelId"), "published": sn.get("publishedAt"),
                "views": int(st["viewCount"]) if "viewCount" in st else None,
                "likes": int(st["likeCount"]) if "likeCount" in st else None,
                "comments": int(st["commentCount"]) if "commentCount" in st else None,
                "duration_s": self.parse_duration(v.get("contentDetails", {}).get("duration")),
                "subs": subs.get(sn.get("channelId")), "thumb": thumb,
                "url": f"https://www.youtube.com/watch?v={v['id']}",
                "channel_url": f"https://www.youtube.com/channel/{sn.get('channelId')}",
                "sample": False})
        return out

    def resolve_channel(self, ident):
        ident = ident.strip()
        m = re.search(r"youtube\.com/(?:channel/(UC[\w-]{20,})|@([\w.\-]+))", ident)
        params = {"part": "snippet,statistics"}
        if m and m.group(1):
            params["id"] = m.group(1)
        elif m and m.group(2):
            params["forHandle"] = "@" + m.group(2)
        elif ident.startswith("UC") and len(ident) > 20:
            params["id"] = ident
        elif ident.startswith("@"):
            params["forHandle"] = ident
        else:
            raise ProviderError("Enter a channel URL, @handle, or channel ID (UC...).")
        data = self._get("channels", params)
        if not data.get("items"):
            raise ProviderError("Channel not found.")
        c = data["items"][0]
        st = c.get("statistics", {})
        return {"id": c["id"], "title": c["snippet"]["title"],
                "subs": None if st.get("hiddenSubscriberCount") else int(st.get("subscriberCount", 0)),
                "video_count": int(st.get("videoCount", 0)),
                "url": f"https://www.youtube.com/channel/{c['id']}",
                "thumb": (c["snippet"].get("thumbnails", {}).get("default") or {}).get("url")}

    def channel_videos(self, channel_id, n=30):
        found = self._get("search", {"part": "snippet", "channelId": channel_id, "type": "video",
                                     "order": "date", "maxResults": min(n, 50)})
        ids = [i["id"]["videoId"] for i in found.get("items", [])]
        return self.videos_by_ids(ids)


yt = YouTubeAdapter()


def llm_json(system, prompt, max_tokens=2500):
    """Structured-JSON LLM call. Returns dict or None (never raises into the UI)."""
    if not LLM_KEY:
        return None
    key = "llm:" + str(hash((system, prompt, LLM_MODEL)))
    hit = cache_get(key)
    if hit is not None:
        return hit
    try:
        r = http_json("POST", "https://api.anthropic.com/v1/messages",
                      headers={"x-api-key": LLM_KEY, "anthropic-version": "2023-06-01",
                               "content-type": "application/json"},
                      json={"model": LLM_MODEL, "max_tokens": max_tokens, "system": system,
                            "messages": [{"role": "user", "content": prompt}]})
        if r.status_code != 200:
            app.logger.warning("LLM error %s", r.status_code)
            return None
        text = "".join(b.get("text", "") for b in r.json().get("content", []))
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0)) if m else None
        if data is not None:
            cache_set(key, data)
        return data
    except (ProviderError, ValueError, KeyError):
        return None


# --------------------------------------------------------------------------- #
# Sample-data mode (explicitly labeled; used only when no YouTube key)
# --------------------------------------------------------------------------- #
def sample_videos(q, n=10):
    """Deterministic SAMPLE records. IDs are not real YouTube IDs; no links."""
    rnd = int(sum(ord(c) for c in q)) or 1
    fmt = ["{q}: complete beginner guide", "I tested {q} for 30 days", "{q} - mistakes to avoid",
           "Top 5 {q} compared", "{q} explained simply", "Why {q} is changing in 2026",
           "{q} on a budget", "Advanced {q} strategies", "{q} vs the alternatives", "{q} Q&A"]
    out = []
    for i in range(min(n, len(fmt))):
        views = (rnd * (i + 3) * 7919) % 900000 + 5000
        subs = (rnd * (i + 5) * 104729) % 900000 + 2000
        age = (rnd * (i + 1) * 31) % 900 + 7
        pub = (datetime.now(timezone.utc) - timedelta(days=age)).isoformat(timespec="seconds")
        out.append({"id": f"SAMPLE_{i+1:03d}", "title": fmt[i].format(q=q).capitalize(),
                    "channel": f"Sample Channel {i+1}", "channel_id": None, "published": pub,
                    "views": views, "likes": views // 25, "comments": views // 400,
                    "duration_s": 240 + 97 * i, "subs": subs, "thumb": None, "url": None,
                    "channel_url": None, "sample": True})
    return out


# --------------------------------------------------------------------------- #
# Keyword expansion, intent, clustering
# --------------------------------------------------------------------------- #
INTENT_RULES = [
    ("Comparison", r"\bvs\b|\bversus\b|compare|alternative"),
    ("Commercial", r"\bbest\b|\btop \d|review|price|cheap|buy|deal|worth it"),
    ("How-to", r"\bhow to\b|tutorial|guide|step by step|for beginners|setup"),
    ("Problem-solving", r"mistake|fix|problem|not working|avoid|why (?:is|does|my)"),
    ("Informational", r"\bwhat\b|\bwhy\b|explained|meaning|history|\bwho\b"),
]


def classify_intent(kw):
    k = kw.lower()
    for label, pat in INTENT_RULES:
        if re.search(pat, k):
            return label
    return "Informational"


TEMPLATE_CLUSTERS = [
    ("How-to & tutorials", "How-to", "Long-form tutorial",
     ["how to {k}", "{k} tutorial", "{k} step by step", "{k} for beginners", "{k} setup guide"]),
    ("Questions", "Informational", "Explainer / Short",
     ["what is {k}", "why {k} matters", "is {k} worth it", "{k} explained", "how does {k} work"]),
    ("Comparisons", "Comparison", "Comparison / review",
     ["{k} vs alternatives", "best {k}", "{k} compared", "{k} pros and cons"]),
    ("Problems & mistakes", "Problem-solving", "Long-form how-to",
     ["{k} mistakes", "{k} problems and fixes", "common {k} mistakes to avoid"]),
    ("Level variations", "Informational", "Series",
     ["{k} for beginners", "advanced {k}", "{k} tips and tricks", "{k} on a budget"]),
    ("Commercial", "Commercial", "Review / listicle",
     ["{k} review", "{k} price", "{k} 2026", "{k} deals"]),
]


def expand_keywords(seed):
    """Return (clusters, method). LLM clusters if configured, else labeled templates."""
    system = ("You are a YouTube SEO analyst. Respond with ONLY valid JSON. Never include "
              "search volumes or statistics; only ideas.")
    prompt = (f'Seed keyword: "{seed}". Return JSON: {{"clusters":[{{"name":str,"intent":str,'
              '"format":str,"angles":[str,str],"why":str,"keywords":[str x5-8]}}]}} with 5-6 '
              "distinct clusters covering questions, how-to, comparisons, problems, beginner/"
              "advanced, commercial and audience-specific variations.")
    res = llm_json(system, prompt)
    if res and isinstance(res.get("clusters"), list) and res["clusters"]:
        clusters = []
        for c in res["clusters"][:8]:
            kws = [str(x)[:120] for x in c.get("keywords", []) if isinstance(x, str)][:10]
            if kws:
                clusters.append({"name": str(c.get("name", "Cluster"))[:80],
                                 "intent": str(c.get("intent", ""))[:40],
                                 "format": str(c.get("format", ""))[:60],
                                 "angles": [str(a)[:160] for a in c.get("angles", [])][:3],
                                 "why": str(c.get("why", ""))[:300], "keywords": kws,
                                 "source": "llm"})
        if clusters:
            return clusters, f"llm:{LLM_MODEL}"
    clusters = []
    for name, intent, fmt, pats in TEMPLATE_CLUSTERS:
        clusters.append({"name": name, "intent": intent, "format": fmt,
                         "angles": [f"Cover '{seed}' from a {name.lower()} perspective"],
                         "why": "Template-generated variation; not derived from search data.",
                         "keywords": [p.format(k=seed) for p in pats], "source": "template"})
    return clusters, "template"


# --------------------------------------------------------------------------- #
# Metrics: derived video stats, difficulty, opportunity (transparent heuristic)
# --------------------------------------------------------------------------- #
def parse_dt(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def enrich_videos(videos, query):
    now = datetime.now(timezone.utc)
    qtok = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    vpds = []
    for v in videos:
        dt = parse_dt(v["published"]) if v.get("published") else None
        age = max((now - dt).days, 1) if dt else None
        v["age_days"] = age
        v["views_per_day"] = round(v["views"] / age, 1) if (age and v.get("views") is not None) else None
        v["views_to_subs"] = round(v["views"] / v["subs"], 2) if (v.get("views") is not None and v.get("subs")) else None
        v["like_ratio"] = round(v["likes"] / v["views"], 4) if (v.get("likes") is not None and v.get("views")) else None
        v["comment_ratio"] = round(v["comments"] / v["views"], 5) if (v.get("comments") is not None and v.get("views")) else None
        title = v["title"].lower()
        v["relevance"] = round(100 * sum(t in title for t in qtok) / len(qtok)) if qtok else None
        if v["views_per_day"] is not None:
            vpds.append(v["views_per_day"])
    med = statistics.median(vpds) if vpds else None
    for v in videos:
        v["outlier_x"] = round(v["views_per_day"] / med, 1) if (med and v["views_per_day"] is not None) else None
    return videos


def metrics_for(videos, weights):
    """Compute each signal with provenance, then the weighted opportunity score."""
    now = now_iso()
    sample = bool(videos) and all(v.get("sample") for v in videos)
    kind = "sample" if sample else None
    src = "Sample data (not real)" if sample else "YouTube Data API (retrieved sample of top results)"
    sig = {}
    vpds = [v["views_per_day"] for v in videos if v.get("views_per_day") is not None]
    n = len(videos)

    # Demand evidence: median views/day of retrieved results (a PROXY, not search volume)
    if len(vpds) >= 5:
        med = statistics.median(vpds)
        label = "Low" if med < 50 else "Moderate" if med < 500 else "High"
        sig["demand"] = {"value": round(min(100, math.log10(1 + med) / 4 * 100)), "label": label,
                         "detail": f"Median {med:,.0f} views/day across {len(vpds)} results. "
                                   "Proxy only; niche-dependent thresholds; not search volume.",
                         "kind": kind or "inferred", "source": src, "retrieved": now}
    else:
        sig["demand"] = None

    # Competition weakness: share of results from channels < 100k subs
    with_subs = [v for v in videos if v.get("subs") is not None]
    if len(with_subs) >= 5:
        small = sum(1 for v in with_subs if v["subs"] < 100_000)
        ratios = [v["views_to_subs"] for v in with_subs if v.get("views_to_subs")]
        w = round(100 * small / len(with_subs))
        sig["competition"] = {"value": w, "label": "Low" if w >= 66 else "Medium" if w >= 33 else "High",
                              "detail": f"{small}/{len(with_subs)} top results come from channels under 100k "
                                        f"subscribers; median views/subs {statistics.median(ratios):.2f}."
                              if ratios else f"{small}/{len(with_subs)} results from channels under 100k subs.",
                              "kind": kind or "inferred", "source": src, "retrieved": now}
    else:
        sig["competition"] = None

    rel = [v["relevance"] for v in videos if v.get("relevance") is not None]
    sig["relevance"] = ({"value": round(statistics.mean(rel)), "label": "Title match",
                         "detail": "Average share of query words present in result titles (text match).",
                         "kind": kind or "measured", "source": src, "retrieved": now} if rel else None)

    ages = [v["age_days"] for v in videos if v.get("age_days")]
    if len(ages) >= 5:
        stale = round(100 * sum(a > 730 for a in ages) / len(ages))
        sig["gap"] = {"value": stale, "label": "Stale results" if stale >= 50 else "Fresh results",
                      "detail": f"{stale}% of results are over 2 years old (possible freshness gap).",
                      "kind": kind or "inferred", "source": src, "retrieved": now}
        recent = round(100 * sum(a <= 90 for a in ages) / len(ages))
        sig["trend"] = {"value": recent, "label": f"{recent}% < 90 days",
                        "detail": "Share of results published in the last 90 days. A single-moment freshness "
                                  "signal, NOT a trend; trends need tracked snapshots.",
                        "kind": kind or "inferred", "source": src, "retrieved": now}
    else:
        sig["gap"] = sig["trend"] = None

    total_w = sum(weights.values()) or 1
    avail = {k: weights.get(k, 0) for k, v in sig.items() if v}
    aw = sum(avail.values())
    score = round(sum(weights[k] * sig[k]["value"] for k in avail) / aw) if aw else None
    cov = aw / total_w
    conf = "Sample" if sample else ("High" if cov >= .85 else "Medium" if cov >= .55 else "Low")
    difficulty = (100 - sig["competition"]["value"]) if sig["competition"] else None
    return {"signals": sig, "opportunity": score, "confidence": conf, "difficulty": difficulty,
            "missing": [k for k, v in sig.items() if not v], "weights": weights,
            "video_count": n, "sample": sample, "computed_at": now,
            "disclaimer": ("Opportunity and difficulty are internal prioritization heuristics computed from "
                           "the retrieved sample. They are not YouTube data and guarantee no views, rankings, "
                           "or revenue. Exact search volume is not available from the YouTube Data API.")}


def get_weights():
    row = db().execute("SELECT weights FROM users WHERE id=?", (uid(),)).fetchone()
    try:
        w = json.loads(row["weights"]) if row and row["weights"] else None
    except ValueError:
        w = None
    return {**DEFAULT_WEIGHTS, **(w or {})}


RANK_MODES = {
    "relevant": ("Most relevant", "Title match to your query, then views/day."),
    "views": ("Most viewed", "Total views. High views do not imply quality or search volume."),
    "recent": ("Recently published", "Newest first."),
    "engagement": ("Strong engagement", "Likes + comments per view (when exposed)."),
    "relative": ("High views vs channel size", "Views divided by subscribers; spots outperformers."),
    "beginner": ("Best for beginner research", "Title mentions beginner/guide/basics, ranked by relevance."),
    "ai": ("AI-recommended", "Blend of relevance, views/day outlier score and channel diversity."),
}


def rank_videos(videos, mode):
    def eng(v):
        return ((v.get("likes") or 0) + 3 * (v.get("comments") or 0)) / v["views"] if v.get("views") else -1
    keyf = {
        "relevant": lambda v: ((v.get("relevance") or 0), v.get("views_per_day") or 0),
        "views": lambda v: v.get("views") or 0,
        "recent": lambda v: v.get("published") or "",
        "engagement": eng,
        "relative": lambda v: v.get("views_to_subs") or -1,
        "beginner": lambda v: (bool(re.search(r"beginner|guide|basics|explained|intro", v["title"], re.I)),
                               v.get("relevance") or 0),
        "ai": lambda v: (v.get("relevance") or 0) * 0.5 + min(v.get("outlier_x") or 0, 5) * 10,
    }.get(mode)
    out = sorted(videos, key=keyf, reverse=True) if keyf else list(videos)
    if mode == "ai":  # diversity: at most 2 videos per channel in the top slots
        seen, top, rest = Counter(), [], []
        for v in out:
            (top if seen[v["channel"]] < 2 else rest).append(v)
            seen[v["channel"]] += 1
        out = top + rest
    return out


def why_recommended(v, mode):
    bits = []
    if v.get("relevance") is not None:
        bits.append(f"{v['relevance']}% of your query words appear in the title")
    if v.get("outlier_x") and v["outlier_x"] >= 1.5:
        bits.append(f"views/day is {v['outlier_x']}x the median of this result set")
    if v.get("views_to_subs") and v["views_to_subs"] >= 1:
        bits.append(f"views exceed the channel's subscriber count ({v['views_to_subs']}x)")
    if v.get("age_days") is not None and v["age_days"] <= 90:
        bits.append("published in the last 90 days")
    if v.get("sample"):
        bits.append("SAMPLE record")
    return "; ".join(bits) or "Retrieved for this query; limited metadata available."


# --------------------------------------------------------------------------- #
# Routes: pages + auth
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return Response(INDEX_HTML.replace("__APP_NAME__", APP_NAME).replace("__TAGLINE__", TAGLINE),
                    mimetype="text/html")


@app.get("/api/me")
def me():
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(32)
    user = None
    if session.get("uid"):
        row = db().execute("SELECT email FROM users WHERE id=?", (uid(),)).fetchone()
        user = row["email"] if row else None
        if not user:
            session.clear()
            session["csrf"] = secrets.token_urlsafe(32)
    return jsonify(user=user, csrf=session["csrf"], app=APP_NAME, tagline=TAGLINE)


@app.post("/api/register")
def register():
    d = request.get_json(silent=True) or {}
    email = (d.get("email") or "").strip().lower()
    pw = d.get("password") or ""
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email) or len(pw) < 8:
        return jsonify(error="Valid email and a password of at least 8 characters are required."), 400
    try:
        cur = db().execute("INSERT INTO users(email,password_hash,created_at) VALUES(?,?,?)",
                           (email, generate_password_hash(pw), now_iso()))
        db().commit()
    except sqlite3.IntegrityError:
        return jsonify(error="That email is already registered."), 409
    session.clear()
    session.update(uid=cur.lastrowid, csrf=secrets.token_urlsafe(32))
    return jsonify(ok=True, csrf=session["csrf"])


@app.post("/api/login")
def login():
    d = request.get_json(silent=True) or {}
    row = db().execute("SELECT * FROM users WHERE email=?", ((d.get("email") or "").strip().lower(),)).fetchone()
    if not row or not check_password_hash(row["password_hash"], d.get("password") or ""):
        return jsonify(error="Invalid email or password."), 401
    session.clear()
    session.update(uid=row["id"], csrf=secrets.token_urlsafe(32))
    return jsonify(ok=True, csrf=session["csrf"])


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify(ok=True)


@app.get("/api/status")
@login_required
def status():
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = db().execute("SELECT kind,n FROM usage WHERE user_id=? AND day=?", (uid(), day)).fetchall()
    return jsonify(
        integrations={
            "youtube": {"configured": yt.configured, "name": "YouTube Data API v3",
                        "setup": "Set YOUTUBE_API_KEY (Google Cloud Console > enable YouTube Data API v3)."},
            "llm": {"configured": bool(LLM_KEY), "name": f"Anthropic ({LLM_MODEL})",
                    "setup": "Set ANTHROPIC_API_KEY for AI keyword expansion and briefs."},
            "trends": {"configured": False, "name": "Trends / keyword-volume provider",
                       "setup": "Not implemented: add an authorized provider adapter (e.g. Google Trends "
                                "or a licensed keyword API). Search volume stays 'Unknown' until then."}},
        usage={r["kind"]: r["n"] for r in rows}, limits=PLAN_LIMITS["free"],
        weights=get_weights(), defaults=DEFAULT_WEIGHTS)


@app.put("/api/weights")
@login_required
def set_weights():
    d = request.get_json(silent=True) or {}
    w = {}
    for k in DEFAULT_WEIGHTS:
        try:
            w[k] = max(0, min(100, int(d.get(k, DEFAULT_WEIGHTS[k]))))
        except (TypeError, ValueError):
            return jsonify(error=f"Weight '{k}' must be a number 0-100."), 400
    if sum(w.values()) == 0:
        return jsonify(error="At least one weight must be above zero."), 400
    db().execute("UPDATE users SET weights=? WHERE id=?", (json.dumps(w), uid()))
    db().commit()
    return jsonify(ok=True, weights=w)


# --------------------------------------------------------------------------- #
# Routes: research
# --------------------------------------------------------------------------- #
def fetch_videos(q, p):
    """Returns (videos, sample_flag, notice)."""
    if not yt.configured:
        return sample_videos(q, p["max"]), True, ("SAMPLE DATA: YOUTUBE_API_KEY is not set, so these videos "
                                                  "are illustrative placeholders, not real YouTube results.")
    after = None
    if p["days"]:
        after = (datetime.now(timezone.utc) - timedelta(days=p["days"])).strftime("%Y-%m-%dT%H:%M:%SZ")
    dur = {"shorts": "short", "long": "long"}.get(p["format"], "any")
    vids = yt.search_videos(q, p["region"], p["lang"], p["max"], p["order"], after, dur)
    if p["format"] == "shorts":
        vids = [v for v in vids if v["duration_s"] is not None and v["duration_s"] <= 60]
    elif p["format"] == "long":
        vids = [v for v in vids if v["duration_s"] is None or v["duration_s"] > 240]
    return vids, False, None


def parse_params(d):
    def num(k, default, lo, hi):
        try:
            return max(lo, min(hi, int(d.get(k, default))))
        except (TypeError, ValueError):
            return default
    return {"region": re.sub(r"[^A-Z]", "", (d.get("region") or "US").upper())[:2] or "US",
            "lang": re.sub(r"[^a-z\-]", "", (d.get("lang") or "en").lower())[:5] or "en",
            "format": d.get("format") if d.get("format") in ("all", "shorts", "long") else "all",
            "order": d.get("order") if d.get("order") in ("relevance", "viewCount", "date", "rating") else "relevance",
            "max": num("max", 15, 5, 50), "days": num("days", 0, 0, 3650),
            "min_views": num("min_views", 0, 0, 10**10),
            "exclude": [x.strip().lower() for x in (d.get("exclude") or "").split(",") if x.strip()][:20],
            "exclude_channels": [x.strip().lower() for x in (d.get("exclude_channels") or "").split(",") if x.strip()][:20]}


@app.post("/api/research")
@login_required
def research():
    d = request.get_json(silent=True) or {}
    seeds = [s.strip() for s in re.split(r"[\n;]", d.get("query") or "") if s.strip()][:5]
    if not seeds:
        return jsonify(error="Enter a keyword or topic."), 400
    if any(len(s) > 150 for s in seeds):
        return jsonify(error="Keywords must be under 150 characters."), 400
    if not bump_usage("searches", "searches_per_day"):
        return jsonify(error="Daily search limit reached for your plan."), 429
    p = parse_params(d)
    weights = get_weights()
    results = []
    for seed in seeds:
        notice, warnings = None, []
        try:
            videos, sample, notice = fetch_videos(seed, p)
        except ProviderError as e:
            return jsonify(error=str(e)), 502
        videos = [v for v in videos
                  if (v.get("views") or 0) >= p["min_views"]
                  and not any(x in v["title"].lower() for x in p["exclude"])
                  and v["channel"].lower() not in p["exclude_channels"]]
        enrich_videos(videos, seed)
        clusters, method = expand_keywords(seed)
        seed_metrics = metrics_for(videos, weights)
        if not videos:
            warnings.append("No videos matched your filters; metrics unavailable.")
        for c in clusters:
            c["items"] = [{"keyword": k, "intent": classify_intent(k), "source": c["source"],
                           "demand": "Unknown", "demand_note": "No search-volume source configured.",
                           "trend": "Unknown"} for k in c["keywords"]]
        mode = d.get("rank") if d.get("rank") in RANK_MODES else "relevant"
        ranked = rank_videos(videos, mode)
        for v in ranked:
            v["why"] = why_recommended(v, mode)
        db().execute("INSERT INTO searches(user_id,query,params,sample,created_at) VALUES(?,?,?,?,?)",
                     (uid(), seed, json.dumps(p), int(sample), now_iso()))
        db().commit()
        results.append({"seed": seed, "sample": sample, "notice": notice, "warnings": warnings,
                        "expansion_method": method, "clusters": clusters, "videos": ranked,
                        "metrics": seed_metrics, "rank_modes": {k: {"label": a, "criteria": b}
                                                                for k, (a, b) in RANK_MODES.items()},
                        "rank": mode, "params": p})
    return jsonify(results=results)


@app.post("/api/keyword/analyze")
@login_required
def analyze_keyword():
    """Fetch real signals for one expanded keyword (costs API quota, so on demand)."""
    d = request.get_json(silent=True) or {}
    kw = (d.get("keyword") or "").strip()
    if not kw or len(kw) > 150:
        return jsonify(error="Keyword required (max 150 chars)."), 400
    if not bump_usage("searches", "searches_per_day"):
        return jsonify(error="Daily search limit reached for your plan."), 429
    p = parse_params(d)
    try:
        videos, sample, notice = fetch_videos(kw, p)
    except ProviderError as e:
        return jsonify(error=str(e)), 502
    enrich_videos(videos, kw)
    m = metrics_for(videos, get_weights())
    return jsonify(keyword=kw, metrics=m, sample=sample, notice=notice, video_count=len(videos),
                   videos=rank_videos(videos, "relevant")[:5])


@app.get("/api/history")
@login_required
def history():
    rows = db().execute("SELECT query, sample, created_at FROM searches WHERE user_id=? "
                        "ORDER BY id DESC LIMIT 20", (uid(),)).fetchall()
    return jsonify(history=[dict(r) for r in rows])


@app.post("/api/export/keywords")
@login_required
def export_keywords():
    d = request.get_json(silent=True) or {}
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["keyword", "intent", "source", "cluster", "demand", "trend"])
    for c in d.get("clusters", []):
        for it in c.get("items", []):
            w.writerow([_csv_safe(it.get("keyword")), it.get("intent"), it.get("source"),
                        _csv_safe(c.get("name")), it.get("demand"), it.get("trend")])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=keywords.csv"})


def _csv_safe(v):
    """Neutralize spreadsheet formula injection from untrusted text."""
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


# --------------------------------------------------------------------------- #
# Routes: playlists (application research playlists, not native YouTube ones)
# --------------------------------------------------------------------------- #
def item_json(r):
    return {"id": r["id"], "video_id": r["video_id"], "video": json.loads(r["video"]),
            "position": r["position"], "note": r["note"], "tags": r["tags"], "status": r["status"],
            "keyword": r["keyword"]}


@app.get("/api/playlists")
@login_required
def list_playlists():
    rows = db().execute("SELECT p.*, (SELECT COUNT(*) FROM playlist_items i WHERE i.playlist_id=p.id) AS n "
                        "FROM playlists p WHERE user_id=? ORDER BY id DESC", (uid(),)).fetchall()
    return jsonify(playlists=[dict(r) for r in rows])


@app.post("/api/playlists")
@login_required
def create_playlist():
    d = request.get_json(silent=True) or {}
    name = (d.get("name") or "").strip()[:100]
    if not name:
        return jsonify(error="Playlist name is required."), 400
    n = db().execute("SELECT COUNT(*) c FROM playlists WHERE user_id=?", (uid(),)).fetchone()["c"]
    if n >= PLAN_LIMITS["free"]["playlists"]:
        return jsonify(error="Playlist limit reached for your plan."), 429
    cur = db().execute("INSERT INTO playlists(user_id,name,objective,created_at) VALUES(?,?,?,?)",
                       (uid(), name, (d.get("objective") or "")[:500], now_iso()))
    db().commit()
    return jsonify(id=cur.lastrowid, name=name)


@app.get("/api/playlists/<int:pid>")
@login_required
def get_playlist(pid):
    pl = own_playlist(pid)
    items = db().execute("SELECT * FROM playlist_items WHERE playlist_id=? ORDER BY position, id", (pid,)).fetchall()
    return jsonify(playlist=dict(pl), items=[item_json(r) for r in items])


@app.patch("/api/playlists/<int:pid>")
@login_required
def update_playlist(pid):
    pl = own_playlist(pid)
    d = request.get_json(silent=True) or {}
    name = (d.get("name", pl["name"]) or "").strip()[:100] or pl["name"]
    db().execute("UPDATE playlists SET name=?, objective=? WHERE id=?",
                 (name, (d.get("objective", pl["objective"]) or "")[:500], pid))
    db().commit()
    return jsonify(ok=True)


@app.delete("/api/playlists/<int:pid>")
@login_required
def delete_playlist(pid):
    own_playlist(pid)
    db().execute("DELETE FROM playlists WHERE id=?", (pid,))
    db().commit()
    return jsonify(ok=True)


@app.post("/api/playlists/<int:pid>/duplicate")
@login_required
def duplicate_playlist(pid):
    pl = own_playlist(pid)
    cur = db().execute("INSERT INTO playlists(user_id,name,objective,created_at) VALUES(?,?,?,?)",
                       (uid(), (pl["name"] + " (copy)")[:100], pl["objective"], now_iso()))
    new = cur.lastrowid
    db().execute("INSERT INTO playlist_items(playlist_id,video_id,video,position,note,tags,status,keyword,added_at) "
                 "SELECT ?,video_id,video,position,note,tags,status,keyword,? FROM playlist_items WHERE playlist_id=?",
                 (new, now_iso(), pid))
    db().commit()
    return jsonify(id=new)


@app.post("/api/playlists/<int:pid>/items")
@login_required
def add_items(pid):
    own_playlist(pid)
    d = request.get_json(silent=True) or {}
    vids = d.get("videos") or []
    if not isinstance(vids, list) or not vids or len(vids) > 50:
        return jsonify(error="Provide 1-50 videos."), 400
    pos = db().execute("SELECT COALESCE(MAX(position),-1)+1 p FROM playlist_items WHERE playlist_id=?", (pid,)).fetchone()["p"]
    added = dup = 0
    for v in vids:
        vid = str(v.get("id", ""))[:40]
        if not vid:
            continue
        keep = {k: v.get(k) for k in ("id", "title", "channel", "channel_id", "published", "views", "likes",
                                       "comments", "duration_s", "subs", "thumb", "url", "channel_url", "sample")}
        try:
            db().execute("INSERT INTO playlist_items(playlist_id,video_id,video,position,keyword,added_at) "
                         "VALUES(?,?,?,?,?,?)",
                         (pid, vid, json.dumps(keep), pos, str(d.get("keyword", ""))[:150], now_iso()))
            pos += 1
            added += 1
        except sqlite3.IntegrityError:
            dup += 1  # duplicate video detection
    db().commit()
    return jsonify(added=added, duplicates=dup)


def own_item(pid, iid):
    own_playlist(pid)
    row = db().execute("SELECT * FROM playlist_items WHERE id=? AND playlist_id=?", (iid, pid)).fetchone()
    if not row:
        from werkzeug.exceptions import NotFound
        raise NotFound("Item not found.")
    return row


@app.patch("/api/playlists/<int:pid>/items/<int:iid>")
@login_required
def update_item(pid, iid):
    row = own_item(pid, iid)
    d = request.get_json(silent=True) or {}
    status = d.get("status", row["status"])
    if status not in ("Not Reviewed", "Reviewed", "Used for Research"):
        return jsonify(error="Invalid status."), 400
    db().execute("UPDATE playlist_items SET note=?, tags=?, status=? WHERE id=?",
                 (str(d.get("note", row["note"]))[:2000], str(d.get("tags", row["tags"]))[:200], status, iid))
    db().commit()
    return jsonify(ok=True)


@app.delete("/api/playlists/<int:pid>/items/<int:iid>")
@login_required
def delete_item(pid, iid):
    own_item(pid, iid)
    db().execute("DELETE FROM playlist_items WHERE id=?", (iid,))
    db().commit()
    return jsonify(ok=True)


@app.post("/api/playlists/<int:pid>/reorder")
@login_required
def reorder(pid):
    own_playlist(pid)
    order = (request.get_json(silent=True) or {}).get("order") or []
    valid = {r["id"] for r in db().execute("SELECT id FROM playlist_items WHERE playlist_id=?", (pid,))}
    if not isinstance(order, list) or set(order) != valid:
        return jsonify(error="Order must list every item in the playlist exactly once."), 400
    for i, iid in enumerate(order):
        db().execute("UPDATE playlist_items SET position=? WHERE id=? AND playlist_id=?", (i, iid, pid))
    db().commit()
    return jsonify(ok=True)


@app.get("/api/playlists/<int:pid>/export.<fmt>")
@login_required
def export_playlist(pid, fmt):
    pl = own_playlist(pid)
    items = [item_json(r) for r in db().execute(
        "SELECT * FROM playlist_items WHERE playlist_id=? ORDER BY position, id", (pid,))]
    if fmt == "json":
        return Response(json.dumps({"playlist": dict(pl), "items": items}, indent=2), mimetype="application/json",
                        headers={"Content-Disposition": f"attachment; filename=playlist-{pid}.json"})
    if fmt != "csv":
        return jsonify(error="Format must be csv or json."), 400
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["position", "title", "channel", "url", "views", "published", "status", "tags", "keyword", "note"])
    for it in items:
        v = it["video"]
        w.writerow([it["position"] + 1, _csv_safe(v.get("title")), _csv_safe(v.get("channel")),
                    v.get("url") or "(sample - no link)", v.get("views"), v.get("published"), it["status"],
                    _csv_safe(it["tags"]), _csv_safe(it["keyword"]), _csv_safe(it["note"])])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=playlist-{pid}.csv"})


@app.post("/api/playlists/<int:pid>/summary")
@login_required
def playlist_summary(pid):
    """Evidence-grounded playlist summary. LLM if configured; else counted patterns (labeled)."""
    own_playlist(pid)
    items = [item_json(r) for r in db().execute("SELECT * FROM playlist_items WHERE playlist_id=?", (pid,))]
    if not items:
        return jsonify(error="Add videos first."), 400
    titles = [i["video"]["title"] for i in items]
    words = Counter(w for t in titles for w in re.findall(r"[a-z]{4,}", t.lower())
                    if w not in {"with", "that", "this", "from", "your", "what", "have", "will", "about"})
    patterns = [w for w, c in words.most_common(8) if c >= 2]
    res = llm_json("You analyze YouTube research playlists. Use ONLY the provided titles. Respond with JSON.",
                   'Titles: ' + json.dumps(titles[:40]) +
                   '\nReturn {"themes":[str],"repeated_angles":[str],"unexplored_angles":[str],"video_ideas":[str]}')
    if res:
        return jsonify(method=f"llm:{LLM_MODEL}", summary=res, based_on=len(titles))
    return jsonify(method="template (no LLM configured)", based_on=len(titles),
                   summary={"themes": patterns or ["Not enough repeated terms yet"],
                            "note": "Counted repeated title words only. Configure ANTHROPIC_API_KEY for "
                                    "AI summaries and idea generation."})


# --------------------------------------------------------------------------- #
# Routes: competitor intelligence
# --------------------------------------------------------------------------- #
def title_structure(t):
    t = t.lower()
    for name, pat in [("How to...", r"^how to"), ("Numbered list", r"^\d+|\btop \d+|\b\d+ (?:best|ways|tips)"),
                      ("Question", r"\?$|^(?:what|why|is|are|can|should)\b"), ("Versus", r"\bvs\b"),
                      ("First-person test", r"^i (?:tried|tested|made|spent)")]:
        if re.search(pat, t):
            return name
    return "Other"


def analyze_channel(ch, videos):
    enrich_videos(videos, "")
    dates = sorted(parse_dt(v["published"]) for v in videos if v.get("published"))
    span = (dates[-1] - dates[0]).days if len(dates) > 1 else None
    durs = [v["duration_s"] for v in videos if v.get("duration_s")]
    words = Counter(w for v in videos for w in re.findall(r"[a-z]{4,}", v["title"].lower()))
    return {"channel": ch, "sample_size": len(videos),
            "sample_note": f"Based on the {len(videos)} most recent videos retrieved, NOT the full channel history.",
            "uploads_per_week": round(7 * (len(dates) - 1) / span, 2) if span else None,
            "median_duration_s": statistics.median(durs) if durs else None,
            "shorts_share": round(100 * sum(d <= 60 for d in durs) / len(durs)) if durs else None,
            "top_viewed": sorted(videos, key=lambda v: v.get("views") or 0, reverse=True)[:5],
            "title_structures": Counter(title_structure(v["title"]) for v in videos).most_common(),
            "themes": [w for w, c in words.most_common(10) if c >= 2],
            "median_views": statistics.median([v["views"] for v in videos if v.get("views") is not None] or [0])}


@app.post("/api/competitors")
@login_required
def competitors():
    if not yt.configured:
        return jsonify(error="Competitor analysis needs YOUTUBE_API_KEY (no sample mode: channel data "
                             "cannot be simulated honestly).", setup_required=True), 400
    idents = [x.strip() for x in re.split(r"[\n,]", (request.get_json(silent=True) or {}).get("channels", "")) if x.strip()][:4]
    if not idents:
        return jsonify(error="Enter at least one channel URL, @handle or ID."), 400
    out = []
    try:
        for ident in idents:
            ch = yt.resolve_channel(ident)
            out.append(analyze_channel(ch, yt.channel_videos(ch["id"], 30)))
    except ProviderError as e:
        return jsonify(error=str(e)), 502
    shared = []
    if len(out) > 1:
        sets = [set(o["themes"]) for o in out]
        shared = sorted(set.intersection(*sets))
    gaps = {}
    if len(out) > 1:
        for o in out:
            others = set().union(*[set(x["themes"]) for x in out if x is not o])
            gaps[o["channel"]["title"]] = sorted(others - set(o["themes"]))[:8]
    return jsonify(channels=out, shared_themes=shared, theme_gaps=gaps,
                   caveat="Public data only. Impressions, CTR, retention, revenue and private search "
                          "terms are not available.")


# --------------------------------------------------------------------------- #
# Routes: tracking (dated snapshots; trends only with enough history)
# --------------------------------------------------------------------------- #
MIN_SNAPSHOTS_FOR_TREND = 3


@app.get("/api/tracked")
@login_required
def list_tracked():
    out = []
    for t in db().execute("SELECT * FROM tracked WHERE user_id=? ORDER BY id DESC", (uid(),)):
        snaps = db().execute("SELECT * FROM snapshots WHERE tracked_id=? ORDER BY taken_at", (t["id"],)).fetchall()
        series = [{"at": s["taken_at"], "sample": bool(s["sample"]),
                   "opportunity": (json.loads(s["metrics"]) or {}).get("opportunity"),
                   "video_count": s["video_count"]} for s in snaps]
        note, change = "No snapshots yet.", None
        real = [s for s in series if not s["sample"] and s["opportunity"] is not None]
        if len(real) >= MIN_SNAPSHOTS_FOR_TREND:
            change = real[-1]["opportunity"] - real[0]["opportunity"]
            note = f"{len(real)} snapshots. Opportunity score change since first: {change:+d}."
        elif snaps:
            note = (f"{len(snaps)} snapshot(s). Need {MIN_SNAPSHOTS_FOR_TREND}+ real snapshots on different "
                    "dates before showing a trend; differences between two searches can be sampling noise.")
        new_top = None
        if len(snaps) >= 2:
            a, b = json.loads(snaps[-2]["top_video_ids"] or "[]"), json.loads(snaps[-1]["top_video_ids"] or "[]")
            new_top = [x for x in b if x not in a]
        out.append({"id": t["id"], "keyword": t["keyword"], "region": t["region"], "series": series,
                    "note": note, "change": change, "new_top_videos": new_top})
    return jsonify(tracked=out, min_snapshots=MIN_SNAPSHOTS_FOR_TREND)


@app.post("/api/tracked")
@login_required
def add_tracked():
    d = request.get_json(silent=True) or {}
    kw = (d.get("keyword") or "").strip()[:150]
    if not kw:
        return jsonify(error="Keyword required."), 400
    n = db().execute("SELECT COUNT(*) c FROM tracked WHERE user_id=?", (uid(),)).fetchone()["c"]
    if n >= PLAN_LIMITS["free"]["tracked"]:
        return jsonify(error="Tracked-keyword limit reached for your plan."), 429
    try:
        cur = db().execute("INSERT INTO tracked(user_id,keyword,region,created_at) VALUES(?,?,?,?)",
                           (uid(), kw, parse_params(d)["region"], now_iso()))
        db().commit()
    except sqlite3.IntegrityError:
        return jsonify(error="Already tracking that keyword."), 409
    return jsonify(id=cur.lastrowid)


@app.post("/api/tracked/<int:tid>/snapshot")
@login_required
def snapshot(tid):
    t = own_tracked(tid)
    try:
        videos, sample, _ = fetch_videos(t["keyword"], parse_params({"region": t["region"], "max": 20}))
    except ProviderError as e:
        return jsonify(error=str(e)), 502
    enrich_videos(videos, t["keyword"])
    m = metrics_for(videos, get_weights())
    db().execute("INSERT INTO snapshots(tracked_id,taken_at,sample,video_count,top_video_ids,metrics) VALUES(?,?,?,?,?,?)",
                 (tid, now_iso(), int(sample), len(videos), json.dumps([v["id"] for v in videos[:10]]), json.dumps(m)))
    db().commit()
    return jsonify(ok=True, sample=sample)


@app.delete("/api/tracked/<int:tid>")
@login_required
def delete_tracked(tid):
    own_tracked(tid)
    db().execute("DELETE FROM tracked WHERE id=?", (tid,))
    db().commit()
    return jsonify(ok=True)


# --------------------------------------------------------------------------- #
# Routes: AI Content Studio + calendar
# --------------------------------------------------------------------------- #
TITLE_APPROACHES = ["Search-focused", "Curiosity-driven", "Educational", "Problem-solving",
                    "Comparison-based", "Story-driven", "Audience-specific"]


def template_brief(kw, titles):
    k = kw
    return {"titles": [
        {"approach": "Search-focused", "title": f"{k.title()}: Complete Guide"},
        {"approach": "Search-focused", "title": f"How to Get Started With {k.title()}"},
        {"approach": "Curiosity-driven", "title": f"What Most People Get Wrong About {k.title()}"},
        {"approach": "Educational", "title": f"{k.title()} Explained Step by Step"},
        {"approach": "Educational", "title": f"The Basics of {k.title()} in 10 Minutes"},
        {"approach": "Problem-solving", "title": f"Struggling With {k.title()}? Fix These Mistakes"},
        {"approach": "Comparison-based", "title": f"{k.title()}: Options Compared Honestly"},
        {"approach": "Story-driven", "title": f"What I Learned Trying {k.title()} for a Month"},
        {"approach": "Audience-specific", "title": f"{k.title()} for Absolute Beginners"},
        {"approach": "Audience-specific", "title": f"{k.title()} on a Tight Budget"}],
        "outline": {"hooks": [f"State the problem the viewer has with {k} in one sentence.",
                              "Promise one concrete outcome you can actually deliver."],
                    "intro": "Who this is for and what they will be able to do by the end.",
                    "sections": ["Core concepts", "Step-by-step walkthrough", "Common mistakes",
                                 "Tools or options and trade-offs", "Worked example"],
                    "conclusion": "Recap, next step, related video.", "cta": "Invite a specific question in comments."},
        "differentiation_plan": ["List what the saved research videos all cover; deliberately add one thing they omit.",
                                 "Use your own examples, footage and data rather than reusing competitors' structure."],
        "description_draft": f"In this video we cover {k}. Chapters below. Sources and tools mentioned are linked.",
        "thumbnail_concepts": ["One clear subject, three words max of text", "Contrast the before and after"],
        "faq": [f"What is {k}?", f"Is {k} worth it for beginners?", f"What are the biggest mistakes with {k}?"],
        "fact_check_checklist": ["Verify every number against a primary source", "Date-stamp anything that changes"],
        "research_checklist": ["Watch the saved videos fully", "Note unanswered questions in comments"],
        "reference_titles": titles}


@app.post("/api/brief/generate")
@login_required
def generate_brief():
    d = request.get_json(silent=True) or {}
    kw = (d.get("keyword") or "").strip()[:150]
    if not kw:
        return jsonify(error="Keyword required."), 400
    titles = [str(t)[:150] for t in (d.get("video_titles") or [])][:25]
    notes = str(d.get("notes", ""))[:1500]
    res = llm_json(
        "You are a YouTube content strategist. Ground suggestions ONLY in the supplied titles and notes. "
        "Do not claim to have watched videos or read transcripts. No clickbait or unsupported claims. "
        "Do not copy competitor phrasing. Respond with JSON only.",
        f'Keyword: "{kw}"\nResearch video titles: {json.dumps(titles)}\nUser notes: {notes}\n'
        'Return {"titles":[{"approach":str,"title":str} x10 using Search-focused, Curiosity-driven, '
        'Educational, Problem-solving, Comparison-based, Story-driven, Audience-specific],'
        '"outline":{"hooks":[str],"intro":str,"sections":[str],"conclusion":str,"cta":str},'
        '"differentiation_plan":[str],"description_draft":str,"thumbnail_concepts":[str],'
        '"faq":[str],"fact_check_checklist":[str],"research_checklist":[str]}', max_tokens=3500)
    if res and isinstance(res.get("titles"), list):
        res["reference_titles"] = titles
        method = f"llm:{LLM_MODEL}"
    else:
        res, method = template_brief(kw, titles), "template (no LLM configured)"
    cur = db().execute("INSERT INTO briefs(user_id,keyword,content,method,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                       (uid(), kw, json.dumps(res), method, now_iso(), now_iso()))
    db().commit()
    return jsonify(id=cur.lastrowid, keyword=kw, brief=res, method=method)


@app.get("/api/briefs")
@login_required
def list_briefs():
    rows = db().execute("SELECT id,keyword,method,scheduled_for,updated_at FROM briefs WHERE user_id=? "
                        "ORDER BY id DESC", (uid(),)).fetchall()
    return jsonify(briefs=[dict(r) for r in rows])


@app.get("/api/briefs/<int:bid>")
@login_required
def get_brief(bid):
    r = own_brief(bid)
    return jsonify(id=r["id"], keyword=r["keyword"], brief=json.loads(r["content"]), method=r["method"],
                   scheduled_for=r["scheduled_for"])


@app.put("/api/briefs/<int:bid>")
@login_required
def update_brief(bid):
    own_brief(bid)
    d = request.get_json(silent=True) or {}
    sched = d.get("scheduled_for")
    if sched and not re.match(r"^\d{4}-\d{2}-\d{2}$", sched):
        return jsonify(error="Date must be YYYY-MM-DD."), 400
    if "brief" in d:
        db().execute("UPDATE briefs SET content=?, updated_at=? WHERE id=?",
                     (json.dumps(d["brief"]), now_iso(), bid))
    db().execute("UPDATE briefs SET scheduled_for=?, updated_at=? WHERE id=?", (sched or None, now_iso(), bid))
    db().commit()
    return jsonify(ok=True)


@app.delete("/api/briefs/<int:bid>")
@login_required
def delete_brief(bid):
    own_brief(bid)
    db().execute("DELETE FROM briefs WHERE id=?", (bid,))
    db().commit()
    return jsonify(ok=True)


@app.get("/api/account/export")
@login_required
def export_account():
    """User data export (privacy)."""
    d = db()
    data = {"playlists": [], "briefs": [], "tracked": [], "searches": []}
    for p in d.execute("SELECT * FROM playlists WHERE user_id=?", (uid(),)):
        items = [item_json(r) for r in d.execute("SELECT * FROM playlist_items WHERE playlist_id=?", (p["id"],))]
        data["playlists"].append({**dict(p), "items": items})
    data["briefs"] = [dict(r) for r in d.execute("SELECT * FROM briefs WHERE user_id=?", (uid(),))]
    data["tracked"] = [dict(r) for r in d.execute("SELECT * FROM tracked WHERE user_id=?", (uid(),))]
    data["searches"] = [dict(r) for r in d.execute("SELECT * FROM searches WHERE user_id=?", (uid(),))]
    return Response(json.dumps(data, indent=2), mimetype="application/json",
                    headers={"Content-Disposition": "attachment; filename=tubepilot-export.json"})


@app.delete("/api/account")
@login_required
def delete_account():
    db().execute("DELETE FROM users WHERE id=?", (uid(),))  # cascades to all user data
    db().commit()
    session.clear()
    return jsonify(ok=True)


# --------------------------------------------------------------------------- #
# Frontend (single page, no build step). All dynamic text is inserted via
# textContent / esc() because video metadata and AI output are untrusted.
# --------------------------------------------------------------------------- #
INDEX_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__APP_NAME__</title>
<style>
:root{--bg:#f7f8fa;--panel:#fff;--ink:#14171f;--mut:#5b6472;--line:#e3e6ec;--acc:#3b5bdb;--acc2:#e8edff;--ok:#1f7a3d;--warn:#9a5b00;--bad:#b42318}
@media(prefers-color-scheme:dark){:root{--bg:#0f1218;--panel:#171b23;--ink:#e8ebf2;--mut:#9aa3b2;--line:#272d39;--acc:#7c93ff;--acc2:#222a45;--ok:#5bd68a;--warn:#f0b04a;--bad:#ff8a80}}
:root[data-theme=light]{--bg:#f7f8fa;--panel:#fff;--ink:#14171f;--mut:#5b6472;--line:#e3e6ec;--acc:#3b5bdb;--acc2:#e8edff;--ok:#1f7a3d;--warn:#9a5b00;--bad:#b42318}
:root[data-theme=dark]{--bg:#0f1218;--panel:#171b23;--ink:#e8ebf2;--mut:#9aa3b2;--line:#272d39;--acc:#7c93ff;--acc2:#222a45;--ok:#5bd68a;--warn:#f0b04a;--bad:#ff8a80}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
button,input,select,textarea{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.app{display:flex;min-height:100vh}nav{width:210px;background:var(--panel);border-right:1px solid var(--line);padding:14px;position:sticky;top:0;height:100vh;flex-shrink:0}
nav h1{font-size:16px;margin:0 0 2px}nav small{color:var(--mut);display:block;margin-bottom:14px;font-size:11px}
nav button{display:block;width:100%;text-align:left;background:none;border:0;padding:8px 10px;border-radius:6px;cursor:pointer}
nav button:hover{background:var(--acc2)}nav button.on{background:var(--acc2);color:var(--acc);font-weight:600}
main{flex:1;padding:22px;max-width:1200px;min-width:0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
h2{margin:0 0 10px;font-size:18px}h3{margin:0 0 8px;font-size:14px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.grow{flex:1;min-width:180px}
input,select,textarea{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:7px 9px}
textarea{width:100%;min-height:70px}input.grow{width:100%}
.btn{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:7px 12px;cursor:pointer}
.btn.sec{background:var(--acc2);color:var(--acc)}.btn.sm{padding:3px 8px;font-size:12px}.btn.bad{background:var(--bad)}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11px;background:var(--acc2);color:var(--acc);margin-right:4px}
.pill.sample{background:#fff3cd;color:#7a5200}.pill.llm{background:#e6f4ea;color:#1f7a3d}.pill.tpl{background:var(--line);color:var(--mut)}
.notice{border-left:4px solid var(--warn);background:var(--panel);padding:8px 12px;margin-bottom:12px;border-radius:4px;color:var(--warn)}
.err{color:var(--bad)}.mut{color:var(--mut)}.ok{color:var(--ok)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.vid{border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--panel);display:flex;flex-direction:column}
.vid .th{aspect-ratio:16/9;background:var(--line);display:flex;align-items:center;justify-content:center;color:var(--mut);font-size:12px}
.vid img{width:100%;height:100%;object-fit:cover;display:block}.vid .b{padding:10px;display:flex;flex-direction:column;gap:4px;flex:1}
.vid .t{font-weight:600;line-height:1.3}.stats{font-size:12px;color:var(--mut)}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{cursor:pointer;font-size:12px;color:var(--mut)}
.score{font-size:28px;font-weight:700}.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--acc)}
.sk{background:linear-gradient(90deg,var(--line),var(--acc2),var(--line));background-size:200% 100%;animation:s 1.2s infinite;height:16px;border-radius:4px;margin:6px 0}@keyframes s{to{background-position:-200% 0}}
.hide{display:none}.auth{max-width:380px;margin:12vh auto}
@media(max-width:760px){.app{flex-direction:column}nav{width:100%;height:auto;position:static;display:flex;overflow-x:auto;gap:4px;padding:8px}nav h1,nav small{display:none}nav button{white-space:nowrap;width:auto}main{padding:12px}}
</style></head><body>
<div id="authv" class="hide auth card"><h2 id="brand">__APP_NAME__</h2><p class="mut">__TAGLINE__</p>
<div class="row"><input id="em" class="grow" type="email" placeholder="Email" aria-label="Email" autocomplete="username"></div><br>
<div class="row"><input id="pw" class="grow" type="password" placeholder="Password (8+ characters)" aria-label="Password" autocomplete="current-password"></div><br>
<div class="row"><button class="btn" id="li">Sign in</button><button class="btn sec" id="su">Create account</button></div><p id="aerr" class="err"></p></div>
<div id="appv" class="app hide"><nav aria-label="Primary"><h1 id="nb"></h1><small>__TAGLINE__</small><div id="navb"></div>
<br><button id="theme">Toggle theme</button><button id="lo">Sign out</button></nav><main id="main"></main></div>
<script>
let CSRF="",ME=null,STATE={res:null,tab:"explore"};
const $=s=>document.querySelector(s);
function esc(s){return String(s==null?"":s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
async function api(path,method="GET",body){const o={method,headers:{"X-CSRF-Token":CSRF}};
 if(body!==undefined){o.headers["Content-Type"]="application/json";o.body=JSON.stringify(body)}
 const r=await fetch(path,o);let d={};try{d=await r.json()}catch(e){}
 if(r.status===401){boot();throw new Error("Please sign in.")}
 if(!r.ok)throw new Error(d.error||("Request failed ("+r.status+")"));return d}
function toast(m,bad){const n=document.createElement("div");n.textContent=m;n.setAttribute("role","status");
 n.style.cssText="position:fixed;bottom:16px;right:16px;padding:10px 14px;border-radius:8px;color:#fff;z-index:9;background:"+(bad?"#b42318":"#1f7a3d");
 document.body.appendChild(n);setTimeout(()=>n.remove(),3500)}
async function boot(){const d=await (await fetch("/api/me")).json();CSRF=d.csrf;ME=d.user;
 $("#authv").classList.toggle("hide",!!ME);$("#appv").classList.toggle("hide",!ME);
 if(ME){$("#nb").textContent=d.app;buildNav();show(STATE.tab)}}
async function auth(kind){try{const d=await api("/api/"+kind,"POST",{email:$("#em").value,password:$("#pw").value});CSRF=d.csrf;boot()}catch(e){$("#aerr").textContent=e.message}}
$("#li").onclick=()=>auth("login");$("#su").onclick=()=>auth("register");
$("#lo").onclick=async()=>{await api("/api/logout","POST",{});STATE.res=null;boot()};
$("#theme").onclick=()=>{const r=document.documentElement;const dark=getComputedStyle(r).getPropertyValue("--bg").trim()==="#0f1218";r.dataset.theme=dark?"light":"dark"};
const TABS=[["explore","Keyword Explorer"],["playlists","Research Playlists"],["competitors","Competitor Intelligence"],
 ["tracking","Trends & Tracking"],["studio","AI Content Studio"],["settings","Settings & Integrations"]];
function buildNav(){const n=$("#navb");n.innerHTML="";TABS.forEach(([k,l])=>{const b=document.createElement("button");b.textContent=l;b.dataset.k=k;b.onclick=()=>show(k);n.appendChild(b)})}
function show(t){STATE.tab=t;document.querySelectorAll("#navb button").forEach(b=>b.classList.toggle("on",b.dataset.k===t));
 ({explore:vExplore,playlists:vPlaylists,competitors:vComp,tracking:vTrack,studio:vStudio,settings:vSettings})[t]()}
function el(html){const d=document.createElement("div");d.innerHTML=html;return d}
const fmtN=n=>n==null?"unavailable":Number(n).toLocaleString();
const fmtD=s=>s==null?"unavailable":Math.floor(s/60)+":"+String(s%60).padStart(2,"0");

/* ---------- Explorer ---------- */
function vExplore(){const m=$("#main");m.innerHTML=`<h2>Keyword Explorer</h2><div class="card">
<div class="row"><input id="q" class="grow" placeholder="Keyword, topic or question (one per line for multiple)" aria-label="Keyword"><button class="btn" id="go">Research</button>
<button class="btn sec" id="adv">Filters</button></div>
<div id="advp" class="row hide" style="margin-top:10px">
<label>Region <input id="rg" size="2" value="US"></label><label>Language <input id="lg" size="3" value="en"></label>
<label>Format <select id="fm"><option value="all">All</option><option value="shorts">Shorts</option><option value="long">Long-form</option></select></label>
<label>Published within <select id="dy"><option value="0">Any time</option><option value="30">30 days</option><option value="90">90 days</option><option value="365">1 year</option></select></label>
<label>Results <select id="mx"><option>10</option><option selected>15</option><option>25</option><option>50</option></select></label>
<label>Min views <input id="mv" type="number" min="0" value="0" style="width:90px"></label>
<label>Exclude terms <input id="ex" placeholder="comma separated"></label>
<label>Exclude channels <input id="exc" placeholder="exact names, comma separated"></label></div></div>
<div id="out"><div class="card"><h3>How it works</h3><p class="mut">1) Enter a seed keyword. 2) We retrieve videos from the YouTube Data API and expand the keyword into labeled clusters. 3) Scores are transparent heuristics from the retrieved sample, not YouTube data. 4) Save the best videos to a research playlist.</p><button class="btn sec" id="ex1">Try: budget gaming laptop</button></div></div>
<div id="hist" class="mut"></div>`;
 $("#adv").onclick=()=>$("#advp").classList.toggle("hide");
 $("#go").onclick=run;$("#q").addEventListener("keydown",e=>{if(e.key==="Enter")run()});
 $("#ex1").onclick=()=>{$("#q").value="budget gaming laptop";run()};
 api("/api/history").then(d=>{$("#hist").textContent=d.history.length?"Recent: "+[...new Set(d.history.map(h=>h.query))].slice(0,8).join(" · "):""});
 if(STATE.res)render(STATE.res)}
async function run(rank){const q=$("#q").value.trim();if(!q)return toast("Enter a keyword",1);
 $("#out").innerHTML='<div class="card"><div class="sk"></div><div class="sk" style="width:70%"></div><div class="sk" style="width:85%"></div></div>';
 try{const body={query:q,region:$("#rg").value,lang:$("#lg").value,format:$("#fm").value,days:$("#dy").value,max:$("#mx").value,
  min_views:$("#mv").value,exclude:$("#ex").value,exclude_channels:$("#exc").value};if(typeof rank==="string")body.rank=rank;
  const d=await api("/api/research","POST",body);STATE.res=d.results;render(d.results)}catch(e){$("#out").innerHTML='<div class="card err"></div>';$("#out").firstChild.textContent=e.message}}
function sig(label,s){if(!s)return `<tr><td>${label}</td><td colspan=2 class="mut">Unavailable (not enough data)</td></tr>`;
 return `<tr><td>${label}</td><td><b>${esc(s.label)}</b> <span class="pill ${s.kind==="sample"?"sample":""}">${esc(s.kind)}</span></td><td class="mut">${esc(s.detail)}<br>Source: ${esc(s.source)} · ${esc(s.retrieved)}</td></tr>`}
function render(results){const out=$("#out");out.innerHTML="";results.forEach((r,ri)=>{const m=r.metrics,s=m.signals;
 const box=el(`<div>${r.notice?'<div class="notice">'+esc(r.notice)+"</div>":""}
 <div class="card"><h2>${esc(r.seed)}</h2><div class="row"><div><div class="score">${m.opportunity==null?"n/a":m.opportunity}</div><div class="mut">Opportunity (heuristic, 0-100)</div></div>
 <div><b>Confidence: ${esc(m.confidence)}</b><div class="mut">Difficulty est.: ${m.difficulty==null?"unavailable":m.difficulty}</div><div class="mut">Missing: ${m.missing.length?esc(m.missing.join(", ")):"none"}</div></div>
 <div class="grow"><span class="pill">Search volume: Unknown</span><div class="mut">YouTube exposes no public search volume; none is invented. Add a licensed provider to enable it.</div></div></div>
 <table style="margin-top:10px"><tr><th>Signal</th><th>Value</th><th>Evidence</th></tr>${sig("Demand proxy",s.demand)}${sig("Competition weakness",s.competition)}${sig("Title relevance",s.relevance)}${sig("Content gap (stale results)",s.gap)}${sig("Freshness (not a trend)",s.trend)}</table>
 <p class="mut">${esc(m.disclaimer)}</p><div class="row"><button class="btn sm sec" data-trk>Track keyword</button></div></div>
 <div class="card"><h3>Keyword clusters <span class="pill ${r.expansion_method.startsWith("llm")?"llm":"tpl"}">${esc(r.expansion_method.startsWith("llm")?"AI-generated":"template-generated, not AI")}</span></h3><div class="mut">Suggestions are ideas, not retrieved search data. Demand shown only when a provider supplies it.</div><div id="cl"></div>
 <button class="btn sm sec" data-csv>Export CSV</button></div>
 <div class="card"><div class="row"><h3 class="grow">Videos (${r.videos.length})</h3><label>Rank by <select data-rank></select></label><button class="btn sm sec" data-bulk>Save selected to playlist</button></div><div class="mut" data-crit></div><div class="grid" data-vg></div></div></div>`);
 const cl=box.querySelector("#cl");r.clusters.forEach(c=>{const d=el(`<details open><summary><b></b> <span class="pill"></span></summary><div class="mut"></div><table><tr><th>Keyword</th><th>Intent</th><th>Source</th><th>Demand</th><th></th></tr></table></details>`);
  d.querySelector("b").textContent=c.name;d.querySelector(".pill").textContent=c.intent+" · "+c.format;d.querySelector(".mut").textContent=(c.why||"")+(c.angles&&c.angles.length?" Angles: "+c.angles.join("; "):"");
  const t=d.querySelector("table");c.items.forEach(it=>{const tr=document.createElement("tr");
   tr.innerHTML=`<td></td><td></td><td><span class="pill ${it.source==="llm"?"llm":"tpl"}"></span></td><td class="mut">Unknown</td><td><button class="btn sm sec">Analyze</button></td>`;
   tr.children[0].textContent=it.keyword;tr.children[1].textContent=it.intent;tr.querySelector(".pill").textContent=it.source;
   tr.querySelector("button").onclick=async e=>{e.target.textContent="…";try{const a=await api("/api/keyword/analyze","POST",{keyword:it.keyword,region:r.params.region,lang:r.params.lang,max:15});
    e.target.replaceWith(document.createTextNode("Opp "+(a.metrics.opportunity==null?"n/a":a.metrics.opportunity)+" ("+a.metrics.confidence+")"+(a.sample?" SAMPLE":"")))}catch(x){e.target.textContent="Analyze";toast(x.message,1)}};t.appendChild(tr)});cl.appendChild(d)});
 const sel=box.querySelector("[data-rank]");Object.entries(r.rank_modes).forEach(([k,v])=>{const o=document.createElement("option");o.value=k;o.textContent=v.label;if(k===r.rank)o.selected=true;sel.appendChild(o)});
 const crit=box.querySelector("[data-crit]");const setC=()=>{crit.textContent="Criteria: "+r.rank_modes[sel.value].criteria};setC();
 sel.onchange=async()=>{setC();try{const d=await api("/api/research","POST",{query:r.seed,...r.params,rank:sel.value,days:r.params.days});STATE.res[ri]=d.results[0];render(STATE.res)}catch(e){toast(e.message,1)}};
 const vg=box.querySelector("[data-vg]");r.videos.forEach(v=>vg.appendChild(vcard(v)));
 box.querySelector("[data-trk]").onclick=async()=>{try{await api("/api/tracked","POST",{keyword:r.seed,region:r.params.region});toast("Tracking started")}catch(e){toast(e.message,1)}};
 box.querySelector("[data-csv]").onclick=async()=>{const rr=await fetch("/api/export/keywords",{method:"POST",headers:{"X-CSRF-Token":CSRF,"Content-Type":"application/json"},body:JSON.stringify({clusters:r.clusters})});
  const b=await rr.blob();const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="keywords.csv";a.click()};
 box.querySelector("[data-bulk]").onclick=()=>{const sel=[...box.querySelectorAll("[data-vg] input:checked")].map(x=>r.videos.find(v=>v.id===x.value));if(!sel.length)return toast("Select videos first",1);pickPlaylist(sel,r.seed)};
 out.appendChild(box)})}
function vcard(v){const d=document.createElement("div");d.className="vid";
 const th=document.createElement("div");th.className="th";
 if(v.thumb&&v.url){const a=document.createElement("a");a.href=v.url;a.target="_blank";a.rel="noopener noreferrer";const i=document.createElement("img");i.src=v.thumb;i.alt="";i.loading="lazy";a.appendChild(i);th.appendChild(a)}else th.textContent=v.sample?"SAMPLE – no thumbnail":"No thumbnail";
 const b=document.createElement("div");b.className="b";
 const t=document.createElement("div");t.className="t";t.textContent=v.title;
 const ch=document.createElement("div");ch.className="stats";ch.textContent=v.channel+" · "+(v.published?v.published.slice(0,10):"date unavailable");
 const s=document.createElement("div");s.className="stats";s.textContent=`Views ${fmtN(v.views)} · Likes ${fmtN(v.likes)} · Comments ${fmtN(v.comments)} · ${fmtD(v.duration_s)} · Subs ${fmtN(v.subs)}`;
 const dm=document.createElement("div");dm.className="stats";dm.textContent=`Views/day ${v.views_per_day==null?"n/a":fmtN(v.views_per_day)} · Views/subs ${v.views_to_subs==null?"n/a":v.views_to_subs} · vs median ${v.outlier_x==null?"n/a":v.outlier_x+"x"} (derived, this sample only)`;
 const w=document.createElement("div");w.className="stats";w.textContent="Why: "+(v.why||"");
 const r=document.createElement("div");r.className="row";
 const cb=document.createElement("input");cb.type="checkbox";cb.value=v.id;cb.setAttribute("aria-label","Select video");r.appendChild(cb);
 const sv=document.createElement("button");sv.className="btn sm";sv.textContent="Save to playlist";sv.onclick=()=>pickPlaylist([v],"");r.appendChild(sv);
 if(v.url){const o=document.createElement("a");o.href=v.url;o.target="_blank";o.rel="noopener noreferrer";o.textContent="Open on YouTube";r.appendChild(o)}else{const p=document.createElement("span");p.className="pill sample";p.textContent="SAMPLE";r.appendChild(p)}
 b.append(t,ch,s,dm,w,r);d.append(th,b);return d}
async function pickPlaylist(videos,kw){const d=await api("/api/playlists");
 const name=prompt("Save to which playlist? Type an existing number or a new name:\n"+d.playlists.map((p,i)=>(i+1)+". "+p.name).join("\n"));if(!name)return;
 let id;const idx=parseInt(name,10);if(idx>=1&&d.playlists[idx-1]&&String(idx)===name.trim())id=d.playlists[idx-1].id;
 else{try{id=(await api("/api/playlists","POST",{name})).id}catch(e){return toast(e.message,1)}}
 try{const r=await api("/api/playlists/"+id+"/items","POST",{videos,keyword:kw});toast(`Added ${r.added}${r.duplicates?`, ${r.duplicates} duplicate(s) skipped`:""}`)}catch(e){toast(e.message,1)}}

/* ---------- Playlists ---------- */
async function vPlaylists(){const m=$("#main");m.innerHTML=`<h2>Research Playlists</h2><p class="mut">Application research playlists. They are not YouTube playlists.</p>
<div class="card row"><input id="pn" class="grow" placeholder="New playlist name"><input id="po" class="grow" placeholder="Research objective (optional)"><button class="btn" id="pc">Create</button></div><div id="pl"></div><div id="pd"></div>`;
 $("#pc").onclick=async()=>{try{await api("/api/playlists","POST",{name:$("#pn").value,objective:$("#po").value});vPlaylists()}catch(e){toast(e.message,1)}};
 const d=await api("/api/playlists");const pl=$("#pl");
 if(!d.playlists.length)pl.innerHTML='<div class="card mut">No playlists yet. Create one above or save videos from the Explorer.</div>';
 d.playlists.forEach(p=>{const c=el(`<div class="card row"><div class="grow"><b></b><div class="mut"></div></div><button class="btn sm sec" data-o>Open</button><button class="btn sm sec" data-d>Duplicate</button><button class="btn sm bad" data-x>Delete</button></div>`);
  c.querySelector("b").textContent=p.name;c.querySelector(".mut").textContent=p.n+" videos"+(p.objective?" · "+p.objective:"");
  c.querySelector("[data-o]").onclick=()=>openPl(p.id);c.querySelector("[data-d]").onclick=async()=>{await api(`/api/playlists/${p.id}/duplicate`,"POST",{});vPlaylists()};
  c.querySelector("[data-x]").onclick=async()=>{if(confirm("Delete this playlist?")){await api("/api/playlists/"+p.id,"DELETE");vPlaylists()}};pl.appendChild(c)})}
async function openPl(id){const d=await api("/api/playlists/"+id);const box=$("#pd");box.innerHTML="";
 const h=el(`<div class="card"><div class="row"><h3 class="grow"></h3><button class="btn sm sec" data-r>Rename</button><a href="/api/playlists/${id}/export.csv">CSV</a><a href="/api/playlists/${id}/export.json">JSON</a><button class="btn sm" data-s>AI summary</button></div><div class="mut"></div><div data-sum></div></div>`);
 h.querySelector("h3").textContent=d.playlist.name;h.querySelector(".mut").textContent=d.playlist.objective||"";
 h.querySelector("[data-r]").onclick=async()=>{const n=prompt("New name",d.playlist.name);if(n){await api("/api/playlists/"+id,"PATCH",{name:n});vPlaylists()}};
 h.querySelector("[data-s]").onclick=async()=>{try{const s=await api(`/api/playlists/${id}/summary`,"POST",{});h.querySelector("[data-sum]").textContent=s.method+": "+JSON.stringify(s.summary)}catch(e){toast(e.message,1)}};
 box.appendChild(h);const ids=d.items.map(i=>i.id);
 d.items.forEach((it,idx)=>{const v=it.video;const c=el(`<div class="card"><div class="row"><b class="grow"></b><button class="btn sm sec" data-u>↑</button><button class="btn sm sec" data-dn>↓</button><button class="btn sm bad" data-x>Remove</button></div><div class="stats"></div>
  <div class="row" style="margin-top:6px"><select><option>Not Reviewed</option><option>Reviewed</option><option>Used for Research</option></select><input class="grow" placeholder="Tags"><button class="btn sm" data-sv>Save</button></div><textarea placeholder="Notes"></textarea></div>`);
  c.querySelector("b").textContent=(idx+1)+". "+v.title;c.querySelector(".stats").textContent=v.channel+" · "+fmtN(v.views)+" views"+(v.sample?" · SAMPLE":"");
  if(v.url){const a=document.createElement("a");a.href=v.url;a.target="_blank";a.rel="noopener noreferrer";a.textContent="Open on YouTube";c.querySelector(".stats").appendChild(document.createTextNode(" · "));c.querySelector(".stats").appendChild(a)}
  const sel=c.querySelector("select");sel.value=it.status;const tg=c.querySelector("input");tg.value=it.tags;const nt=c.querySelector("textarea");nt.value=it.note;
  c.querySelector("[data-sv]").onclick=async()=>{try{await api(`/api/playlists/${id}/items/${it.id}`,"PATCH",{status:sel.value,tags:tg.value,note:nt.value});toast("Saved")}catch(e){toast(e.message,1)}};
  c.querySelector("[data-x]").onclick=async()=>{await api(`/api/playlists/${id}/items/${it.id}`,"DELETE");openPl(id)};
  const mv=async(o)=>{const j=idx+o;if(j<0||j>=ids.length)return;const n=ids.slice();[n[idx],n[j]]=[n[j],n[idx]];await api(`/api/playlists/${id}/reorder`,"POST",{order:n});openPl(id)};
  c.querySelector("[data-u]").onclick=()=>mv(-1);c.querySelector("[data-dn]").onclick=()=>mv(1);box.appendChild(c)})}

/* ---------- Competitors ---------- */
function vComp(){$("#main").innerHTML=`<h2>Competitor Intelligence</h2><div class="card"><textarea id="ch" placeholder="Channel URLs, @handles or IDs (one per line, up to 4)"></textarea><button class="btn" id="cg">Analyze</button></div><div id="co"></div>`;
 $("#cg").onclick=async()=>{const o=$("#co");o.innerHTML='<div class="card"><div class="sk"></div><div class="sk"></div></div>';
  try{const d=await api("/api/competitors","POST",{channels:$("#ch").value});o.innerHTML="";const cav=el('<div class="notice"></div>');cav.firstChild.textContent=d.caveat;o.appendChild(cav.firstChild);
   d.channels.forEach(c=>{const x=el('<div class="card"><h3></h3><div class="mut" data-n></div><div data-b></div></div>');x.querySelector("h3").textContent=c.channel.title+" · subs "+fmtN(c.channel.subs);x.querySelector("[data-n]").textContent=c.sample_note;
    x.querySelector("[data-b]").textContent=`Uploads/week ${c.uploads_per_week??"n/a"} · Median duration ${fmtD(c.median_duration_s)} · Shorts share ${c.shorts_share??"n/a"}% · Median views ${fmtN(c.median_views)} · Themes: ${c.themes.join(", ")||"n/a"} · Title patterns: ${c.title_structures.map(t=>t[0]+" ("+t[1]+")").join(", ")}`;o.appendChild(x)});
   if(d.shared_themes.length){const s=el('<div class="card"><b>Shared themes:</b> <span></span></div>');s.querySelector("span").textContent=d.shared_themes.join(", ");o.appendChild(s)}
   Object.entries(d.theme_gaps).forEach(([k,v])=>{if(v.length){const s=el('<div class="card"><span></span></div>');s.firstChild.textContent=`Themes others cover that ${k} does not (in sampled videos): ${v.join(", ")}`;o.appendChild(s)}})}
  catch(e){o.innerHTML='<div class="card err"></div>';o.firstChild.textContent=e.message}}}

/* ---------- Tracking ---------- */
async function vTrack(){const m=$("#main");m.innerHTML='<h2>Trends & Tracking</h2><p class="mut">Dated snapshots. A trend is shown only after 3+ real snapshots.</p><div class="card row"><input id="tk" class="grow" placeholder="Keyword to track"><button class="btn" id="ta">Track</button></div><div id="tl"></div>';
 $("#ta").onclick=async()=>{try{await api("/api/tracked","POST",{keyword:$("#tk").value});vTrack()}catch(e){toast(e.message,1)}};
 const d=await api("/api/tracked");const l=$("#tl");if(!d.tracked.length)l.innerHTML='<div class="card mut">Nothing tracked yet.</div>';
 d.tracked.forEach(t=>{const c=el('<div class="card"><div class="row"><b class="grow"></b><button class="btn sm" data-s>Take snapshot</button><button class="btn sm bad" data-x>Remove</button></div><div class="mut" data-n></div><div class="stats" data-h></div></div>');
  c.querySelector("b").textContent=t.keyword+" ("+t.region+")";c.querySelector("[data-n]").textContent=t.note;
  c.querySelector("[data-h]").textContent=t.series.map(s=>s.at.slice(0,10)+": opp "+(s.opportunity==null?"n/a":s.opportunity)+(s.sample?" (SAMPLE)":"")).join(" | ");
  c.querySelector("[data-s]").onclick=async()=>{try{await api(`/api/tracked/${t.id}/snapshot`,"POST",{});vTrack()}catch(e){toast(e.message,1)}};
  c.querySelector("[data-x]").onclick=async()=>{await api("/api/tracked/"+t.id,"DELETE");vTrack()};l.appendChild(c)})}

/* ---------- Studio ---------- */
async function vStudio(){const m=$("#main");m.innerHTML='<h2>AI Content Studio</h2><div class="card"><input id="bk" class="grow" placeholder="Keyword"><br><br><textarea id="bt" placeholder="Reference video titles from your research (one per line) - optional"></textarea><textarea id="bn" placeholder="Your research notes - optional"></textarea><button class="btn" id="bg">Generate brief</button></div><div id="bo"></div><h3>Saved briefs</h3><div id="bl"></div>';
 $("#bg").onclick=async()=>{const o=$("#bo");o.innerHTML='<div class="card"><div class="sk"></div><div class="sk"></div></div>';
  try{const d=await api("/api/brief/generate","POST",{keyword:$("#bk").value,video_titles:$("#bt").value.split("\n").filter(Boolean),notes:$("#bn").value});showBrief(d);vBriefs()}catch(e){o.innerHTML='<div class="card err"></div>';o.firstChild.textContent=e.message}};vBriefs()}
function showBrief(d){const o=$("#bo");o.innerHTML="";const c=el('<div class="card"><div class="row"><h3 class="grow"></h3><span class="pill"></span></div><textarea style="min-height:320px" data-j></textarea><div class="row"><label>Publish date <input type="date" data-d></label><button class="btn" data-s>Save edits</button></div></div>');
 c.querySelector("h3").textContent=d.keyword;c.querySelector(".pill").textContent=d.method;c.querySelector("[data-j]").value=JSON.stringify(d.brief,null,2);c.querySelector("[data-d]").value=d.scheduled_for||"";
 c.querySelector("[data-s]").onclick=async()=>{try{await api("/api/briefs/"+d.id,"PUT",{brief:JSON.parse(c.querySelector("[data-j]").value),scheduled_for:c.querySelector("[data-d]").value||null});toast("Saved");vBriefs()}catch(e){toast(e.message,1)}};o.appendChild(c)}
async function vBriefs(){const d=await api("/api/briefs");const l=$("#bl");l.innerHTML=d.briefs.length?"":'<div class="mut">None yet.</div>';
 d.briefs.forEach(b=>{const c=el('<div class="card row"><span class="grow"></span><button class="btn sm sec" data-o>Open</button><button class="btn sm bad" data-x>Delete</button></div>');c.querySelector("span").textContent=b.keyword+(b.scheduled_for?" · scheduled "+b.scheduled_for:"")+" · "+b.updated_at.slice(0,10);
  c.querySelector("[data-o]").onclick=async()=>showBrief(await api("/api/briefs/"+b.id));c.querySelector("[data-x]").onclick=async()=>{await api("/api/briefs/"+b.id,"DELETE");vBriefs()};l.appendChild(c)})}

/* ---------- Settings ---------- */
async function vSettings(){const d=await api("/api/status");const m=$("#main");m.innerHTML='<h2>Settings & Integrations</h2><div id="ig"></div><div class="card"><h3>Usage today</h3><div id="us" class="mut"></div></div><div class="card"><h3>Opportunity score weights</h3><div class="row" id="wt"></div><p class="mut">Weights are normalized over available signals. Missing signals are excluded, never treated as zero.</p><button class="btn" id="ws">Save weights</button></div><div class="card"><h3>Privacy</h3><div class="row"><a href="/api/account/export">Export my data</a><button class="btn bad sm" id="da">Delete account</button></div></div>';
 Object.values(d.integrations).forEach(i=>{const c=el('<div class="card"><b></b> <span class="pill"></span><div class="mut"></div></div>');c.querySelector("b").textContent=i.name;const p=c.querySelector(".pill");p.textContent=i.configured?"Connected":"Setup required";if(!i.configured)p.classList.add("sample");c.querySelector(".mut").textContent=i.configured?"Active (key stored server-side).":i.setup;$("#ig").appendChild(c)});
 $("#us").textContent=`Searches ${d.usage.searches||0}/${d.limits.searches_per_day} · limits are configurable in PLAN_LIMITS`;
 Object.entries(d.weights).forEach(([k,v])=>{const l=document.createElement("label");l.textContent=k+" ";const i=document.createElement("input");i.type="number";i.min=0;i.max=100;i.value=v;i.dataset.k=k;i.style.width="64px";l.appendChild(i);$("#wt").appendChild(l)});
 $("#ws").onclick=async()=>{const w={};document.querySelectorAll("#wt input").forEach(i=>w[i.dataset.k]=i.value);try{await api("/api/weights","PUT",w);toast("Weights saved")}catch(e){toast(e.message,1)}};
 $("#da").onclick=async()=>{if(confirm("Permanently delete your account and all data?")){await api("/api/account","DELETE");boot()}}}
boot();
</script></body></html>"""

init_db()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
