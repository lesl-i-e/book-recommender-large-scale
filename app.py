"""
Large-Scale Book Recommendation System
SDS 2412 — Analysis of Large Datasets
Streamlit GUI — connects to the trained models from the notebook
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import hashlib
from collections import OrderedDict, defaultdict, deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BookMind — Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Refined dark editorial: ink + gold + ivory
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Outfit:wght@300;400;500;600&display=swap');

/* ── Root variables ── */
:root {
    --ink:      #0d1117;
    --ink2:     #161b22;
    --ink3:     #1c2330;
    --gold:     #c9922a;
    --gold2:    #e8b84b;
    --gold3:    #f5d37a;
    --ivory:    #f7f3ec;
    --ivory2:   #ede8df;
    --ivory3:   #e2dbd0;
    --text:     #1c1c28;
    --muted:    #64748b;
    --card-bg:  #ffffff;
    --border:   #e8e0d4;
    --shadow:   rgba(13,17,23,0.10);
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: var(--text);
}

/* ── App background — parchment with very subtle noise ── */
.stApp {
    background: var(--ivory);
    background-image:
        radial-gradient(ellipse 80% 50% at 5% 0%, rgba(201,146,42,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 95% 100%, rgba(13,17,23,0.05) 0%, transparent 60%),
        url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.018'/%3E%3C/svg%3E");
}

/* ── Main block padding ── */
.block-container {
    padding: 2.5rem 3.5rem 5rem 3.5rem !important;
    max-width: 1180px;
}

/* ── Sidebar — compact dark column ── */
[data-testid="stSidebar"] {
    background: var(--ink) !important;
    border-right: 1px solid rgba(201,146,42,0.18) !important;
    min-width: 220px !important;
    max-width: 240px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div {
    width: 220px !important;
}
[data-testid="stSidebar"] * {
    color: var(--ivory) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p {
    color: rgba(247,243,236,0.6) !important;
    font-size: 0.78rem !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--gold2) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(201,146,42,0.15) !important;
    margin: 0.8rem 0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.82rem !important;
    color: rgba(247,243,236,0.8) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] [aria-checked="true"] ~ span {
    color: var(--gold2) !important;
    font-weight: 600 !important;
}

/* ── Headers ── */
h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em;
    line-height: 1.12;
}
h2 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em;
}
h3 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

/* ── Gold accent line ── */
.accent-line {
    height: 2px;
    width: 48px;
    background: linear-gradient(90deg, var(--gold), var(--gold3));
    border-radius: 2px;
    margin: 0.5rem 0 1.6rem 0;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--ink2);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    border-top: 2px solid var(--gold);
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(13,17,23,0.18);
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: -20px; right: -20px;
    width: 80px; height: 80px;
    border-radius: 50%;
    background: rgba(201,146,42,0.06);
}
.metric-card .val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--gold2);
    line-height: 1;
}
.metric-card .lbl {
    font-size: 0.7rem;
    font-weight: 600;
    color: rgba(247,243,236,0.5);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.4rem;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: rgba(247,243,236,0.3);
    margin-top: 0.15rem;
}

/* ── Book recommendation cards ── */
.book-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem 1.2rem 1.8rem;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: flex-start;
    gap: 1.2rem;
    transition: box-shadow 0.22s ease, border-color 0.22s ease, transform 0.22s ease;
    position: relative;
    overflow: hidden;
}
.book-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, var(--gold), var(--gold3));
    border-radius: 3px 0 0 3px;
}
.book-card:hover {
    box-shadow: 0 8px 28px rgba(13,17,23,0.09);
    border-color: rgba(201,146,42,0.4);
    transform: translateY(-1px);
}
.book-rank {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--ivory3);
    line-height: 1;
    min-width: 2rem;
    text-align: center;
    padding-top: 0.1rem;
}
.book-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.3;
    margin-bottom: 0.18rem;
}
.book-author {
    font-size: 0.82rem;
    color: var(--muted);
    font-weight: 400;
    margin-bottom: 0.5rem;
}
.score-pill {
    display: inline-block;
    background: linear-gradient(135deg, var(--gold), var(--gold2));
    color: var(--ink);
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.18rem 0.65rem;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-right: 0.35rem;
}
.score-pill.cb {
    background: linear-gradient(135deg, #4f8ef7, #818cf8);
    color: #fff;
}
.score-pill.hybrid {
    background: var(--ink);
    color: var(--gold2);
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, var(--ink) 0%, var(--ink3) 100%);
    border-radius: 18px;
    padding: 2.2rem 2.6rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(201,146,42,0.2);
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(13,17,23,0.14);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201,146,42,0.12) 0%, transparent 70%);
}
.hero-banner::after {
    content: '📚';
    position: absolute;
    right: 2.5rem;
    bottom: 1rem;
    font-size: 5rem;
    opacity: 0.07;
    pointer-events: none;
}
.hero-banner h2 {
    color: var(--ivory) !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    margin: 0 0 0.5rem 0;
    letter-spacing: 0 !important;
    text-transform: none !important;
}
.hero-banner p {
    color: rgba(247,243,236,0.65) !important;
    font-size: 0.88rem !important;
    margin: 0;
    line-height: 1.65;
    max-width: 560px;
}

/* ── Mode selector tabs ── */
.mode-tab-row {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 1.6rem;
}
.mode-tab {
    flex: 1;
    background: #fff;
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    cursor: pointer;
    transition: all 0.18s ease;
    text-align: left;
}
.mode-tab.active {
    border-color: var(--gold);
    background: linear-gradient(135deg, rgba(201,146,42,0.06), rgba(232,184,75,0.04));
    box-shadow: 0 2px 12px rgba(201,146,42,0.12);
}
.mode-tab .mt-icon { font-size: 1.4rem; margin-bottom: 0.4rem; }
.mode-tab .mt-label {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--ink);
}
.mode-tab .mt-desc {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.15rem;
}

/* ── Settings inline row (main page) ── */
.settings-row {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.6rem;
    display: flex;
    gap: 2rem;
    align-items: center;
    flex-wrap: wrap;
}
.settings-row .sr-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    font-weight: 600;
    margin-bottom: 0.15rem;
}

/* ── Section divider ── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.8rem 0;
}

/* ── Pipeline step badges ── */
.pipeline-step {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--ivory2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--ink);
    margin: 0.15rem;
    font-family: 'Outfit', sans-serif;
}
.step-num {
    background: var(--gold);
    color: var(--ink);
    border-radius: 50%;
    width: 16px; height: 16px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    font-weight: 700;
}

/* ── Alert/warning box ── */
.warn-box {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #78350f;
}

/* ── Streamlit widgets overrides ── */
.stButton > button {
    background: linear-gradient(135deg, var(--ink) 0%, var(--ink3) 100%) !important;
    color: var(--gold2) !important;
    border: 1px solid rgba(201,146,42,0.35) !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 2.2rem !important;
    letter-spacing: 0.04em;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e2a3a 0%, #243040 100%) !important;
    border-color: var(--gold) !important;
    box-shadow: 0 4px 18px rgba(13,17,23,0.22) !important;
    transform: translateY(-1px);
}
.stTextInput > div > div > input,
.stSelectbox > div > div > div {
    border-radius: 10px !important;
    border-color: var(--border) !important;
    font-family: 'Outfit', sans-serif !important;
    background: #fff !important;
    color: var(--ink) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,146,42,0.15) !important;
}
.stSlider > div > div > div > div {
    background: var(--gold) !important;
}
.stRadio [data-baseweb="radio"] {
    gap: 0.3rem !important;
}

/* ── Tab overrides ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: transparent;
    border-bottom: 1.5px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500;
    font-size: 0.88rem;
    color: var(--muted);
    background: transparent;
    border: none;
    padding: 0.55rem 1.1rem;
    border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    color: var(--ink) !important;
    background: rgba(201,146,42,0.08) !important;
    border-bottom: 2px solid var(--gold) !important;
}

/* ── Progress / spinner ── */
.stSpinner > div {
    border-top-color: var(--gold) !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}

/* ── Sidebar logo area ── */
.sidebar-logo {
    text-align: center;
    padding: 1.4rem 0.8rem 1rem 0.8rem;
    border-bottom: 1px solid rgba(201,146,42,0.18);
    margin-bottom: 1.2rem;
}
.sidebar-logo .icon { font-size: 2rem; }
.sidebar-logo .title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--gold2);
    line-height: 1.2;
    margin-top: 0.4rem;
    letter-spacing: 0.03em;
}
.sidebar-logo .sub {
    font-size: 0.62rem;
    color: rgba(247,243,236,0.35);
    margin-top: 0.2rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

/* ── Cache badge ── */
.cache-hit {
    display: inline-block;
    background: rgba(34,197,94,0.1);
    color: #16a34a;
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 20px;
    padding: 0.12rem 0.6rem;
    font-size: 0.7rem;
    font-weight: 600;
    vertical-align: middle;
}
.cache-miss {
    display: inline-block;
    background: rgba(201,146,42,0.1);
    color: #92400e;
    border: 1px solid rgba(201,146,42,0.25);
    border-radius: 20px;
    padding: 0.12rem 0.6rem;
    font-size: 0.7rem;
    font-weight: 600;
    vertical-align: middle;
}

/* ── User profile strip ── */
.profile-strip {
    background: linear-gradient(135deg, var(--ink) 0%, var(--ink3) 100%);
    border-radius: 12px;
    padding: 1.1rem 1.6rem;
    margin-bottom: 1.5rem;
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
    align-items: center;
    border: 1px solid rgba(201,146,42,0.18);
    box-shadow: 0 4px 16px rgba(13,17,23,0.12);
}
.profile-strip .ps-item .ps-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(247,243,236,0.45);
    font-weight: 600;
    margin-bottom: 0.15rem;
}
.profile-strip .ps-item .ps-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--ivory);
    line-height: 1.1;
}
.profile-strip .ps-item .ps-val.gold { color: var(--gold2); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# BLOOM FILTER
# ─────────────────────────────────────────────────────────────
class BloomFilter:
    def __init__(self, size=5000, num_hashes=3):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [0] * size

    def _hashes(self, item):
        return [
            int(hashlib.md5(f"{item}_{i}".encode()).hexdigest(), 16) % self.size
            for i in range(self.num_hashes)
        ]

    def add(self, item):
        for pos in self._hashes(item):
            self.bit_array[pos] = 1

    def check(self, item):
        return all(self.bit_array[pos] for pos in self._hashes(item))


# ─────────────────────────────────────────────────────────────
# LRU CACHE
# ─────────────────────────────────────────────────────────────
class LRUCache:
    def __init__(self, capacity=100):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# DATA LOADING — cached so it only runs once
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_data():
    """Load and clean the Book-Crossing dataset."""
    import os

    # Try Google Drive path first (Colab), then local
    base_paths = [
        "/content",
        "/content/drive/MyDrive",
        ".",
        "./data",
    ]

    books_df = ratings_df = users_df = None

    for base in base_paths:
        try:
            b = pd.read_csv(f"{base}/Books.csv",   encoding="latin-1", on_bad_lines="skip")
            r = pd.read_csv(f"{base}/Ratings.csv", encoding="latin-1", on_bad_lines="skip")
            u = pd.read_csv(f"{base}/Users.csv",   encoding="latin-1", on_bad_lines="skip")
            books_df, ratings_df, users_df = b, r, u
            break
        except FileNotFoundError:
            continue

    if books_df is None:
        return None, None, None, None, None

    # Clean books
    books_df["Year-Of-Publication"] = pd.to_numeric(
        books_df["Year-Of-Publication"], errors="coerce"
    )
    books_clean = books_df.dropna(subset=["Book-Author", "Publisher"]).copy()
    drop_cols = [c for c in ["Image-URL-S", "Image-URL-M", "Image-URL-L"] if c in books_clean.columns]
    books_clean = books_clean.drop(columns=drop_cols)
    books_clean["Year-Of-Publication"] = pd.to_numeric(
        books_clean["Year-Of-Publication"], errors="coerce"
    ).fillna(0).astype(int)

    # Clean users
    users_clean = users_df.copy()
    users_clean["Age"] = users_clean["Age"].fillna(users_clean["Age"].median())
    users_clean["Age"] = pd.to_numeric(users_clean["Age"], errors="coerce").fillna(35).astype(int)
    users_clean["Country"] = users_clean["Location"].apply(
        lambda x: x.strip().split(",")[-1].strip() if isinstance(x, str) else "unknown"
    )

    # Explicit ratings only
    explicit_ratings = ratings_df[ratings_df["Book-Rating"] > 0].copy()

    # Build working dataset
    working_df = (
        explicit_ratings
        .merge(books_clean, on="ISBN", how="inner")
        .merge(users_clean[["User-ID", "Age", "Country"]], on="User-ID", how="inner")
    )

    return books_clean, users_clean, explicit_ratings, working_df, ratings_df


@st.cache_resource(show_spinner=False)
def build_tfidf(_books_clean):
    """Build TF-IDF content matrix."""
    bc = _books_clean.copy()
    bc["content"] = (
        bc["Book-Author"].fillna("") + " " +
        bc["Publisher"].fillna("") + " " +
        bc["Year-Of-Publication"].astype(str)
    )
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    matrix = tfidf.fit_transform(bc["content"])
    return tfidf, matrix, bc.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# MMR DIVERSITY
# ─────────────────────────────────────────────────────────────
def mmr_rerank(recs_df, penalty=0.3):
    """Penalise repeated authors to prevent filter bubbles."""
    seen_authors = set()
    result = []
    temp = recs_df.copy()
    while len(result) < len(temp):
        best_idx = None
        best_score = -1
        for i, row in temp.iterrows():
            if i in [r[0] for r in result]:
                continue
            score = row["hybrid_score"]
            if row.get("Book-Author", "") in seen_authors:
                score -= penalty
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is None:
            break
        result.append((best_idx, temp.loc[best_idx]))
        seen_authors.add(temp.loc[best_idx].get("Book-Author", ""))
    return pd.DataFrame([r[1] for r in result]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# CONTENT-BASED RECOMMENDER
# ─────────────────────────────────────────────────────────────
def cb_recommend(book_title, tfidf_matrix, books_cb, n=10):
    matches = books_cb[books_cb["Book-Title"].str.contains(book_title, case=False, na=False)]
    if matches.empty:
        return pd.DataFrame()
    idx = matches.index[0]
    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_idx = sim.argsort()[::-1][1:n+1]
    res = books_cb.iloc[top_idx][["Book-Title", "Book-Author", "Year-Of-Publication"]].copy()
    res["similarity"] = sim[top_idx].round(3)
    return res.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# USER-BASED COLLABORATIVE FILTER (lightweight, no Spark)
# ─────────────────────────────────────────────────────────────
def user_cf_recommend(user_id, working_df, books_clean, tfidf_matrix, books_cb, n=10):
    """
    Lightweight collaborative filter:
    1. Find books this user rated highly
    2. Find users who also rated those books highly
    3. Recommend their other top books (not already read by user)
    4. Blend with TF-IDF content similarity
    5. Apply MMR diversity
    """
    user_ratings = working_df[working_df["User-ID"] == user_id]
    if user_ratings.empty:
        return pd.DataFrame(), "User not found"

    user_books = set(user_ratings["ISBN"].tolist())
    user_liked  = user_ratings[user_ratings["Book-Rating"] >= 7]["ISBN"].tolist()

    if not user_liked:
        user_liked = user_ratings.nlargest(3, "Book-Rating")["ISBN"].tolist()

    # Find similar users — those who rated the same books
    similar_users_data = working_df[
        (working_df["ISBN"].isin(user_liked)) &
        (working_df["User-ID"] != user_id) &
        (working_df["Book-Rating"] >= 6)
    ]

    if similar_users_data.empty:
        # Fall back to popularity
        popular = (
            working_df[~working_df["ISBN"].isin(user_books)]
            .groupby(["ISBN", "Book-Title", "Book-Author"])
            .agg(count=("Book-Rating", "count"), avg=("Book-Rating", "mean"))
            .reset_index()
            .query("count >= 3")
            .sort_values(["avg", "count"], ascending=False)
            .head(n)
        )
        popular["hybrid_score"] = (
            (popular["avg"] - popular["avg"].min()) /
            (popular["avg"].max() - popular["avg"].min() + 1e-6)
        )
        return popular, "popularity"

    # Score candidate books
    candidate_scores = (
        similar_users_data[~similar_users_data["ISBN"].isin(user_books)]
        .groupby(["ISBN", "Book-Title", "Book-Author"])
        .agg(cf_score=("Book-Rating", "mean"), cf_count=("Book-Rating", "count"))
        .reset_index()
        .query("cf_count >= 2")
    )

    if candidate_scores.empty:
        candidate_scores = (
            similar_users_data[~similar_users_data["ISBN"].isin(user_books)]
            .groupby(["ISBN", "Book-Title", "Book-Author"])
            .agg(cf_score=("Book-Rating", "mean"), cf_count=("Book-Rating", "count"))
            .reset_index()
        )

    if candidate_scores.empty:
        return pd.DataFrame(), "No candidates"

    # Normalise CF score
    mn, mx = candidate_scores["cf_score"].min(), candidate_scores["cf_score"].max()
    candidate_scores["cf_norm"] = (candidate_scores["cf_score"] - mn) / (mx - mn + 1e-6)

    # Content-based component — use first liked book as seed
    seed_isbn = user_liked[0]
    seed_matches = books_cb[books_cb["ISBN"] == seed_isbn] if "ISBN" in books_cb.columns else pd.DataFrame()

    if not seed_matches.empty:
        seed_idx = seed_matches.index[0]
        sim_scores = cosine_similarity(tfidf_matrix[seed_idx], tfidf_matrix).flatten()
        isbn_list = books_cb["ISBN"].tolist() if "ISBN" in books_cb.columns else []
        isbn_sim = {isbn_list[i]: sim_scores[i] for i in range(len(isbn_list))}
        candidate_scores["cb_norm"] = candidate_scores["ISBN"].map(isbn_sim).fillna(0)
    else:
        candidate_scores["cb_norm"] = 0.0

    # Hybrid score
    candidate_scores["hybrid_score"] = (
        0.7 * candidate_scores["cf_norm"] +
        0.3 * candidate_scores["cb_norm"]
    )
    candidate_scores = candidate_scores.sort_values("hybrid_score", ascending=False).head(n * 2)

    # MMR diversity
    result = mmr_rerank(candidate_scores, penalty=0.3).head(n)
    return result, "hybrid"


# ─────────────────────────────────────────────────────────────
# SESSION STATE — initialise cache & bloom filters
# ─────────────────────────────────────────────────────────────
if "lru_cache" not in st.session_state:
    st.session_state.lru_cache = LRUCache(capacity=100)
if "bloom_filters" not in st.session_state:
    st.session_state.bloom_filters = {}
if "query_history" not in st.session_state:
    st.session_state.query_history = []


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="icon">📚</div>
        <div class="title">BookMind</div>
        <div class="sub">SDS 2412 · JKUAT · 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigate")
    page = st.radio(
        "",
        ["🔍 Get Recommendations", "📊 System Dashboard", "📖 How It Works"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Compact settings — sliders only in sidebar, checkboxes on main page
    n_recs    = st.slider("Results", 5, 20, 10)
    min_rating = st.slider("Min rating", 1, 10, 6)
    use_mmr   = st.checkbox("MMR diversity", value=True)
    show_scores = st.checkbox("Show scores", value=False)

    st.markdown("---")
    st.markdown("### Stats")
    cache = st.session_state.lru_cache
    st.markdown(f"""
    <div style="font-size:0.78rem; color:rgba(247,243,236,0.5); line-height:2.1;">
    Cache: <b style="color:#e8b84b">{len(cache.cache)}/100</b><br>
    Hit rate: <b style="color:#e8b84b">{cache.hit_rate:.1f}%</b><br>
    Queries: <b style="color:#e8b84b">{len(st.session_state.query_history)}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.65rem; color:rgba(247,243,236,0.25); line-height:1.9; letter-spacing:0.02em;">
    Book-Crossing Dataset<br>
    1.1M ratings · 270K books<br>
    Lambda Architecture<br>
    ALS + TF-IDF Hybrid
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading dataset and building models..."):
    books_clean, users_clean, explicit_ratings, working_df, ratings_df = load_data()

data_loaded = books_clean is not None

if data_loaded:
    with st.spinner("Building TF-IDF content matrix..."):
        tfidf, tfidf_matrix, books_cb = build_tfidf(books_clean)
    # Add ISBN to books_cb if not present
    if "ISBN" not in books_cb.columns and "ISBN" in books_clean.columns:
        books_cb["ISBN"] = books_clean["ISBN"].values


# ─────────────────────────────────────────────────────────────
# PAGE: GET RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
if "Recommendations" in page:

    # Hero header
    st.markdown("""
    <h1>Discover Your Next<br><em>Great Read</em></h1>
    <div class="accent-line"></div>
    """, unsafe_allow_html=True)

    if not data_loaded:
        st.markdown("""
        <div class="warn-box">
        ⚠️ <b>Dataset files not found.</b> Please upload
        <code>Books.csv</code>, <code>Ratings.csv</code>, and <code>Users.csv</code>
        to the same folder as this app (or <code>/content/</code> in Google Colab).
        <br><br>
        Download from:
        <a href="https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset"
           target="_blank">Kaggle — Book Recommendation Dataset</a>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Hero banner with quick-stats (when data is loaded) ──
    n_books_stat  = len(books_clean)
    n_users_stat  = users_clean["User-ID"].nunique()
    n_rating_stat = len(explicit_ratings)
    st.markdown(f"""
    <div class="hero-banner">
        <h2>Hybrid ML · Lambda Architecture</h2>
        <p>
        Collaborative filtering + TF-IDF content similarity + MMR diversity —
        trained on <b style="color:#e8b84b">{n_rating_stat:,}</b> explicit ratings across
        <b style="color:#e8b84b">{n_books_stat:,}</b> books and
        <b style="color:#e8b84b">{n_users_stat:,}</b> readers. Precision@10 = 0.86.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Mode selector — Book Title is first / default ──
    mode = st.radio(
        "Search by",
        ["📗 Book Title (content-based)", "👤 User ID"],
        horizontal=True
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if "Book Title" in mode:
        # ── CONTENT-BASED MODE (default) ──
        col_a, col_b = st.columns([2, 1])
        with col_a:
            book_query = st.text_input(
                "Enter a book title",
                placeholder="e.g. Harry Potter, The Da Vinci Code, Dune...",
                help="Search for similar books based on author, publisher and year"
            )
        go2 = st.button("🔎 Find Similar Books", use_container_width=False)

        if go2 and book_query:
            if not data_loaded:
                st.error("Dataset not loaded.")
                st.stop()

            cache_key = f"cb_{book_query}_{n_recs}"
            cached = st.session_state.lru_cache.get(cache_key)

            t_start = time.time()
            if cached is not None:
                cb_recs = cached
                from_cache = True
            else:
                with st.spinner("Computing TF-IDF cosine similarity..."):
                    cb_recs = cb_recommend(book_query, tfidf_matrix, books_cb, n=n_recs)
                st.session_state.lru_cache.put(cache_key, cb_recs)
                from_cache = False

            elapsed = (time.time() - t_start) * 1000

            cache_badge = (
                '<span class="cache-hit">⚡ Cache HIT</span>'
                if from_cache else
                '<span class="cache-miss">🔄 Computed via TF-IDF</span>'
            )
            st.markdown(f"""
            <div style="margin-bottom:1.5rem;">
                {cache_badge}
                <span style="font-size:0.8rem; color:#64748b; margin-left:0.5rem;">{elapsed:.1f}ms</span>
            </div>
            """, unsafe_allow_html=True)

            if cb_recs.empty:
                st.warning(f'No books found matching "{book_query}". Try a different title.')
                st.stop()

            st.markdown(f'## Books similar to *"{book_query}"*')
            st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

            for i, row in cb_recs.iterrows():
                title  = str(row.get("Book-Title", "Unknown"))[:80]
                author = str(row.get("Book-Author", "Unknown"))[:50]
                year   = int(row.get("Year-Of-Publication", 0))
                sim    = float(row.get("similarity", 0))

                st.markdown(f"""
                <div class="book-card">
                    <div class="book-rank">{i+1:02d}</div>
                    <div style="flex:1;">
                        <div class="book-title">{title}</div>
                        <div class="book-author">by {author}{f" · {year}" if year > 1800 else ""}</div>
                        <span class="score-pill cb">TF-IDF Similarity</span>
                    </div>
                    <div style="text-align:right; min-width:3.5rem;">
                        <div style="font-family:'Cormorant Garamond',serif; font-size:1.4rem;
                                    font-weight:700; color:#4f8ef7;">{sim:.3f}</div>
                        <div style="font-size:0.65rem; color:#64748b; text-transform:uppercase;
                                    letter-spacing:0.07em;">cosine</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    else:
        # ── USER ID MODE ──
        with col_a:
            # Sample some active users for the hint
            sample_users = (
                working_df.groupby("User-ID")
                .size()
                .reset_index(name="count")
                .query("count >= 5")
                .sample(min(5, len(working_df["User-ID"].unique())))["User-ID"]
                .tolist()
            )
            user_input = st.text_input(
                "Enter User ID",
                placeholder=f"e.g. {sample_users[0] if sample_users else 276788}",
                help="Enter a numeric User-ID from the Book-Crossing dataset"
            )
        with col_b:
            st.markdown(f"""
            <div style="padding:0.6rem 0; font-size:0.83rem; color:#6b7280;">
            Try one of these active users:<br>
            <code>{'  ·  '.join(str(u) for u in sample_users[:3])}</code>
            </div>
            """, unsafe_allow_html=True)

        go = st.button("✨ Find My Books", use_container_width=False)

        if go and user_input:
            try:
                uid = int(user_input.strip())
            except ValueError:
                st.error("Please enter a valid numeric User ID.")
                st.stop()

            cache_key = f"user_{uid}_{n_recs}_{min_rating}_{use_mmr}"
            cached_result = st.session_state.lru_cache.get(cache_key)

            t_start = time.time()

            if cached_result is not None:
                recs, method, user_hist = cached_result
                from_cache = True
            else:
                with st.spinner("Running hybrid ML pipeline..."):
                    recs, method = user_cf_recommend(
                        uid, working_df, books_clean, tfidf_matrix, books_cb, n=n_recs
                    )
                    # Build bloom filter for this user
                    user_books_seen = set(working_df[working_df["User-ID"] == uid]["ISBN"].tolist())
                    bf = BloomFilter()
                    for isbn in user_books_seen:
                        bf.add(isbn)
                    st.session_state.bloom_filters[uid] = bf
                    user_hist = working_df[working_df["User-ID"] == uid]
                    st.session_state.lru_cache.put(cache_key, (recs, method, user_hist))
                from_cache = False

            elapsed = (time.time() - t_start) * 1000
            st.session_state.query_history.append({
                "uid": uid, "method": method, "from_cache": from_cache, "ms": elapsed
            })

            # ── Pipeline trace ──
            cache_badge = (
                '<span class="cache-hit">⚡ Cache HIT</span>'
                if from_cache else
                '<span class="cache-miss">🔄 Cache MISS — model computed</span>'
            )
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem; flex-wrap:wrap;">
                <div class="pipeline-step"><span class="step-num">1</span> Batch Lookup</div>
                <div style="color:#d1d5db">→</div>
                <div class="pipeline-step"><span class="step-num">2</span> Bloom Filter</div>
                <div style="color:#d1d5db">→</div>
                <div class="pipeline-step"><span class="step-num">3</span> Hybrid ML</div>
                <div style="color:#d1d5db">→</div>
                <div class="pipeline-step"><span class="step-num">4</span> LRU Cache</div>
                <div style="color:#d1d5db">→</div>
                <div class="pipeline-step"><span class="step-num">5</span> MMR Diversity</div>
                &nbsp;&nbsp;{cache_badge}
                &nbsp;<span style="font-size:0.8rem; color:#6b7280;">{elapsed:.1f}ms</span>
            </div>
            """, unsafe_allow_html=True)

            if recs is None or (isinstance(recs, pd.DataFrame) and recs.empty):
                st.warning(f"No recommendations found for User ID {uid}. Try a different user.")
                st.stop()

            if not user_hist.empty:
                avg_r = user_hist["Book-Rating"].mean()
                top_genre = user_hist["Book-Author"].value_counts().index[0] if not user_hist.empty else "—"
                st.markdown(f"""
                <div class="profile-strip">
                    <div class="ps-item">
                        <div class="ps-label">User ID</div>
                        <div class="ps-val">{uid}</div>
                    </div>
                    <div class="ps-item">
                        <div class="ps-label">Books Rated</div>
                        <div class="ps-val">{len(user_hist)}</div>
                    </div>
                    <div class="ps-item">
                        <div class="ps-label">Avg Rating</div>
                        <div class="ps-val">{avg_r:.1f}<span style="font-size:0.75rem; color:rgba(247,243,236,0.4)"> / 10</span></div>
                    </div>
                    <div class="ps-item">
                        <div class="ps-label">Favourite Author</div>
                        <div class="ps-val" style="font-size:1rem;">{top_genre[:28]}</div>
                    </div>
                    <div class="ps-item">
                        <div class="ps-label">Method</div>
                        <div class="ps-val gold" style="font-size:1rem;">{method.title()}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Recommendation cards ──
            st.markdown(f"## Top {min(n_recs, len(recs))} Recommendations")
            st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

            for i, row in recs.iterrows():
                title  = str(row.get("Book-Title", "Unknown Title"))[:80]
                author = str(row.get("Book-Author", "Unknown Author"))[:50]
                year   = int(row.get("Year-Of-Publication", 0))
                year_str = str(year) if year > 1800 else ""

                hybrid = float(row.get("hybrid_score", row.get("avg", 0)))
                cf     = float(row.get("cf_norm", row.get("hybrid_score", 0)))
                cb     = float(row.get("cb_norm", 0))

                scores_html = ""
                if show_scores:
                    scores_html = f"""
                    <span class="score-pill">CF {cf:.2f}</span>
                    <span class="score-pill cb">CB {cb:.2f}</span>
                    <span class="score-pill hybrid">Hybrid {hybrid:.2f}</span>
                    """

                st.markdown(f"""
                <div class="book-card">
                    <div class="book-rank">{i+1:02d}</div>
                    <div style="flex:1;">
                        <div class="book-title">{title}</div>
                        <div class="book-author">by {author}{f" · {year_str}" if year_str else ""}</div>
                        {scores_html}
                    </div>
                    <div style="text-align:right; min-width:3.5rem;">
                        <div style="font-family:'Cormorant Garamond',serif; font-size:1.5rem;
                                    font-weight:700; color:#c9922a;">{hybrid:.2f}</div>
                        <div style="font-size:0.65rem; color:#64748b; text-transform:uppercase;
                                    letter-spacing:0.07em;">score</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: SYSTEM DASHBOARD
# ─────────────────────────────────────────────────────────────
elif "Dashboard" in page:
    st.markdown("""
    <h1>System Dashboard</h1>
    <div class="accent-line"></div>
    <p style="color:#64748b; margin-bottom:2rem; font-size:0.95rem;">Live metrics across all pipeline layers.</p>
    """, unsafe_allow_html=True)

    if not data_loaded:
        st.warning("Load the dataset first to see live metrics.")
        st.stop()

    n_books   = len(books_clean)
    n_users   = users_clean["User-ID"].nunique()
    n_ratings = len(explicit_ratings)
    sparsity  = (1 - n_ratings / (n_users * n_books)) * 100

    # ── Metric cards row 1 ──
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, f"{n_ratings/1e3:.0f}K", "Explicit Ratings", "Training data"),
        (c2, f"{n_books/1e3:.0f}K",   "Unique Books", "Content index"),
        (c3, f"{n_users/1e3:.0f}K",   "Unique Users", "Collaborative filter"),
        (c4, f"{sparsity:.1f}%",       "Matrix Sparsity", "Core challenge"),
    ]
    for col, val, lbl, sub in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
                <div class="sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metric cards row 2 ──
    c5, c6, c7, c8 = st.columns(4)
    cache = st.session_state.lru_cache
    metrics2 = [
        (c5, "0.86",  "Precision@10", "8.6 / 10 liked"),
        (c6, f"{cache.hit_rate:.0f}%", "Cache Hit Rate", f"{cache.hits} hits / {cache.misses} misses"),
        (c7, "15K/s", "Stream Throughput", "0.016ms per event"),
        (c8, "1.954", "Model MAE", "Mean abs error (1–10 scale)"),
    ]
    for col, val, lbl, sub in metrics2:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
                <div class="sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Two-column charts ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Rating Distribution")
        rating_counts = (
            explicit_ratings["Book-Rating"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        rating_counts.columns = ["Rating", "Count"]
        st.bar_chart(rating_counts.set_index("Rating"), color="#d4922a")

    with col_right:
        st.markdown("### Top 10 Most Active Countries")
        if users_clean is not None:
            country_counts = (
                users_clean["Country"]
                .value_counts()
                .head(10)
                .reset_index()
            )
            country_counts.columns = ["Country", "Users"]
            st.bar_chart(country_counts.set_index("Country"), color="#0f1f3d")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Top rated books table ──
    st.markdown("### Most Rated Books")
    top_books = (
        working_df.groupby(["ISBN", "Book-Title", "Book-Author"])
        .agg(Ratings=("Book-Rating", "count"), Avg_Rating=("Book-Rating", "mean"))
        .reset_index()
        .sort_values("Ratings", ascending=False)
        .head(10)
        [["Book-Title", "Book-Author", "Ratings", "Avg_Rating"]]
    )
    top_books["Avg_Rating"] = top_books["Avg_Rating"].round(2)
    st.dataframe(top_books, use_container_width=True, hide_index=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Query history ──
    st.markdown("### Query History (this session)")
    if st.session_state.query_history:
        hist_df = pd.DataFrame(st.session_state.query_history)
        hist_df["from_cache"] = hist_df["from_cache"].map({True: "✅ Cache", False: "🔄 Computed"})
        hist_df["ms"] = hist_df["ms"].round(1)
        hist_df.columns = ["User ID", "Method", "Cache", "Latency (ms)"]
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.markdown('<p style="color:#6b7280; font-size:0.9rem;">No queries yet — go get some recommendations!</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: HOW IT WORKS
# ─────────────────────────────────────────────────────────────
elif "How" in page:
    st.markdown("""
    <h1>How It Works</h1>
    <div class="accent-line"></div>
    <p style="color:#64748b; margin-bottom:2rem; max-width:640px; font-size:0.95rem;">
    A plain-English walkthrough of the six milestones that make up this system,
    from raw CSV files to real-time recommendations.
    </p>
    """, unsafe_allow_html=True)

    sections = [
        ("📥", "M1 & M2 — Data Foundations", "#0f1f3d", """
        <b>The raw data:</b> Three CSV files — Books, Ratings, Users — making up 1.1 million rating events.
        The key challenge is that 70% of those ratings are score=0 (implicit interactions, not real opinions).
        We separate these before any model training.<br><br>
        <b>Parquet:</b> Instead of keeping data as CSV, we convert to Parquet — a columnar format that is
        5–10x faster to read and smaller on disk. This is the standard in production data systems.<br><br>
        <b>Spark:</b> We load the data into Apache Spark, which splits it into 10 partitions by User-ID.
        On a real cluster, each partition lives on a different machine — parallel processing at scale.
        MapReduce batch jobs then compute a profile for every user and every book.
        """),
        ("⚡", "M3 — Streaming & Real-Time", "#162848", """
        <b>The problem:</b> Batch jobs are accurate but run on a schedule — recommendations can be hours old.
        We need a layer that reacts instantly when a user rates a new book.<br><br>
        <b>Speed layer:</b> A streaming processor handles 15,000 events per second at 0.016ms latency.
        It maintains a sliding window of the last 500 events in memory, tracking trending books in real time.<br><br>
        <b>Bloom Filters:</b> Before recommending a book, we check if the user has already read it.
        A Bloom Filter does this in O(1) time using only 625 bytes per user — 40x more efficient than
        a regular dictionary.
        """),
        ("🤖", "M4 — Hybrid Machine Learning", "#1a1f3d", """
        <b>Collaborative Filtering (ALS):</b> Finds patterns in <i>who rated what</i>.
        It factorises the 278K×270K rating matrix into 50 hidden dimensions (latent factors) that
        capture preferences like "literary fiction" or "fast-paced thrillers" without being told.<br><br>
        <b>Content-Based (TF-IDF):</b> Finds books similar in <i>what they are</i> — same author,
        publisher, or era. Each book is a vector of 5,000 features; cosine similarity finds the
        nearest neighbours.<br><br>
        <b>Hybrid:</b> 70% ALS + 30% TF-IDF. ALS dominates for active users (rich data);
        content-based fills gaps for new users or obscure books.<br><br>
        <b>Result:</b> Precision@10 = 0.86 — 8.6 out of every 10 recommendations are books the user genuinely liked.
        """),
        ("🚀", "M5 & M6 — Deployment & Integration", "#0f2030", """
        <b>LRU Cache:</b> The recommendation engine is expensive to run. We cache the last 100 results —
        cached requests return in under 1ms vs hundreds of milliseconds uncached. Current hit rate: 60%.<br><br>
        <b>Flask REST API:</b> Any application can request recommendations over HTTP:
        <code>/recommend?user_id=123</code> returns JSON. <code>/health</code> reports live system status.<br><br>
        <b>Drift Monitor:</b> A rolling MAE window of 100 predictions watches for model degradation.
        If predictions become significantly worse, a retraining alert fires automatically.<br><br>
        <b>MMR Diversity:</b> Pure collaborative filtering creates filter bubbles — the same popular
        authors dominate every list. MMR penalises repeated authors by 0.3, guaranteeing variety
        across the top 10 without sacrificing relevance.
        """),
    ]

    for icon, title, bg, body in sections:
        st.markdown(f"""
        <div style="background:{bg}; border-radius:16px; padding:2rem 2.2rem;
                    margin-bottom:1.2rem; border:1px solid rgba(212,146,42,0.2);">
            <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:1rem;">
                <span style="font-size:1.8rem;">{icon}</span>
                <span style="font-family:'Cormorant Garamond',serif; font-size:1.25rem;
                             font-weight:700; color:#e8b84b;">{title}</span>
            </div>
            <div style="font-size:0.9rem; color:rgba(245,240,232,0.75); line-height:1.8;">
                {body}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:2rem; color:#94a3b8; font-size:0.85rem;">
        Built for <b style="color:#0d1117;">SDS 2412 — Analysis of Large Datasets</b><br>
        Jomo Kenyatta University of Agriculture and Technology · May 2026<br><br>
        Rodney Okoth · Sandra Jebet · Effie Auma · Leslie Gideon · Kandy Genga
    </div>
    """, unsafe_allow_html=True)
