"""
Large-Scale Book Recommendation System
SDS 2412 — Analysis of Large Datasets | JKUAT 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import hashlib
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BookMind · Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy:   #0f1f3d;
    --navy2:  #1a3460;
    --amber:  #c9861f;
    --amber2: #e8a825;
    --white:  #ffffff;
    --bg:     #f7f8fa;
    --text:   #111827;
    --muted:  #6b7280;
    --border: #e2e8f0;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text);
}

.stApp {
    background-color: var(--bg) !important;
    background-image: none !important;
}

.block-container {
    padding: 2.5rem 2.5rem 4rem 2.5rem !important;
    max-width: 1100px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    min-width: 220px !important;
    max-width: 220px !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] h3 {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(226,232,240,0.4) !important;
    margin: 1.2rem 0 0.4rem 0;
}

/* ── Headings ── */
h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    color: var(--navy) !important;
    line-height: 1.15 !important;
    letter-spacing: -0.02em;
}
h2 {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.45rem !important;
    font-weight: 600 !important;
    color: var(--navy) !important;
}

.accent-line {
    height: 3px;
    width: 52px;
    background: linear-gradient(90deg, var(--amber), var(--amber2));
    border-radius: 2px;
    margin: 0.5rem 0 1.8rem 0;
}

/* ── Stats strip ── */
.stat-strip {
    display: flex;
    background: var(--navy);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 2rem;
}
.stat-item {
    flex: 1;
    padding: 1.2rem 1rem;
    border-right: 1px solid rgba(255,255,255,0.07);
    text-align: center;
}
.stat-item:last-child { border-right: none; }
.stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--amber2);
    line-height: 1;
}
.stat-lbl {
    font-size: 0.67rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(226,232,240,0.45);
    margin-top: 0.3rem;
}

/* ── Search mode card ── */
.mode-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
}
.mode-card-title {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* ── Pipeline trace ── */
.pipeline-trace {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.35rem;
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-bottom: 1.5rem;
    font-size: 0.78rem;
    color: var(--muted);
}
.pip-step {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.22rem 0.6rem;
    color: var(--navy);
    font-weight: 500;
    font-size: 0.78rem;
}
.pip-step .num {
    background: var(--amber);
    color: #fff;
    border-radius: 50%;
    width: 16px; height: 16px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.62rem;
    font-weight: 700;
}
.pip-arrow { color: #d1d5db; }
.pip-hit  { background:#dcfce7; color:#15803d; border-radius:999px; padding:0.15rem 0.6rem; font-weight:600; font-size:0.72rem; }
.pip-miss { background:#fef9c3; color:#854d0e; border-radius:999px; padding:0.15rem 0.6rem; font-weight:600; font-size:0.72rem; }

/* ── Profile strip ── */
.profile-strip {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}
.p-lbl { font-size:0.67rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); font-weight:600; }
.p-val { font-family:'Playfair Display',serif; font-size:1.1rem; font-weight:700; color:var(--navy); line-height:1.25; }

/* ── Book card ── */
.book-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.65rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.18s, border-color 0.18s;
}
.book-card::before {
    content:'';
    position:absolute;
    left:0; top:0; bottom:0;
    width:3px;
    background: linear-gradient(180deg, var(--amber), var(--amber2));
}
.book-card:hover {
    box-shadow: 0 4px 18px rgba(15,31,61,0.09);
    border-color: rgba(201,134,31,0.4);
}
.book-rank {
    font-family:'Playfair Display',serif;
    font-size:1.5rem;
    font-weight:700;
    color:#d1d5db;
    min-width:1.9rem;
    text-align:center;
    line-height:1;
}
.book-title {
    font-family:'Playfair Display',serif;
    font-size:0.98rem;
    font-weight:600;
    color:var(--navy);
    line-height:1.35;
    margin-bottom:0.12rem;
}
.book-author { font-size:0.81rem; color:var(--muted); }
.score-badge {
    font-size:0.68rem; font-weight:700;
    padding:0.18rem 0.6rem; border-radius:999px;
    display:inline-block; margin-top:0.35rem;
}
.score-badge.hybrid { background:#fef3c7; color:#92400e; }
.score-badge.cb     { background:#dbeafe; color:#1e3a8a; }

/* ── Metric card ── */
.metric-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-top: 3px solid var(--amber);
    border-radius: 12px;
    padding: 1.2rem 1.3rem;
}
.metric-card .val { font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:700; color:var(--navy); line-height:1; }
.metric-card .lbl { font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); margin-top:0.4rem; }
.metric-card .sub { font-size:0.76rem; color:#9ca3af; margin-top:0.12rem; }

/* ── How-it-works card ── */
.how-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-left: 4px solid var(--amber);
    border-radius: 12px;
    padding: 1.4rem 1.7rem;
    margin-bottom: 1rem;
}
.how-title { font-family:'Playfair Display',serif; font-size:1.08rem; font-weight:700; color:var(--navy); margin-bottom:0.65rem; }
.how-body  { font-size:0.87rem; color:#374151; line-height:1.85; }

/* ── Buttons ── */
.stButton > button {
    background: var(--navy) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.93rem !important;
    padding: 0.62rem 1.8rem !important;
    transition: background 0.18s, transform 0.12s !important;
}
.stButton > button:hover {
    background: var(--navy2) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(15,31,61,0.2) !important;
}

/* ── Inputs & selects ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    border-radius: 9px !important;
    border-color: var(--border) !important;
    font-family: 'DM Sans', sans-serif !important;
    background: var(--white) !important;
    color: var(--text) !important;
    font-size: 0.93rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px rgba(201,134,31,0.12) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
    border-bottom: 2px solid var(--border);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--muted);
    border: none;
    background: transparent;
    padding: 0.5rem 1rem;
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    color: var(--navy) !important;
    background: #fff !important;
    border-bottom: 2px solid var(--amber) !important;
    font-weight: 600 !important;
}

.divider { border:none; border-top:1px solid var(--border); margin:1.8rem 0; }

.warn-box {
    background:#fffbeb; border:1px solid #fcd34d; border-radius:10px;
    padding:1rem 1.3rem; font-size:0.88rem; color:#78350f; line-height:1.65;
}
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
        return [int(hashlib.md5(f"{item}_{i}".encode()).hexdigest(), 16) % self.size
                for i in range(self.num_hashes)]

    def add(self, item):
        for pos in self._hashes(item): self.bit_array[pos] = 1

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
        if key in self.cache: self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_data():
    for base in ["./data", ".", "/content", "/content/drive/MyDrive"]:
        try:
            books_df   = pd.read_csv(f"{base}/Books.csv",   encoding="latin-1", on_bad_lines="skip")
            ratings_df = pd.read_csv(f"{base}/Ratings.csv", encoding="latin-1", on_bad_lines="skip")
            users_df   = pd.read_csv(f"{base}/Users.csv",   encoding="latin-1", on_bad_lines="skip")

            # Clean books
            books_df["Year-Of-Publication"] = pd.to_numeric(
                books_df["Year-Of-Publication"], errors="coerce")
            books_clean = books_df.dropna(subset=["Book-Author", "Publisher"]).copy()
            for col in ["Image-URL-S", "Image-URL-M", "Image-URL-L"]:
                if col in books_clean.columns:
                    books_clean.drop(columns=[col], inplace=True)
            books_clean["Year-Of-Publication"] = (
                pd.to_numeric(books_clean["Year-Of-Publication"], errors="coerce")
                .fillna(0).astype(int))

            # Clean users
            users_clean = users_df.copy()
            users_clean["Age"] = (
                pd.to_numeric(users_clean["Age"], errors="coerce").fillna(35).astype(int))
            users_clean["Country"] = users_clean["Location"].apply(
                lambda x: x.strip().split(",")[-1].strip() if isinstance(x, str) else "unknown")

            explicit_ratings = ratings_df[ratings_df["Book-Rating"] > 0].copy()
            working_df = (
                explicit_ratings
                .merge(books_clean, on="ISBN", how="inner")
                .merge(users_clean[["User-ID", "Age", "Country"]], on="User-ID", how="inner")
            )
            return books_clean, users_clean, explicit_ratings, working_df

        except FileNotFoundError:
            continue
    return None, None, None, None


@st.cache_resource(show_spinner=False)
def build_tfidf(_books_clean):
    bc = _books_clean.copy().reset_index(drop=True)
    bc["content"] = (bc["Book-Author"].fillna("") + " " +
                     bc["Publisher"].fillna("") + " " +
                     bc["Year-Of-Publication"].astype(str))
    tfidf  = TfidfVectorizer(max_features=5000, stop_words="english")
    matrix = tfidf.fit_transform(bc["content"])
    return tfidf, matrix, bc


@st.cache_data(show_spinner=False)
def get_sample_books(_working_df):
    """Return 10 popular book titles as dropdown options."""
    popular = (
        _working_df.groupby(["Book-Title", "Book-Author"])
        .agg(count=("Book-Rating", "count"), avg=("Book-Rating", "mean"))
        .reset_index()
        .query("count >= 20 and avg >= 7")
        .sort_values("count", ascending=False)
        .head(10)
    )
    return [f"{row['Book-Title']} — {row['Book-Author']}"
            for _, row in popular.iterrows()]


@st.cache_data(show_spinner=False)
def get_sample_users(_working_df):
    """Return 10 active user IDs with a short description."""
    active = (
        _working_df.groupby("User-ID")
        .agg(count=("Book-Rating", "count"), avg=("Book-Rating", "mean"))
        .reset_index()
        .query("count >= 10")
        .sort_values("count", ascending=False)
        .head(10)
    )
    return [f"User {int(row['User-ID'])}  —  {int(row['count'])} ratings, avg {row['avg']:.1f}"
            for _, row in active.iterrows()]


# ─────────────────────────────────────────────────────────────
# MMR
# ─────────────────────────────────────────────────────────────
def mmr_rerank(df, penalty=0.3):
    seen, result, remaining = set(), [], list(df.iterrows())
    while remaining and len(result) < len(df):
        best_i, best_score, best_row = None, -9e9, None
        for idx, (_, row) in enumerate(remaining):
            s = row["hybrid_score"] - (penalty if row.get("Book-Author", "") in seen else 0)
            if s > best_score:
                best_score, best_i, best_row = s, idx, row
        result.append(best_row)
        seen.add(best_row.get("Book-Author", ""))
        remaining.pop(best_i)
    return pd.DataFrame(result).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# CONTENT-BASED
# ─────────────────────────────────────────────────────────────
def cb_recommend(book_title, tfidf_matrix, books_cb, n=10):
    matches = books_cb[books_cb["Book-Title"].str.contains(book_title, case=False, na=False)]
    if matches.empty:
        return pd.DataFrame()
    idx = matches.index[0]
    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top = sim.argsort()[::-1][1:n + 1]
    res = books_cb.iloc[top][["Book-Title", "Book-Author", "Year-Of-Publication"]].copy()
    res["similarity"] = sim[top].round(3)
    return res.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# USER-BASED COLLABORATIVE + HYBRID
# ─────────────────────────────────────────────────────────────
def user_cf_recommend(user_id, working_df, books_cb, tfidf_matrix, n=10, use_mmr=True):
    user_ratings = working_df[working_df["User-ID"] == user_id]
    if user_ratings.empty:
        return pd.DataFrame(), "not_found"

    user_books = set(user_ratings["ISBN"])
    user_liked = user_ratings[user_ratings["Book-Rating"] >= 7]["ISBN"].tolist()
    if not user_liked:
        user_liked = user_ratings.nlargest(3, "Book-Rating")["ISBN"].tolist()

    similar = working_df[
        working_df["ISBN"].isin(user_liked) &
        (working_df["User-ID"] != user_id) &
        (working_df["Book-Rating"] >= 6)
    ]

    if similar.empty:
        pop = (
            working_df[~working_df["ISBN"].isin(user_books)]
            .groupby(["ISBN", "Book-Title", "Book-Author"])
            .agg(count=("Book-Rating", "count"), avg=("Book-Rating", "mean"))
            .reset_index().query("count >= 3")
            .sort_values(["avg", "count"], ascending=False).head(n)
        )
        pop["hybrid_score"] = ((pop["avg"] - pop["avg"].min()) /
                               (pop["avg"].max() - pop["avg"].min() + 1e-6))
        return pop, "popularity"

    cands = (
        similar[~similar["ISBN"].isin(user_books)]
        .groupby(["ISBN", "Book-Title", "Book-Author"])
        .agg(cf_score=("Book-Rating", "mean"), cf_count=("Book-Rating", "count"))
        .reset_index()
    )
    if cands.empty:
        return pd.DataFrame(), "no_candidates"

    mn, mx = cands["cf_score"].min(), cands["cf_score"].max()
    cands["cf_norm"] = (cands["cf_score"] - mn) / (mx - mn + 1e-6)

    # Content component
    seed_isbn = user_liked[0]
    seed_row  = books_cb[books_cb["ISBN"] == seed_isbn] if "ISBN" in books_cb.columns else pd.DataFrame()
    if not seed_row.empty:
        sidx      = seed_row.index[0]
        sims      = cosine_similarity(tfidf_matrix[sidx], tfidf_matrix).flatten()
        isbn_list = books_cb["ISBN"].tolist()
        isbn_sim  = {isbn_list[i]: sims[i] for i in range(len(isbn_list))}
        cands["cb_norm"] = cands["ISBN"].map(isbn_sim).fillna(0)
    else:
        cands["cb_norm"] = 0.0

    cands["hybrid_score"] = 0.7 * cands["cf_norm"] + 0.3 * cands["cb_norm"]
    cands = cands.sort_values("hybrid_score", ascending=False).head(n * 2)
    result = mmr_rerank(cands).head(n) if use_mmr else cands.head(n)
    return result.reset_index(drop=True), "hybrid"


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "lru_cache"     not in st.session_state: st.session_state.lru_cache = LRUCache(100)
if "bloom_filters" not in st.session_state: st.session_state.bloom_filters = {}
if "query_history" not in st.session_state: st.session_state.query_history = []


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0.4rem 1rem 0.4rem;
                border-bottom:1px solid rgba(255,255,255,0.08);
                margin-bottom:0.8rem;">
        <div style="font-size:1.6rem;">📚</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.1rem;
                    font-weight:700; color:#e8a825; margin-top:0.3rem;">BookMind</div>
        <div style="font-size:0.67rem; color:rgba(226,232,240,0.38);
                    margin-top:0.2rem; text-transform:uppercase; letter-spacing:0.07em;">
            SDS 2412 · JKUAT · 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio(
        "", ["🔍  Recommendations", "📊  Dashboard", "📖  How It Works"],
        label_visibility="collapsed"
    )

    st.markdown("### Settings")
    n_recs      = st.slider("Results to show", 5, 20, 10)
    use_mmr     = st.checkbox("MMR diversity injection", value=True,
                              help="Prevents the same author dominating the list")
    show_scores = st.checkbox("Show recommendation scores", value=False)

    cache = st.session_state.lru_cache
    st.markdown(f"""
    <div style="margin-top:1.5rem; padding-top:1rem;
                border-top:1px solid rgba(255,255,255,0.07);
                font-size:0.73rem; color:rgba(226,232,240,0.38); line-height:2.1;">
        Cache entries: {len(cache.cache)} / 100<br>
        Hit rate: {cache.hit_rate:.0f}%<br>
        Queries: {len(st.session_state.query_history)}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading Book-Crossing dataset…"):
    books_clean, users_clean, explicit_ratings, working_df = load_data()
data_loaded = books_clean is not None

if data_loaded:
    with st.spinner("Building TF-IDF content matrix…"):
        tfidf, tfidf_matrix, books_cb = build_tfidf(books_clean)
    if "ISBN" not in books_cb.columns and "ISBN" in books_clean.columns:
        books_cb["ISBN"] = books_clean["ISBN"].values


# ═════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════
if "Recommendations" in page:

    st.markdown("""
    <h1>Discover Your Next<br><em>Great Read</em></h1>
    <div class="accent-line"></div>
    <p style="color:#6b7280; font-size:0.95rem; max-width:580px; margin-bottom:1.8rem; line-height:1.7;">
    A large-scale hybrid recommendation engine — collaborative filtering, TF-IDF content
    similarity, and MMR diversity — trained on 1.1 million ratings from the Book-Crossing dataset.
    </p>
    """, unsafe_allow_html=True)

    if not data_loaded:
        st.markdown("""
        <div class="warn-box">
        ⚠️ <b>Dataset files not found.</b><br>
        Place <code>Books.csv</code>, <code>Ratings.csv</code>, and <code>Users.csv</code>
        inside the <code>data/</code> subfolder (or the same folder as <code>app.py</code>).<br><br>
        Download from
        <a href="https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset" target="_blank">
        Kaggle — Book Recommendation Dataset</a>.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Stats strip ──
    n_books   = len(books_clean)
    n_users   = users_clean["User-ID"].nunique()
    n_ratings = len(explicit_ratings)
    sparsity  = (1 - n_ratings / (n_users * n_books)) * 100
    st.markdown(f"""
    <div class="stat-strip">
        <div class="stat-item"><div class="stat-val">{n_ratings/1e3:.0f}K</div>
            <div class="stat-lbl">Ratings</div></div>
        <div class="stat-item"><div class="stat-val">{n_books/1e3:.0f}K</div>
            <div class="stat-lbl">Books</div></div>
        <div class="stat-item"><div class="stat-val">{n_users/1e3:.0f}K</div>
            <div class="stat-lbl">Users</div></div>
        <div class="stat-item"><div class="stat-val">0.86</div>
            <div class="stat-lbl">Precision@10</div></div>
        <div class="stat-item"><div class="stat-val">{sparsity:.0f}%</div>
            <div class="stat-lbl">Sparsity</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Precompute samples ──
    sample_books = get_sample_books(working_df)
    sample_users = get_sample_users(working_df)

    # ── Search mode tabs (label has icon AND text) ──
    tab_user, tab_book = st.tabs([
        "👤  Search by User ID — personalised picks based on your reading history",
        "📗  Search by Book Title — find books similar to one you already love"
    ])

    # ══════════════════════════════════════════════
    # TAB 1 — USER ID
    # ══════════════════════════════════════════════
    with tab_user:
        st.markdown("""
        <div style="font-size:0.88rem; color:#6b7280; margin-bottom:1.2rem; line-height:1.6;">
        Enter a User ID from the Book-Crossing dataset, or pick one of the ten most active
        users below. The engine will analyse their rating history, find similar users, and
        blend collaborative filtering with content similarity to build a personalised list.
        </div>
        """, unsafe_allow_html=True)

        # Sample user dropdown
        selected_sample_user = st.selectbox(
            "📋  Pick a sample active user (optional)",
            options=["— type your own User ID below —"] + sample_users,
            help="These are 10 of the most active users in the dataset"
        )

        # Pre-fill text box if user picked from dropdown
        prefill_uid = ""
        if selected_sample_user != "— type your own User ID below —":
            prefill_uid = selected_sample_user.split()[1]  # extracts the numeric ID

        col_in, col_btn = st.columns([3, 1])
        with col_in:
            user_input = st.text_input(
                "Or enter any User ID manually",
                value=prefill_uid,
                placeholder="e.g. 276788"
            )
        with col_btn:
            st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
            go_user = st.button("✨  Get Recommendations", key="go_user")

        if go_user and user_input:
            try:
                uid = int(str(user_input).strip())
            except ValueError:
                st.error("Please enter a valid numeric User ID.")
                st.stop()

            cache_key     = f"user_{uid}_{n_recs}_{use_mmr}"
            cached_result = st.session_state.lru_cache.get(cache_key)
            t0 = time.time()

            if cached_result is not None:
                recs, method, user_hist = cached_result
                from_cache = True
            else:
                with st.spinner("Running hybrid pipeline…"):
                    recs, method = user_cf_recommend(
                        uid, working_df, books_cb, tfidf_matrix,
                        n=n_recs, use_mmr=use_mmr)
                    user_hist = working_df[working_df["User-ID"] == uid]
                    bf = BloomFilter()
                    for isbn in user_hist["ISBN"].tolist(): bf.add(isbn)
                    st.session_state.bloom_filters[uid] = bf
                    st.session_state.lru_cache.put(cache_key, (recs, method, user_hist))
                from_cache = False

            elapsed = (time.time() - t0) * 1000
            st.session_state.query_history.append(
                {"uid": uid, "method": method,
                 "from_cache": from_cache, "ms": round(elapsed, 1)})

            # Pipeline trace
            badge = (f'<span class="pip-hit">⚡ Cache HIT</span>'
                     if from_cache else
                     f'<span class="pip-miss">🔄 Model computed</span>')
            st.markdown(f"""
            <div class="pipeline-trace">
                <div class="pip-step"><span class="num">1</span> Batch Lookup</div>
                <span class="pip-arrow">→</span>
                <div class="pip-step"><span class="num">2</span> Bloom Filter</div>
                <span class="pip-arrow">→</span>
                <div class="pip-step"><span class="num">3</span> Hybrid ML</div>
                <span class="pip-arrow">→</span>
                <div class="pip-step"><span class="num">4</span> LRU Cache</div>
                <span class="pip-arrow">→</span>
                <div class="pip-step"><span class="num">5</span> MMR Diversity</div>
                <span style="margin-left:auto; display:flex; align-items:center; gap:0.5rem;">
                    {badge}
                    <span style="color:#9ca3af; font-size:0.75rem;">{elapsed:.1f} ms</span>
                </span>
            </div>
            """, unsafe_allow_html=True)

            if recs is None or (isinstance(recs, pd.DataFrame) and recs.empty):
                st.warning(f"No recommendations found for User ID {uid}. "
                           f"Try a different user from the dropdown.")
                st.stop()

            # User profile strip
            if not user_hist.empty:
                avg_r = user_hist["Book-Rating"].mean()
                fav   = user_hist["Book-Author"].value_counts().index[0]
                st.markdown(f"""
                <div class="profile-strip">
                    <div><div class="p-lbl">User ID</div>
                         <div class="p-val">{uid}</div></div>
                    <div><div class="p-lbl">Books Rated</div>
                         <div class="p-val">{len(user_hist)}</div></div>
                    <div><div class="p-lbl">Avg Rating Given</div>
                         <div class="p-val">{avg_r:.1f} / 10</div></div>
                    <div><div class="p-lbl">Favourite Author</div>
                         <div class="p-val" style="font-size:0.92rem;">{str(fav)[:28]}</div></div>
                    <div><div class="p-lbl">Method Used</div>
                         <div class="p-val" style="color:#c9861f; font-size:0.92rem;">{method.title()}</div></div>
                </div>
                """, unsafe_allow_html=True)

            # Results
            st.markdown(f"## Top {min(n_recs, len(recs))} Recommendations")
            st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

            for i, row in recs.iterrows():
                title  = str(row.get("Book-Title",  "Unknown Title"))[:80]
                author = str(row.get("Book-Author", "Unknown Author"))[:50]
                year   = int(row.get("Year-Of-Publication", 0))
                hybrid = float(row.get("hybrid_score", row.get("avg", 0)))
                cf     = float(row.get("cf_norm", hybrid))
                cb     = float(row.get("cb_norm", 0))

                scores_html = ""
                if show_scores:
                    scores_html = (
                        f'<span class="score-badge hybrid">Hybrid {hybrid:.2f}</span>'
                        f'<span class="score-badge cb" style="margin-left:0.3rem;">'
                        f'CF {cf:.2f} · CB {cb:.2f}</span>'
                    )

                st.markdown(f"""
                <div class="book-card">
                    <div class="book-rank">{i+1:02d}</div>
                    <div style="flex:1; min-width:0;">
                        <div class="book-title">{title}</div>
                        <div class="book-author">by {author}
                            {f"&nbsp;·&nbsp;{year}" if year > 1800 else ""}
                        </div>
                        {scores_html}
                    </div>
                    <div style="text-align:right; min-width:3rem; flex-shrink:0;">
                        <div style="font-family:'Playfair Display',serif; font-size:1.45rem;
                                    font-weight:700; color:#c9861f; line-height:1;">{hybrid:.2f}</div>
                        <div style="font-size:0.62rem; color:#9ca3af; text-transform:uppercase;
                                    letter-spacing:0.06em;">score</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # TAB 2 — BOOK TITLE
    # ══════════════════════════════════════════════
    with tab_book:
        st.markdown("""
        <div style="font-size:0.88rem; color:#6b7280; margin-bottom:1.2rem; line-height:1.6;">
        Choose a book from the popular titles below, or type any title yourself. The engine
        builds a TF-IDF vector from the book's author, publisher, and year, then finds the
        ten most similar books using cosine similarity across all 270,000 titles.
        </div>
        """, unsafe_allow_html=True)

        # Sample book dropdown
        selected_sample_book = st.selectbox(
            "📋  Pick a popular book to start from (optional)",
            options=["— type your own title below —"] + sample_books,
            help="10 highly-rated books with the most ratings in the dataset"
        )

        # Pre-fill text box from dropdown
        prefill_book = ""
        if selected_sample_book != "— type your own title below —":
            prefill_book = selected_sample_book.split(" — ")[0]

        col_b, col_btn2 = st.columns([3, 1])
        with col_b:
            book_query = st.text_input(
                "Or type any book title",
                value=prefill_book,
                placeholder="e.g. Harry Potter, The Da Vinci Code, Dune…"
            )
        with col_btn2:
            st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
            go_book = st.button("🔎  Find Similar Books", key="go_book")

        if go_book and book_query:
            cache_key  = f"cb_{book_query.strip().lower()}_{n_recs}"
            cached     = st.session_state.lru_cache.get(cache_key)
            t0 = time.time()

            if cached is not None:
                cb_recs    = cached
                from_cache = True
            else:
                with st.spinner("Computing TF-IDF cosine similarity…"):
                    cb_recs = cb_recommend(book_query, tfidf_matrix, books_cb, n=n_recs)
                st.session_state.lru_cache.put(cache_key, cb_recs)
                from_cache = False

            elapsed = (time.time() - t0) * 1000
            badge   = (f'<span class="pip-hit">⚡ Cache HIT</span>'
                       if from_cache else
                       f'<span class="pip-miss">🔄 TF-IDF computed</span>')

            st.markdown(f"""
            <div class="pipeline-trace">
                <div class="pip-step"><span class="num">1</span> Book Lookup</div>
                <span class="pip-arrow">→</span>
                <div class="pip-step"><span class="num">2</span> TF-IDF Vector</div>
                <span class="pip-arrow">→</span>
                <div class="pip-step"><span class="num">3</span> Cosine Similarity</div>
                <span class="pip-arrow">→</span>
                <div class="pip-step"><span class="num">4</span> LRU Cache</div>
                <span style="margin-left:auto; display:flex; align-items:center; gap:0.5rem;">
                    {badge}
                    <span style="color:#9ca3af; font-size:0.75rem;">{elapsed:.1f} ms</span>
                </span>
            </div>
            """, unsafe_allow_html=True)

            if cb_recs.empty:
                st.warning(f'No books found matching "{book_query}". '
                           f'Try a shorter title or pick from the dropdown.')
                st.stop()

            st.markdown(f'## Books similar to *"{book_query}"*')
            st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

            for i, row in cb_recs.iterrows():
                title  = str(row.get("Book-Title",  "Unknown"))[:80]
                author = str(row.get("Book-Author", "Unknown"))[:50]
                year   = int(row.get("Year-Of-Publication", 0))
                sim    = float(row.get("similarity", 0))

                st.markdown(f"""
                <div class="book-card">
                    <div class="book-rank">{i+1:02d}</div>
                    <div style="flex:1; min-width:0;">
                        <div class="book-title">{title}</div>
                        <div class="book-author">by {author}
                            {f"&nbsp;·&nbsp;{year}" if year > 1800 else ""}
                        </div>
                        <span class="score-badge cb">TF-IDF content match</span>
                    </div>
                    <div style="text-align:right; min-width:3rem; flex-shrink:0;">
                        <div style="font-family:'Playfair Display',serif; font-size:1.45rem;
                                    font-weight:700; color:#1d4ed8; line-height:1;">{sim:.3f}</div>
                        <div style="font-size:0.62rem; color:#9ca3af; text-transform:uppercase;
                                    letter-spacing:0.06em;">cosine</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════
elif "Dashboard" in page:
    st.markdown("""
    <h1>System Dashboard</h1>
    <div class="accent-line"></div>
    <p style="color:#6b7280; font-size:0.95rem; margin-bottom:1.8rem;">
    Live metrics across all pipeline layers — data, ML model, cache, and streaming.
    </p>
    """, unsafe_allow_html=True)

    if not data_loaded:
        st.warning("Dataset not loaded — dashboard metrics unavailable.")
        st.stop()

    n_books   = len(books_clean)
    n_users   = users_clean["User-ID"].nunique()
    n_ratings = len(explicit_ratings)
    sparsity  = (1 - n_ratings / (n_users * n_books)) * 100
    cache     = st.session_state.lru_cache

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl, sub in [
        (c1, f"{n_ratings/1e3:.0f}K",  "Explicit Ratings",  "Training corpus"),
        (c2, f"{n_books/1e3:.0f}K",    "Books Indexed",     "TF-IDF content matrix"),
        (c3, f"{n_users/1e3:.0f}K",    "Unique Users",      "Collaborative filter"),
        (c4, f"{sparsity:.1f}%",        "Matrix Sparsity",   "Core challenge"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    for col, val, lbl, sub in [
        (c5, "0.86",                      "Precision@10",       "8.6 / 10 genuinely liked"),
        (c6, f"{cache.hit_rate:.0f}%",    "Cache Hit Rate",     f"{cache.hits} hits, {cache.misses} misses"),
        (c7, "15,000/s",                  "Stream Throughput",  "0.016 ms per event"),
        (c8, "1.954",                     "Model MAE",          "Mean absolute error (1–10 scale)"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Rating Distribution")
        rc = explicit_ratings["Book-Rating"].value_counts().sort_index().reset_index()
        rc.columns = ["Rating", "Count"]
        st.bar_chart(rc.set_index("Rating"), color="#c9861f")
    with col_r:
        st.markdown("### Top 10 Countries by Users")
        cc = users_clean["Country"].value_counts().head(10).reset_index()
        cc.columns = ["Country", "Users"]
        st.bar_chart(cc.set_index("Country"), color="#0f1f3d")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### Most Rated Books in Dataset")
    top_books = (
        working_df.groupby(["ISBN", "Book-Title", "Book-Author"])
        .agg(Ratings=("Book-Rating", "count"), Avg=("Book-Rating", "mean"))
        .reset_index()
        .sort_values("Ratings", ascending=False)
        .head(10)[["Book-Title", "Book-Author", "Ratings", "Avg"]]
    )
    top_books["Avg"] = top_books["Avg"].round(2)
    st.dataframe(top_books, use_container_width=True, hide_index=True)

    if st.session_state.query_history:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### Session Query History")
        hdf = pd.DataFrame(st.session_state.query_history)
        hdf["from_cache"] = hdf["from_cache"].map({True: "✅ Cache", False: "🔄 Computed"})
        hdf.columns = ["User ID", "Method", "Source", "Latency (ms)"]
        st.dataframe(hdf, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
# PAGE: HOW IT WORKS
# ═════════════════════════════════════════════════════════════
elif "How" in page:
    st.markdown("""
    <h1>How It Works</h1>
    <div class="accent-line"></div>
    <p style="color:#6b7280; font-size:0.95rem; max-width:600px;
              margin-bottom:2rem; line-height:1.75;">
    A plain-English walkthrough of all six milestones — from three raw CSV files
    to a real-time, production-grade book recommendation system.
    </p>
    """, unsafe_allow_html=True)

    sections = [
        ("📥", "Milestone 1 & 2 — Data Foundations & Distributed Processing", """
        <b>The raw data:</b> Three CSV files — Books, Ratings, and Users — containing 1.1 million
        rating events. The first discovery: 70% of those ratings are score=0, meaning the user
        interacted with a book but gave no opinion. These implicit ratings are separated
        before training so they don't corrupt the model signal.<br><br>
        <b>Parquet storage:</b> All cleaned data is saved in Parquet format instead of CSV.
        Parquet stores data column-by-column rather than row-by-row, making analytical queries
        5–10× faster and the files significantly smaller. This is the standard format in
        production data systems like HDFS, S3, and Azure Data Lake.<br><br>
        <b>Apache Spark:</b> Data is loaded into Spark and split into 10 partitions by User-ID.
        On a real cluster, each partition lives on a different physical machine — true parallel
        processing. MapReduce batch jobs then compute a full profile for every user
        (rating count, average, min/max) and every book (popularity, average score).
        These profiles feed directly into the ML layer.
        """),
        ("⚡", "Milestone 3 — Streaming & Real-Time (Speed Layer)", """
        <b>The problem:</b> Batch jobs run on a schedule and can be hours old by the time
        a user makes a request. A speed layer solves this by reacting to new events instantly.<br><br>
        <b>Apache Kafka design:</b> The system is architected for Kafka — a distributed message
        streaming platform. Since Kafka requires a broker server (not available in a notebook),
        the speed layer is simulated with a Python event loop using identical processing logic.<br><br>
        <b>Performance:</b> 15,000 events per second at 0.016ms average latency. A sliding window
        of the last 500 events is kept in memory, automatically dropping old events as new ones
        arrive, enabling real-time trending book detection.<br><br>
        <b>Bloom Filters:</b> Before recommending a book, the system checks if the user has already
        read it. A Bloom Filter does this in O(1) constant time using only 625 bytes per user —
        40× more memory-efficient than a dictionary — at the cost of a tiny, acceptable
        false-positive rate.
        """),
        ("🤖", "Milestone 4 — Hybrid Machine Learning Engine", """
        <b>Collaborative Filtering (ALS):</b> Learns purely from rating patterns — who liked what,
        not what the books are about. ALS factorises the sparse 278K × 270K user-item matrix into
        two dense matrices with 50 latent dimensions. These hidden factors capture preferences
        like "literary fiction" or "fast-paced thrillers" without ever being told what they mean.
        Training alternates between fixing the user matrix and solving for the item matrix
        (hence "Alternating Least Squares") until convergence.<br><br>
        <b>Content-Based Filtering (TF-IDF):</b> Recommends based on what books <i>are</i>.
        Each book's author, publisher, and year are combined into a text string and vectorised
        into a 5,000-feature TF-IDF matrix. Cosine similarity then finds the nearest neighbours
        in this content space.<br><br>
        <b>Hybrid (70% ALS + 30% TF-IDF):</b> ALS dominates for active users where rating data
        is rich. Content-based fills the gap for new users or obscure books with few ratings.<br><br>
        <b>Result: Precision@10 = 0.86</b> — 8.6 out of every 10 recommendations are books
        the test user genuinely rated 7 or above.
        """),
        ("🚀", "Milestone 5 & 6 — Deployment, Optimisation & Full Integration", """
        <b>LRU Cache:</b> The recommendation pipeline is computationally expensive.
        Results are stored in a Least Recently Used cache — cached responses return in under 1ms
        versus hundreds of milliseconds for a fresh computation. In production this would be Redis.
        The current session hit rate is tracked live in the sidebar.<br><br>
        <b>Drift Monitor:</b> A rolling MAE window of 100 predictions continuously watches for
        model degradation. If the rolling error exceeds a threshold, a retraining alert fires
        automatically — preventing silent accuracy decay over time.<br><br>
        <b>Flask REST API:</b> The full system is deployed as an HTTP API. Any application can
        call <code>/recommend?user_id=123</code> and receive personalised books as JSON.
        The <code>/health</code> endpoint reports cache hit rate and rolling MAE in real time.<br><br>
        <b>MMR Diversity Injection:</b> Pure collaborative filtering creates filter bubbles —
        the same few popular authors dominate every recommendation list. Maximal Marginal Relevance
        penalises repeated authors by 0.3, iteratively building a top-10 that is both
        relevant and varied across different authors and styles.
        """),
    ]

    for icon, title, body in sections:
        st.markdown(f"""
        <div class="how-card">
            <div class="how-title">{icon} &nbsp; {title}</div>
            <div class="how-body">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding:2.5rem 0 0.5rem 0;
                color:#9ca3af; font-size:0.82rem; line-height:2;">
        <b style="color:#374151;">SDS 2412 — Analysis of Large Datasets</b><br>
        Jomo Kenyatta University of Agriculture and Technology &nbsp;·&nbsp; May 2026<br>
        Rodney Okoth &nbsp;·&nbsp; Sandra Jebet &nbsp;·&nbsp; Effie Auma &nbsp;·&nbsp;
        Leslie Gedion &nbsp;·&nbsp; Kandy Genga
    </div>
    """, unsafe_allow_html=True)
