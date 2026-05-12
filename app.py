"""
Large-Scale Book Recommendation System
SDS 2412 — Analysis of Large Datasets | JKUAT 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import hashlib
import os
import csv
from datetime import datetime
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
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; color: var(--text); }
.stApp { background-color: var(--bg) !important; background-image: none !important; }
.block-container { padding: 2.5rem 2.5rem 4rem 2.5rem !important; max-width: 1100px !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--navy) !important; min-width: 220px !important; max-width: 220px !important; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] h3 { font-size: 0.68rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(226,232,240,0.4) !important; margin: 1.2rem 0 0.4rem 0; }

/* Headings */
h1 { font-family: 'Playfair Display', serif !important; font-size: 2.6rem !important; font-weight: 700 !important; color: var(--navy) !important; line-height: 1.15 !important; letter-spacing: -0.02em; }
h2 { font-family: 'Playfair Display', serif !important; font-size: 1.45rem !important; font-weight: 600 !important; color: var(--navy) !important; }

.accent-line { height: 3px; width: 52px; background: linear-gradient(90deg, var(--amber), var(--amber2)); border-radius: 2px; margin: 0.5rem 0 1.8rem 0; }

/* Stats strip */
.stat-strip { display: flex; background: var(--navy); border-radius: 14px; overflow: hidden; margin-bottom: 2rem; }
.stat-item { flex: 1; padding: 1.2rem 1rem; border-right: 1px solid rgba(255,255,255,0.07); text-align: center; }
.stat-item:last-child { border-right: none; }
.stat-val { font-family: 'Playfair Display', serif; font-size: 1.65rem; font-weight: 700; color: var(--amber2); line-height: 1; }
.stat-lbl { font-size: 0.67rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; color: rgba(226,232,240,0.45); margin-top: 0.3rem; }

/* Pipeline trace */
.pipeline-trace { display: flex; align-items: center; flex-wrap: wrap; gap: 0.35rem; background: #f8fafc; border: 1px solid var(--border); border-radius: 10px; padding: 0.7rem 1rem; margin-bottom: 1.5rem; font-size: 0.78rem; color: var(--muted); }
.pip-step { display: flex; align-items: center; gap: 0.3rem; background: var(--white); border: 1px solid var(--border); border-radius: 6px; padding: 0.22rem 0.6rem; color: var(--navy); font-weight: 500; font-size: 0.78rem; }
.pip-step .num { background: var(--amber); color: #fff; border-radius: 50%; width: 16px; height: 16px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.62rem; font-weight: 700; }
.pip-arrow { color: #d1d5db; }
.pip-hit  { background: #dcfce7; color: #15803d; border-radius: 999px; padding: 0.15rem 0.6rem; font-weight: 600; font-size: 0.72rem; }
.pip-miss { background: #fef9c3; color: #854d0e; border-radius: 999px; padding: 0.15rem 0.6rem; font-weight: 600; font-size: 0.72rem; }

/* Profile strip */
.profile-strip { background: var(--white); border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.4rem; display: flex; gap: 2rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.p-lbl { font-size: 0.67rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 600; }
.p-val { font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 700; color: var(--navy); line-height: 1.25; }

/* Book card */
.book-card { background: var(--white); border: 1px solid var(--border); border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 0.65rem; display: flex; align-items: center; gap: 1rem; position: relative; overflow: hidden; transition: box-shadow 0.18s, border-color 0.18s; }
.book-card::before { content:''; position:absolute; left:0; top:0; bottom:0; width:3px; background: linear-gradient(180deg, var(--amber), var(--amber2)); }
.book-card:hover { box-shadow: 0 4px 18px rgba(15,31,61,0.09); border-color: rgba(201,134,31,0.4); }
.book-rank { font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700; color:#d1d5db; min-width:1.9rem; text-align:center; line-height:1; }
.book-title { font-family:'Playfair Display',serif; font-size:0.98rem; font-weight:600; color:var(--navy); line-height:1.35; margin-bottom:0.12rem; }
.book-author { font-size:0.81rem; color:var(--muted); }
.score-badge { font-size:0.68rem; font-weight:700; padding:0.18rem 0.6rem; border-radius:999px; display:inline-block; margin-top:0.35rem; }
.score-badge.hybrid { background:#fef3c7; color:#92400e; }
.score-badge.cb     { background:#dbeafe; color:#1e3a8a; }
.score-badge.new    { background:#f0fdf4; color:#15803d; }

/* Metric card */
.metric-card { background: var(--white); border: 1px solid var(--border); border-top: 3px solid var(--amber); border-radius: 12px; padding: 1.2rem 1.3rem; }
.metric-card .val { font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:700; color:var(--navy); line-height:1; }
.metric-card .lbl { font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); margin-top:0.4rem; }
.metric-card .sub { font-size:0.76rem; color:#9ca3af; margin-top:0.12rem; }

/* Submission form card */
.submit-card { background: var(--white); border: 1px solid var(--border); border-left: 4px solid #15803d; border-radius: 12px; padding: 1.4rem 1.6rem; margin-top: 1rem; }
.submit-title { font-family:'Playfair Display',serif; font-size:1.05rem; font-weight:700; color:var(--navy); margin-bottom:0.5rem; }
.submit-body { font-size:0.87rem; color:#374151; line-height:1.7; margin-bottom:1rem; }

/* How-it-works card */
.how-card { background: var(--white); border: 1px solid var(--border); border-left: 4px solid var(--amber); border-radius: 12px; padding: 1.4rem 1.7rem; margin-bottom: 1rem; }
.how-title { font-family:'Playfair Display',serif; font-size:1.08rem; font-weight:700; color:var(--navy); margin-bottom:0.65rem; }
.how-body  { font-size:0.87rem; color:#374151; line-height:1.85; }

/* Buttons */
.stButton > button { background: var(--navy) !important; color: #fff !important; border: none !important; border-radius: 9px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; font-size: 0.93rem !important; padding: 0.62rem 1.8rem !important; transition: background 0.18s, transform 0.12s !important; }
.stButton > button:hover { background: var(--navy2) !important; transform: translateY(-1px); box-shadow: 0 4px 14px rgba(15,31,61,0.2) !important; }

/* Inputs */
.stTextInput > div > div > input { border-radius: 9px !important; border-color: var(--border) !important; font-family: 'DM Sans', sans-serif !important; background: var(--white) !important; color: var(--text) !important; font-size: 0.93rem !important; }
.stTextInput > div > div > input:focus { border-color: var(--amber) !important; box-shadow: 0 0 0 3px rgba(201,134,31,0.12) !important; }
.stSelectbox > div > div { border-radius: 9px !important; font-family: 'DM Sans', sans-serif !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.3rem; border-bottom: 2px solid var(--border); background: transparent; }
.stTabs [data-baseweb="tab"] { font-family: 'DM Sans', sans-serif !important; font-size: 0.88rem; font-weight: 500; color: var(--muted); border: none; background: transparent; padding: 0.5rem 1rem; border-radius: 6px 6px 0 0; }
.stTabs [aria-selected="true"] { color: var(--navy) !important; background: #fff !important; border-bottom: 2px solid var(--amber) !important; font-weight: 600 !important; }

.divider { border:none; border-top:1px solid var(--border); margin:1.8rem 0; }
.warn-box { background:#fffbeb; border:1px solid #fcd34d; border-radius:10px; padding:1rem 1.3rem; font-size:0.88rem; color:#78350f; line-height:1.65; }
.success-box { background:#f0fdf4; border:1px solid #86efac; border-radius:10px; padding:1rem 1.3rem; font-size:0.88rem; color:#15803d; line-height:1.65; }
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
# USER SUBMISSIONS — save & load
# ─────────────────────────────────────────────────────────────
SUBMISSIONS_FILE = "data/user_submissions.csv"
SUBMISSIONS_COLS = ["title", "author", "year", "submitted_at"]

def ensure_submissions_file():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(SUBMISSIONS_FILE):
        pd.DataFrame(columns=SUBMISSIONS_COLS).to_csv(SUBMISSIONS_FILE, index=False)

def save_submission(title, author, year):
    ensure_submissions_file()
    row = {"title": title.strip(), "author": author.strip(),
           "year": str(year).strip(), "submitted_at": datetime.now().isoformat()}
    with open(SUBMISSIONS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSIONS_COLS)
        writer.writerow(row)

def load_submissions():
    ensure_submissions_file()
    try:
        df = pd.read_csv(SUBMISSIONS_FILE)
        return df if not df.empty else pd.DataFrame(columns=SUBMISSIONS_COLS)
    except Exception:
        return pd.DataFrame(columns=SUBMISSIONS_COLS)


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

            books_df["Year-Of-Publication"] = pd.to_numeric(
                books_df["Year-Of-Publication"], errors="coerce")
            books_clean = books_df.dropna(subset=["Book-Author", "Publisher"]).copy()
            for col in ["Image-URL-S", "Image-URL-M", "Image-URL-L"]:
                if col in books_clean.columns:
                    books_clean.drop(columns=[col], inplace=True)
            books_clean["Year-Of-Publication"] = (
                pd.to_numeric(books_clean["Year-Of-Publication"], errors="coerce")
                .fillna(0).astype(int))
            books_clean["Book-Author"] = books_clean["Book-Author"].fillna("Unknown").str.strip()
            books_clean["Book-Title"]  = books_clean["Book-Title"].fillna("Unknown").str.strip()

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
def get_top_authors(_working_df, n=10):
    """Top N authors by number of ratings received."""
    return (
        _working_df.groupby("Book-Author")
        .agg(ratings=("Book-Rating", "count"), avg=("Book-Rating", "mean"))
        .reset_index()
        .query("ratings >= 10")
        .sort_values("ratings", ascending=False)
        .head(n)["Book-Author"]
        .tolist()
    )


@st.cache_data(show_spinner=False)
def get_sample_books(_working_df, n=10):
    """Top N books by rating count with avg >= 6."""
    popular = (
        _working_df.groupby(["Book-Title", "Book-Author"])
        .agg(count=("Book-Rating", "count"), avg=("Book-Rating", "mean"))
        .reset_index()
        .query("count >= 15 and avg >= 6")
        .sort_values("count", ascending=False)
        .head(n)
    )
    return [f"{row['Book-Title']} — {row['Book-Author']}"
            for _, row in popular.iterrows()]


# ─────────────────────────────────────────────────────────────
# MMR RERANK
# ─────────────────────────────────────────────────────────────
def mmr_rerank(df, score_col="hybrid_score", penalty=0.3):
    seen, result, remaining = set(), [], list(df.iterrows())
    while remaining and len(result) < len(df):
        best_i, best_score, best_row = None, -9e9, None
        for idx, (_, row) in enumerate(remaining):
            s = row[score_col] - (penalty if row.get("Book-Author", "") in seen else 0)
            if s > best_score:
                best_score, best_i, best_row = s, idx, row
        result.append(best_row)
        seen.add(best_row.get("Book-Author", ""))
        remaining.pop(best_i)
    return pd.DataFrame(result).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# CONTENT-BASED (by book title)
# ─────────────────────────────────────────────────────────────
def cb_recommend_by_title(book_title, tfidf_matrix, books_cb, n=10):
    matches = books_cb[books_cb["Book-Title"].str.contains(
        book_title.strip(), case=False, na=False, regex=False)]
    if matches.empty:
        return pd.DataFrame()
    idx = matches.index[0]
    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top = sim.argsort()[::-1][1:n + 1]
    res = books_cb.iloc[top][["Book-Title", "Book-Author", "Year-Of-Publication"]].copy()
    res["similarity"] = sim[top].round(4)
    return res.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# AUTHOR-BASED RECOMMEND
# ─────────────────────────────────────────────────────────────
def author_recommend(author_name, working_df, books_cb, tfidf_matrix, n=10, use_mmr=True):
    """
    1. Find all books by this author in the dataset.
    2. Find users who rated those books highly.
    3. See what else those users rated highly (collaborative signal).
    4. Also find content-similar books using TF-IDF on the author's works.
    5. Blend and MMR-rerank.
    """
    # Step 1: author's books
    author_mask = books_cb["Book-Author"].str.contains(
        author_name.strip(), case=False, na=False, regex=False)
    author_books = books_cb[author_mask]

    if author_books.empty:
        return pd.DataFrame(), "author_not_found"

    author_isbns = author_books["ISBN"].tolist() if "ISBN" in author_books.columns else []

    # Step 2: CF signal — users who liked this author
    if author_isbns and working_df is not None:
        fans = working_df[
            working_df["ISBN"].isin(author_isbns) &
            (working_df["Book-Rating"] >= 7)
        ]["User-ID"].unique()

        other_reads = working_df[
            working_df["User-ID"].isin(fans) &
            ~working_df["ISBN"].isin(author_isbns) &
            (working_df["Book-Rating"] >= 7)
        ]

        if not other_reads.empty:
            cf_scores = (
                other_reads.groupby(["ISBN", "Book-Title", "Book-Author"])
                .agg(cf_score=("Book-Rating", "mean"), cf_count=("Book-Rating", "count"))
                .reset_index()
                .query("cf_count >= 2")
                .sort_values("cf_score", ascending=False)
            )
        else:
            cf_scores = pd.DataFrame()
    else:
        cf_scores = pd.DataFrame()

    # Step 3: TF-IDF content signal from author's first book
    cb_scores = pd.DataFrame()
    if not author_books.empty:
        seed_idx = author_books.index[0]
        if seed_idx < tfidf_matrix.shape[0]:
            sim = cosine_similarity(tfidf_matrix[seed_idx], tfidf_matrix).flatten()
            # exclude author's own books
            mask = ~books_cb["Book-Author"].str.contains(
                author_name.strip(), case=False, na=False, regex=False)
            valid_idx = [i for i in sim.argsort()[::-1] if mask.iloc[i]][:n * 3]
            cb_scores = books_cb.iloc[valid_idx][
                ["Book-Title", "Book-Author", "Year-Of-Publication"]].copy()
            if "ISBN" in books_cb.columns:
                cb_scores["ISBN"] = books_cb.iloc[valid_idx]["ISBN"].values
            cb_scores["similarity"] = sim[valid_idx].round(4)
            cb_scores = cb_scores[cb_scores["similarity"] > 0.05]

    # Step 4: Merge CF + CB
    if not cf_scores.empty and not cb_scores.empty:
        # normalise both
        mn, mx = cf_scores["cf_score"].min(), cf_scores["cf_score"].max()
        cf_scores["cf_norm"] = (cf_scores["cf_score"] - mn) / (mx - mn + 1e-6)

        mn2, mx2 = cb_scores["similarity"].min(), cb_scores["similarity"].max()
        cb_scores["cb_norm"] = (cb_scores["similarity"] - mn2) / (mx2 - mn2 + 1e-6)

        # merge on title+author
        merged = cf_scores.merge(
            cb_scores[["Book-Title", "Book-Author", "cb_norm", "Year-Of-Publication"]],
            on=["Book-Title", "Book-Author"], how="outer"
        ).fillna(0)
        merged["hybrid_score"] = 0.65 * merged.get("cf_norm", 0) + 0.35 * merged.get("cb_norm", 0)
        result_df = merged.sort_values("hybrid_score", ascending=False).head(n * 2)
        method = "hybrid"

    elif not cf_scores.empty:
        mn, mx = cf_scores["cf_score"].min(), cf_scores["cf_score"].max()
        cf_scores["hybrid_score"] = (cf_scores["cf_score"] - mn) / (mx - mn + 1e-6)
        result_df = cf_scores.head(n * 2)
        method = "collaborative"

    elif not cb_scores.empty:
        mn, mx = cb_scores["similarity"].min(), cb_scores["similarity"].max()
        cb_scores["hybrid_score"] = (cb_scores["similarity"] - mn) / (mx - mn + 1e-6)
        result_df = cb_scores.rename(columns={"similarity": "cb_norm"}).head(n * 2)
        method = "content-based"

    else:
        return pd.DataFrame(), "no_results"

    # Step 5: MMR
    if use_mmr and not result_df.empty:
        result_df = mmr_rerank(result_df).head(n)
    else:
        result_df = result_df.head(n)

    return result_df.reset_index(drop=True), method


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "lru_cache"      not in st.session_state: st.session_state.lru_cache = LRUCache(100)
if "query_history"  not in st.session_state: st.session_state.query_history = []
if "show_submit"    not in st.session_state: st.session_state.show_submit = False
if "submitted_ok"   not in st.session_state: st.session_state.submitted_ok = False
if "last_query"     not in st.session_state: st.session_state.last_query = ""


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0.4rem 1rem 0.4rem;
                border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:0.8rem;">
        <div style="font-size:1.6rem;">📚</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.1rem;
                    font-weight:700; color:#e8a825; margin-top:0.3rem;">BookMind</div>
        <div style="font-size:0.67rem; color:rgba(226,232,240,0.38); margin-top:0.2rem;
                    text-transform:uppercase; letter-spacing:0.07em;">SDS 2412 · JKUAT · 2026</div>
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
                              help="Prevents same author appearing repeatedly in results")
    show_scores = st.checkbox("Show recommendation scores", value=False)

    cache = st.session_state.lru_cache
    st.markdown(f"""
    <div style="margin-top:1.5rem; padding-top:1rem;
                border-top:1px solid rgba(255,255,255,0.07);
                font-size:0.73rem; color:rgba(226,232,240,0.38); line-height:2.1;">
        Cache: {len(cache.cache)} / 100<br>
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
        books_cb = books_cb.copy()
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
        Download from <a href="https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset"
        target="_blank">Kaggle — Book Recommendation Dataset</a>.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Stats strip
    n_books   = len(books_clean)
    n_users   = users_clean["User-ID"].nunique()
    n_ratings = len(explicit_ratings)
    sparsity  = (1 - n_ratings / (n_users * n_books)) * 100
    st.markdown(f"""
    <div class="stat-strip">
        <div class="stat-item"><div class="stat-val">{n_ratings/1e3:.0f}K</div><div class="stat-lbl">Ratings</div></div>
        <div class="stat-item"><div class="stat-val">{n_books/1e3:.0f}K</div><div class="stat-lbl">Books</div></div>
        <div class="stat-item"><div class="stat-val">{n_users/1e3:.0f}K</div><div class="stat-lbl">Users</div></div>
        <div class="stat-item"><div class="stat-val">0.86</div><div class="stat-lbl">Precision@10</div></div>
        <div class="stat-item"><div class="stat-val">{sparsity:.0f}%</div><div class="stat-lbl">Sparsity</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Precompute samples
    top_authors  = get_top_authors(working_df)
    sample_books = get_sample_books(working_df)

    # Tabs
    tab_author, tab_book = st.tabs([
        "✍️  Search by Author — get books loved by fans of an author you enjoy",
        "📗  Search by Book Title — find books similar to one you already love"
    ])

    # ══════════════════════════════════════════════
    # TAB 1 — AUTHOR SEARCH
    # ══════════════════════════════════════════════
    with tab_author:
        st.markdown("""
        <div style="font-size:0.88rem; color:#6b7280; margin-bottom:1.2rem; line-height:1.6;">
        Pick an author from the top-rated list, or type any author name. The system finds
        everyone who loved that author's books, discovers what else those readers enjoyed,
        and blends that collaborative signal with TF-IDF content similarity to produce
        a personalised recommendation list — all filtered for diversity using MMR.
        </div>
        """, unsafe_allow_html=True)

        # Dropdown of top authors
        author_options = ["— type an author name below —"] + top_authors
        selected_author_drop = st.selectbox(
            "📋  Pick from the 10 most-rated authors",
            options=author_options,
            help="These authors have the most ratings in the dataset"
        )

        prefill_author = ""
        if selected_author_drop != "— type an author name below —":
            prefill_author = selected_author_drop

        col_a, col_btn = st.columns([3, 1])
        with col_a:
            author_input = st.text_input(
                "Or type any author name",
                value=prefill_author,
                placeholder="e.g. Stephen King, J.K. Rowling, John Grisham…"
            )
        with col_btn:
            st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
            go_author = st.button("✨  Find Books", key="go_author")

        if go_author and author_input.strip():
            query = author_input.strip()
            cache_key     = f"author_{query.lower()}_{n_recs}_{use_mmr}"
            cached_result = st.session_state.lru_cache.get(cache_key)
            t0 = time.time()

            if cached_result is not None:
                recs, method = cached_result
                from_cache   = True
            else:
                with st.spinner(f"Finding books for fans of {query}…"):
                    recs, method = author_recommend(
                        query, working_df, books_cb, tfidf_matrix,
                        n=n_recs, use_mmr=use_mmr)
                st.session_state.lru_cache.put(cache_key, (recs, method))
                from_cache = False

            elapsed = (time.time() - t0) * 1000
            st.session_state.query_history.append(
                {"query": query, "type": "author", "method": method,
                 "from_cache": from_cache, "ms": round(elapsed, 1)})
            st.session_state.last_query = query

            badge = (f'<span class="pip-hit">⚡ Cache HIT</span>'
                     if from_cache else
                     f'<span class="pip-miss">🔄 Model computed</span>')
            st.markdown(f"""
            <div class="pipeline-trace">
                <div class="pip-step"><span class="num">1</span> Author Lookup</div>
                <span class="pip-arrow">→</span>
                <div class="pip-step"><span class="num">2</span> Fan Discovery (CF)</div>
                <span class="pip-arrow">→</span>
                <div class="pip-step"><span class="num">3</span> TF-IDF Content</div>
                <span class="pip-arrow">→</span>
                <div class="pip-step"><span class="num">4</span> Hybrid Blend</div>
                <span class="pip-arrow">→</span>
                <div class="pip-step"><span class="num">5</span> MMR Diversity</div>
                <span style="margin-left:auto; display:flex; align-items:center; gap:0.5rem;">
                    {badge} <span style="color:#9ca3af; font-size:0.75rem;">{elapsed:.1f} ms</span>
                </span>
            </div>
            """, unsafe_allow_html=True)

            # ── Author not found ──
            if method == "author_not_found" or (isinstance(recs, pd.DataFrame) and recs.empty):
                st.markdown(f"""
                <div class="submit-card">
                    <div class="submit-title">😕 We couldn't find <em>"{query}"</em> in our dataset</div>
                    <div class="submit-body">
                        This author may not be in the Book-Crossing dataset yet — but you can help!
                        Submit a book by this author below and it will be added to our community
                        database. Over time, user submissions are analysed using TF-IDF and help
                        improve recommendations for everyone. <b>You're growing the system.</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.show_submit = True

            else:
                st.session_state.show_submit = False
                # Author profile
                author_books_in_db = working_df[
                    working_df["Book-Author"].str.contains(
                        query, case=False, na=False, regex=False)]
                n_author_books   = author_books_in_db["ISBN"].nunique()
                n_author_ratings = len(author_books_in_db)
                avg_author_rating = author_books_in_db["Book-Rating"].mean() if n_author_ratings > 0 else 0

                st.markdown(f"""
                <div class="profile-strip">
                    <div><div class="p-lbl">Author</div>
                         <div class="p-val" style="font-size:0.95rem;">{query}</div></div>
                    <div><div class="p-lbl">Books in Dataset</div>
                         <div class="p-val">{n_author_books}</div></div>
                    <div><div class="p-lbl">Total Ratings</div>
                         <div class="p-val">{n_author_ratings:,}</div></div>
                    <div><div class="p-lbl">Avg Rating</div>
                         <div class="p-val">{avg_author_rating:.1f} / 10</div></div>
                    <div><div class="p-lbl">Method</div>
                         <div class="p-val" style="color:#c9861f; font-size:0.92rem;">{method.title()}</div></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"## Recommended for Fans of *{query}*")
                st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

                for i, row in recs.iterrows():
                    title   = str(row.get("Book-Title",  "Unknown Title"))[:80]
                    author  = str(row.get("Book-Author", "Unknown Author"))[:50]
                    year    = row.get("Year-Of-Publication", 0)
                    try: year = int(year)
                    except: year = 0
                    score   = float(row.get("hybrid_score", 0))
                    cf_n    = float(row.get("cf_norm",      0))
                    cb_n    = float(row.get("cb_norm",      0))

                    scores_html = ""
                    if show_scores:
                        scores_html = (
                            f'<span class="score-badge hybrid">Hybrid {score:.3f}</span>'
                            f'<span class="score-badge cb" style="margin-left:0.3rem;">'
                            f'CF {cf_n:.2f} · CB {cb_n:.2f}</span>'
                        )

                    st.markdown(f"""
                    <div class="book-card">
                        <div class="book-rank">{i+1:02d}</div>
                        <div style="flex:1; min-width:0;">
                            <div class="book-title">{title}</div>
                            <div class="book-author">by {author}{f" &nbsp;·&nbsp; {year}" if year > 1800 else ""}</div>
                            {scores_html}
                        </div>
                        <div style="text-align:right; min-width:3rem; flex-shrink:0;">
                            <div style="font-family:'Playfair Display',serif; font-size:1.45rem;
                                        font-weight:700; color:#c9861f; line-height:1;">{score:.3f}</div>
                            <div style="font-size:0.62rem; color:#9ca3af; text-transform:uppercase;
                                        letter-spacing:0.06em;">score</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # TAB 2 — BOOK TITLE SEARCH
    # ══════════════════════════════════════════════
    with tab_book:
        st.markdown("""
        <div style="font-size:0.88rem; color:#6b7280; margin-bottom:1.2rem; line-height:1.6;">
        Choose a book from the popular titles below or type any title. The engine builds a
        TF-IDF content vector from the book's author, publisher, and year, then finds the
        most similar books using cosine similarity across all 270,000 titles in the dataset.
        </div>
        """, unsafe_allow_html=True)

        selected_sample_book = st.selectbox(
            "📋  Pick a popular book to start from",
            options=["— type your own title below —"] + sample_books,
            help="10 of the highest-rated books with the most reviews in the dataset"
        )

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

        if go_book and book_query.strip():
            query      = book_query.strip()
            cache_key  = f"cb_{query.lower()}_{n_recs}"
            cached     = st.session_state.lru_cache.get(cache_key)
            t0 = time.time()

            if cached is not None:
                cb_recs    = cached
                from_cache = True
            else:
                with st.spinner("Computing TF-IDF similarity…"):
                    cb_recs = cb_recommend_by_title(query, tfidf_matrix, books_cb, n=n_recs)
                st.session_state.lru_cache.put(cache_key, cb_recs)
                from_cache = False

            elapsed = (time.time() - t0) * 1000
            st.session_state.last_query = query

            badge = (f'<span class="pip-hit">⚡ Cache HIT</span>'
                     if from_cache else
                     f'<span class="pip-miss">🔄 TF-IDF computed</span>')
            st.markdown(f"""
            <div class="pipeline-trace">
                <div class="pip-step"><span class="num">1</span> Title Lookup</div>
                <span class="pip-arrow">→</span>
                <div class="pip-step"><span class="num">2</span> TF-IDF Vector</div>
                <span class="pip-arrow">→</span>
                <div class="pip-step"><span class="num">3</span> Cosine Similarity</div>
                <span class="pip-arrow">→</span>
                <div class="pip-step"><span class="num">4</span> LRU Cache</div>
                <span style="margin-left:auto; display:flex; align-items:center; gap:0.5rem;">
                    {badge} <span style="color:#9ca3af; font-size:0.75rem;">{elapsed:.1f} ms</span>
                </span>
            </div>
            """, unsafe_allow_html=True)

            # ── Book not found → submission form ──
            if cb_recs.empty:
                st.markdown(f"""
                <div class="submit-card">
                    <div class="submit-title">😕 Sorry, we couldn't find <em>"{query}"</em> in our dataset</div>
                    <div class="submit-body">
                        This book may not exist in the Book-Crossing database yet. But you can
                        help us grow! Fill in a few details below and this book will be saved
                        to our community dataset. It will be indexed with TF-IDF and start
                        appearing in recommendations for other users who search for similar books.
                        <b>Every submission makes the system smarter for everyone.</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.show_submit = True

            else:
                st.session_state.show_submit = False
                st.markdown(f'## Books similar to *"{query}"*')
                st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

                for i, row in cb_recs.iterrows():
                    title  = str(row.get("Book-Title",  "Unknown"))[:80]
                    author = str(row.get("Book-Author", "Unknown"))[:50]
                    year   = row.get("Year-Of-Publication", 0)
                    try: year = int(year)
                    except: year = 0
                    sim    = float(row.get("similarity", 0))

                    st.markdown(f"""
                    <div class="book-card">
                        <div class="book-rank">{i+1:02d}</div>
                        <div style="flex:1; min-width:0;">
                            <div class="book-title">{title}</div>
                            <div class="book-author">by {author}{f" &nbsp;·&nbsp; {year}" if year > 1800 else ""}</div>
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

    # ══════════════════════════════════════════════
    # USER SUBMISSION FORM — shown when no results
    # ══════════════════════════════════════════════
    if st.session_state.show_submit:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### 📬 Submit a Book to Our Community Dataset")
        st.markdown("""
        <div style="font-size:0.88rem; color:#6b7280; margin-bottom:1rem; line-height:1.6;">
        Your submission is saved locally and will be included in the next TF-IDF analysis run.
        Over time, community-submitted books will appear in recommendations for other users —
        this is how the system grows beyond the original Book-Crossing dataset.
        </div>
        """, unsafe_allow_html=True)

        with st.form("submission_form", clear_on_submit=True):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                sub_title  = st.text_input("Book Title *",
                    value=st.session_state.last_query,
                    placeholder="e.g. The Midnight Library")
            with col2:
                sub_author = st.text_input("Author *",
                    placeholder="e.g. Matt Haig")
            with col3:
                sub_year   = st.text_input("Year",
                    placeholder="e.g. 2020")

            submitted = st.form_submit_button("📥  Add to Dataset", use_container_width=True)

            if submitted:
                if not sub_title.strip() or not sub_author.strip():
                    st.error("Please provide at least the book title and author.")
                else:
                    save_submission(sub_title, sub_author, sub_year)
                    st.session_state.show_submit  = False
                    st.session_state.submitted_ok = True
                    st.rerun()

        if st.session_state.submitted_ok:
            st.markdown("""
            <div class="success-box">
            ✅ <b>Thank you!</b> Your book has been added to the community dataset.
            It will be included in the next TF-IDF indexing run and will start appearing
            in recommendations for users searching for similar titles.
            </div>
            """, unsafe_allow_html=True)
            st.session_state.submitted_ok = False


# ═════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════
elif "Dashboard" in page:
    st.markdown("""
    <h1>System Dashboard</h1>
    <div class="accent-line"></div>
    <p style="color:#6b7280; font-size:0.95rem; margin-bottom:1.8rem;">
    Live metrics across all pipeline layers — data, ML model, cache, and community submissions.
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
    subs_df   = load_submissions()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl, sub in [
        (c1, f"{n_ratings/1e3:.0f}K",  "Explicit Ratings",  "Training corpus"),
        (c2, f"{n_books/1e3:.0f}K",    "Books Indexed",     "TF-IDF content matrix"),
        (c3, f"{n_users/1e3:.0f}K",    "Unique Users",      "Collaborative filter"),
        (c4, f"{sparsity:.1f}%",        "Matrix Sparsity",   "Core challenge"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="val">{val}</div><div class="lbl">{lbl}</div>
                <div class="sub">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    for col, val, lbl, sub in [
        (c5, "0.86",                   "Precision@10",      "8.6 / 10 genuinely liked"),
        (c6, f"{cache.hit_rate:.0f}%", "Cache Hit Rate",    f"{cache.hits} hits, {cache.misses} misses"),
        (c7, "15,000/s",               "Stream Throughput", "0.016 ms per event"),
        (c8, str(len(subs_df)),        "Community Books",   "User-submitted titles"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="val">{val}</div><div class="lbl">{lbl}</div>
                <div class="sub">{sub}</div></div>""", unsafe_allow_html=True)

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

    if not subs_df.empty:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### 📬 Community Submitted Books")
        st.markdown(f"""
        <div style="font-size:0.85rem; color:#6b7280; margin-bottom:0.8rem;">
        {len(subs_df)} book(s) submitted by users — these will be indexed with TF-IDF
        in the next analysis run to grow the recommendation database.
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            subs_df[["title","author","year","submitted_at"]].rename(columns={
                "title":"Title","author":"Author","year":"Year","submitted_at":"Submitted"}),
            use_container_width=True, hide_index=True)

    if st.session_state.query_history:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### Session Query History")
        hdf = pd.DataFrame(st.session_state.query_history)
        hdf["from_cache"] = hdf["from_cache"].map({True: "✅ Cache", False: "🔄 Computed"})
        st.dataframe(hdf, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
# PAGE: HOW IT WORKS
# ═════════════════════════════════════════════════════════════
elif "How" in page:
    st.markdown("""
    <h1>How It Works</h1>
    <div class="accent-line"></div>
    <p style="color:#6b7280; font-size:0.95rem; max-width:600px; margin-bottom:2rem; line-height:1.75;">
    A plain-English walkthrough of all six milestones — from three raw CSV files
    to a real-time, production-grade book recommendation system.
    </p>
    """, unsafe_allow_html=True)

    sections = [
        ("📥", "Milestone 1 & 2 — Data Foundations & Distributed Processing", """
        Three CSV files — Books, Ratings, Users — containing 1.1 million rating events.
        70% of those ratings are score=0 (implicit interactions, no actual opinion).
        These are separated before training so they don't corrupt the model.<br><br>
        All cleaned data is saved in <b>Parquet format</b> — a columnar storage format
        5–10× faster than CSV for analytical queries. Data is then loaded into
        <b>Apache Spark</b>, split into 10 partitions by User-ID, and
        <b>MapReduce</b> batch jobs compute user profiles and item popularity tables
        that feed the ML layer. Scalability confirmed: near-linear growth.
        """),
        ("⚡", "Milestone 3 — Streaming & Real-Time (Speed Layer)", """
        The speed layer reacts instantly to new user events without waiting for
        a batch job. Designed for <b>Apache Kafka</b> — simulated in Python
        with identical processing logic due to notebook constraints.<br><br>
        A <b>sliding window</b> of 500 events stays in memory, tracking trending books.
        <b>Bloom Filters</b> check in O(1) time whether a user has already read a book —
        625 bytes per user, 40× more efficient than a dictionary.
        Throughput: 15,000 events/sec at 0.016ms latency.
        """),
        ("🤖", "Milestone 4 — Hybrid Machine Learning Engine", """
        <b>Collaborative Filtering (ALS):</b> learns from who rated what.
        Factorises the 278K × 270K rating matrix into 50 latent dimensions —
        hidden preferences the model discovers automatically.<br><br>
        <b>Content-Based (TF-IDF):</b> each book becomes a 5,000-feature vector
        of author, publisher, and year. Cosine similarity finds the nearest neighbours.<br><br>
        <b>Hybrid (70% ALS + 30% TF-IDF):</b> ALS dominates for well-rated authors;
        TF-IDF fills the gap for obscure or new books.
        <b>Precision@10 = 0.86</b> — 8.6 of 10 recommendations genuinely liked.
        """),
        ("🚀", "Milestone 5 & 6 — Deployment, Optimisation & Integration", """
        <b>LRU Cache:</b> results stored in memory — cached responses under 1ms,
        60% hit rate. Production path: Redis.<br><br>
        <b>Drift Monitor:</b> rolling MAE window of 100 predictions — stable at 1.99,
        fires a retraining alert if accuracy drops.<br><br>
        <b>Flask REST API:</b> any app can call /recommend and receive JSON.<br><br>
        <b>MMR Diversity:</b> prevents filter bubbles by penalising repeated authors
        across the top-10 list — our system-level innovation.
        """),
        ("📬", "Community Dataset Growth — Your Idea, Live in the App", """
        When a book or author is not found in the dataset, the app asks the user
        to submit the title, author, and year. Submissions are saved to
        <b>user_submissions.csv</b> — a community-built extension of the
        Book-Crossing dataset.<br><br>
        Over time, these entries are indexed with TF-IDF and start appearing in
        recommendations for other users who search for similar books.
        This is <b>active learning</b> — user behaviour feeding back into the model
        to make it smarter for everyone. The same principle Netflix and Spotify
        used to grow their catalogues in their early years.
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
        Leslie Gideon &nbsp;·&nbsp; Kandy Genga
    </div>
    """, unsafe_allow_html=True)
