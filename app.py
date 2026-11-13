import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import time
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import html

# ── NLP / ML imports ──────────────────────────────────────────────────────────
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PDF / DOCX reading
import pdfplumber
import docx

# ── NLTK downloads (run once) ──────────────────────────────────────────────────
for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger",
            "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ResumeRank · AI-Powered Hiring Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0b0f1a;
    --surface: #111827;
    --surface2: #1a2235;
    --border: #1f2d45;
    --accent: #4f8ef7;
    --accent2: #7c3aed;
    --green: #10b981;
    --yellow: #f59e0b;
    --red: #ef4444;
    --text: #e2e8f0;
    --muted: #64748b;
    --radius: 14px;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
}
.stApp { background: var(--bg); }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -40%; left: -10%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(79,142,247,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    background: rgba(79,142,247,0.15);
    border: 1px solid rgba(79,142,247,0.3);
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0 0 0.6rem 0;
    background: linear-gradient(120deg, #e2e8f0, #4f8ef7, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.hero p {
    color: var(--muted);
    font-size: 1rem;
    margin: 0;
    max-width: 600px;
}

/* ── Stat Cards ── */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 140px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 var(--radius) var(--radius);
}
.stat-card.blue::after  { background: var(--accent); }
.stat-card.purple::after{ background: var(--accent2); }
.stat-card.green::after { background: var(--green); }
.stat-card.yellow::after{ background: var(--yellow); }
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.stat-label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Cards / Panels ── */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
}
.panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Score Badge ── */
.score-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 54px; height: 54px;
    border-radius: 50%;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    flex-shrink: 0;
}
.score-high   { background: rgba(16,185,129,0.18); color: #10b981; border: 2px solid rgba(16,185,129,0.4); }
.score-medium { background: rgba(245,158,11,0.18); color: #f59e0b; border: 2px solid rgba(245,158,11,0.4); }
.score-low    { background: rgba(239,68,68,0.18);  color: #ef4444; border: 2px solid rgba(239,68,68,0.4); }

/* ── Candidate Row ── */
.candidate-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    transition: border-color 0.2s;
}
.candidate-card:hover { border-color: var(--accent); }
.candidate-info { flex: 1; }
.candidate-name {
    font-weight: 600;
    font-size: 1rem;
    margin: 0 0 0.25rem 0;
}
.candidate-meta { font-size: 0.8rem; color: var(--muted); }
.rank-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--muted);
    min-width: 28px;
}
.rank-1 { color: #f59e0b; }
.rank-2 { color: #94a3b8; }
.rank-3 { color: #b45309; }

/* ── Progress bar override ── */
.stProgress > div > div { background: var(--accent) !important; }

/* ── Keyword pill ── */
.pill {
    display: inline-block;
    background: rgba(79,142,247,0.12);
    border: 1px solid rgba(79,142,247,0.25);
    color: var(--accent);
    font-size: 0.72rem;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    margin: 0.15rem;
}
.pill-miss {
    background: rgba(239,68,68,0.1);
    border-color: rgba(239,68,68,0.25);
    color: #f87171;
}
.pill-match {
    background: rgba(16,185,129,0.1);
    border-color: rgba(16,185,129,0.25);
    color: #34d399;
}

/* ── Section divider ── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* ── Streamlit widget overrides ── */
.stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(79,142,247,0.15) !important;
}
.stFileUploader {
    background: var(--surface2);
    border: 1px dashed var(--border);
    border-radius: 10px;
    padding: 0.5rem;
}
div[data-testid="stFileUploadDropzone"] {
    background: var(--surface2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stSelectbox div[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
}
.stSlider .stSlider { color: var(--accent) !important; }
[data-testid="stMetric"] {
    background: var(--surface2);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    border: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

# Common tech / domain skill keywords
SKILL_KEYWORDS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "sql", "nosql", "r", "scala", "kotlin", "swift", "php", "ruby",
    "machine learning", "deep learning", "nlp", "computer vision", "ai",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "spark", "hadoop", "kafka", "airflow",
    "react", "angular", "vue", "node", "django", "flask", "fastapi",
    "spring", "docker", "kubernetes", "aws", "gcp", "azure", "git",
    "data analysis", "data science", "statistics", "visualization",
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "agile", "scrum", "devops", "ci/cd", "linux", "api", "rest", "graphql",
}

DOMAIN_KEYWORDS = {
    "finance", "banking", "healthcare", "medical", "education", "ecommerce",
    "retail", "manufacturing", "logistics", "supply chain", "marketing",
    "sales", "hr", "legal", "insurance", "real estate", "telecommunications",
    "gaming", "media", "automotive", "energy", "consulting",
}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        pass
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract plain text from a DOCX file."""
    text = ""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception:
        pass
    return text.strip()


def read_resume_file(uploaded_file) -> str:
    """Read text from an uploaded resume file (PDF, DOCX, or TXT)."""
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    else:  # treat as plain text
        return file_bytes.decode("utf-8", errors="ignore")


# ── NLP helpers ───────────────────────────────────────────────────────────────

lemmatizer = WordNetLemmatizer()

def clean_and_tokenize(text: str) -> list[str]:
    """
    Basic NLP pre-processing:
      1. Lowercase
      2. Remove special characters
      3. Tokenize
      4. Remove stopwords
      5. Lemmatize
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#]", " ", text)  # keep + and # for C++/C#
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return tokens


def extract_years_of_experience(text: str) -> float:
    """
    Simple regex heuristic to pull years of experience from text.
    Patterns: '5 years', '5+ years', '3-5 years experience', etc.
    """
    patterns = [
        r"(\d+)\+?\s*(?:–|-|to)\s*(\d+)\s*years?",   # range: 3-5 years
        r"(\d+)\+\s*years?",                            # 5+ years
        r"(\d+)\s*years?\s*(?:of\s*)?experience",       # 5 years of experience
    ]
    found = []
    text_lower = text.lower()
    for pat in patterns:
        for m in re.finditer(pat, text_lower):
            nums = [int(g) for g in m.groups() if g is not None]
            found.append(sum(nums) / len(nums))  # use average for ranges
    return max(found) if found else 0.0


def extract_skills(text: str) -> set[str]:
    """Match known tech skills against the resume/JD text."""
    text_lower = text.lower()
    matched = set()
    for skill in SKILL_KEYWORDS:
        # word-boundary safe match
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            matched.add(skill)
    return matched


def extract_domain(text: str) -> list[str]:
    """Find domain / industry keywords mentioned in text."""
    text_lower = text.lower()
    found = [d for d in DOMAIN_KEYWORDS if re.search(r"\b" + re.escape(d) + r"\b", text_lower)]
    return found


def get_top_keywords(text: str, n: int = 15) -> list[str]:
    """Return the top-n meaningful keywords from text using token frequency."""
    tokens = clean_and_tokenize(text)

    # keep Counter intact while filtering
    freq = Counter(t for t in tokens if len(t) > 2)

    return [w for w, _ in freq.most_common(n)]


# ── Core scoring ──────────────────────────────────────────────────────────────

def compute_tfidf_similarity(jd_text: str, resume_text: str) -> float:
    """
    TF-IDF cosine similarity between JD and a single resume.
    Returns a score in [0, 1].
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=5000,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(score)
    except Exception:
        return 0.0


def compute_skill_overlap(jd_skills: set, resume_skills: set) -> float:
    """
    Jaccard-like skill overlap:  |intersection| / |jd_skills|
    Gives a ratio of how many JD-required skills the resume covers.
    """
    if not jd_skills:
        return 0.0
    matched = jd_skills & resume_skills
    return len(matched) / len(jd_skills)


def compute_experience_score(jd_yoe: float, resume_yoe: float) -> float:
    """
    Score how well the candidate's YOE meets the JD requirement.
    - Meets or exceeds → 1.0
    - Below required → proportional penalty
    - Way over (2x+) → slight penalty for over-qualification
    """
    if jd_yoe <= 0:
        return 0.8  # JD didn't specify; give neutral score
    if resume_yoe >= jd_yoe:
        # slight over-qualification penalty above 2x
        if resume_yoe > jd_yoe * 2:
            return max(0.7, 1.0 - (resume_yoe - jd_yoe * 2) * 0.02)
        return 1.0
    return resume_yoe / jd_yoe  # proportional to how close they are


def compute_domain_score(jd_domains: list, resume_domains: list) -> float:
    """Binary domain match: at least one shared domain."""
    if not jd_domains:
        return 0.7  # JD domain unclear → neutral
    jd_set = set(jd_domains)
    res_set = set(resume_domains)
    overlap = len(jd_set & res_set)
    return min(1.0, overlap / max(1, len(jd_set)))


def rank_resumes(jd_text: str, resumes: list[dict],
                 w_tfidf: float = 0.4, w_skill: float = 0.35,
                 w_yoe: float = 0.15, w_domain: float = 0.1) -> list[dict]:
    """
    Rank all resumes against the JD.
    weights: tfidf, skill_overlap, yoe, domain (must sum to 1)
    Returns a sorted list of result dicts.
    """
    jd_skills  = extract_skills(jd_text)
    jd_yoe     = extract_years_of_experience(jd_text)
    jd_domains = extract_domain(jd_text)
    jd_keywords = get_top_keywords(jd_text, 20)

    results = []
    for r in resumes:
        res_text    = r["text"]
        res_skills  = extract_skills(res_text)
        res_yoe     = extract_years_of_experience(res_text)
        res_domains = extract_domain(res_text)

        tfidf_score  = compute_tfidf_similarity(jd_text, res_text)
        skill_score  = compute_skill_overlap(jd_skills, res_skills)
        yoe_score    = compute_experience_score(jd_yoe, res_yoe)
        domain_score = compute_domain_score(jd_domains, res_domains)

        final_score = (
            w_tfidf  * tfidf_score  +
            w_skill  * skill_score  +
            w_yoe    * yoe_score    +
            w_domain * domain_score
        ) * 100  # convert to 0-100

        matched_skills  = jd_skills & res_skills
        missing_skills  = jd_skills - res_skills

        results.append({
            "name":            r["name"],
            "text":            res_text,
            "final_score":     round(final_score, 2),
            "tfidf_score":     round(tfidf_score * 100, 2),
            "skill_score":     round(skill_score * 100, 2),
            "yoe_score":       round(yoe_score * 100, 2),
            "domain_score":    round(domain_score * 100, 2),
            "res_yoe":         res_yoe,
            "matched_skills":  matched_skills,
            "missing_skills":  missing_skills,
            "res_domains":     res_domains,
            "res_skills":      res_skills,
            "jd_keywords":     jd_keywords,
            "jd_yoe":          jd_yoe,
            "jd_skills":       jd_skills,
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def score_class(score: float) -> str:
    if score >= 65:  return "score-high"
    if score >= 40:  return "score-medium"
    return "score-low"


def score_label(score: float) -> str:
    if score >= 65:  return "Strong Match"
    if score >= 40:  return "Moderate Match"
    return "Weak Match"


def render_hero():
    st.markdown("""
    <div class="hero">
        <div class="hero-tag">⚡ NLP-Powered · TF-IDF · Cosine Similarity</div>
        <h1>ResumeRank</h1>
        <p>Upload hundreds of resumes, paste a Job Description, and get instant AI-powered rankings with skill gap analysis — all in seconds.</p>
    </div>
    """, unsafe_allow_html=True)


def render_stat_row(n_resumes, n_matched, avg_score, top_score):
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card blue">
            <div class="stat-num">{n_resumes}</div>
            <div class="stat-label">Resumes Analyzed</div>
        </div>
        <div class="stat-card green">
            <div class="stat-num">{n_matched}</div>
            <div class="stat-label">Strong Matches (≥65)</div>
        </div>
        <div class="stat-card yellow">
            <div class="stat-num">{avg_score:.1f}</div>
            <div class="stat-label">Average Score</div>
        </div>
        <div class="stat-card purple">
            <div class="stat-num">{top_score:.1f}</div>
            <div class="stat-label">Top Score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_candidate_card(rank: int, result: dict):
    rank_cls = f"rank-{rank}" if rank <= 3 else ""
    sc = result["final_score"]
    sclass = score_class(sc)
    label  = score_label(sc)

    matched_pills = " ".join(f'<span class="pill pill-match">{s}</span>'
                             for s in sorted(result["matched_skills"])[:6])
    missing_pills = " ".join(f'<span class="pill pill-miss">{s}</span>'
                             for s in sorted(result["missing_skills"])[:4])

    domain_text = ", ".join(result["res_domains"]).title() or "Not detected"
    yoe_text = f"{result['res_yoe']:.0f} yrs" if result["res_yoe"] > 0 else "N/A"

    st.markdown(f"""
    <div class="candidate-card">
        <div class="rank-num {rank_cls}">#{rank}</div>
        <div class="score-badge {sclass}">{sc:.0f}</div>
        <div class="candidate-info">
            <div class="candidate-name">{result['name']}</div>
            <div class="candidate-meta">
                🗂 Domain: {domain_text} &nbsp;·&nbsp;
                ⏱ Experience: {yoe_text} &nbsp;·&nbsp;
                📊 {label}
            </div>
            <div style="margin-top:0.45rem;">
                {matched_pills}
                {missing_pills}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_detail_panel(result: dict, rank: int):
    """Expanded detail view for a selected candidate."""
    sc = result["final_score"]
    sclass = score_class(sc)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">📊 Score Breakdown</div>
        </div>
        """, unsafe_allow_html=True)

        categories = ["Text Similarity (TF-IDF)", "Skill Match", "Experience", "Domain Fit"]
        values     = [result["tfidf_score"], result["skill_score"],
                      result["yoe_score"],   result["domain_score"]]
        colors     = ["#4f8ef7", "#7c3aed", "#10b981", "#f59e0b"]

        fig = go.Figure(go.Bar(
            x=values, y=categories,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}" for v in values],
            textposition="inside",
            textfont=dict(color="white", size=12),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            xaxis=dict(range=[0, 100], color="#64748b", showgrid=True,
                       gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(color="#e2e8f0"),
            margin=dict(l=0, r=0, t=10, b=10),
            height=200,
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        fig2 = go.Figure(go.Scatterpolar(
            r=[result["tfidf_score"], result["skill_score"],
               result["yoe_score"], result["domain_score"],
               result["tfidf_score"]],
            theta=["TF-IDF", "Skills", "Experience", "Domain", "TF-IDF"],
            fill="toself",
            line=dict(color="#4f8ef7"),
            fillcolor="rgba(79,142,247,0.18)",
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], color="#64748b",
                                gridcolor="rgba(255,255,255,0.08)"),
                angularaxis=dict(color="#e2e8f0"),
            ),
            margin=dict(l=20, r=20, t=30, b=10),
            height=250,
            font=dict(family="Space Grotesk", color="#e2e8f0"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Skills
        st.markdown('<div class="panel"><div class="panel-title">✅ Matched Skills</div>', unsafe_allow_html=True)
        if result["matched_skills"]:
            pills = " ".join(f'<span class="pill pill-match">{s}</span>'
                             for s in sorted(result["matched_skills"]))
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#64748b">No skill overlap detected</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel" style="margin-top:1rem"><div class="panel-title">❌ Missing Skills</div>', unsafe_allow_html=True)
        if result["missing_skills"]:
            pills = " ".join(f'<span class="pill pill-miss">{s}</span>'
                             for s in sorted(result["missing_skills"]))
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#10b981">All required skills matched! 🎉</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Summary text
        jd_yoe = result["jd_yoe"]
        res_yoe = result["res_yoe"]
        yoe_note = ""
        if jd_yoe > 0:
            if res_yoe >= jd_yoe:
                yoe_note = f"✅ Meets {jd_yoe:.0f}yr requirement (has {res_yoe:.0f} yrs)"
            else:
                yoe_note = f"⚠️ Short by {jd_yoe - res_yoe:.0f} yr(s) (has {res_yoe:.0f} yrs)"
        else:
            yoe_note = f"Detected {res_yoe:.0f} yrs experience"

        dom_text = ", ".join(result["res_domains"]).title() or "Not detected"

        st.markdown(f"""
        <div class="panel" style="margin-top:1rem">
            <div class="panel-title">📋 Quick Summary</div>
            <div style="font-size:0.88rem; line-height:1.9; color:#cbd5e1">
                <b>Rank:</b> #{rank}<br/>
                <b>Overall Score:</b> {sc:.1f} / 100<br/>
                <b>Experience:</b> {yoe_note}<br/>
                <b>Domain:</b> {dom_text}<br/>
                <b>Skills Coverage:</b> {len(result['matched_skills'])} / {len(result['jd_skills'])} required skills<br/>
                <b>Recommendation:</b> {'Shortlist for interview' if sc >= 65 else ('Consider with caution' if sc >= 40 else 'Not recommended')}<br/>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800;
                background:linear-gradient(120deg,#4f8ef7,#7c3aed);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                margin-bottom:1.5rem;">
        ⚡ ResumeRank
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**⚖️ Scoring Weights**")
    st.caption("Adjust how each factor contributes to the final score.")

    w_tfidf  = st.slider("Text Similarity (TF-IDF)", 0.1, 0.7, 0.40, 0.05,
                          help="Overall textual similarity using TF-IDF + cosine similarity")
    w_skill  = st.slider("Skill Overlap",             0.1, 0.6, 0.35, 0.05,
                          help="How many JD-required skills the resume covers")
    w_yoe    = st.slider("Years of Experience",       0.0, 0.4, 0.15, 0.05,
                          help="Whether the candidate meets the YOE requirement")
    w_domain = st.slider("Domain / Industry Fit",     0.0, 0.3, 0.10, 0.05,
                          help="Industry domain alignment")

    total_w = w_tfidf + w_skill + w_yoe + w_domain
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"⚠️ Weights sum to {total_w:.2f}. Will be auto-normalised.")

    st.markdown("<hr style='border-color:#1f2d45;margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown("**🔍 Filter & Sort**")
    min_score = st.slider("Minimum Score Threshold", 0, 80, 0, 5,
                           help="Hide candidates below this score")
    top_n = st.selectbox("Show Top N Candidates", [10, 25, 50, 100, "All"], index=2)

    st.markdown("<hr style='border-color:#1f2d45;margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#475569; line-height:1.8">
        <b style="color:#64748b">NLP Pipeline</b><br/>
        → Text extraction (PDF/DOCX)<br/>
        → Tokenization & Lemmatization<br/>
        → TF-IDF Vectorisation<br/>
        → Cosine Similarity<br/>
        → Skill keyword matching<br/>
        → Regex YOE extraction<br/>
        → Domain detection<br/>
        → Weighted scoring
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

render_hero()

tab_input, tab_results, tab_analytics, tab_export = st.tabs(
    ["📝 Input", "🏆 Rankings", "📊 Analytics", "💾 Export"])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 · INPUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_input:
    col_jd, col_res = st.columns([1, 1], gap="large")

    with col_jd:
        st.markdown('<div class="panel-title">📄 Job Description</div>', unsafe_allow_html=True)
        jd_text = st.text_area(
            "Paste the full Job Description here",
            height=420,
            placeholder="We are looking for a Senior Data Scientist with 5+ years of experience in Python, machine learning, NLP, and cloud platforms (AWS/GCP). Experience in the healthcare domain preferred...",
            label_visibility="collapsed",
        )
        if jd_text:
            jd_skills  = extract_skills(jd_text)
            jd_yoe     = extract_years_of_experience(jd_text)
            jd_domains = extract_domain(jd_text)

            st.markdown('<div class="panel" style="margin-top:1rem">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title" style="font-size:0.9rem">🔍 JD Analysis Preview</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Skills Detected",    len(jd_skills))
            c2.metric("Required YOE",       f"{jd_yoe:.0f} yrs" if jd_yoe else "N/A")
            c3.metric("Domains Found",      len(jd_domains))

            if jd_skills:
                pills = " ".join(f'<span class="pill">{s}</span>' for s in sorted(jd_skills))
                st.markdown(f"<div style='margin-top:0.6rem'>{pills}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        st.markdown('<div class="panel-title">📂 Upload Resumes</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload resumes (PDF, DOCX, TXT) — batch upload supported",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        st.caption("💡 Tip: Select all files at once in your file browser for bulk upload")

        if uploaded_files:
            st.markdown(f"""
            <div class="panel" style="margin-top:0.8rem">
                <div class="panel-title" style="font-size:0.9rem">📁 {len(uploaded_files)} Resume(s) Queued</div>
            """, unsafe_allow_html=True)
            # Show first 8
            for f in uploaded_files[:8]:
                ext  = f.name.split(".")[-1].upper()
                size = f.size / 1024
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            font-size:0.8rem;padding:0.3rem 0;border-bottom:1px solid #1f2d45">
                    <span>📄 {f.name}</span>
                    <span style="color:#64748b">{ext} · {size:.1f} KB</span>
                </div>
                """, unsafe_allow_html=True)
            if len(uploaded_files) > 8:
                st.markdown(f'<div style="font-size:0.78rem;color:#64748b;margin-top:0.4rem">+{len(uploaded_files)-8} more files</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        run_btn = st.button("⚡ Rank Resumes", use_container_width=True)

    if run_btn:
        if not jd_text.strip():
            st.error("❌ Please paste a Job Description first.")
        elif not uploaded_files:
            st.error("❌ Please upload at least one resume.")
        else:
            # Normalize weights
            w_total = w_tfidf + w_skill + w_yoe + w_domain
            wt = w_tfidf / w_total
            ws = w_skill  / w_total
            wy = w_yoe    / w_total
            wd = w_domain / w_total

            progress_bar = st.progress(0, text="Extracting text from resumes…")
            resumes = []

            for i, f in enumerate(uploaded_files):
                text = read_resume_file(f)
                if text.strip():
                    resumes.append({"name": f.name.rsplit(".", 1)[0], "text": text})
                progress_bar.progress((i + 1) / len(uploaded_files),
                                      text=f"Reading {f.name}…")

            progress_bar.progress(1.0, text="Running NLP scoring…")
            time.sleep(0.3)

            with st.spinner(""):
                ranked = rank_resumes(jd_text, resumes, wt, ws, wy, wd)

            progress_bar.empty()
            st.session_state["ranked"] = ranked
            st.session_state["jd_text"] = jd_text
            st.success(f"✅ Ranked {len(ranked)} resumes! Switch to the **Rankings** tab.")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 · RANKINGS
# ─────────────────────────────────────────────────────────────────────────────
with tab_results:
    if "ranked" not in st.session_state:
        st.info("⬅️ Go to the **Input** tab, upload resumes and click **Rank Resumes**.")
    else:
        ranked = st.session_state["ranked"]

        # Filter
        filtered = [r for r in ranked if r["final_score"] >= min_score]
        if top_n != "All":
            filtered = filtered[:int(top_n)]

        n_strong = sum(1 for r in ranked if r["final_score"] >= 65)
        avg_sc   = np.mean([r["final_score"] for r in ranked])
        top_sc   = ranked[0]["final_score"] if ranked else 0

        render_stat_row(len(ranked), n_strong, avg_sc, top_sc)

        # Candidate list
        st.markdown('<div class="panel-title">🏆 Ranked Candidates</div>',unsafe_allow_html=True)

        for i, result in enumerate(filtered):
            render_candidate_card(i + 1, result)

        # Detail drill-down
        st.markdown("<hr class='divider'>")
        st.markdown('<div class="panel-title">🔬 Candidate Deep Dive</div>')

        names = [f"#{i+1} · {r['name']}" for i, r in enumerate(filtered)]
        chosen = st.selectbox("Select a candidate to inspect", names, label_visibility="visible")
        chosen_idx = names.index(chosen)
        render_detail_panel(filtered[chosen_idx], chosen_idx + 1)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 · ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab_analytics:
    if "ranked" not in st.session_state:
        st.info("⬅️ Run the ranking first.")
    else:
        ranked = st.session_state["ranked"]
        scores = [r["final_score"] for r in ranked]
        names  = [r["name"][:25] for r in ranked]

        col1, col2 = st.columns(2)

        with col1:
            # Score distribution histogram
            fig_hist = px.histogram(
                x=scores, nbins=20,
                title="Score Distribution",
                color_discrete_sequence=["#4f8ef7"],
                labels={"x": "Score", "y": "Count"},
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font=dict(family="Space Grotesk", color="#e2e8f0"),
                title_font=dict(family="Syne", size=16),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Match category pie
            labels_pie = ["Strong (≥65)", "Moderate (40-64)", "Weak (<40)"]
            vals_pie   = [
                sum(1 for s in scores if s >= 65),
                sum(1 for s in scores if 40 <= s < 65),
                sum(1 for s in scores if s < 40),
            ]
            fig_pie = px.pie(
                names=labels_pie, values=vals_pie,
                title="Candidate Quality Breakdown",
                color_discrete_sequence=["#10b981", "#f59e0b", "#ef4444"],
                hole=0.5,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Space Grotesk", color="#e2e8f0"),
                title_font=dict(family="Syne", size=16),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Top 15 bar chart
            top15 = ranked[:15]
            bar_colors = ["#10b981" if r["final_score"] >= 65
                          else "#f59e0b" if r["final_score"] >= 40
                          else "#ef4444" for r in top15]

            fig_bar = go.Figure(go.Bar(
                y=[r["name"][:22] for r in top15],
                x=[r["final_score"] for r in top15],
                orientation="h",
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=[f"{r['final_score']:.1f}" for r in top15],
                textposition="inside",
                textfont=dict(color="white"),
            ))
            fig_bar.update_layout(
                title="Top 15 Candidates",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font=dict(family="Space Grotesk", color="#e2e8f0"),
                title_font=dict(family="Syne", size=16),
                xaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.05)", color="#64748b"),
                yaxis=dict(color="#e2e8f0", autorange="reversed"),
                margin=dict(l=0, r=0, t=40, b=0),
                height=420,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Component scores comparison for top 5
            top5 = ranked[:5]
            fig_grouped = go.Figure()
            comp_colors = {"TF-IDF": "#4f8ef7", "Skills": "#7c3aed",
                           "Experience": "#10b981", "Domain": "#f59e0b"}
            for comp, key, clr in [("TF-IDF", "tfidf_score", "#4f8ef7"),
                                    ("Skills", "skill_score", "#7c3aed"),
                                    ("Experience", "yoe_score", "#10b981"),
                                    ("Domain", "domain_score", "#f59e0b")]:
                fig_grouped.add_trace(go.Bar(
                    name=comp,
                    x=[r["name"][:18] for r in top5],
                    y=[r[key] for r in top5],
                    marker_color=clr,
                ))
            fig_grouped.update_layout(
                title="Top 5 · Score Components",
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font=dict(family="Space Grotesk", color="#e2e8f0"),
                title_font=dict(family="Syne", size=16),
                xaxis=dict(color="#e2e8f0"),
                yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.05)", color="#64748b"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=0, r=0, t=40, b=0),
                height=300,
            )
            st.plotly_chart(fig_grouped, use_container_width=True)

        # Skill frequency across all resumes
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🛠️ Skill Frequency Across All Resumes</div>', unsafe_allow_html=True)

        all_skills = Counter()
        for r in ranked:
            for s in r["res_skills"]:
                all_skills[s] += 1

        if all_skills:
            top_skills = dict(all_skills.most_common(20))
            jd_skills_set = ranked[0]["jd_skills"] if ranked else set()

            fig_skills = go.Figure(go.Bar(
                x=list(top_skills.keys()),
                y=list(top_skills.values()),
                marker=dict(
                    color=["#4f8ef7" if s in jd_skills_set else "#374151"
                           for s in top_skills.keys()],
                    line=dict(width=0),
                ),
            ))
            fig_skills.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font=dict(family="Space Grotesk", color="#e2e8f0"),
                xaxis=dict(color="#e2e8f0", tickangle=-30),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#64748b", title="# Resumes"),
                margin=dict(l=0, r=0, t=10, b=0),
                height=280,
            )
            st.plotly_chart(fig_skills, use_container_width=True)
            st.caption("🔵 Blue bars = skills required by the JD")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 · EXPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab_export:
    if "ranked" not in st.session_state:
        st.info("⬅️ Run the ranking first.")
    else:
        ranked = st.session_state["ranked"]

        st.markdown('<div class="panel-title">💾 Export Results</div>', unsafe_allow_html=True)

        # Build dataframe
        rows = []
        for i, r in enumerate(ranked):
            rows.append({
                "Rank":             i + 1,
                "Candidate":        r["name"],
                "Overall Score":    r["final_score"],
                "TF-IDF Score":     r["tfidf_score"],
                "Skill Score":      r["skill_score"],
                "Experience Score": r["yoe_score"],
                "Domain Score":     r["domain_score"],
                "Detected YOE":     r["res_yoe"],
                "Domains":          ", ".join(r["res_domains"]),
                "Matched Skills":   ", ".join(sorted(r["matched_skills"])),
                "Missing Skills":   ", ".join(sorted(r["missing_skills"])),
                "Match Level":      score_label(r["final_score"]),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=350)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name="resume_rankings.csv",
            mime="text/csv",
            use_container_width=False,
        )

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📋 Full Analysis Report</div>', unsafe_allow_html=True)

        report_lines = [
            "# ResumeRank · Analysis Report\n",
            f"Total Resumes: {len(ranked)}",
            f"Strong Matches (≥65): {sum(1 for r in ranked if r['final_score']>=65)}",
            f"Average Score: {np.mean([r['final_score'] for r in ranked]):.2f}\n",
            "---\n",
        ]
        for i, r in enumerate(ranked):
            report_lines.append(f"## #{i+1} · {r['name']}  (Score: {r['final_score']:.1f})")
            report_lines.append(f"- Match Level: {score_label(r['final_score'])}")
            report_lines.append(f"- TF-IDF Similarity: {r['tfidf_score']:.1f}")
            report_lines.append(f"- Skill Match: {r['skill_score']:.1f}")
            report_lines.append(f"- Experience Score: {r['yoe_score']:.1f}")
            report_lines.append(f"- Domain Score: {r['domain_score']:.1f}")
            report_lines.append(f"- Detected YOE: {r['res_yoe']:.0f} yrs")
            report_lines.append(f"- Matched Skills: {', '.join(sorted(r['matched_skills'])) or 'None'}")
            report_lines.append(f"- Missing Skills: {', '.join(sorted(r['missing_skills'])) or 'None'}\n")

        report_text = "\n".join(report_lines)
        st.download_button(
            "⬇️ Download Full Report (.txt)",
            data=report_text.encode("utf-8"),
            file_name="resume_rank_report.txt",
            mime="text/plain",
        )