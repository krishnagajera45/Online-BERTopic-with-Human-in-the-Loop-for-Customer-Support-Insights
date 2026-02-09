"""
📖 Project Overview — About the system, motivation, architecture, and data.
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.components.theme import inject_custom_css, page_header, render_footer

st.set_page_config(page_title="Project Overview", page_icon="📖", layout="wide")
inject_custom_css()

page_header(
    "Project Overview",
    "Online BERTopic with Human-in-the-Loop for Customer Support Insights",
    "📖",
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_intro, tab_arch, tab_data, tab_method, tab_eval, tab_team = st.tabs([
    "🎯 Introduction",
    "🏗️ Architecture",
    "📊 Data",
    "🔬 Methodology",
    "📈 Evaluation",
    "👥 About",
])

# ── INTRODUCTION ──────────────────────────────────────────────────────────────
with tab_intro:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        ## Motivation

        Customer support channels on social media generate **millions of
        messages daily**.  Manually categorizing these messages is
        impractical — yet understanding the *topics* customers talk about is
        critical for improving products, reducing churn, and routing tickets
        to the right teams.

        Traditional topic models (LDA, NMF) struggle with:
        - Short, noisy social-media text
        - Evolving vocabulary over time
        - Lack of interpretable topic labels

        **BERTopic** solves these problems by combining contextual language
        models (Sentence-BERT) with density-based clustering, producing
        coherent and interpretable topics without the bag-of-words
        limitations.

        ## Objective

        This project implements an **end-to-end online topic modeling
        pipeline** that:

        1. **Ingests** customer support tweets in configurable time-window
           batches
        2. **Discovers topics** using BERTopic (Sentence-BERT → UMAP →
           HDBSCAN → c-TF-IDF)
        3. **Detects drift** between consecutive models (prevalence shifts,
           centroid drift, keyword divergence)
        4. **Enables human oversight** — domain experts can merge or relabel
           topics through an interactive UI (Human-in-the-Loop)
        5. **Tracks experiments** via MLflow & Prefect orchestration
        6. **Serves results** through a FastAPI + Streamlit stack
        """)
    with c2:
        st.markdown("""
        ## 🔑 Key Benefits

        <div class="info-card">
            <h3>🚀 Real-Time Insights</h3>
            <p>New batches are processed incrementally — no need to retrain from scratch.</p>
        </div>
        <div class="info-card">
            <h3>🧑‍🔬 Human-in-the-Loop</h3>
            <p>Experts refine topics via merge & relabel with full audit trail.</p>
        </div>
        <div class="info-card">
            <h3>📉 Drift Detection</h3>
            <p>Automatic alerts when topics shift — centroid, prevalence, and JS divergence.</p>
        </div>
        <div class="info-card">
            <h3>📊 Model Versioning</h3>
            <p>Every model version is archived — compare, rollback, and reproduce.</p>
        </div>
        """, unsafe_allow_html=True)

# ── ARCHITECTURE ──────────────────────────────────────────────────────────────
with tab_arch:
    st.markdown("## System Architecture")
    st.markdown("""
    The system follows a **layered architecture** with clear separation of
    concerns.  Each layer communicates through well-defined interfaces.
    """)

    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────────┐
    │                     PRESENTATION LAYER                          │
    │                                                                  │
    │   Streamlit Dashboard  ◄──── REST API ────►  FastAPI Server     │
    │   (Port 8501)                                (Port 8000)         │
    └──────────────────────────────────┬───────────────────────────────┘
                                       │
    ┌──────────────────────────────────┴───────────────────────────────┐
    │                  ORCHESTRATION / BUSINESS LOGIC                  │
    │                                                                  │
    │   Prefect Flows          ETL Tasks          Model Tasks         │
    │   ├─ data_ingestion      ├─ data_tasks      ├─ train_seed      │
    │   ├─ model_training      ├─ drift_tasks     ├─ batch_merge     │
    │   ├─ drift_detection                        ├─ save / load     │
    │   └─ complete_pipeline                      └─ metadata        │
    └──────────────────────────────────┬───────────────────────────────┘
                                       │
    ┌──────────────────────────────────┴───────────────────────────────┐
    │                       CORE ML COMPONENTS                        │
    │                                                                  │
    │   Sentence-BERT ─► UMAP ─► HDBSCAN ─► c-TF-IDF ─► BERTopic   │
    │   (Embeddings)    (Dim Red) (Cluster) (Repr.)    (Wrapper)      │
    └──────────────────────────────────┬───────────────────────────────┘
                                       │
    ┌──────────────────────────────────┴───────────────────────────────┐
    │                        STORAGE LAYER                            │
    │                                                                  │
    │   Models (.pkl)    Metadata (JSON)    Tabular (CSV / Parquet)   │
    │   ├─ current/      ├─ topics.json     ├─ assignments.csv       │
    │   ├─ previous/     ├─ state.json      ├─ alerts.csv            │
    │   └─ archive/ts/   └─ model_meta      └─ audit_log.csv        │
    │                                                                  │
    │   MLflow (mlruns/)  ─  experiment tracking & metrics            │
    └──────────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("### Component Descriptions")

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
        <div class="info-card">
            <h3>📥 Data Ingestion Flow</h3>
            <p>Reads raw TwCS CSV, filters by time-window, cleans text (URL/mention
            removal, emoji stripping, phone masking), and saves processed parquet.</p>
        </div>
        <div class="info-card">
            <h3>🤖 Model Training Flow</h3>
            <p><strong>Seed mode:</strong> fit_transform on first batch.<br/>
            <strong>Online mode:</strong> train a fresh model on the new batch, then
            <code>merge_models()</code> with the base model so that HITL edits persist.</p>
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div class="info-card">
            <h3>📉 Drift Detection Flow</h3>
            <p>Compares current vs. previous model using:<br/>
            • <strong>Prevalence change</strong> (TVD)<br/>
            • <strong>Centroid shift</strong> (cosine distance in embedding space)<br/>
            • <strong>JS divergence</strong> on keyword distributions<br/>
            • <strong>New / disappeared topics</strong></p>
        </div>
        <div class="info-card">
            <h3>🧑‍🔬 HITL Module</h3>
            <p>Merge and relabel topics directly in the BERTopic model.
            Every action is versioned — archived model + audit log entry.</p>
        </div>
        """, unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("## Dataset — Twitter Customer Support (TwCS)")
    st.markdown("""
    The [TwCS dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
    contains **~3 million tweets** exchanged between customers and support
    agents across major brands on Twitter/X.
    """)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">📨</div>
            <div class="metric-value">~3M</div>
            <div class="metric-label">Total Tweets</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🏢</div>
            <div class="metric-value">108</div>
            <div class="metric-label">Brands Covered</div>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">📅</div>
            <div class="metric-value">2017</div>
            <div class="metric-label">Collection Year</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Key Columns")

    st.markdown("""
    | Column | Description |
    |--------|-------------|
    | `tweet_id` | Unique identifier for each tweet |
    | `author_id` | Anonymized user who sent the tweet |
    | `inbound` | `True` = customer message, `False` = agent reply |
    | `created_at` | Timestamp (used for time-windowed batching) |
    | `text` | Raw tweet body |
    | `response_tweet_id` | ID of the reply (links conversations) |
    | `in_response_to_tweet_id` | Parent tweet ID |
    """)

    st.markdown("### Preprocessing Pipeline")
    st.markdown("""
    1. **Filter** inbound tweets only (customer messages)
    2. **Sort** by `created_at` for chronological batching
    3. **Clean text**: remove URLs, @mentions, emojis; mask phone numbers and version strings;
       normalize unicode and repeated punctuation
    4. **Drop** empty / near-empty documents
    5. **Save** as partitioned Parquet for fast read
    """)

# ── METHODOLOGY ───────────────────────────────────────────────────────────────
with tab_method:
    st.markdown("## Technical Methodology")

    st.markdown("### BERTopic Pipeline")
    st.markdown("""
    BERTopic combines four modular steps:

    | Step | Component | Purpose |
    |------|-----------|---------|
    | 1 | **Sentence-BERT** (`all-MiniLM-L6-v2`) | Encode each document into a 384-dim dense vector |
    | 2 | **UMAP** (5 components, cosine metric) | Non-linear dimensionality reduction preserving local structure |
    | 3 | **HDBSCAN** (density-based clustering) | Discover clusters of arbitrary shape — outliers mapped to topic -1 |
    | 4 | **c-TF-IDF** (class-based TF-IDF) | Extract representative keywords per cluster → human-readable topics |
    """)

    st.markdown("### Online / Incremental Learning Strategy")
    st.markdown("""
    Rather than re-processing the entire corpus each time:

    1. **New batch arrives** → train a *fresh* BERTopic model on it
    2. **Merge** the new model with the existing *base* model using
       `BERTopic.merge_models()` with configurable `min_similarity`
    3. The base model **accumulates** all historical topics *and* any
       HITL edits (merges / relabels)
    4. Previous model is archived for drift comparison
    """)

    st.markdown("### Drift Detection Metrics")
    st.markdown("""
    After each batch we compare **current** vs **previous** model:

    | Metric | Formula / Idea | Threshold |
    |--------|---------------|-----------|
    | **Prevalence Change** | Total Variation Distance between topic distributions | 0.15 |
    | **Centroid Shift** | 1 − cosine_similarity(centroid_curr, centroid_prev) | 0.25 |
    | **JS Divergence** | Jensen-Shannon divergence on keyword weight distributions | 0.30 |
    | **New Topics** | Topics in current but not in previous | >5 triggers alert |
    | **Disappeared Topics** | Topics in previous but not in current | >3 triggers alert |

    Alerts are generated at **high / medium / low** severity levels.
    """)

    st.markdown("### Human-in-the-Loop Workflow")
    st.markdown("""
    ```
    Expert reviews topics in Dashboard
        │
        ├── Merge similar topics ─── BERTopic.merge_topics(docs, topics_to_merge)
        │                                └─ Model is re-saved, previous version archived
        │
        └── Relabel topics ────────── BERTopic.set_topic_labels({id: label})
                                          └─ Metadata + model updated
    ```
    Every action is logged to `hitl_audit_log.csv` with timestamp, old/new
    topics, and optional user note.
    """)

# ── EVALUATION ────────────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("## Evaluation Framework")
    st.markdown("""
    Topic model quality is assessed using both **intrinsic** and
    **extrinsic** measures.
    """)

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("""
        <div class="info-card">
            <h3>Intrinsic Metrics</h3>
            <p>
            <strong>Topic Coherence (C_v)</strong> — measures semantic
            similarity among top keywords using sliding window and
            normalized pointwise mutual information.<br/><br/>
            <strong>Topic Diversity</strong> — fraction of unique words
            across all topic representations (higher = less redundancy).
            </p>
        </div>
        """, unsafe_allow_html=True)
    with e2:
        st.markdown("""
        <div class="info-card">
            <h3>Extrinsic / Operational Metrics</h3>
            <p>
            <strong>Silhouette Score</strong> — cluster separation in
            embedding space (−1 to 1, higher is better).<br/><br/>
            <strong>Outlier Ratio</strong> — fraction of documents assigned
            to topic −1 (lower indicates better coverage).<br/><br/>
            <strong>Drift Scores</strong> — prevalence TVD, centroid shift,
            JS divergence tracked per batch.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ### Planned Comparative Analysis

    | Model | Embedding | Clustering | Representation |
    |-------|-----------|------------|----------------|
    | **BERTopic** (ours) | Sentence-BERT | HDBSCAN | c-TF-IDF |
    | **LDA (Gensim)** | BoW | Dirichlet prior | Word distributions |
    | **NMF** | TF-IDF | Non-negative factorization | Weight vectors |

    The *Model Benchmarking* page allows side-by-side comparison of
    coherence, diversity, and silhouette across these approaches.
    """)

# ── ABOUT ─────────────────────────────────────────────────────────────────────
with tab_team:
    st.markdown("## About This Project")
    st.markdown("""
    <div class="info-card">
        <h3>🎓 Academic Context</h3>
        <p>
        <strong>Course:</strong> COMP-EE 798 — Final Year Project<br/>
        <strong>Title:</strong> Online BERTopic with Human-in-the-Loop for
        Customer Support Insights<br/>
        <strong>Year:</strong> 2025 / 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Tech Stack")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("""
        **ML / NLP**
        - BERTopic
        - Sentence-Transformers
        - UMAP / HDBSCAN
        - scikit-learn
        """)
    with t2:
        st.markdown("""
        **Backend**
        - FastAPI
        - Prefect (orchestration)
        - MLflow (tracking)
        - Ollama (LLM labels)
        """)
    with t3:
        st.markdown("""
        **Frontend**
        - Streamlit
        - Plotly
        - Custom CSS theme
        """)
    with t4:
        st.markdown("""
        **Infrastructure**
        - Docker Compose
        - YAML config
        - CSV / Parquet / JSON
        - Git version control
        """)

    st.markdown("""
    ### Repository

    🔗 [github.com/krishnagajera45/Online-BERTopic-with-Human-in-the-Loop-for-Customer-Support-Insights](
    https://github.com/krishnagajera45/Online-BERTopic-with-Human-in-the-Loop-for-Customer-Support-Insights)
    """)

render_footer()
