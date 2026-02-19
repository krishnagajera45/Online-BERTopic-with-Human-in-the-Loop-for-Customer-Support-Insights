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
tab_intro, tab_arch, tab_data, tab_method, tab_eval, tab_compare, tab_team = st.tabs([
    "🎯 Introduction",
    "🏗️ Architecture",
    "📊 Data",
    "🔬 Methodology",
    "📈 Evaluation",
    "⚖️ LDA vs BERTopic",
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
            <p>Reads raw TwCS CSV, filters by configurable time-window batches (e.g., hourly, daily),
            cleans text (URL/mention removal, emoji stripping, phone/version masking), normalizes
            unicode, and saves processed data as Parquet. Filters inbound (customer) tweets only.</p>
        </div>
        <div class="info-card">
            <h3>🤖 Model Training Flow</h3>
            <p><strong>Seed mode:</strong> fit_transform on first batch to establish base model.<br/>
            <strong>Online/incremental mode:</strong> train fresh model on new batch data only, then
            merge with cumulative base model via <code>merge_models()</code>. This preserves all
            historical topics and HITL edits while incorporating new discoveries.</p>
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div class="info-card">
            <h3>📉 Drift Detection Flow</h3>
            <p>Compares current vs. previous model after each batch using:<br/>
            • <strong>Prevalence change</strong> (TVD of topic distributions)<br/>
            • <strong>Centroid shift</strong> (cosine distance in embedding space)<br/>
            • <strong>JS divergence</strong> (Jensen-Shannon on keyword distributions)<br/>
            • <strong>New / disappeared topics</strong> (excluding outlier topic -1)<br/>
            Alerts are stored in CSV with severity levels and JSON metrics for analysis.</p>
        </div>
        <div class="info-card">
            <h3>🧑‍🔬 HITL Module</h3>
            <p>Experts merge similar topics or relabel them directly in BERTopic model.
            Every action triggers model re-save and creates:
            • Archived version (timestamped .pkl)<br/>
            • Audit log entry (CSV with old/new topics, user note, timestamp)<br/>
            Supports full version history and rollback capability.</p>
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
    1. **Filter** inbound tweets only (customer messages where `inbound = True`)
    2. **Sort** by `created_at` timestamp for chronological time-window batching
    3. **Clean text** using comprehensive pipeline:
       - Remove URLs, @mentions, hashtag symbols
       - Strip emojis and special characters
       - Mask phone numbers (XXX-XXX-XXXX) and version strings (v1.2.3)
       - Normalize unicode characters (NFD decomposition)
       - Remove repeated punctuation and extra whitespace
    4. **Drop** empty or near-empty documents (< 3 characters after cleaning)
    5. **Save** as partitioned Parquet for fast I/O in batch processing
    6. **State tracking** via `processing_state.json` for resumable processing
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

    1. **New batch arrives** → train a *fresh* BERTopic model on new data only
    2. **Merge** the new batch model with the existing *cumulative* base model using
       `BERTopic.merge_models()` with `min_similarity` threshold
    3. The merged model **accumulates**:
       - All historical topics from previous batches
       - New topics discovered in the current batch
       - All HITL edits (topic merges and custom labels)
    4. Previous model is archived with timestamp for:
       - Drift comparison and alerting
       - Version history and rollback capability
    5. Topic -1 (outliers) are consistently excluded from metrics and counts

    This approach preserves human expertise while enabling continuous learning.
    """)

    st.markdown("### Drift Detection Metrics")
    st.markdown("""
    After each batch we compare **current** vs **previous** model:

    | Metric | Formula / Idea | Threshold (Alert Trigger) |
    |--------|---------------|---------------------------|
    | **Prevalence Change** | Total Variation Distance between topic distributions | 0.25 (High: >0.30, Med: >0.15, Low: >0.05) |
    | **Centroid Shift** | 1 − cosine_similarity(centroid_curr, centroid_prev) | 0.55 (High: >0.40, Med: >0.25, Low: >0.10) |
    | **JS Divergence** | Jensen-Shannon divergence on keyword weight distributions | 0.40 (High: >0.50, Med: >0.30, Low: >0.10) |
    | **New Topics** | Topics in current but not in previous (excluding outlier -1) | >10 new topics |
    | **Disappeared Topics** | Topics in previous but not in current | >6 disappeared topics |

    Alerts are generated at **high / medium / low** severity levels based on configurable thresholds.
    All thresholds are defined in `config/drift_thresholds.yaml` and can be tuned based on your data.
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

# ── LDA vs BERTopic ──────────────────────────────────────────────────────────
with tab_compare:
    st.markdown("## ⚖️ LDA vs BERTopic — Implementation Comparison")
    st.caption("Based on the exact code in `src/etl/tasks/lda_tasks.py` and `src/etl/tasks/model_tasks.py`.")

    st.divider()

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    st.markdown("### Step 1 — Text Preprocessing")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #636E72;">
            <h3>🗃️ LDA (Gensim)</h3>
            <p>
            <strong>Library:</strong> <code>gensim.utils.simple_preprocess</code>, NLTK<br/><br/>
            <strong>Pipeline (preprocess_documents_for_lda_task):</strong><br/>
            1. <code>simple_preprocess(doc, deacc=True, min_len=3)</code> — tokenize + remove accents<br/>
            2. Remove English stopwords via <code>stopwords.words('english')</code><br/>
            3. Lemmatize every token: <code>WordNetLemmatizer().lemmatize(token)</code><br/>
            4. Discard tokens shorter than 3 characters after lemmatization<br/>
            5. Build Gensim <code>Dictionary</code> + filter extremes:<br/>
            &nbsp;&nbsp;&bull; <code>no_below=5</code> (min 5 docs)<br/>
            &nbsp;&nbsp;&bull; <code>no_above=0.5</code> (max 50% of corpus)<br/>
            &nbsp;&nbsp;&bull; <code>keep_n=10000</code> (vocabulary cap)<br/>
            6. Convert to Bag-of-Words via <code>doc2bow</code><br/><br/>
            <strong>Why heavy preprocessing?</strong> BoW only captures word counts — common words like "the", "is" would dominate without aggressive filtering.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #6C5CE7;">
            <h3>🤖 BERTopic (Sentence-BERT)</h3>
            <p>
            <strong>Library:</strong> <code>sentence_transformers</code><br/><br/>
            <strong>Pipeline (initialize_bertopic_model_task):</strong><br/>
            1. Basic text cleaning only — remove URLs, @mentions, emojis<br/>
            2. <code>SentenceTransformer('all-MiniLM-L6-v2')</code> encodes the full sentence<br/>
            3. No tokenization, no stopword removal, no lemmatization needed<br/>
            4. Full sentence passed to transformer as-is<br/><br/>
            <strong>Why minimal preprocessing?</strong> BERT is pre-trained on billions of sentences — it inherently understands grammar, context, and semantics. Stopwords like "not" carry real meaning (e.g., "not happy" ≠ "happy").
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Step 2: Document Representation ───────────────────────────────────────
    st.markdown("### Step 2 — Document Representation")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #636E72;">
            <h3>🗃️ LDA — Bag-of-Words</h3>
            <p>
            <strong>Output:</strong> Sparse integer vector (~10,000 dims)<br/><br/>
            <strong>Example tweet:</strong> <em>"App keeps crashing!"</em><br/>
            After preprocessing: <code>['app', 'keep', 'crash']</code><br/>
            BoW: <code>{app: 1, keep: 1, crash: 1, ...rest: 0}</code><br/><br/>
            <strong>Limitations:</strong><br/>
            &bull; Word order lost — "not happy" = "happy not"<br/>
            &bull; No semantics — "crash" ≠ "freeze" ≠ "stop working"<br/>
            &bull; Very sparse for short tweets (mostly zeros)
            </p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #6C5CE7;">
            <h3>🤖 BERTopic — 384-dim Dense Embeddings</h3>
            <p>
            <strong>Output:</strong> Dense float vector of 384 dimensions<br/>
            <strong>Model:</strong> <code>all-MiniLM-L6-v2</code>, batch_size=32<br/><br/>
            <strong>Example tweet:</strong> <em>"App keeps crashing!"</em><br/>
            Embedding: <code>[0.23, -0.45, 0.12, ..., 0.89]</code> (384 values)<br/><br/>
            <strong>Advantages:</strong><br/>
            &bull; Semantic similarity: "crash" ≈ "freeze" ≈ "stop working"<br/>
            &bull; Context-aware: "not happy" ≠ "happy"<br/>
            &bull; Robust on short text — rich signal even from 3 words<br/>
            &bull; Semantically similar sentences map to nearby vector space points
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Step 3: Topic Discovery ────────────────────────────────────────────────
    st.markdown("### Step 3 — Topic Discovery")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #636E72;">
            <h3>🗃️ LDA — Dirichlet + Variational Bayes</h3>
            <p>
            <strong>Exact config (from lda_tasks.py):</strong><br/>
            <code>LdaModel(corpus, id2word=dictionary,<br/>
            &nbsp;num_topics=N, alpha='auto', eta='auto',<br/>
            &nbsp;passes=10, iterations=200,<br/>
            &nbsp;update_every=1, chunksize=100,<br/>
            &nbsp;per_word_topics=True)</code><br/><br/>
            <strong>How it works:</strong><br/>
            &bull; Assumes: document = mixture of K topics<br/>
            &bull; Assumes: topic = distribution over vocabulary<br/>
            &bull; <code>alpha='auto'</code>: learns document-topic concentration<br/>
            &bull; <code>eta='auto'</code>: learns topic-word concentration<br/>
            &bull; <strong>K must be fixed upfront</strong> (set equal to BERTopic's count for fair comparison)<br/>
            &bull; Each doc gets a <em>soft</em> probability vector over all topics
            </p>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #6C5CE7;">
            <h3>🤖 BERTopic — UMAP → HDBSCAN</h3>
            <p>
            <strong>Stage A — UMAP (model_config.yaml):</strong><br/>
            <code>UMAP(n_neighbors=15, n_components=5,<br/>
            &nbsp;min_dist=0.0, metric='cosine',<br/>
            &nbsp;random_state=42)</code><br/>
            Reduces 384-dim → 5-dim preserving local neighbourhood structure<br/><br/>
            <strong>Stage B — HDBSCAN:</strong><br/>
            <code>HDBSCAN(min_cluster_size=15, min_samples=5,<br/>
            &nbsp;metric='euclidean',<br/>
            &nbsp;cluster_selection_method='eom',<br/>
            &nbsp;prediction_data=True)</code><br/><br/>
            &bull; Density-based: finds clusters of any shape<br/>
            &bull; <strong>K auto-detected</strong> from data density<br/>
            &bull; Outliers assigned to Topic -1 (excluded from all metrics)<br/>
            &bull; <em>Hard</em> assignment (1 topic per document)
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Step 4: Topic Representation ──────────────────────────────────────────
    st.markdown("### Step 4 — Topic Representation & Labels")
    rp1, rp2 = st.columns(2)
    with rp1:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #636E72;">
            <h3>🗃️ LDA — Top-N Weighted Words</h3>
            <p>
            Each topic is described by the highest-probability words from the learned Dirichlet distribution.<br/><br/>
            <strong>Example output:</strong><br/>
            Topic 3: <code>['flight', 'cancel', 'refund', 'book', 'ticket']</code><br/><br/>
            <strong>Limitation:</strong> Labels must be inferred manually — the model provides only words, not a human-readable title. No bigrams.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with rp2:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #6C5CE7;">
            <h3>🤖 BERTopic — c-TF-IDF + Ollama (DeepSeek-R1:1.5b)</h3>
            <p>
            <strong>CountVectorizer (model_config.yaml):</strong><br/>
            <code>stop_words='english', min_df=5, max_df=0.95,<br/>
            &nbsp;ngram_range=(1, 2)</code> — includes bigrams!<br/><br/>
            <strong>ClassTfidfTransformer:</strong><br/>
            <code>bm25_weighting=False, reduce_frequent_words=False</code><br/>
            Scores words by how distinctive they are in each cluster vs all others.<br/><br/>
            <strong>LLM Labels (Ollama):</strong> DeepSeek-R1:1.5b reads the top keywords and generates a concise human-readable topic name automatically.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Step 5: Online / Incremental Learning ─────────────────────────────────
    st.markdown("### Step 5 — Online / Incremental Learning")
    ol1, ol2 = st.columns(2)
    with ol1:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #636E72;">
            <h3>🗃️ LDA — Full Corpus Retrain Each Batch</h3>
            <p>
            LDA has no native online learning for the full model structure.<br/><br/>
            <strong>Strategy used in this project:</strong><br/>
            After each new batch, retrain LDA on the <em>entire</em> cumulative corpus from scratch:<br/>
            <code>LdaModel(full_cumulative_corpus,<br/>
            &nbsp;num_topics=N, passes=10, ...)</code><br/><br/>
            <strong>Drawback:</strong> Training time grows with each batch. All historical documents are reprocessed every run.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with ol2:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #6C5CE7;">
            <h3>🤖 BERTopic — merge_models()</h3>
            <p>
            <strong>Strategy used in this project (model_tasks.py):</strong><br/>
            1. Train a fresh BERTopic on the new batch only<br/>
            2. Merge with existing cumulative model:<br/>
            <code>base_model.merge_models([new_batch_model],<br/>
            &nbsp;min_similarity=...)</code><br/>
            3. Archive previous model version with timestamp<br/><br/>
            <strong>What is preserved after merge:</strong><br/>
            &bull; All historical topics from prior batches<br/>
            &bull; HITL edits (merged topics, custom labels)<br/>
            &bull; New topics discovered in the current batch<br/><br/>
            <strong>Advantage:</strong> Only new data is processed per run.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Quick Comparison Table ─────────────────────────────────────────────────
    st.markdown("### 📊 Side-by-Side Parameter Reference")
    st.markdown("""
| Parameter | **LDA (Gensim)** | **BERTopic (Our Implementation)** |
|-----------|------------------|-----------------------------------|
| **Library** | `gensim.models.LdaModel` | `bertopic.BERTopic` |
| **Embedding** | Bag-of-Words (~10k dims, sparse) | `all-MiniLM-L6-v2` (384 dims, dense) |
| **Preprocessing** | Tokenize → stopwords → lemmatize → BoW | URL/mention/emoji removal only |
| **Vocabulary filter** | no_below=5, no_above=0.5, keep_n=10000 | min_df=5, max_df=0.95 (CountVectorizer) |
| **Dimensionality reduction** | None | UMAP: n_neighbors=15, n_components=5, metric=cosine |
| **Clustering** | Dirichlet (Variational Bayes) | HDBSCAN: min_cluster_size=15, min_samples=5 |
| **Ngrams** | Unigrams only | Unigrams + bigrams (1, 2) |
| **Topic count** | Fixed (set = BERTopic's auto-detected count) | Auto-detected by HDBSCAN |
| **Assignment type** | Soft (probability per topic) | Hard (1 topic per document) |
| **Outlier handling** | None | Topic -1 excluded from all metrics |
| **Topic labels** | Top-N weighted words only | c-TF-IDF keywords + Ollama LLM labels |
| **Training passes** | passes=10, iterations=200, chunksize=100 | Single forward pass (fit_transform) |
| **Online strategy** | Full corpus retrain per batch | merge_models() — new batch only |
| **HITL support** | None | merge_topics(), set_topic_labels(), versioned archive |
    """)

render_footer()
