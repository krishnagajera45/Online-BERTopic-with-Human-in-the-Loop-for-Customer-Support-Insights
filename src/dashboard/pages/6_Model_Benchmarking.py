"""Model Benchmarking — Temporal evaluation analysis of BERTopic vs LDA."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.utils.api_client import APIClient
from src.dashboard.components.theme import (
    inject_custom_css, page_header, metric_card, render_footer,
)

st.set_page_config(page_title="Model Benchmarking", page_icon="⚖️", layout="wide")
inject_custom_css()

if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()
api = st.session_state.api_client

page_header(
    "Model Benchmarking",
    "Temporal evaluation analysis — coherence, diversity, and silhouette over batches for BERTopic vs LDA.",
    "⚖️",
)

# ── Load metrics history (with auto-refresh on batch completion) ─────────────────
@st.cache_data(ttl=60)
def load_metrics_history():
    """Load temporal metrics for both models. TTL=60s for refresh after pipeline runs."""
    import json
    from pathlib import Path
    
    # Try API first, fallback to local files
    try:
        bertopic = api.get_bertopic_metrics_history()
        lda = api.get_lda_metrics_history()
        return bertopic.get("batches", []), lda.get("batches", [])
    except Exception:
        # Fallback: read metrics files directly
        bt_batches, lda_batches = [], []
        try:
            bt_path = Path("outputs/metrics/bertopic_metrics.json")
            if bt_path.exists():
                with open(bt_path) as f:
                    bt_data = json.load(f)
                    bt_batches = bt_data.get("batches", [])
        except Exception:
            pass
        
        try:
            lda_path = Path("outputs/metrics/lda_metrics.json")
            if lda_path.exists():
                with open(lda_path) as f:
                    lda_data = json.load(f)
                    lda_batches = lda_data.get("batches", [])
        except Exception:
            pass
        
        return bt_batches, lda_batches

bertopic_batches, lda_batches = load_metrics_history()

# ── Model cards: average metrics so far ────────────────────────────────────────
st.markdown("### 📊 Model Performance Summary")
st.caption("Average evaluation metrics across all batches processed so far.")

def safe_avg(vals):
    vals = [v for v in vals if v is not None and isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0

bt_coherence = [b.get("coherence_c_v") for b in bertopic_batches]
bt_diversity = [b.get("diversity") for b in bertopic_batches]
bt_silhouette = [b.get("silhouette_score") for b in bertopic_batches]
lda_coherence = [b.get("coherence_c_v") for b in lda_batches]
lda_diversity = [b.get("diversity") for b in lda_batches]
lda_silhouette = [b.get("silhouette_score") for b in lda_batches]

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="info-card" style="border-left: 4px solid #6C5CE7;">
        <h3>📐 Coherence (C_v)</h3>
        <p style="font-size:1.4rem; margin:0.5rem 0;">
            <span style="color:#6C5CE7;">BERTopic</span> {bt:.3f} &nbsp;·&nbsp;
            <span style="color:#636E72;">LDA</span> {lda:.3f}
        </p>
        <p style="font-size:0.8rem; color:#636E72;">Higher = more interpretable topics</p>
    </div>
    """.format(
        bt=safe_avg(bt_coherence) if bertopic_batches else 0,
        lda=safe_avg(lda_coherence) if lda_batches else 0
    ), unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="info-card" style="border-left: 4px solid #00CEC9;">
        <h3>🎯 Diversity</h3>
        <p style="font-size:1.4rem; margin:0.5rem 0;">
            <span style="color:#6C5CE7;">BERTopic</span> {bt:.3f} &nbsp;·&nbsp;
            <span style="color:#636E72;">LDA</span> {lda:.3f}
        </p>
        <p style="font-size:0.8rem; color:#636E72;">Higher = less keyword overlap</p>
    </div>
    """.format(
        bt=safe_avg(bt_diversity) if bertopic_batches else 0,
        lda=safe_avg(lda_diversity) if lda_batches else 0
    ), unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="info-card" style="border-left: 4px solid #FD79A8;">
        <h3>🧮 Silhouette Score</h3>
        <p style="font-size:1.4rem; margin:0.5rem 0;">
            <span style="color:#6C5CE7;">BERTopic</span> {bt:.3f} &nbsp;·&nbsp;
            <span style="color:#636E72;">LDA</span> {lda:.3f}
        </p>
        <p style="font-size:0.8rem; color:#636E72;">Higher = better cluster separation</p>
    </div>
    """.format(
        bt=safe_avg(bt_silhouette) if bertopic_batches else 0,
        lda=safe_avg(lda_silhouette) if lda_batches else 0
    ), unsafe_allow_html=True)

st.divider()

# ── Temporal evaluation charts ─────────────────────────────────────────────────
st.markdown("### 📈 Temporal Evaluation Analysis")
refresh_col, _ = st.columns([1, 4])
with refresh_col:
    if st.button("🔄 Refresh Metrics", help="Reload metrics after pipeline run"):
        load_metrics_history.clear()
        st.rerun()
st.markdown("""
<div class="info-card" style="background: linear-gradient(135deg, rgba(108,92,231,0.12), rgba(0,206,201,0.08)); border: 2px solid rgba(108,92,231,0.35); padding: 1rem; margin-bottom: 1rem;">
    <p style="margin:0; color:#DFE6E9; font-size:0.9rem;">
    📊 <strong>Live charts</strong> — Metrics are computed from merged/cumulative models after each batch.
    Click <strong>Refresh Metrics</strong> above after running the pipeline to see updated trends.
    </p>
</div>
""", unsafe_allow_html=True)

# Build unified batch order (chronological by timestamp)
bertopic_map = {b["batch_id"]: b for b in bertopic_batches if b.get("batch_id")}
lda_map = {b["batch_id"]: b for b in lda_batches if b.get("batch_id")}
all_batch_ids = list(set(bertopic_map.keys()) | set(lda_map.keys()))

def sort_key(bid):
    ts = bertopic_map.get(bid, lda_map.get(bid, {})).get("timestamp") or ""
    return (ts, bid)

batch_ids = sorted(all_batch_ids, key=sort_key)

if not batch_ids:
    st.info("📊 **No temporal data yet.** Run the pipeline to process batches — metrics will appear here after each run.")
else:
    # Extract timestamps for x-axis (use actual training/execution time)
    from datetime import datetime
    
    def get_timestamp(batch_id):
        """Get timestamp from either BERTopic or LDA batch."""
        bt_batch = bertopic_map.get(batch_id, {})
        lda_batch = lda_map.get(batch_id, {})
        ts = bt_batch.get("timestamp") or lda_batch.get("timestamp")
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except:
                return None
        return None
    
    timestamps = [get_timestamp(bid) for bid in batch_ids]
    # Filter out None values
    valid_indices = [i for i, ts in enumerate(timestamps) if ts is not None]
    valid_batch_ids = [batch_ids[i] for i in valid_indices]
    valid_timestamps = [timestamps[i] for i in valid_indices]
    
    if not valid_timestamps:
        st.warning("⚠️ No valid timestamp data found in metrics.")
    else:
        def make_temporal_chart(metric_key, title, ylabel):
            bt_vals = [bertopic_map.get(bid, {}).get(metric_key) for bid in valid_batch_ids]
            lda_vals = [lda_map.get(bid, {}).get(metric_key) for bid in valid_batch_ids]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=valid_timestamps, y=bt_vals, mode="lines+markers", name="BERTopic",
                line=dict(color="#6C5CE7", width=3), marker=dict(size=8),
                hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>" + ylabel + ": %{y:.4f}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=valid_timestamps, y=lda_vals, mode="lines+markers", name="LDA",
                line=dict(color="#636E72", width=3), marker=dict(size=8),
                hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>" + ylabel + ": %{y:.4f}<extra></extra>"
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=dict(text=title, font=dict(size=14, color="#B2BEC3")),
                xaxis=dict(
                    title="Training Time",
                    showgrid=True,
                    gridcolor="rgba(99,110,114,0.15)",
                    tickformat="%m/%d\n%H:%M",  # Date on top line, time on bottom
                ),
                yaxis=dict(title=ylabel, showgrid=True, gridcolor="rgba(99,110,114,0.15)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                height=320,
                margin=dict(t=50, b=60, l=50, r=20),
                hovermode="x unified"
            )
            return fig
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(
            make_temporal_chart("coherence_c_v", "Topic Coherence (C_v) Over Time", "Coherence"),
            width="stretch"
        )
    with col2:
        st.plotly_chart(
            make_temporal_chart("diversity", "Topic Diversity Over Time", "Diversity"),
            width="stretch"
        )
    with col3:
        st.plotly_chart(
            make_temporal_chart("silhouette_score", "Silhouette Score Over Time", "Silhouette"),
            width="stretch"
        )

st.divider()

# ── Documentation tab ───────────────────────────────────────────────────────────
st.markdown("### 📝 Evaluation Methodology & Definitions")

with st.expander("📐 Topic Coherence (C_v)", expanded=True):
    st.markdown("""
    **Method:** Sliding window + Normalized Pointwise Mutual Information (NPMI) + cosine similarity over top-K words per topic.
    
    **Implementation:** Uses Gensim's `CoherenceModel` with C_v metric on the original documents and computed embeddings.
    
    **Range:** 0 → 1 (higher is better)
    
    **Interpretation:** Measures how semantically related the top keywords in a topic are to each other.
    Higher coherence indicates more interpretable, human-readable topics that cluster semantically related words.
    
    **Why it matters:** BERTopic leverages contextual transformer embeddings (Sentence-BERT) which capture richer
    semantic relationships compared to bag-of-words approaches, typically resulting in higher coherence scores.
    """)

with st.expander("🎯 Topic Diversity"):
    st.markdown("""
    **Formula:** |unique_words_across_all_topics| / (top_k_words_per_topic × num_topics)
    
    **Range:** 0 → 1 (higher is better)
    
    **Interpretation:** Proportion of unique words across all topic representations. Higher diversity score
    means less keyword overlap between topics — each topic is more distinct and specialized.
    
    **Why it matters:** Diversity prevents topics from being too similar or redundant. A model with high
    coherence but low diversity might have many overlapping topics. The ideal model balances both metrics.
    
    **Implementation:** Computed by collecting top-N keywords from all topics (excluding -1) and calculating
    the ratio of unique terms to total possible terms (N × num_topics).
    """)

with st.expander("🧮 Silhouette Score"):
    st.markdown("""
    **Formula:** For each document: (b − a) / max(a, b), then averaged across all documents.
    - **a** = mean intra-cluster distance (how far the document is from others in its topic)
    - **b** = mean nearest-cluster distance (how far the document is from the nearest other topic)
    
    **Range:** −1 → 1 (higher is better)
    - **> 0.5:** Strong, well-separated clusters
    - **0.2 - 0.5:** Reasonable structure
    - **< 0.2:** Weak or overlapping clusters
    
    **Interpretation:** Measures cluster quality in embedding space: how similar documents are to their
    assigned topic vs. other topics. Higher scores indicate tighter, more cohesive topic clusters.
    
    **Why it matters:** BERTopic's HDBSCAN clustering on UMAP-reduced embeddings typically produces
    tighter clusters compared to probabilistic approaches like LDA, often resulting in higher silhouette scores.
    
    **Note:** Computed on the embedding space (384-dim for Sentence-BERT or 5-dim UMAP), not raw text.
    """)

with st.expander("⚙️ Experimental Setup"):
    st.markdown("""
    | Parameter | BERTopic (Ours) | LDA (Baseline) |
    |-----------|-----------------|----------------|
    | **Dataset** | Twitter Customer Support (TwCS) inbound only | Same |
    | **Preprocessing** | Minimal (URLs, mentions, emojis removed) | Full NLP (tokenize + stopwords + lemmatize) |
    | **Embedding** | all-MiniLM-L6-v2 (384-dim, Sentence-BERT) | Bag-of-Words (BoW) / TF-IDF |
    | **Dim Reduction** | UMAP (n_components=5, metric=cosine) | N/A (LDA operates on BoW) |
    | **Clustering** | HDBSCAN (density-based, auto min_cluster_size) | Variational Bayes (Dirichlet prior) |
    | **Topic Representation** | c-TF-IDF + Ollama (DeepSeek-R1:1.5b) for labels | Top-N weighted words per topic |
    | **Topic Count** | Auto-detected via HDBSCAN | Fixed to match BERTopic for fair comparison |
    | **Online Strategy** | merge_models() for cumulative learning | Full corpus retrain (cumulative) |
    | **Outlier Handling** | Topic -1 excluded from all metrics | No explicit outlier concept |
    | **Evaluation Scope** | Cumulative (all batches merged) | Cumulative (trained on full history) |
    
    ### Fair Comparison Methodology
    
    **Temporal Alignment:** Both models are evaluated on the same cumulative corpus after each batch:
    - **BERTopic:** Uses `merge_models()` to accumulate topics across batches
    - **LDA:** Retrains on the full cumulative corpus up to current batch
    
    **Topic Count Alignment:** LDA is configured to use the same number of topics as BERTopic discovers
    (excluding outlier topic -1) to ensure fair coherence and diversity comparisons.
    
    **Metrics Timing:** All metrics (coherence, diversity, silhouette) are computed after each batch
    completes and models are merged/retrained. Timestamps represent actual training completion time.
    
    **Why This Matters:** This setup ensures we compare apples-to-apples — both models see the same
    data, both use cumulative learning, and metrics reflect the same evaluation window.
    """)

render_footer()
