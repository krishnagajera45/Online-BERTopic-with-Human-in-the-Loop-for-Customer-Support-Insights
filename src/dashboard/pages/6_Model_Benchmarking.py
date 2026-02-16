"""Model Benchmarking — BERTopic vs LDA comparative analysis."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.utils.api_client import APIClient
from src.dashboard.components.theme import (
    inject_custom_css, page_header, metric_card, status_badge, render_footer,
)

st.set_page_config(page_title="Model Benchmarking", page_icon="⚖️", layout="wide")
inject_custom_css()

if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()
api = st.session_state.api_client

page_header(
    "Model Benchmarking",
    "Comparative analysis of BERTopic vs LDA across coherence, diversity, and cluster-quality metrics.",
    "⚖️",
)

# ── Helper — load real BERTopic stats from topics API ────────────────────────
@st.cache_data(ttl=120)
def load_bertopic_live_stats():
    """Pull live BERTopic numbers from the running system."""
    try:
        topics = api.get_topics()
        if not topics:
            return None
        num_topics = len(topics)
        all_keywords = []
        for t in topics:
            all_keywords.extend([w for w in t.get("top_words", [])])
        unique_kw = len(set(all_keywords))
        total_kw = len(all_keywords) if all_keywords else 1
        diversity = unique_kw / total_kw if total_kw else 0
        avg_kw = total_kw / num_topics if num_topics else 0
        return {
            "num_topics": num_topics,
            "diversity": diversity,
            "avg_keywords_per_topic": avg_kw,
            "unique_keywords": unique_kw,
        }
    except Exception:
        return None


@st.cache_data(ttl=120)
def load_lda_live_stats():
    """Pull live LDA metrics from the API."""
    try:
        response = api.get_lda_metrics()
        if response and response.get('status') != 'not_available':
            return {
                "num_topics": response.get('num_topics', 0),
                "coherence_c_v": response.get('coherence_c_v', 0.0),
                "diversity": response.get('diversity', 0.0),
                "silhouette_score": response.get('silhouette_score', 0.0),
                "training_time_seconds": response.get('training_time_seconds', 0.0),
            }
        return None
    except Exception:
        return None


@st.cache_data(ttl=120)
def load_bertopic_metrics():
    """Pull live BERTopic evaluation metrics from the API."""
    try:
        response = api.get_bertopic_metrics()
        if response and response.get('status') != 'not_available':
            return {
                "coherence_c_v": response.get('coherence_c_v'),
                "silhouette_score": response.get('silhouette_score'),
                "training_time_seconds": response.get('training_time_seconds'),
            }
        return None
    except Exception:
        return None


live = load_bertopic_live_stats()
lda_live = load_lda_live_stats()
bertopic_metrics_live = load_bertopic_metrics()

# ── Helper function for formatting values ────────────────────────────────────
def fmt_val(val, precision=3, suffix=""):
    """Format value or show N/A if None."""
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        if precision == 0:
            return f"{int(val)}{suffix}"
        return f"{val:.{precision}f}{suffix}"
    return str(val)

# ── Benchmark data ───────────────────────────────────────────────────────────
# Use ONLY live data - no hardcoded defaults

# BERTopic values
bertopic_diversity = live["diversity"] if live else None
bertopic_topics = live["num_topics"] if live else None
bertopic_avg_keywords = live["avg_keywords_per_topic"] if live else None
bertopic_coherence = bertopic_metrics_live["coherence_c_v"] if bertopic_metrics_live else None
bertopic_silhouette = bertopic_metrics_live["silhouette_score"] if bertopic_metrics_live else None
bertopic_training_time = bertopic_metrics_live["training_time_seconds"] if bertopic_metrics_live else None

# LDA values - use live data only
lda_coherence = lda_live["coherence_c_v"] if lda_live else None
lda_diversity = lda_live["diversity"] if lda_live else None
lda_silhouette = lda_live["silhouette_score"] if lda_live else None
lda_topics = lda_live["num_topics"] if lda_live else None
lda_training_time = lda_live["training_time_seconds"] if lda_live else None

metrics = pd.DataFrame({
    "Metric": [
        "Topic Coherence (C_v)",
        "Topic Diversity",
        "Silhouette Score",
        "Topic Count",
        "Avg Keywords per Topic",
        "Embedding Quality",
        "Online Learning",
        "Training Time (s)",
    ],
    "BERTopic": [
        fmt_val(bertopic_coherence, 3),
        fmt_val(bertopic_diversity, 3),
        fmt_val(bertopic_silhouette, 3),
        fmt_val(bertopic_topics, 0),
        fmt_val(bertopic_avg_keywords, 1),
        "Sentence-BERT",
        "✅ Yes",
        fmt_val(bertopic_training_time, 1, "s"),
    ],
    "LDA (Gensim)": [
        fmt_val(lda_coherence, 3),
        fmt_val(lda_diversity, 3),
        fmt_val(lda_silhouette, 3),
        fmt_val(lda_topics, 0),
        "N/A",  # LDA doesn't track this
        "BoW + TF-IDF",
        "❌ No",
        fmt_val(lda_training_time, 1, "s"),
    ],
})

# Check if we have enough data for comparison charts
has_comparison_data = (
    bertopic_diversity is not None and 
    lda_coherence is not None and 
    lda_diversity is not None and 
    lda_silhouette is not None
)

# Numeric sub-table for charts (only if we have data)
if has_comparison_data:
    numeric_metrics = pd.DataFrame({
        "Metric": [
            "Topic Coherence (C_v)",
            "Topic Diversity",
            "Silhouette Score",
        ],
        "BERTopic": [
            bertopic_coherence if bertopic_coherence is not None else 0,
            bertopic_diversity,
            bertopic_silhouette if bertopic_silhouette is not None else 0
        ],
        "LDA": [lda_coherence, lda_diversity, lda_silhouette],
    })
else:
    numeric_metrics = None

# ── KPI row ──────────────────────────────────────────────────────────────────
# Show data availability status
if not live and not lda_live:
    st.warning("⚠️ **No live metrics available**. Run the pipeline to generate comparison data: `./run_full_system.sh`")
elif not lda_live:
    st.info("ℹ️ **LDA metrics not yet available**. Enable LDA in config and run pipeline to see comparison.")
    
k1, k2, k3, k4 = st.columns(4)
with k1:
    status = "Live Data ✓" if (live and lda_live) else "Awaiting Data"
    metric_card("🏆", status, "Comparison Status")
with k2:
    metrics_available = sum([
        lda_coherence is not None,
        lda_diversity is not None,
        lda_silhouette is not None
    ])
    metric_card("📏", str(metrics_available) + "/3", "LDA Metrics Available")
with k3:
    metric_card("🧩", fmt_val(bertopic_topics, 0), "BERTopic Topics")
with k4:
    diversity_pct = f"{bertopic_diversity:.0%}" if bertopic_diversity is not None else "N/A"
    metric_card("📐", diversity_pct, "BERTopic Diversity")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_compare, tab_radar, tab_detail, tab_trade, tab_method = st.tabs([
    "📊 Comparison Table",
    "🕸️ Radar Chart",
    "📈 Metric Deep-Dive",
    "⚖️ Trade-offs",
    "📝 Methodology Notes",
])

# ── Tab 1: Comparison Table ──────────────────────────────────────────────────
with tab_compare:
    st.markdown("### Full Comparison")
    st.markdown("""
    <style>
    div[data-testid="stDataFrame"] table { font-size: 1rem; }
    </style>
    """, unsafe_allow_html=True)
    st.dataframe(metrics, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Bar chart side-by-side (only if we have numeric data)
    if has_comparison_data and numeric_metrics is not None:
        bar_df = numeric_metrics.melt(id_vars="Metric", var_name="Model", value_name="Score")
        fig_bar = px.bar(
            bar_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            color_discrete_map={"BERTopic": "#6C5CE7", "LDA": "#636E72"},
        )
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="",
            yaxis_title="Score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=380,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("📊 Charts will appear after running the pipeline with LDA enabled.")

# ── Tab 2: Radar Chart ──────────────────────────────────────────────────────
with tab_radar:
    st.markdown("### Multi-Axis Radar Comparison")

    if has_comparison_data and numeric_metrics is not None:
        categories = numeric_metrics["Metric"].tolist()
        bert_vals = numeric_metrics["BERTopic"].tolist()
        lda_vals = numeric_metrics["LDA"].tolist()
        # close polygon
        categories += [categories[0]]
        bert_vals += [bert_vals[0]]
        lda_vals += [lda_vals[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=bert_vals, theta=categories, fill="toself", name="BERTopic",
            line_color="#6C5CE7", fillcolor="rgba(108,92,231,0.25)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=lda_vals, theta=categories, fill="toself", name="LDA",
            line_color="#636E72", fillcolor="rgba(99,110,114,0.20)",
        ))
        fig_radar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], color="#636E72"),
                angularaxis=dict(color="#DFE6E9"),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            height=460,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Dynamic insight based on actual data
        if lda_diversity is not None and bertopic_diversity is not None:
            if bertopic_diversity > lda_diversity:
                st.info("⬆️ Larger area = better overall quality. BERTopic shows higher diversity and better cluster separation.")
            else:
                st.info("⬆️ Larger area = better overall quality. Compare the performance across different metrics.")
    else:
        st.warning("🕸️ **Radar chart will appear after running the pipeline.**")
        st.markdown("""
        To generate the radar chart:
        1. Ensure `lda.enabled: true` in `config/config.yaml`
        2. Run: `./run_full_system.sh`
        3. Refresh this page
        """)

# ── Tab 3: Metric Deep-Dive ─────────────────────────────────────────────────
with tab_detail:
    st.markdown("### Metric Explanations")

    detail_data = [
        {
            "name": "Topic Coherence (C_v)",
            "icon": "📐",
            "bert": bertopic_coherence,
            "lda": lda_coherence,
            "desc": "Measures how semantically similar the top words in a topic are. Higher means more interpretable topics. BERTopic leverages transformer embeddings for richer semantics.",
        },
        {
            "name": "Topic Diversity",
            "icon": "🎯",
            "bert": round(bertopic_diversity, 3),
            "lda": round(lda_diversity, 3),
            "desc": "Proportion of unique words across all topics. Higher diversity means less keyword overlap between topics — each topic is more distinct.",
        },
        {
            "name": "Silhouette Score",
            "icon": "🧮",
            "bert": bertopic_silhouette,
            "lda": lda_silhouette,
            "desc": "Measures cluster quality: how similar documents are to their own topic vs. other topics. Range −1 to 1; higher is better. BERTopic's HDBSCAN produces tighter clusters.",
        },
    ]

    for d in detail_data:
        with st.container():
            st.markdown(f"#### {d['icon']} {d['name']}")
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(d["desc"])
                
                # Only show winner comparison if both values are available
                if d["bert"] is not None and d["lda"] is not None:
                    winner = "BERTopic" if d["bert"] >= d["lda"] else "LDA"
                    diff_pct = abs(d["bert"] - d["lda"]) / max(d["lda"], 0.01) * 100
                    st.markdown(f"**Winner**: {status_badge(winner, 'success' if winner == 'BERTopic' else 'medium')}   (+{diff_pct:.0f}% improvement)", unsafe_allow_html=True)
                else:
                    st.markdown("**Status**: Awaiting live metrics from pipeline")
                    
            with c2:
                if d["bert"] is not None and d["lda"] is not None:
                    fig_mini = go.Figure()
                    fig_mini.add_trace(go.Bar(x=["BERTopic"], y=[d["bert"]], marker_color="#6C5CE7", name="BERTopic", text=[round(d["bert"], 3)], textposition="outside"))
                    fig_mini.add_trace(go.Bar(x=["LDA"], y=[d["lda"]], marker_color="#636E72", name="LDA", text=[round(d["lda"], 3)], textposition="outside"))
                    fig_mini.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=200,
                        margin=dict(l=20, r=20, t=10, b=30),
                        showlegend=False,
                        yaxis=dict(range=[0, max(d["bert"], d["lda"]) * 1.35]),
                    )
                    st.plotly_chart(fig_mini, use_container_width=True)
                else:
                    st.info("📊 Chart will appear after running pipeline")
            st.markdown("---")

# ── Tab 4: Trade-offs ────────────────────────────────────────────────────────
with tab_trade:
    st.markdown("### Trade-Off Analysis")

    trade_col1, trade_col2 = st.columns(2)

    with trade_col1:
        st.markdown("""
        <div class="info-card">
            <h3>✅ BERTopic Strengths</h3>
            <ul>
                <li><strong>Semantic Understanding</strong> — Sentence-BERT captures meaning, not just word frequency</li>
                <li><strong>Online Learning</strong> — Incrementally merges new batches without full retraining</li>
                <li><strong>Automatic Topic Count</strong> — HDBSCAN discovers the natural number of topics</li>
                <li><strong>Rich Representations</strong> — c-TF-IDF gives class-based importance, not just frequency</li>
                <li><strong>Drift Detection</strong> — Track centroid shifts and prevalence changes over time</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>⚠️ BERTopic Limitations</h3>
            <ul>
                <li><strong>Higher Compute Cost</strong> — Transformer embeddings are GPU-hungry</li>
                <li><strong>Slower Inference</strong> — Each prediction requires embedding + nearest-neighbour search</li>
                <li><strong>Outlier Topics</strong> — HDBSCAN can create a large outlier topic (−1)</li>
                <li><strong>Black-Box Embeddings</strong> — Harder to explain why a document maps to a topic</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with trade_col2:
        st.markdown("""
        <div class="info-card">
            <h3>✅ LDA Strengths</h3>
            <ul>
                <li><strong>Interpretability</strong> — Probabilistic model with clear generative story</li>
                <li><strong>Fast Training</strong> — Variational inference on BoW is lightweight</li>
                <li><strong>Low Memory</strong> — No large transformer model needed</li>
                <li><strong>Established</strong> — Well-understood evaluation & tuning methods</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>⚠️ LDA Limitations</h3>
            <ul>
                <li><strong>No Online Merging</strong> — Must retrain on full corpus when data grows</li>
                <li><strong>Fixed K</strong> — Number of topics must be specified upfront</li>
                <li><strong>Bag-of-Words</strong> — Ignores word order and context</li>
                <li><strong>Lower Coherence</strong> — Tends to produce less interpretable topics on short texts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Compute-vs-quality scatterplot (only if we have the data)
    st.markdown("#### Compute vs Quality")
    
    if (bertopic_training_time is not None and lda_training_time is not None and 
        lda_coherence is not None):
        cq_df = pd.DataFrame({
            "Model": ["BERTopic", "LDA"],
            "Training Time (s)": [bertopic_training_time or 0, lda_training_time],
            "Coherence": [bertopic_coherence or 0, lda_coherence],
        })
        fig_cq = px.scatter(
            cq_df, x="Training Time (s)", y="Coherence",
            color="Model", size=[30, 30], text="Model",
            color_discrete_map={"BERTopic": "#6C5CE7", "LDA": "#636E72"},
        )
        fig_cq.update_traces(textposition="top center")
        fig_cq.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_cq, use_container_width=True)
        
        # Dynamic caption based on actual values
        if bertopic_training_time and lda_training_time and bertopic_coherence and lda_coherence:
            time_ratio = bertopic_training_time / lda_training_time
            coherence_diff_pct = ((bertopic_coherence - lda_coherence) / lda_coherence * 100)
            st.caption(f"💡 **Live Metrics**: BERTopic trades ~{time_ratio:.1f}× training time for {coherence_diff_pct:.0f}% higher coherence.")
        else:
            st.caption("ℹ️ **Note**: Some metrics are not yet available. Run pipeline for complete comparison.")
    else:
        st.warning("⏱️ **Training time data not yet available**. Run the pipeline to see compute vs quality trade-offs.")
        st.markdown("""
        This chart will show:
        - Training time comparison
        - Coherence score comparison
        - Performance/quality trade-off analysis
        """)

# ── Tab 5: Methodology Notes ────────────────────────────────────────────────
with tab_method:
    st.markdown("### Evaluation Methodology")

    # Get actual config values dynamically
    bertopic_topics_str = fmt_val(bertopic_topics, 0) if bertopic_topics else "Auto-detected"
    lda_topics_str = fmt_val(lda_topics, 0) if lda_topics else "Configured"
    
    st.markdown(f"""
    <div class="info-card">
        <h3>📋 Experimental Setup</h3>
        <table class="comparison-table">
            <tr><th>Parameter</th><th>BERTopic</th><th>LDA (Gensim)</th></tr>
            <tr><td>Dataset</td><td colspan="2">Twitter Customer Support (TwCS)</td></tr>
            <tr><td>Preprocessing</td><td>Minimal (BERT handles context)</td><td>Tokenize + stopwords + lemmatize</td></tr>
            <tr><td>Embedding</td><td>all-MiniLM-L6-v2 (384-dim)</td><td>BoW / TF-IDF</td></tr>
            <tr><td>Dim Reduction</td><td>UMAP (5 components, cosine)</td><td>N/A</td></tr>
            <tr><td>Clustering</td><td>HDBSCAN (auto min_cluster)</td><td>Variational Bayes</td></tr>
            <tr><td>Topic Count</td><td>{bertopic_topics_str}</td><td>{lda_topics_str}</td></tr>
            <tr><td>Representation</td><td>c-TF-IDF + Ollama labels</td><td>Top-N words per topic</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>📊 Metrics Definitions</h3>
        <table class="comparison-table">
            <tr><th>Metric</th><th>Formula / Method</th><th>Range</th></tr>
            <tr><td>Coherence (C_v)</td><td>Sliding window + NPMI + cosine similarity</td><td>0 → 1</td></tr>
            <tr><td>Diversity</td><td>|unique words in top-K| / (K × N_topics)</td><td>0 → 1</td></tr>
            <tr><td>Silhouette</td><td>(b − a) / max(a, b) per sample, averaged</td><td>−1 → 1</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>📝 Notes for Evaluation Report</h3>
        <ul>
            <li>Coherence and diversity are computed on the <strong>top 10 keywords per topic</strong>.</li>
            <li>Silhouette is measured on <strong>UMAP-reduced embeddings</strong> (BERTopic) vs. TF-IDF vectors (LDA).</li>
            <li>BERTopic uses <strong>online learning</strong> with merge_models for batch updates.</li>
            <li>LDA is trained on <strong>current batch only</strong> (no accumulation across batches).</li>
            <li>Both models use the <strong>same preprocessed data</strong> from upstream flow.</li>
            <li>All experiments are logged via <strong>MLflow</strong> for reproducibility.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

render_footer()
