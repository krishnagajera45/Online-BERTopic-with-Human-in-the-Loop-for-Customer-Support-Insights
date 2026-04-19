"""Model Benchmarking - Three-way temporal evaluation: BERTopic vs LDA vs NMF."""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.utils.api_client import APIClient
from src.dashboard.components.theme import (
    inject_custom_css, page_header, render_footer,
)
from src.utils.metrics_paths import (
    bertopic_metrics_path,
    lda_metrics_path,
    nmf_metrics_path,
    read_json_first_existing,
)

st.set_page_config(page_title="Model Benchmarking", page_icon="⚖️", layout="wide")
inject_custom_css()

if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()
api = st.session_state.api_client

page_header(
    "Model Benchmarking",
    "Three-way temporal evaluation — BERTopic (primary) · LDA (baseline 1) · NMF (baseline 2)",
    "⚖️",
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Display Settings")
    st.caption("Toggle which comparison baselines are shown alongside BERTopic.")

    show_lda = st.toggle(
        "Show LDA (Baseline 1 — Dirichlet generative)",
        value=True,
        help="Latent Dirichlet Allocation: bag-of-words generative model.",
    )
    show_nmf = st.toggle(
        "Show NMF (Baseline 2 — Matrix factorisation)",
        value=True,
        help="Non-negative Matrix Factorisation on TF-IDF: deterministic factorisation.",
    )

    st.divider()
    st.markdown("### Chart Style")
    chart_height = st.slider("Chart height (px)", 260, 480, 340, step=20)

    st.divider()
    st.markdown("### Model Guide")
    st.markdown("""
| Model | Paradigm | Features |
|---|---|---|
| **BERTopic** | Neural / Embeddings | Sentence-BERT + HDBSCAN |
| **LDA** | Generative (Dirichlet) | BoW + Variational Bayes |
| **NMF** | Matrix Factorisation | TF-IDF + Coordinate Descent |
    """)


# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "BERTopic": "#6C5CE7",
    "LDA":      "#00B894",
    "NMF":      "#E17055",
}


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_all_metrics():
    """Load temporal metrics for all three models. TTL=60 s for live refresh."""
    bt_batches, lda_batches, nmf_batches = [], [], []
    try:
        bt_batches  = api.get_bertopic_metrics_history().get("batches", [])
        lda_batches = api.get_lda_metrics_history().get("batches", [])
        nmf_batches = api.get_nmf_metrics_history().get("batches", [])
    except Exception:
        def _batches(primary, legacy_name):
            data = read_json_first_existing(primary, legacy_name)
            return data.get("batches", []) if data else []

        bt_batches = _batches(bertopic_metrics_path(), "bertopic_metrics.json")
        lda_batches = _batches(lda_metrics_path(), "lda_metrics.json")
        nmf_batches = _batches(nmf_metrics_path(), "nmf_metrics.json")
    return bt_batches, lda_batches, nmf_batches


bt_batches, lda_batches, nmf_batches = load_all_metrics()


def safe_avg(vals):
    vals = [v for v in vals if v is not None and isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def metric_vals(batches, key):
    return [b.get(key) for b in batches]


def safe_avg_training(batches, key):
    """Mean of numeric ``key`` over batches (skips missing)."""
    vals = [b.get(key) for b in batches if isinstance(b.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None


def mean_and_count(batches, key):
    """Arithmetic mean and count of batch rows with a numeric ``key``."""
    vals = [b.get(key) for b in batches if isinstance(b.get(key), (int, float))]
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


# Shared batch alignment (temporal charts + training-time plot)
bt_map = {b["batch_id"]: b for b in bt_batches if b.get("batch_id")}
lda_map = {b["batch_id"]: b for b in lda_batches if b.get("batch_id")}
nmf_map = {b["batch_id"]: b for b in nmf_batches if b.get("batch_id")}
all_ids = set(bt_map) | set(lda_map) | set(nmf_map)


def _sort_key_batches(bid):
    ts = (bt_map.get(bid) or lda_map.get(bid) or nmf_map.get(bid) or {}).get("timestamp", "")
    return (ts, bid)


batch_ids = sorted(all_ids, key=_sort_key_batches)


# ── Refresh ───────────────────────────────────────────────────────────────────
r_col, _ = st.columns([1, 5])
with r_col:
    if st.button("🔄 Refresh Metrics", help="Reload after pipeline run"):
        load_all_metrics.clear()
        st.rerun()

st.info(
    "📊 **Live dashboard** — metrics are written after each pipeline batch. "
    "Click **Refresh Metrics** above to reload after a new run.",
    icon="ℹ️",
)

st.divider()

# ── Summary cards ─────────────────────────────────────────────────────────────
st.markdown("### 📊 Model Performance Summary")
st.caption("Average evaluation metrics across all batches processed so far.")

bt_c   = metric_vals(bt_batches,  "coherence_c_v")
bt_d   = metric_vals(bt_batches,  "diversity")
bt_s   = metric_vals(bt_batches,  "silhouette_score")
lda_c  = metric_vals(lda_batches, "coherence_c_v")
lda_d  = metric_vals(lda_batches, "diversity")
lda_s  = metric_vals(lda_batches, "silhouette_score")
nmf_c  = metric_vals(nmf_batches, "coherence_c_v")
nmf_d  = metric_vals(nmf_batches, "diversity")
nmf_s  = metric_vals(nmf_batches, "silhouette_score")


def _card(title, icon, rows_html, note):
    inner = "<br>".join(rows_html)
    return (
        f'<div class="info-card" style="border-left:4px solid #6C5CE7;padding:1rem;">'
        f'<h3>{icon} {title}</h3>'
        f'<p style="font-size:1.15rem;margin:0.4rem 0;line-height:1.8;">{inner}</p>'
        f'<p style="font-size:0.78rem;color:#636E72;">{note}</p></div>'
    )


c1, c2, c3 = st.columns(3)
with c1:
    rows = [f'<span style="color:{COLORS["BERTopic"]};font-weight:700;">BERTopic</span> {safe_avg(bt_c):.3f}']
    if show_lda:
        rows.append(f'<span style="color:{COLORS["LDA"]};font-weight:700;">LDA</span> {safe_avg(lda_c):.3f}')
    if show_nmf:
        rows.append(f'<span style="color:{COLORS["NMF"]};font-weight:700;">NMF</span> {safe_avg(nmf_c):.3f}')
    st.markdown(_card("Coherence (C_v)", "📐", rows, "Higher = more interpretable topics"), unsafe_allow_html=True)

with c2:
    rows = [f'<span style="color:{COLORS["BERTopic"]};font-weight:700;">BERTopic</span> {safe_avg(bt_d):.3f}']
    if show_lda:
        rows.append(f'<span style="color:{COLORS["LDA"]};font-weight:700;">LDA</span> {safe_avg(lda_d):.3f}')
    if show_nmf:
        rows.append(f'<span style="color:{COLORS["NMF"]};font-weight:700;">NMF</span> {safe_avg(nmf_d):.3f}')
    st.markdown(_card("Topic Diversity", "🎯", rows, "Higher = less keyword overlap"), unsafe_allow_html=True)

with c3:
    rows = [f'<span style="color:{COLORS["BERTopic"]};font-weight:700;">BERTopic</span> {safe_avg(bt_s):.3f}']
    if show_lda:
        rows.append(f'<span style="color:{COLORS["LDA"]};font-weight:700;">LDA</span> {safe_avg(lda_s):.3f}')
    if show_nmf:
        rows.append(f'<span style="color:{COLORS["NMF"]};font-weight:700;">NMF</span> {safe_avg(nmf_s):.3f}')
    st.markdown(_card("Silhouette Score", "🧮", rows, "Higher = better cluster separation (−1 → 1)"), unsafe_allow_html=True)

st.divider()

# ── Training time (fair comparison) ───────────────────────────────────────────
st.markdown("### ⏱️ Training time")

# Per-model mean over all batches that record that metric (each model may differ in n).
avg_bt_excl, n_bt = mean_and_count(bt_batches, "training_time_seconds_excl_ollama")
if avg_bt_excl is None:
    avg_bt_excl, n_bt = mean_and_count(bt_batches, "training_time_seconds")
avg_lda_t, n_lda = (mean_and_count(lda_batches, "training_time_seconds") if show_lda else (None, 0))
avg_nmf_t, n_nmf = (mean_and_count(nmf_batches, "training_time_seconds") if show_nmf else (None, 0))

has_bt = avg_bt_excl is not None
has_lda = show_lda and avg_lda_t is not None
has_nmf = show_nmf and avg_nmf_t is not None
has_baselines = has_lda or has_nmf

if has_bt or has_baselines:
    parts = []
    if has_bt:
        parts.append(f"BERTopic n={n_bt}")
    if has_lda and n_lda:
        parts.append(f"LDA n={n_lda}")
    if has_nmf and n_nmf:
        parts.append(f"NMF n={n_nmf}")
    title_sub = " · ".join(parts)

    if has_bt and has_baselines:
        fig_t = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.52, 0.48],
            vertical_spacing=0.14,
            subplot_titles=(
                "BERTopic — mean seconds (excl. Ollama)",
                "Baselines — mean fit time (seconds)",
            ),
        )
        fig_t.add_trace(
            go.Bar(
                x=["BERTopic (excl. Ollama)"],
                y=[avg_bt_excl],
                marker_color=[COLORS["BERTopic"]],
                hovertemplate="<b>%{x}</b><br>Mean: %{y:.2f} s<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        bl_names, bl_secs, bl_colors = [], [], []
        if has_lda:
            bl_names.append("LDA (fit)")
            bl_secs.append(avg_lda_t)
            bl_colors.append(COLORS["LDA"])
        if has_nmf:
            bl_names.append("NMF (fit)")
            bl_secs.append(avg_nmf_t)
            bl_colors.append(COLORS["NMF"])
        fig_t.add_trace(
            go.Bar(
                x=bl_names,
                y=bl_secs,
                marker_color=bl_colors,
                hovertemplate="<b>%{x}</b><br>Mean: %{y:.4f} s<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        ymax_bl = max(bl_secs) if bl_secs else 1.0
        pad = max(ymax_bl * 0.2, 0.05)
        fig_t.update_yaxes(
            title_text="Seconds (mean)",
            gridcolor="rgba(99,110,114,0.15)",
            rangemode="tozero",
            row=1,
            col=1,
        )
        fig_t.update_yaxes(
            title_text="Seconds (mean)",
            gridcolor="rgba(99,110,114,0.15)",
            rangemode="tozero",
            range=[0, ymax_bl + pad],
            row=2,
            col=1,
        )
        fig_t.update_xaxes(tickangle=-12, row=1, col=1)
        fig_t.update_xaxes(tickangle=-12, row=2, col=1)
        fig_h = min(chart_height + 120, 520)
    elif has_bt:
        fig_t = go.Figure(
            data=[
                go.Bar(
                    x=["BERTopic (excl. Ollama)"],
                    y=[avg_bt_excl],
                    marker_color=[COLORS["BERTopic"]],
                    hovertemplate="<b>%{x}</b><br>Mean: %{y:.2f} s<extra></extra>",
                )
            ]
        )
        fig_t.update_layout(
            xaxis=dict(title="", tickangle=-12),
            yaxis=dict(title="Seconds (mean)", showgrid=True, gridcolor="rgba(99,110,114,0.15)"),
        )
        fig_h = min(chart_height + 40, 420)
    else:
        bl_names, bl_secs, bl_colors = [], [], []
        if has_lda:
            bl_names.append("LDA (fit)")
            bl_secs.append(avg_lda_t)
            bl_colors.append(COLORS["LDA"])
        if has_nmf:
            bl_names.append("NMF (fit)")
            bl_secs.append(avg_nmf_t)
            bl_colors.append(COLORS["NMF"])
        fig_t = go.Figure(
            data=[
                go.Bar(
                    x=bl_names,
                    y=bl_secs,
                    marker_color=bl_colors,
                    hovertemplate="<b>%{x}</b><br>Mean: %{y:.4f} s<extra></extra>",
                )
            ]
        )
        fig_t.update_layout(
            xaxis=dict(title="", tickangle=-12),
            yaxis=dict(title="Seconds (mean)", showgrid=True, gridcolor="rgba(99,110,114,0.15)"),
        )
        fig_h = min(chart_height + 40, 420)

    fig_t.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text=f"Mean training time per model ({title_sub})",
            font=dict(size=13, color="#B2BEC3"),
        ),
        height=fig_h,
        margin=dict(t=50, b=80, l=50, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig_t, use_container_width=True)
else:
    st.info("No training-time fields in metrics yet. Run a pipeline with the latest code to record timings.")

st.divider()

# ── Temporal line charts ──────────────────────────────────────────────────────
st.markdown("### 📈 Temporal Evaluation Analysis")
st.caption("Metrics plotted over training time — each point is one pipeline batch.")

if not batch_ids:
    st.info("📊 **No temporal data yet.** Run the pipeline to process batches.")
else:
    from datetime import datetime

    def _ts(bid):
        for m in (bt_map, lda_map, nmf_map):
            ts = m.get(bid, {}).get("timestamp")
            if ts:
                try:
                    return datetime.fromisoformat(ts)
                except Exception:
                    pass
        return None

    timestamps = [_ts(bid) for bid in batch_ids]
    valid_idx = [i for i, ts in enumerate(timestamps) if ts is not None]
    vids  = [batch_ids[i] for i in valid_idx]
    vtime = [timestamps[i] for i in valid_idx]

    if not vids:
        st.warning("⚠️ No valid timestamp data found in metrics.")
    else:
        def make_chart(metric_key, title, ylabel, height=chart_height):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=vtime, y=[bt_map.get(bid, {}).get(metric_key) for bid in vids],
                mode="lines+markers", name="BERTopic",
                line=dict(color=COLORS["BERTopic"], width=3), marker=dict(size=8),
                hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>" + ylabel + ": %{y:.4f}<extra></extra>",
            ))
            if show_lda:
                fig.add_trace(go.Scatter(
                    x=vtime, y=[lda_map.get(bid, {}).get(metric_key) for bid in vids],
                    mode="lines+markers", name="LDA",
                    line=dict(color=COLORS["LDA"], width=3), marker=dict(size=8),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>" + ylabel + ": %{y:.4f}<extra></extra>",
                ))
            if show_nmf:
                fig.add_trace(go.Scatter(
                    x=vtime, y=[nmf_map.get(bid, {}).get(metric_key) for bid in vids],
                    mode="lines+markers", name="NMF",
                    line=dict(color=COLORS["NMF"], width=3, dash="dot"),
                    marker=dict(size=8, symbol="diamond"),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>" + ylabel + ": %{y:.4f}<extra></extra>",
                ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                title=dict(text=title, font=dict(size=13, color="#B2BEC3")),
                xaxis=dict(title="Training Time", showgrid=True, gridcolor="rgba(99,110,114,0.15)", tickformat="%m/%d\n%H:%M"),
                yaxis=dict(title=ylabel, showgrid=True, gridcolor="rgba(99,110,114,0.15)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                height=height, margin=dict(t=55, b=60, l=50, r=20), hovermode="x unified",
            )
            return fig

        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(make_chart("coherence_c_v", "Topic Coherence (C_v) Over Time", "Coherence"), use_container_width=True)
        with col2:
            st.plotly_chart(make_chart("diversity", "Topic Diversity Over Time", "Diversity"), use_container_width=True)
        with col3:
            st.plotly_chart(make_chart("silhouette_score", "Silhouette Score Over Time", "Silhouette"), use_container_width=True)

        def _bt_train_sec(bid):
            b = bt_map.get(bid, {})
            v = b.get("training_time_seconds_excl_ollama")
            if isinstance(v, (int, float)):
                return v
            return b.get("training_time_seconds")

        y_bt = [_bt_train_sec(bid) for bid in vids]
        y_lda = [lda_map.get(bid, {}).get("training_time_seconds") for bid in vids] if show_lda else None
        y_nmf = [nmf_map.get(bid, {}).get("training_time_seconds") for bid in vids] if show_nmf else None
        has_bl_lines = (y_lda and any(isinstance(v, (int, float)) for v in y_lda)) or (
            y_nmf and any(isinstance(v, (int, float)) for v in y_nmf)
        )

        if has_bl_lines and any(isinstance(v, (int, float)) for v in y_bt):
            fig_time = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.52, 0.48],
                vertical_spacing=0.12,
                subplot_titles=(
                    "BERTopic — seconds per batch (excl. Ollama)",
                    "Baselines — fit time per batch (seconds)",
                ),
            )
            fig_time.add_trace(
                go.Scatter(
                    x=vtime,
                    y=y_bt,
                    mode="lines+markers",
                    name="BERTopic (excl. Ollama)",
                    line=dict(color=COLORS["BERTopic"], width=3),
                    marker=dict(size=8),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Seconds: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            if show_lda:
                fig_time.add_trace(
                    go.Scatter(
                        x=vtime,
                        y=y_lda,
                        mode="lines+markers",
                        name="LDA (fit)",
                        line=dict(color=COLORS["LDA"], width=3),
                        marker=dict(size=8),
                        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Seconds: %{y:.4f}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )
            if show_nmf:
                fig_time.add_trace(
                    go.Scatter(
                        x=vtime,
                        y=y_nmf,
                        mode="lines+markers",
                        name="NMF (fit)",
                        line=dict(color=COLORS["NMF"], width=3, dash="dot"),
                        marker=dict(size=8, symbol="diamond"),
                        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Seconds: %{y:.4f}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )
            bl_vals = [v for v in (y_lda or []) + (y_nmf or []) if isinstance(v, (int, float))]
            ymax_bl = max(bl_vals) if bl_vals else 1.0
            pad = max(ymax_bl * 0.15, 0.02)
            fig_time.update_yaxes(
                title_text="Seconds",
                gridcolor="rgba(99,110,114,0.15)",
                rangemode="tozero",
                row=1,
                col=1,
            )
            fig_time.update_yaxes(
                title_text="Seconds",
                gridcolor="rgba(99,110,114,0.15)",
                rangemode="tozero",
                range=[0, ymax_bl + pad],
                row=2,
                col=1,
            )
            fig_time.update_xaxes(
                title="Batch time",
                showgrid=True,
                gridcolor="rgba(99,110,114,0.15)",
                tickformat="%m/%d\n%H:%M",
                row=1,
                col=1,
            )
            fig_time.update_xaxes(
                title="Batch time",
                showgrid=True,
                gridcolor="rgba(99,110,114,0.15)",
                tickformat="%m/%d\n%H:%M",
                row=2,
                col=1,
            )
            _th = min(chart_height + 100, 480)
        else:
            fig_time = go.Figure()
            fig_time.add_trace(
                go.Scatter(
                    x=vtime,
                    y=y_bt,
                    mode="lines+markers",
                    name="BERTopic (excl. Ollama)",
                    line=dict(color=COLORS["BERTopic"], width=3),
                    marker=dict(size=8),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Seconds: %{y:.2f}<extra></extra>",
                )
            )
            if show_lda:
                fig_time.add_trace(
                    go.Scatter(
                        x=vtime,
                        y=y_lda,
                        mode="lines+markers",
                        name="LDA (fit)",
                        line=dict(color=COLORS["LDA"], width=3),
                        marker=dict(size=8),
                        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Seconds: %{y:.4f}<extra></extra>",
                    )
                )
            if show_nmf:
                fig_time.add_trace(
                    go.Scatter(
                        x=vtime,
                        y=y_nmf,
                        mode="lines+markers",
                        name="NMF (fit)",
                        line=dict(color=COLORS["NMF"], width=3, dash="dot"),
                        marker=dict(size=8, symbol="diamond"),
                        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Seconds: %{y:.4f}<extra></extra>",
                    )
                )
            fig_time.update_layout(
                xaxis=dict(
                    title="Batch time",
                    showgrid=True,
                    gridcolor="rgba(99,110,114,0.15)",
                    tickformat="%m/%d\n%H:%M",
                ),
                yaxis=dict(title="Seconds", showgrid=True, gridcolor="rgba(99,110,114,0.15)"),
            )
            _th = chart_height

        fig_time.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(
                text="Training time per batch (seconds)",
                font=dict(size=13, color="#B2BEC3"),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=_th,
            margin=dict(t=55, b=60, l=50, r=20),
            hovermode="x unified",
        )
        st.plotly_chart(fig_time, use_container_width=True)

st.divider()

# ── Radar chart ───────────────────────────────────────────────────────────────
st.markdown("### 🕸️ Multi-Metric Radar Comparison")
st.caption("Average scores across all three metrics on a single chart.")

categories   = ["Coherence (C_v)", "Diversity", "Silhouette Score"]
bt_vals_avg  = [safe_avg(bt_c),  safe_avg(bt_d),  safe_avg(bt_s)]
lda_vals_avg = [safe_avg(lda_c), safe_avg(lda_d), safe_avg(lda_s)]
nmf_vals_avg = [safe_avg(nmf_c), safe_avg(nmf_d), safe_avg(nmf_s)]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=bt_vals_avg + [bt_vals_avg[0]], theta=categories + [categories[0]],
    fill="toself", name="BERTopic",
    line=dict(color=COLORS["BERTopic"], width=2), fillcolor="rgba(108,92,231,0.15)",
))
if show_lda:
    fig_radar.add_trace(go.Scatterpolar(
        r=lda_vals_avg + [lda_vals_avg[0]], theta=categories + [categories[0]],
        fill="toself", name="LDA",
        line=dict(color=COLORS["LDA"], width=2), fillcolor="rgba(0,184,148,0.15)",
    ))
if show_nmf:
    fig_radar.add_trace(go.Scatterpolar(
        r=nmf_vals_avg + [nmf_vals_avg[0]], theta=categories + [categories[0]],
        fill="toself", name="NMF",
        line=dict(color=COLORS["NMF"], width=2, dash="dot"), fillcolor="rgba(225,112,85,0.15)",
    ))
fig_radar.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 1], showticklabels=True, gridcolor="rgba(99,110,114,0.25)"),
        angularaxis=dict(gridcolor="rgba(99,110,114,0.25)"),
    ),
    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    height=420, margin=dict(t=30, b=90, l=60, r=60),
)
st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ── Batch table ───────────────────────────────────────────────────────────────
st.markdown("### 📋 Batch-Level Metrics Table")
st.caption("All processed batches with per-model scores. Toggle LDA / NMF on the sidebar to filter columns.")

if batch_ids:
    rows = []
    for bid in batch_ids:
        ts_raw = (bt_map.get(bid) or lda_map.get(bid) or nmf_map.get(bid) or {}).get("timestamp", "")
        row = {"Batch ID": bid, "Timestamp": ts_raw[:19].replace("T", " ") if ts_raw else "—"}
        for prefix, bmap, enabled in [
            ("BERTopic", bt_map, True),
            ("LDA",      lda_map, show_lda),
            ("NMF",      nmf_map, show_nmf),
        ]:
            if enabled:
                b = bmap.get(bid, {})
                row[f"{prefix} Coherence"]  = round(b.get("coherence_c_v")   or 0, 4)
                row[f"{prefix} Diversity"]  = round(b.get("diversity")        or 0, 4)
                row[f"{prefix} Silhouette"] = round(b.get("silhouette_score") or 0, 4)
                row[f"{prefix} Topics"]     = b.get("num_topics", "—")
        rows.append(row)
    
    # Add average row
    avg_row = {"Batch ID": "───── AVERAGE ─────", "Timestamp": "—"}
    for prefix, enabled, coh, div, sil in [
        ("BERTopic", True, bt_c, bt_d, bt_s),
        ("LDA", show_lda, lda_c, lda_d, lda_s),
        ("NMF", show_nmf, nmf_c, nmf_d, nmf_s),
    ]:
        if enabled:
            avg_row[f"{prefix} Coherence"]  = round(safe_avg(coh), 4)
            avg_row[f"{prefix} Diversity"]  = round(safe_avg(div), 4)
            avg_row[f"{prefix} Silhouette"] = round(safe_avg(sil), 4)
            avg_row[f"{prefix} Topics"]     = "—"
    rows.append(avg_row)
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Run the pipeline to populate the batch metrics table.")

st.divider()

# ── Methodology documentation ─────────────────────────────────────────────────
st.markdown("### 📝 Evaluation Methodology & Definitions")

with st.expander("📐 Topic Coherence (C_v)", expanded=True):
    st.markdown("""
**Method:** Sliding window + NPMI + cosine similarity over top-K words per topic.

- **BERTopic** — c-TF-IDF top-K words → Gensim CoherenceModel
- **LDA** — Dirichlet top-K words → Gensim CoherenceModel
- **NMF** — H-matrix top-K words → Gensim CoherenceModel

**Range:** 0 → 1  ·  Higher = more interpretable topics
    """)

with st.expander("🎯 Topic Diversity"):
    st.markdown("""
**Formula:** |unique words across topics| / (top_k × num_topics)  ·  **Range:** 0 → 1

⚠️ Raw diversity numbers are **not directly comparable** across model types:

| Model | Scoring | Expected |
|---|---|---|
| BERTopic | c-TF-IDF — explicitly penalises shared words | ~0.90–0.98 |
| NMF | TF-IDF H-matrix — partial overlap suppression | ~0.50–0.85 |
| LDA | P(word\\|topic) — shares generic words by design | ~0.15–0.35 |
    """)

with st.expander("🧮 Silhouette Score"):
    st.markdown("""
**Formula:** (b−a)/max(a,b) per document, averaged  ·  **Range:** −1 → 1

| Model | Distance Space |
|---|---|
| BERTopic | UMAP 5-dim embedding space (cosine) |
| LDA | TF-IDF vectors, topic = argmax P(topic\\|doc) (cosine) |
| NMF | TF-IDF vectors, topic = argmax W row (cosine) |
    """)

with st.expander("⚙️ Experimental Setup — Three-Way Comparison"):
    st.markdown("""
| Parameter | BERTopic | LDA | NMF |
|-----------|----------|-----|-----|
| **Dataset** | Active dataset | Same | Same |
| **Preprocessing** | Minimal | Full NLP | TF-IDF stopwords |
| **Features** | Sentence-BERT 384-dim | BoW | TF-IDF (5k) |
| **Algorithm** | HDBSCAN | Variational Bayes | Coord. Descent |
| **Topic Count** | Auto (HDBSCAN) | Matched | Matched |
| **Online** | merge_models() | Cumulative retrain | Cumulative retrain |

**Why three models?**
- BERTopic vs LDA → neural embeddings vs generative BoW (classical benchmark)
- BERTopic vs NMF → confirms advantage beyond TF-IDF; NMF often outperforms LDA on short texts
- LDA vs NMF → algebraic vs probabilistic anchor point
    """)

render_footer()
