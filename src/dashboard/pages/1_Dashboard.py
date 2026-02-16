"""
Dashboard — TwCS Online Topic Modeling System.

Central command center: cumulative vs current-batch statistics, batch explorer
with synced bubble timeline, topic distribution, keyword heatmap, and trends.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, json, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.utils.api_client import APIClient
from src.dashboard.components.theme import (
    inject_custom_css,
    page_header,
    metric_card,
    render_footer,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard · TwCS Topic Modeling",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()

# ── Session state / API ──────────────────────────────────────────────────────
if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()
api = st.session_state.api_client

page_header(
    "Dashboard",
    "Cumulative and per-batch statistics for the customer-support topic landscape — powered by BERTopic.",
    "📊",
)

# ── Connectivity ──────────────────────────────────────────────────────────────
if not api.health_check():
    st.error("⚠️ Cannot reach the FastAPI backend at **http://localhost:8000**.")
    st.info("Start it with: `python -m src.api.main`")
    render_footer()
    st.stop()

# ── Fetch all data ────────────────────────────────────────────────────────────
try:
    topics = api.get_topics()
except Exception as e:
    st.error(f"Could not load topics: {e}")
    render_footer()
    st.stop()

try:
    batch_stats = api.get_batch_stats()
except Exception:
    batch_stats = {"cumulative": {}, "batches": []}

try:
    trends_raw = api.get_trends()
except Exception:
    trends_raw = []

cumulative = batch_stats.get("cumulative", {})
batches_info = batch_stats.get("batches", [])

# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — Overall vs Current Batch Statistics  (side-by-side)
# ──────────────────────────────────────────────────────────────────────────────
overall_col, divider_col, batch_col = st.columns([1, 0.02, 1])

# ── LEFT: Overall / Cumulative ───────────────────────────────────────────────
with overall_col:
    st.markdown("### 🌐 Overall Statistics")
    st.caption("Aggregated across every batch processed so far.")

    o1, o2 = st.columns(2)
    with o1:
        metric_card("📌", cumulative.get("total_topics", len(topics)), "Total Topics Discovered")
    with o2:
        metric_card("📄", f"{cumulative.get('total_docs', 0):,}", "Total Docs Processed")

    o3, o4 = st.columns(2)
    with o3:
        metric_card("📦", cumulative.get("total_batches", len(batches_info)), "Batches Processed")
    with o4:
        last_run = cumulative.get("last_run") or "—"
        if last_run != "—":
            # Show just date+time nicely
            try:
                from datetime import datetime as _dt
                lr = _dt.fromisoformat(last_run)
                last_run = lr.strftime("%b %d, %Y %H:%M")
            except Exception:
                pass
        metric_card("🕑", last_run, "Last Pipeline Run")

    # ── Mini sparkline — docs processed per batch (cumulative line) ───────
    if batches_info:
        spark_df = pd.DataFrame(batches_info)
        spark_df["cum_docs"] = spark_df["docs"].cumsum()
        fig_spark = go.Figure()
        fig_spark.add_trace(go.Scatter(
            x=list(range(1, len(spark_df) + 1)),
            y=spark_df["cum_docs"],
            mode="lines+markers+text",
            text=spark_df["cum_docs"].astype(str),
            textposition="top center",
            line=dict(color="#6C5CE7", width=3),
            marker=dict(size=8, color="#00CEC9"),
            fill="tozeroy",
            fillcolor="rgba(108,92,231,0.10)",
        ))
        fig_spark.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=190,
            margin=dict(l=30, r=10, t=25, b=30),
            xaxis=dict(
                title="Batch #",
                dtick=1,
                showgrid=False,
            ),
            yaxis=dict(title="Cumulative Docs", showgrid=True, gridcolor="rgba(99,110,114,0.15)"),
            showlegend=False,
            title=dict(text="Documents Processed Over Time", font=dict(size=13, color="#B2BEC3")),
        )
        st.plotly_chart(fig_spark, use_container_width=True)
    
    # ── Topic evolution sparkline — topics per batch ───────
    if batches_info and trends_raw:
        # Calculate topics per batch from trends
        tdf = pd.DataFrame(trends_raw)
        topic_counts = tdf.groupby("batch_id")["topic_id"].nunique().reset_index()
        topic_counts.columns = ["batch_id", "topics"]
        
        # Merge with batches to maintain order
        batch_df = pd.DataFrame(batches_info)
        batch_df = batch_df.merge(topic_counts, on="batch_id", how="left", suffixes=("", "_from_trends"))
        batch_df["topics_from_trends"] = batch_df["topics_from_trends"].fillna(0).astype(int)
        
        fig_topic_evo = go.Figure()
        fig_topic_evo.add_trace(go.Scatter(
            x=list(range(1, len(batch_df) + 1)),
            y=batch_df["topics_from_trends"],
            mode="lines+markers+text",
            text=batch_df["topics_from_trends"].astype(str),
            textposition="top center",
            line=dict(color="#00CEC9", width=3),
            marker=dict(size=8, color="#6C5CE7"),
            fill="tozeroy",
            fillcolor="rgba(0,206,201,0.10)",
        ))
        fig_topic_evo.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=190,
            margin=dict(l=30, r=10, t=25, b=30),
            xaxis=dict(
                title="Batch #",
                dtick=1,
                showgrid=False,
            ),
            yaxis=dict(title="Topics Discovered", showgrid=True, gridcolor="rgba(99,110,114,0.15)"),
            showlegend=False,
            title=dict(text="Topic Evolution Over Batches", font=dict(size=13, color="#B2BEC3")),
        )
        st.plotly_chart(fig_topic_evo, use_container_width=True)

# ── DIVIDER ───────────────────────────────────────────────────────────────────
with divider_col:
    st.markdown(
        '<div style="border-left:2px solid #2D3142;height:100%;min-height:380px;margin:auto;width:1px;"></div>',
        unsafe_allow_html=True,
    )

# ── RIGHT: Current (latest) batch ────────────────────────────────────────────
with batch_col:
    st.markdown("### 🔄 Current Batch")
    if batches_info and trends_raw:
        # Calculate topics from trends data for accuracy (exclude outlier topic -1)
        trends_df_current = pd.DataFrame(trends_raw)
        
        # Find the most recent batch with topics > 0 (skip test batches)
        latest = None
        for batch in reversed(batches_info):
            batch_id = batch.get("batch_id")
            # Filter out outlier topic -1 for accurate count
            batch_trends = trends_df_current[
                (trends_df_current["batch_id"] == batch_id) & 
                (trends_df_current["topic_id"] != -1)
            ]
            topics_count = len(batch_trends["topic_id"].unique())
            if topics_count > 0:
                latest = batch
                latest["topics_actual"] = topics_count
                break
        
        # Fallback to absolute latest if no batch has topics
        if not latest:
            latest = batches_info[-1]
            batch_id = latest.get("batch_id")
            # Filter out outlier topic -1
            batch_trends = trends_df_current[
                (trends_df_current["batch_id"] == batch_id) & 
                (trends_df_current["topic_id"] != -1)
            ]
            latest["topics_actual"] = len(batch_trends["topic_id"].unique())
        
        st.caption(f"Most recently processed batch with topics.")

        b1, b2 = st.columns(2)
        with b1:
            metric_card("📌", latest["topics_actual"], "Topics in This Batch")
        with b2:
            metric_card("📄", f"{latest['docs']:,}", "Docs in This Batch")

        b3, b4 = st.columns(2)
        with b3:
            ws = latest.get("window_start") or latest.get("timestamp") or "—"
            metric_card("🕐", ws, "Window Start")
        with b4:
            we = latest.get("window_end") or "—"
            metric_card("🕑", we, "Window End")

        # Per-batch doc bar chart
        if len(batches_info) > 1:
            bdf = pd.DataFrame(batches_info)
            short_labels = [f"B{i+1}" for i in range(len(bdf))]
            fig_bbar = go.Figure()
            fig_bbar.add_trace(go.Bar(
                x=short_labels,
                y=bdf["docs"],
                marker_color=["#2D3142"] * (len(bdf) - 1) + ["#6C5CE7"],
                text=bdf["docs"],
                textposition="outside",
            ))
            fig_bbar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=190,
                margin=dict(l=30, r=10, t=25, b=30),
                yaxis=dict(showgrid=True, gridcolor="rgba(99,110,114,0.15)"),
                xaxis=dict(title="Batch"),
                showlegend=False,
                title=dict(text="Docs per Batch (latest highlighted)", font=dict(size=13, color="#B2BEC3")),
            )
            st.plotly_chart(fig_bbar, use_container_width=True)
    else:
        st.info("No batches processed yet. Run the pipeline to start.")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — Batch Explorer  (shared navigation drives everything below)
# ──────────────────────────────────────────────────────────────────────────────
if trends_raw:
    trends_df = pd.DataFrame(trends_raw)
    sorted_batches = sorted(trends_df["batch_id"].unique())

    # Initialise shared batch index to latest batch with topics
    if "batch_idx" not in st.session_state:
        # Find the latest batch that has topics > 0
        latest_idx = len(sorted_batches) - 1
        for i in range(len(sorted_batches) - 1, -1, -1):
            batch_id = sorted_batches[i]
            batch_info = next((b for b in batches_info if b["batch_id"] == batch_id), {})
            if batch_info.get("topics", 0) > 0:
                latest_idx = i
                break
        st.session_state.batch_idx = latest_idx

    st.markdown("### 🎛️ Batch Control Panel")
    st.markdown("""
    <div class="info-card" style="background: linear-gradient(135deg, rgba(108,92,231,0.15), rgba(0,206,201,0.10)); border: 2px solid rgba(108,92,231,0.4); padding: 1.5rem; margin-bottom: 1rem;">
        <p style="margin:0; color:#DFE6E9; font-size:0.95rem;">
        ⚡ <strong>Interactive Filter:</strong> Select a batch below to update all visualizations (distribution, bubbles, trends, heatmap) in sync.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Handle single batch vs multiple batches
    if len(sorted_batches) == 1:
        # Single batch - just show info, no slider/buttons needed
        st.info("📌 Currently viewing the only batch available. Run the pipeline to process more batches.")
    else:
        # Multiple batches - show slider + navigation buttons
        slider_col, btn_col = st.columns([4, 1])
        with slider_col:
            batch_idx_new = st.slider(
                "Select Batch",
                min_value=0,
                max_value=len(sorted_batches) - 1,
                value=st.session_state.batch_idx,
                format="Batch %d",
            )
            if batch_idx_new != st.session_state.batch_idx:
                st.session_state.batch_idx = batch_idx_new
                st.rerun()

        with btn_col:
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                if st.button("⏮", use_container_width=True, key="first_btn", help="First batch"):
                    st.session_state.batch_idx = 0
                    st.rerun()
            with b2:
                if st.button("◀", use_container_width=True, key="prev_btn", help="Previous batch"):
                    if st.session_state.batch_idx > 0:
                        st.session_state.batch_idx -= 1
                        st.rerun()
            with b3:
                if st.button("▶", use_container_width=True, key="next_btn", help="Next batch"):
                    if st.session_state.batch_idx < len(sorted_batches) - 1:
                        st.session_state.batch_idx += 1
                        st.rerun()
            with b4:
                if st.button("⏭", use_container_width=True, key="latest_btn", help="Latest batch"):
                    # Find the latest batch that has topics > 0
                    latest_idx = len(sorted_batches) - 1
                    for i in range(len(sorted_batches) - 1, -1, -1):
                        batch_id = sorted_batches[i]
                        batch_info = next((b for b in batches_info if b["batch_id"] == batch_id), {})
                        if batch_info.get("topics", 0) > 0:
                            latest_idx = i
                            break
                    st.session_state.batch_idx = latest_idx
                    st.rerun()

    sel_batch = sorted_batches[st.session_state.batch_idx]

    # Build info panel showing batch metadata
    batch_meta = next((b for b in batches_info if b["batch_id"] == sel_batch), {})
    ws = batch_meta.get("window_start") or batch_meta.get("timestamp") or "—"
    we = batch_meta.get("window_end") or "—"
    docs_in_batch = batch_meta.get("docs", 0)
    # Get actual topic count from trends data for this batch
    topics_in_batch = len(trends_df[trends_df["batch_id"] == sel_batch]["topic_id"].unique())

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1A1D23, #22262E); border: 1px solid #6C5CE7; border-radius: 12px; padding: 1rem 1.5rem; margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 200px;">
                <div style="font-size: 0.75rem; color: #B2BEC3; text-transform: uppercase; letter-spacing: 0.5px;">Selected Batch</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #6C5CE7; margin: 0.2rem 0;">#{st.session_state.batch_idx + 1} <span style="color: #636E72; font-size: 1rem;">of {len(sorted_batches)}</span></div>
            </div>
            <div style="flex: 1; min-width: 200px;">
                <div style="font-size: 0.75rem; color: #B2BEC3; text-transform: uppercase; letter-spacing: 0.5px;">Time Window</div>
                <div style="font-size: 0.95rem; font-weight: 500; color: #DFE6E9; margin: 0.2rem 0;">🕐 {ws}<br/>🕑 {we}</div>
            </div>
            <div style="flex: 1; min-width: 150px; text-align: center;">
                <div style="font-size: 0.75rem; color: #B2BEC3; text-transform: uppercase; letter-spacing: 0.5px;">Documents</div>
                <div style="font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg, #6C5CE7, #00CEC9); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{docs_in_batch:,}</div>
            </div>
            <div style="flex: 1; min-width: 150px; text-align: center;">
                <div style="font-size: 0.75rem; color: #B2BEC3; text-transform: uppercase; letter-spacing: 0.5px;">Topics</div>
                <div style="font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg, #6C5CE7, #00CEC9); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{topics_in_batch}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    batch_data = trends_df[trends_df["batch_id"] == sel_batch]
    topic_label_map = {t["topic_id"]: t.get("custom_label", f"Topic {t['topic_id']}") for t in topics}

    # ── 2a  Topic Treemap for Selected Batch ─────────────────────────────────
    st.markdown("### 🌳 Topic Hierarchy — Batch Overview")
    st.caption("Proportional topic distribution for the selected batch — larger blocks = more documents.")
    
    # Filter topics present in current batch
    batch_topic_ids = set(batch_data["topic_id"].unique())
    batch_topics = [t for t in topics if t["topic_id"] in batch_topic_ids and t.get("count", 0) > 0]
    
    if batch_topics:
        tree_df = pd.DataFrame([
            {
                "label": f"T{t['topic_id']} — {t.get('custom_label', '')}",
                "count": batch_data[batch_data["topic_id"] == t["topic_id"]]["count"].sum(),
                "keywords": ", ".join(t.get("top_words", [])[:5]),
            }
            for t in batch_topics
        ])
        tree_df = tree_df[tree_df["count"] > 0].sort_values("count", ascending=False)
        
        if not tree_df.empty:
            fig_tree = px.treemap(
                tree_df, path=["label"], values="count",
                color="count", color_continuous_scale="Viridis",
                hover_data=["keywords"],
            )
            fig_tree.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=10, l=10, r=10),
                height=400,
            )
            fig_tree.update_traces(textposition="middle center", textfont_size=12)
            st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("No topics found for this batch.")
    
    st.divider()

    # ── 2b  Topic distribution bar + pie ─────────────────────────────────────
    dist_col, pie_col = st.columns([2, 1])
    with dist_col:
        bd_sorted = batch_data.sort_values("count", ascending=False)
        bd_sorted["label"] = bd_sorted["topic_id"].map(topic_label_map).fillna(bd_sorted["topic_id"].astype(str))
        fig_dist = px.bar(
            bd_sorted, x="topic_id", y="count",
            hover_data=["label"],
            color="count", color_continuous_scale="Viridis",
            labels={"topic_id": "Topic ID", "count": "Documents"},
            title=f"Topic Distribution — Batch {st.session_state.batch_idx + 1}",
        )
        fig_dist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=30, l=30, r=10), coloraxis_showscale=False,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with pie_col:
        fig_pie = px.pie(
            batch_data, values="count", names="topic_id",
            title="Topic Share",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=10, l=10, r=10), showlegend=False,
        )
        fig_pie.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── 2c  Bubble Timeline (shows ONLY the selected batch – not overlapping) ─
    st.markdown("### 🫧 Topic Bubble View")
    st.caption("Each bubble represents a topic in the selected batch. Size = document count.")

    bubble_data = batch_data.copy()
    bubble_data["label"] = bubble_data["topic_id"].map(topic_label_map).fillna(bubble_data["topic_id"].astype(str))

    fig_bubble = px.scatter(
        bubble_data,
        x="topic_id",
        y="count",
        size="count",
        color="label",
        hover_data=["label", "count"],
        labels={"topic_id": "Topic ID", "count": "Documents"},
        size_max=55,
    )
    fig_bubble.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=40, l=40, r=10),
        height=420,
        legend=dict(font=dict(size=9), orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        xaxis=dict(title="Topic ID", dtick=1),
        yaxis=dict(title="Document Count"),
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    st.divider()

    # ── 2d  Trend line — topic counts across ALL batches ─────────────────────
    st.markdown("### 📈 Topic Trends Across Batches")
    st.caption("Line chart showing how each topic's document count evolves over batches.")

    trends_df["label"] = trends_df["topic_id"].map(topic_label_map).fillna(trends_df["topic_id"].astype(str))

    # Let user pick which topics to show (top N by total count as default)
    top_tids = (
        trends_df.groupby("topic_id")["count"].sum()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )
    chosen_tids = st.multiselect(
        "Filter topics (leave blank for top 10):",
        options=sorted(trends_df["topic_id"].unique()),
        default=top_tids,
        format_func=lambda x: f"T{x} — {topic_label_map.get(x, '')}",
    )
    if not chosen_tids:
        chosen_tids = top_tids

    filt_trends = trends_df[trends_df["topic_id"].isin(chosen_tids)]

    fig_line = px.line(
        filt_trends, x="batch_id", y="count", color="label",
        markers=True,
        labels={"batch_id": "Batch", "count": "Documents", "label": "Topic"},
    )
    # Highlight the selected batch with a shaded vertical band
    batch_list = sorted(filt_trends["batch_id"].unique())
    if sel_batch in batch_list:
        idx = batch_list.index(sel_batch)
        fig_line.add_vrect(
            x0=idx - 0.4, x1=idx + 0.4,
            fillcolor="rgba(253,121,168,0.12)", line_width=0,
            annotation_text="▼ Selected",
            annotation_position="top",
            annotation_font_color="#FD79A8",
        )
    fig_line.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=40), legend=dict(font=dict(size=9)),
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_line, use_container_width=True)

else:
    st.info("Run the ETL pipeline to generate batch-level data.")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — Top Topics  (always visible)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 🏆 Top Topics by Size")
st.caption("Most significant topics across all batches — keyword highlights and summaries.")

if topics:
    sorted_topics = sorted(topics, key=lambda x: x.get("count", 0), reverse=True)

    # Keyword cards (top 8)
    top8 = sorted_topics[:8]
    cols = st.columns(4)
    for i, topic in enumerate(top8):
        with cols[i % 4]:
            label = topic.get("custom_label", f"Topic {topic['topic_id']}")
            kw = ", ".join(topic.get("top_words", [])[:4])
            gpt = topic.get("gpt_summary") or ""
            st.markdown(f"""
            <div class="info-card">
                <h3>#{topic['topic_id']} — {label}</h3>
                <p><strong>{topic.get('count', 0)}</strong> docs &nbsp;·&nbsp; <em>{kw}</em></p>
                <p style="font-size:0.8rem; color:#636E72;">{gpt[:100]}</p>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — Complete Topics Catalog Table
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 📋 Complete Topics Catalog")
st.caption("Full sortable table with all topic metadata — click column headers to sort.")

if topics:
    catalog_df = pd.DataFrame(topics)
    
    # Select and rename columns for display
    display_cols = ["topic_id", "custom_label", "top_words", "count", "batch_id", 
                    "window_start", "window_end", "gpt_summary"]
    available_cols = [c for c in display_cols if c in catalog_df.columns]
    display_df = catalog_df[available_cols].copy()
    
    # Format top_words as comma-separated string
    if "top_words" in display_df.columns:
        display_df["top_words"] = display_df["top_words"].apply(
            lambda x: ", ".join(x[:8]) if isinstance(x, list) else str(x)
        )
    
    # Truncate GPT summary
    if "gpt_summary" in display_df.columns:
        display_df["gpt_summary"] = display_df["gpt_summary"].fillna("").apply(
            lambda x: x[:150] + ("…" if len(str(x)) > 150 else "")
        )
    
    # Rename columns for readability
    rename_map = {
        "topic_id": "ID",
        "custom_label": "Label",
        "top_words": "Keywords",
        "count": "Docs",
        "batch_id": "Batch",
        "window_start": "Window Start",
        "window_end": "Window End",
        "gpt_summary": "GPT Summary",
    }
    display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns}, inplace=True)
    
    # Display with custom styling
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=500,
        column_config={
            "ID": st.column_config.NumberColumn("ID", width="small"),
            "Docs": st.column_config.NumberColumn("Docs", format="%d"),
        }
    )
    
    # Download button
    csv_data = catalog_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Catalog (CSV)",
        data=csv_data,
        file_name="topics_catalog.csv",
        mime="text/csv",
    )
else:
    st.info("No topics available yet. Run the ETL pipeline to generate topics.")

render_footer()
