"""Drift Detection & Alerts — monitoring topic evolution and scoring."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.utils.api_client import APIClient
from src.dashboard.components.theme import (
    inject_custom_css, page_header, metric_card, status_badge, render_footer,
)

st.set_page_config(page_title="Drift & Alerts", page_icon="⚠️", layout="wide")
inject_custom_css()

if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()
api = st.session_state.api_client

page_header(
    "Drift Detection & Alerts",
    "Monitor topic evolution — prevalence shifts, centroid drift, JS divergence, "
    "new / disappeared topics, and severity-coded alerts.",
    "⚠️",
)

# ── Load alerts ───────────────────────────────────────────────────────────────
try:
    alerts_raw = api.get_alerts(limit=200)
except Exception as e:
    st.error(f"Could not load alerts: {e}")
    render_footer()
    st.stop()

if not alerts_raw:
    st.success("✅ No drift alerts — all topics are stable!")
    st.info("Drift alerts will appear here when significant topic changes are detected across batches.")
    render_footer()
    st.stop()

alerts_df = pd.DataFrame(alerts_raw)

# ── KPI Row ───────────────────────────────────────────────────────────────────
high = len(alerts_df[alerts_df["severity"] == "high"]) if "severity" in alerts_df.columns else 0
med = len(alerts_df[alerts_df["severity"] == "medium"]) if "severity" in alerts_df.columns else 0
low = len(alerts_df[alerts_df["severity"] == "low"]) if "severity" in alerts_df.columns else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card("🚨", len(alerts_df), "Total Alerts")
with k2:
    metric_card("🔴", high, "High Severity")
with k3:
    metric_card("🟠", med, "Medium Severity")
with k4:
    metric_card("🟢", low, "Low Severity")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_timeline, tab_detail, tab_metrics, tab_score = st.tabs([
    "📅 Alert Timeline",
    "📋 Alert Details",
    "📊 Drift Metrics Viz",
    "🎯 Drift Score Card",
])

# ── TIMELINE ──────────────────────────────────────────────────────────────────
with tab_timeline:
    st.markdown("### Alert Timeline")
    st.caption("Each dot is an alert — hover for details.")

    sev_color = {"high": "#E17055", "medium": "#FDCB6E", "low": "#00B894"}
    alerts_df["color"] = alerts_df["severity"].map(sev_color).fillna("#636E72")
    alerts_df["idx"] = range(len(alerts_df))

    fig_tl = px.scatter(
        alerts_df, x="created_at", y="severity",
        color="severity", color_discrete_map=sev_color,
        hover_data=["alert_id", "topic_id", "reason"],
        symbol="severity",
        size_max=12,
    )
    fig_tl.update_traces(marker_size=10)
    fig_tl.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=30),
        yaxis_title="Severity",
        xaxis_title="Time",
        height=320,
    )
    st.plotly_chart(fig_tl, width='stretch')

    # Alerts by reason bar chart
    st.markdown("### Alerts by Reason")
    reason_counts = alerts_df["reason"].value_counts().reset_index()
    reason_counts.columns = ["reason", "count"]
    fig_r = px.bar(
        reason_counts, x="count", y="reason", orientation="h",
        color="count", color_continuous_scale="OrRd",
    )
    fig_r.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_showscale=False,
        yaxis_title="",
        height=max(200, len(reason_counts) * 35),
    )
    st.plotly_chart(fig_r, width='stretch')

# ── DETAIL ────────────────────────────────────────────────────────────────────
with tab_detail:
    st.markdown("### Filter & Inspect Alerts")

    fc1, fc2 = st.columns(2)
    with fc1:
        sev_filter = st.multiselect(
            "Severity:", ["high", "medium", "low"], default=["high", "medium"]
        )
    with fc2:
        reason_opts = alerts_df["reason"].unique().tolist()
        reason_filter = st.multiselect("Reason:", reason_opts, default=reason_opts)

    filtered = alerts_df[
        (alerts_df["severity"].isin(sev_filter)) &
        (alerts_df["reason"].isin(reason_filter))
    ]
    st.caption(f"Showing {len(filtered)} of {len(alerts_df)} alerts")

    sev_icons = {"high": "🔴", "medium": "🟠", "low": "🟢"}

    for _, row in filtered.iterrows():
        icon = sev_icons.get(row["severity"], "⚪")
        with st.expander(f"{icon} {row['severity'].upper()} — {row['reason']}  |  Topic {row['topic_id']}"):
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown(f"**Alert ID:** `{row['alert_id']}`")
                st.markdown(f"**Topic ID:** {row['topic_id']}")
                st.markdown(f"**Window:** {row.get('window_start', '—')}")
            with dc2:
                st.markdown(f"**Severity:** {status_badge(row['severity'].upper(), row['severity'])}", unsafe_allow_html=True)
                st.markdown(f"**Created:** {row.get('created_at', '—')}")

            # Parse metrics
            metrics_str = row.get("metrics_json", "")
            if metrics_str:
                try:
                    metrics = ast.literal_eval(metrics_str) if isinstance(metrics_str, str) else metrics_str
                    st.json(metrics)
                except Exception:
                    st.code(metrics_str)

# ── DRIFT METRICS VIZ ────────────────────────────────────────────────────────
with tab_metrics:
    st.markdown("### Drift Metrics Visualization")
    st.caption("Extracted from alert payloads — centroid shifts, prevalence, and JS divergence.")

    # Parse metrics from alerts
    centroid_data = []
    prevalence_data = []

    for _, row in alerts_df.iterrows():
        try:
            m = ast.literal_eval(row.get("metrics_json", "{}")) if isinstance(row.get("metrics_json"), str) else {}
        except Exception:
            m = {}

        if "centroid_shift" in m:
            centroid_data.append({
                "topic_id": row["topic_id"],
                "centroid_shift": m["centroid_shift"],
                "similarity": m.get("similarity", 0),
                "window": row.get("window_start", ""),
            })
        if "prevalence_change" in m:
            prevalence_data.append({
                "prevalence_change": m["prevalence_change"],
                "threshold": m.get("threshold", 0.15),
                "window": row.get("window_start", ""),
            })

    mc1, mc2 = st.columns(2)

    with mc1:
        if centroid_data:
            cd = pd.DataFrame(centroid_data)
            st.markdown("#### Centroid Shift by Topic")
            fig_cs = px.bar(
                cd.groupby("topic_id")["centroid_shift"].mean().reset_index(),
                x="topic_id", y="centroid_shift",
                color="centroid_shift", color_continuous_scale="Reds",
                labels={"topic_id": "Topic ID", "centroid_shift": "Avg Centroid Shift"},
            )
            fig_cs.add_hline(y=0.25, line_dash="dash", line_color="#FDCB6E",
                             annotation_text="Threshold (0.25)")
            fig_cs.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=30, b=30),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_cs, width='stretch')
        else:
            st.info("No centroid shift data available.")

    with mc2:
        if prevalence_data:
            pd_df = pd.DataFrame(prevalence_data)
            st.markdown("#### Prevalence Change Over Time")
            fig_pc = px.line(
                pd_df, x="window", y="prevalence_change", markers=True,
                color_discrete_sequence=["#E17055"],
                labels={"window": "Window", "prevalence_change": "TVD"},
            )
            if not pd_df.empty:
                fig_pc.add_hline(y=pd_df["threshold"].iloc[0], line_dash="dash",
                                 line_color="#FDCB6E", annotation_text="Threshold")
            fig_pc.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=30, b=30),
            )
            st.plotly_chart(fig_pc, width='stretch')
        else:
            st.info("No prevalence change data available.")

    # Similarity heatmap
    if centroid_data:
        st.markdown("#### Topic Similarity (from centroid analysis)")
        cd = pd.DataFrame(centroid_data)
        # pivot topic vs window
        pivot = cd.pivot_table(index="topic_id", columns="window", values="similarity", aggfunc="mean")
        if not pivot.empty:
            fig_hm = px.imshow(
                pivot, color_continuous_scale="Tealgrn", aspect="auto",
                labels={"x": "Window", "y": "Topic ID", "color": "Similarity"},
            )
            fig_hm.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=30, b=10),
            )
            st.plotly_chart(fig_hm, width='stretch')

# ── DRIFT SCORE CARD ─────────────────────────────────────────────────────────
with tab_score:
    st.markdown("### System Drift Health Score")
    st.caption("An aggregate drift score computed from all alert data.")

    # Compute a simple weighted score
    weights = {"high": 3, "medium": 2, "low": 1}
    total_weight = sum(weights.get(row["severity"], 0) for _, row in alerts_df.iterrows())
    max_possible = len(alerts_df) * 3
    drift_pct = (total_weight / max_possible * 100) if max_possible > 0 else 0
    health_pct = max(0, 100 - drift_pct)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        metric_card("🎯", f"{health_pct:.0f}%", "Health Score")
    with sc2:
        metric_card("📊", f"{drift_pct:.0f}%", "Drift Intensity")
    with sc3:
        if health_pct > 70:
            level = "HEALTHY"
            badge = status_badge(level, "success")
        elif health_pct > 40:
            level = "MODERATE"
            badge = status_badge(level, "medium")
        else:
            level = "AT RISK"
            badge = status_badge(level, "high")
        st.markdown(f"### System Status\n{badge}", unsafe_allow_html=True)

    st.divider()

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_pct,
        title={"text": "Topic Stability Index", "font": {"size": 18, "color": "#DFE6E9"}},
        delta={"reference": 80, "increasing": {"color": "#00B894"}, "decreasing": {"color": "#E17055"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#636E72"},
            "bar": {"color": "#6C5CE7"},
            "bgcolor": "#1A1D23",
            "borderwidth": 2,
            "bordercolor": "#2D3142",
            "steps": [
                {"range": [0, 40], "color": "rgba(225,112,85,0.2)"},
                {"range": [40, 70], "color": "rgba(253,203,110,0.2)"},
                {"range": [70, 100], "color": "rgba(0,184,148,0.2)"},
            ],
            "threshold": {
                "line": {"color": "#E17055", "width": 4},
                "thickness": 0.75,
                "value": 40,
            },
        },
    ))
    fig_gauge.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#DFE6E9"},
        height=300,
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_gauge, width='stretch')

    # Summary table
    st.markdown("### Severity Breakdown")
    sev_summary = alerts_df["severity"].value_counts().reset_index()
    sev_summary.columns = ["Severity", "Count"]
    sev_summary["Weight"] = sev_summary["Severity"].map(weights)
    sev_summary["Score"] = sev_summary["Count"] * sev_summary["Weight"]
    st.dataframe(sev_summary, width='stretch', hide_index=True)

render_footer()

