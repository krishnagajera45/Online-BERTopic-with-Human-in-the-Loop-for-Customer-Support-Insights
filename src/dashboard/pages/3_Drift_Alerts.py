"""Drift Detection & Alerts — monitoring topic evolution and scoring."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast
import sys
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
    "Monitor topic evolution — prevalence shifts, centroid drift, JS divergence, and severity-coded alerts.",
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

# ── Normalise legacy dynamic reason strings (e.g. "26 new topics appeared") ──
import re
def _normalise_reason(r: str) -> str:
    if re.match(r'^\d+ new topics appeared$', str(r)):
        return 'New topics appeared'
    if re.match(r'^\d+ topics disappeared$', str(r)):
        return 'Topics disappeared'
    return r
alerts_df['reason'] = alerts_df['reason'].apply(_normalise_reason)

# ── Severity colour map ───────────────────────────────────────────────────────
SEV_COLOR = {"high": "#E17055", "medium": "#FDCB6E", "low": "#00B894"}
SEV_ICON  = {"high": "🔴", "medium": "🟠", "low": "🟢"}

# ── KPI Row ───────────────────────────────────────────────────────────────────
high = len(alerts_df[alerts_df["severity"] == "high"]) if "severity" in alerts_df.columns else 0
med  = len(alerts_df[alerts_df["severity"] == "medium"]) if "severity" in alerts_df.columns else 0
low  = len(alerts_df[alerts_df["severity"] == "low"])  if "severity" in alerts_df.columns else 0

k1, k2, k3, k4 = st.columns(4)
with k1: metric_card("🚨", len(alerts_df), "Total Alerts")
with k2: metric_card("🔴", high, "High Severity")
with k3: metric_card("🟠", med,  "Medium Severity")
with k4: metric_card("🟢", low,  "Low Severity")

st.divider()

# ── Parse metrics once for use in both tabs ───────────────────────────────────
centroid_data, prevalence_data, jsd_data = [], [], []
for _, row in alerts_df.iterrows():
    try:
        m = ast.literal_eval(row.get("metrics_json", "{}")) if isinstance(row.get("metrics_json"), str) else {}
    except Exception:
        m = {}
    win = row.get("window_start", "")
    if "centroid_shift" in m:
        centroid_data.append({
            "topic_id": row["topic_id"],
            "centroid_shift": m["centroid_shift"],
            "similarity": m.get("similarity", 0),
            "window": win,
        })
    if "prevalence_change" in m:
        prevalence_data.append({
            "prevalence_change": m["prevalence_change"],
            "threshold": m.get("threshold", 0.25),
            "window": win,
        })
    if "js_divergence" in m:
        jsd_data.append({
            "topic_id": row["topic_id"],
            "js_divergence": m["js_divergence"],
            "window": win,
        })

# ══════════════════════════════════════════════════════════════════════════════
# TWO TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_details, tab_analytics = st.tabs([
    "📋 Alert Details",
    "📊 Drift Analytics",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ALERT DETAILS
# ══════════════════════════════════════════════════════════════════════════════
with tab_details:
    st.markdown("### Filter & Inspect Alerts")
    st.caption("Use the filters below to narrow down alerts by severity or type.")

    fc1, fc2 = st.columns(2)
    with fc1:
        sev_filter = st.multiselect(
            "Severity:", ["high", "medium", "low"], default=["high", "medium", "low"]
        )
    with fc2:
        reason_opts = alerts_df["reason"].unique().tolist()
        reason_filter = st.multiselect("Alert Type:", reason_opts, default=reason_opts)

    filtered = alerts_df[
        (alerts_df["severity"].isin(sev_filter)) &
        (alerts_df["reason"].isin(reason_filter))
    ]
    st.caption(f"Showing **{len(filtered)}** of **{len(alerts_df)}** alerts")

    if filtered.empty:
        st.info("No alerts match the selected filters.")
    else:
        for _, row in filtered.iterrows():
            icon = SEV_ICON.get(row["severity"], "⚪")
            with st.expander(
                f"{icon} **{row['severity'].upper()}** — {row['reason']}  |  Topic `{row['topic_id']}`  |  {row.get('created_at', '')}"
            ):
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.markdown(f"**Alert ID:** `{row['alert_id']}`")
                    st.markdown(f"**Topic ID:** `{row['topic_id']}`")
                    st.markdown(f"**Time Window:** {row.get('window_start', '—')}")
                with dc2:
                    st.markdown(
                        f"**Severity:** {status_badge(row['severity'].upper(), row['severity'])}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Detected At:** {row.get('created_at', '—')}")

                # Show parsed metrics in a readable way
                metrics_str = row.get("metrics_json", "")
                if metrics_str:
                    try:
                        metrics = ast.literal_eval(metrics_str) if isinstance(metrics_str, str) else metrics_str
                        st.markdown("**Drift Metrics:**")
                        st.json(metrics)
                    except Exception:
                        st.code(metrics_str)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DRIFT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:

    # ── ROW 1: Alert Trends ──────────────────────────────────────────────────
    st.markdown("### Alert Trends")
    tr1, tr2 = st.columns([3, 2])

    with tr1:
        st.markdown("#### Alerts per Batch Window")
        st.caption("Number of alerts by severity for each pipeline batch — rising bars signal increasing drift.")

        # Count alerts per window_start × severity
        if "window_start" in alerts_df.columns and alerts_df["window_start"].notna().any():
            batch_sev = (
                alerts_df.groupby(["window_start", "severity"])
                .size()
                .reset_index(name="count")
            )
            # Ensure severity order: high on top
            sev_order = ["low", "medium", "high"]
            batch_sev["severity"] = pd.Categorical(batch_sev["severity"], categories=sev_order, ordered=True)
            batch_sev = batch_sev.sort_values(["window_start", "severity"])

            fig_batch = px.bar(
                batch_sev,
                x="window_start", y="count",
                color="severity",
                color_discrete_map=SEV_COLOR,
                category_orders={"severity": sev_order},
                labels={"window_start": "Batch Window", "count": "Alert Count", "severity": "Severity"},
                barmode="stack",
            )
            fig_batch.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=40),
                xaxis_tickangle=-30,
                height=300,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_batch, width='stretch')
        else:
            st.info("No batch window data available.")

    with tr2:
        st.markdown("#### Alerts by Type")
        st.caption("Which kind of drift is most common.")
        reason_counts = alerts_df["reason"].value_counts().reset_index()
        reason_counts.columns = ["reason", "count"]
        fig_r = px.bar(
            reason_counts, x="count", y="reason", orientation="h",
            color="count", color_continuous_scale="OrRd",
            labels={"count": "Count", "reason": ""},
        )
        fig_r.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            coloraxis_showscale=False,
            height=300,
        )
        st.plotly_chart(fig_r, width='stretch')

    st.divider()

    # ── ROW 2: Drift Metric Deep-Dive ─────────────────────────────────────────
    st.markdown("### Drift Metric Deep-Dive")
    dm1, dm2 = st.columns(2)

    with dm1:
        st.markdown("#### Centroid Shift by Topic")
        st.caption("How far each topic moved in semantic space (higher = more drift).")
        if centroid_data:
            cd = pd.DataFrame(centroid_data)
            fig_cs = px.bar(
                cd.groupby("topic_id")["centroid_shift"].mean().reset_index(),
                x="topic_id", y="centroid_shift",
                color="centroid_shift", color_continuous_scale="Reds",
                labels={"topic_id": "Topic ID", "centroid_shift": "Avg Shift"},
            )
            fig_cs.add_hline(y=0.35, line_dash="dash", line_color="#FDCB6E",
                             annotation_text="Threshold (0.35)")
            fig_cs.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=30, b=30),
                coloraxis_showscale=False,
                height=300,
            )
            st.plotly_chart(fig_cs, width='stretch')
        else:
            st.info("No centroid shift data in current alerts.")

    with dm2:
        st.markdown("#### Prevalence Change Over Time")
        st.caption("How much the topic distribution (TVD) shifted each batch.")
        if prevalence_data:
            pd_df = pd.DataFrame(prevalence_data)
            fig_pc = px.line(
                pd_df, x="window", y="prevalence_change", markers=True,
                color_discrete_sequence=["#E17055"],
                labels={"window": "Batch Window", "prevalence_change": "TVD"},
            )
            fig_pc.add_hline(y=pd_df["threshold"].iloc[0], line_dash="dash",
                             line_color="#FDCB6E", annotation_text="Threshold")
            fig_pc.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=30, b=30),
                height=300,
            )
            st.plotly_chart(fig_pc, width='stretch')
        else:
            st.info("No prevalence change data in current alerts.")

    # ── ROW 3: JS Divergence ──────────────────────────────────────────────────
    if jsd_data:
        st.markdown("#### Keyword Divergence (JS) by Topic")
        st.caption("How much each topic's keywords changed (higher = bigger shift).")
        jd = pd.DataFrame(jsd_data)
        fig_jsd = px.bar(
            jd.groupby("topic_id")["js_divergence"].mean().reset_index(),
            x="topic_id", y="js_divergence",
            color="js_divergence", color_continuous_scale="Purples",
            labels={"topic_id": "Topic ID", "js_divergence": "Avg JS Divergence"},
        )
        fig_jsd.add_hline(y=0.4, line_dash="dash", line_color="#FDCB6E",
                          annotation_text="Threshold (0.4)")
        fig_jsd.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30, b=30),
            coloraxis_showscale=False,
            height=300,
        )
        st.plotly_chart(fig_jsd, width='stretch')
    else:
        st.info("No keyword divergence data in current alerts.")

render_footer()

