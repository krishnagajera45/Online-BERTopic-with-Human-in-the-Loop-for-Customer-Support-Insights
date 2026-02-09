"""Human-in-the-Loop Editor — merge, relabel, and curate topics interactively."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.utils.api_client import APIClient
from src.dashboard.components.theme import (
    inject_custom_css, page_header, metric_card, status_badge, render_footer,
)

st.set_page_config(page_title="Human-in-the-Loop", page_icon="✏️", layout="wide")
inject_custom_css()

if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()
api = st.session_state.api_client

page_header(
    "Human-in-the-Loop Editor",
    "Merge similar topics, relabel for clarity, and review the full audit trail. "
    "Every action updates the BERTopic model and is version-archived.",
    "✏️",
)

try:
    topics = api.get_topics()
    if not topics:
        st.warning("No topics found — train a model first.")
        st.stop()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

topics_df = pd.DataFrame(topics)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_merge, tab_relabel, tab_impact, tab_audit, tab_versions = st.tabs([
    "🔗 Merge Topics",
    "🏷️ Relabel Topic",
    "📊 Impact Preview",
    "📋 Audit Log",
    "🗂️ Version History",
])

# ── MERGE ─────────────────────────────────────────────────────────────────────
with tab_merge:
    st.markdown("### Merge Similar Topics")
    st.caption("Select 2+ topics to combine into one. The first selected topic's ID is kept.")

    merge_col1, merge_col2 = st.columns([2, 1])

    with merge_col1:
        topic_opts = {
            f"Topic {t['topic_id']}: {t['custom_label']} ({t['count']} docs)": t["topic_id"]
            for t in topics
        }
        selected_merge = st.multiselect(
            "Select topics to merge:",
            options=list(topic_opts.keys()),
        )

    if len(selected_merge) >= 2:
        merge_ids = [topic_opts[t] for t in selected_merge]

        with merge_col2:
            st.markdown("#### Selected for Merge")
            for tid in merge_ids:
                t = next((x for x in topics if x["topic_id"] == tid), {})
                kw = ", ".join(t.get("top_words", [])[:4])
                st.markdown(
                    f'{status_badge(f"#{tid}", "info")} '
                    f'**{t.get("custom_label", "")}** — {t.get("count", 0)} docs · _{kw}_',
                    unsafe_allow_html=True,
                )

        # Shared keywords preview
        st.markdown("#### Keyword Overlap Preview")
        word_sets = [set(t.get("top_words", [])) for t in topics if t["topic_id"] in merge_ids]
        if word_sets:
            common = set.intersection(*word_sets)
            all_words = set.union(*word_sets)
            overlap_pct = len(common) / max(len(all_words), 1) * 100
            st.progress(min(overlap_pct / 100, 1.0), text=f"Keyword overlap: {overlap_pct:.0f}%")
            if common:
                st.success(f"Common keywords: {', '.join(sorted(common))}")
        
        # Jaccard similarity heatmap
        st.markdown("#### 🔥 Topic Similarity Matrix")
        st.caption("Jaccard similarity between selected topics — higher values = more similar.")
        
        if len(merge_ids) >= 2:
            kw_sets = {tid: set(t.get("top_words", [])) for t in topics if t["topic_id"] in merge_ids for tid in [t["topic_id"]]}
            matrix = []
            for i in merge_ids:
                row = []
                for j in merge_ids:
                    si, sj = kw_sets.get(i, set()), kw_sets.get(j, set())
                    row.append(len(si & sj) / len(si | sj) if (si | sj) else 0)
                matrix.append(row)

            labels = [f"T{i}" for i in merge_ids]
            fig_hm = go.Figure(go.Heatmap(
                z=matrix, x=labels, y=labels,
                colorscale="Viridis", colorbar=dict(title="Jaccard"),
                text=[[f"{val:.2f}" for val in row] for row in matrix],
                texttemplate="%{text}",
                textfont={"size": 12},
            ))
            fig_hm.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=min(400, len(merge_ids) * 80 + 100),
                margin=dict(t=10, b=30, l=10, r=10),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

        # Before / After size visualization
        before_sizes = {tid: next((t["count"] for t in topics if t["topic_id"] == tid), 0) for tid in merge_ids}
        merged_size = sum(before_sizes.values())

        bdf = pd.DataFrame([
            {"label": f"Topic {tid}", "count": c, "stage": "Before"}
            for tid, c in before_sizes.items()
        ] + [{"label": "Merged", "count": merged_size, "stage": "After"}])

        fig_ba = px.bar(
            bdf, x="label", y="count", color="stage",
            color_discrete_map={"Before": "#636E72", "After": "#6C5CE7"},
            barmode="group",
        )
        fig_ba.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20),
            height=280,
        )
        st.plotly_chart(fig_ba, use_container_width=True)

        new_label = st.text_input("New label for the merged topic:", placeholder="e.g., Billing & Payment Issues")
        note = st.text_area("Note (optional):", placeholder="Reason for merging…")

        if st.button("🔗 Merge Topics", type="primary", use_container_width=True):
            if new_label:
                with st.spinner("Merging topics and updating model…"):
                    try:
                        result = api.merge_topics(merge_ids, new_label, note)
                        st.success(f"✅ {result.get('message', 'Merge complete!')}")
                        st.toast("🎉 Topics merged successfully!", icon="✅")
                    except Exception as e:
                        st.error(f"Merge failed: {e}")
            else:
                st.warning("Please provide a new label for the merged topic.")
    elif len(selected_merge) == 1:
        st.info("👆 Select at least **2** topics to merge.")
    else:
        st.info("👆 Pick topics from the dropdown above.")

# ── RELABEL ───────────────────────────────────────────────────────────────────
with tab_relabel:
    st.markdown("### Relabel a Topic")
    st.caption("Update the human-readable label for any topic. The model is re-saved automatically.")

    rl_col1, rl_col2 = st.columns([1, 1])

    with rl_col1:
        rl_opts = {f"Topic {t['topic_id']}: {t['custom_label']}": t["topic_id"] for t in topics}
        rl_selected = st.selectbox("Select topic:", list(rl_opts.keys()), key="rl_sel")
        rl_id = rl_opts[rl_selected]
        rl_topic = next(t for t in topics if t["topic_id"] == rl_id)

        st.markdown(f"**Current label:** {rl_topic['custom_label']}")
        kw = ", ".join(rl_topic.get("top_words", [])[:6])
        st.markdown(f"**Keywords:** {kw}")
        if rl_topic.get("gpt_summary"):
            st.markdown(f"**GPT hint:** _{rl_topic['gpt_summary']}_")

    with rl_col2:
        new_rl = st.text_input("New label:", key="new_rl", placeholder="Enter a descriptive label")
        rl_note = st.text_area("Note:", key="rl_note", placeholder="Reason for relabeling…")

        if st.button("🏷️ Update Label", type="primary", use_container_width=True):
            if new_rl:
                with st.spinner("Updating model…"):
                    try:
                        result = api.relabel_topic(rl_id, new_rl, rl_note)
                        st.success(f"✅ {result.get('message', 'Label updated!')}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Relabel failed: {e}")
            else:
                st.warning("Enter a new label first.")

# ── IMPACT PREVIEW ────────────────────────────────────────────────────────────
with tab_impact:
    st.markdown("### Topic Landscape — Current State")
    st.caption("See the current topic distribution. After merges / relabels, refresh to see the impact.")

    fig_tree = px.treemap(
        topics_df, path=["custom_label"], values="count",
        color="count", color_continuous_scale="Viridis",
    )
    fig_tree.update_layout(
        template="plotly_dark",
        margin=dict(t=30, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # Sunburst
    st.markdown("### Topic Sunburst")
    fig_sun = px.sunburst(
        topics_df, path=["batch_id", "custom_label"], values="count",
        color="count", color_continuous_scale="Tealgrn",
    )
    fig_sun.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig_sun, use_container_width=True)

# ── AUDIT LOG ─────────────────────────────────────────────────────────────────
with tab_audit:
    st.markdown("### HITL Audit Log")
    st.caption("Complete history of all human interventions on the topic model.")

    try:
        audit = api.get_audit_log(limit=100)
        if not audit:
            st.info("No audit entries yet — perform a merge or relabel to get started.")
        else:
            audit_df = pd.DataFrame(audit)
            st.dataframe(audit_df, use_container_width=True, hide_index=True)

            # Timeline visualization
            if "timestamp" in audit_df.columns and "action_type" in audit_df.columns:
                st.markdown("### Action Timeline")
                for _, row in audit_df.iterrows():
                    icon = "🔗" if row.get("action_type") == "merge" else "🏷️"
                    st.markdown(f"""
                    <div class="timeline-item">
                        <div class="tl-title">{icon} {row.get('action_type', '').upper()} — {row.get('timestamp', '')}</div>
                        <div class="tl-desc">Old: {row.get('old_topics', '—')} → New: {row.get('new_topics', '—')}</div>
                        <div class="tl-desc">{row.get('user_note', '') or ''}</div>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load audit log: {e}")

# ── VERSION HISTORY ───────────────────────────────────────────────────────────
with tab_versions:
    st.markdown("### Model Version History")
    st.caption("Every merge/relabel creates an archived version you can inspect or rollback to.")

    try:
        vh = api.get_version_history()
        if vh and vh.get("versions"):
            st.markdown(f"**Current model:** `{vh.get('current_model_path', '—')}`")
            for v in vh["versions"]:
                meta = v.get("metadata", {})
                action = meta.get("action_type", "unknown")
                icon = "🔗" if action == "merge" else ("🏷️" if action == "relabel" else "📦")
                with st.expander(f"{icon} Version {v['timestamp']} — {v.get('size_mb', 0):.1f} MB"):
                    st.json(meta)
                    if st.button(f"⏪ Rollback to {v['timestamp']}", key=f"rb_{v['timestamp']}"):
                        with st.spinner("Rolling back…"):
                            try:
                                result = api.rollback_model(v["timestamp"])
                                st.success(result.get("message", "Rolled back!"))
                                st.rerun()
                            except Exception as e:
                                st.error(f"Rollback failed: {e}")
        else:
            st.info("No archived versions yet.")
    except Exception as e:
        st.warning(f"Could not load version history: {e}")

render_footer()

