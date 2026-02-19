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

    # ── Similar Topics Finder ─────────────────────────────────────────────────
    st.markdown("#### 🔍 Similar Topics Finder")
    st.caption("Jaccard similarity between all topic pairs — identifies merge candidates automatically by comparing keyword overlap.")
    
    # Calculate all pairwise similarities
    similarity_pairs = []
    for i, t1 in enumerate(topics):
        for t2 in topics[i+1:]:
            kw1 = set(t1.get("top_words", []))
            kw2 = set(t2.get("top_words", []))
            jaccard = len(kw1 & kw2) / len(kw1 | kw2) if (kw1 | kw2) else 0
            if jaccard > 0:  # Only show pairs with some similarity
                similarity_pairs.append({
                    "Topic 1": f"T{t1['topic_id']}: {t1['custom_label']}",
                    "Topic 2": f"T{t2['topic_id']}: {t2['custom_label']}",
                    "Topic 1 ID": t1['topic_id'],
                    "Topic 2 ID": t2['topic_id'],
                    "Similarity": jaccard,
                    "Common Keywords": ", ".join(sorted(kw1 & kw2)[:5]),
                    "T1 Docs": t1["count"],
                    "T2 Docs": t2["count"],
                })
    
    if similarity_pairs:
        sim_df = pd.DataFrame(similarity_pairs).sort_values("Similarity", ascending=False)
        
        # Filter controls
        fc1, fc2 = st.columns([1, 1])
        with fc1:
            min_sim = st.slider("Minimum similarity threshold:", 0.0, 1.0, 0.2, 0.05)
        with fc2:
            top_n = st.slider("Show top N pairs:", 5, 50, 20, 5)
        
        filtered_sim = sim_df[sim_df["Similarity"] >= min_sim].head(top_n)
        
        if not filtered_sim.empty:
            # Display table without internal IDs
            display_cols = ["Topic 1", "Topic 2", "Similarity", "Common Keywords", "T1 Docs", "T2 Docs"]
            st.dataframe(
                filtered_sim[display_cols].style.background_gradient(
                    subset=["Similarity"], cmap="YlOrRd", vmin=0, vmax=1
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(f"💡 **Tip:** Higher similarity = better merge candidates. Click on pairs above to identify topics to merge below.")
        else:
            st.info(f"No topic pairs found with similarity ≥ {min_sim:.2f}")
    else:
        st.info("Not enough topics to compare.")
    
    st.divider()

    # ── Manual Topic Selection ────────────────────────────────────────────────
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
            st.plotly_chart(fig_hm, width='stretch')

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
        st.plotly_chart(fig_ba, width='stretch')

        new_label = st.text_input("New label for the merged topic:", placeholder="e.g., Billing & Payment Issues")
        note = st.text_area("Note (optional):", placeholder="Reason for merging…")

        if st.button("🔗 Merge Topics", type="primary", width='stretch'):
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

        if st.button("🏷️ Update Label", type="primary", width='stretch'):
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

    # Prepare data with proper labels for display and hover
    tree_data = []
    for _, topic in topics_df.iterrows():
        tree_data.append({
            'display_label': f"T{topic['topic_id']}: {topic['custom_label']}",
            'topic_id': topic['topic_id'],
            'label': topic['custom_label'],
            'count': topic['count'],
            'keywords': ', '.join(topic.get('top_words', [])[:5]) if 'top_words' in topic else ''
        })
    
    tree_df = pd.DataFrame(tree_data)
    
    fig_tree = px.treemap(
        tree_df, 
        path=["display_label"], 
        values="count",
        color="count", 
        color_continuous_scale="Viridis",
        hover_data={
            'display_label': False,  # Hide from hover (already in label)
            'topic_id': True,
            'label': True, 
            'count': ':,',  # Format with commas
            'keywords': True
        },
        custom_data=['topic_id', 'label', 'keywords']
    )
    fig_tree.update_traces(
        hovertemplate='<b>%{label}</b><br>' +
                      'Topic ID: %{customdata[0]}<br>' +
                      'Label: %{customdata[1]}<br>' +
                      'Documents: %{value:,}<br>' +
                      'Keywords: %{customdata[2]}<br>' +
                      '<extra></extra>'
    )
    fig_tree.update_layout(
        template="plotly_dark",
        margin=dict(t=30, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_tree, width='stretch')

    # Sunburst
    st.markdown("### Topic Sunburst")
    
    # Prepare sunburst data with proper labels
    sun_data = []
    for _, topic in topics_df.iterrows():
        sun_data.append({
            'batch': f"Batch {topic['batch_id']}",
            'topic_label': f"T{topic['topic_id']}: {topic['custom_label']}",
            'topic_id': topic['topic_id'],
            'custom_label': topic['custom_label'],
            'count': topic['count'],
            'keywords': ', '.join(topic.get('top_words', [])[:5]) if 'top_words' in topic else ''
        })
    
    sun_df = pd.DataFrame(sun_data)
    
    fig_sun = px.sunburst(
        sun_df, 
        path=["batch", "topic_label"], 
        values="count",
        color="count", 
        color_continuous_scale="Tealgrn",
        hover_data={
            'batch': False,
            'topic_label': False,
            'topic_id': True,
            'custom_label': True,
            'count': ':,',
            'keywords': True
        },
        custom_data=['topic_id', 'custom_label', 'keywords']
    )
    fig_sun.update_traces(
        hovertemplate='<b>%{label}</b><br>' +
                      'Topic ID: %{customdata[0]}<br>' +
                      'Label: %{customdata[1]}<br>' +
                      'Documents: %{value:,}<br>' +
                      'Keywords: %{customdata[2]}<br>' +
                      '<extra></extra>'
    )
    fig_sun.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig_sun, width='stretch')

# ── AUDIT LOG ─────────────────────────────────────────────────────────────────
with tab_audit:
    st.markdown("### HITL Audit Log")
    st.caption("Complete history of all human interventions on the topic model — merges, relabels, and model updates.")

    try:
        audit = api.get_audit_log(limit=100)
        if not audit or len(audit) == 0:
            st.info("""
            📋 **No audit entries yet**
            
            The audit log will track all HITL actions including:
            - 🔗 Topic merges (combining similar topics)
            - 🏷️ Topic relabels (custom human-readable names)
            - 📦 Archived model versions (for rollback)
            
            **Get started:** Switch to the "Merge Topics" or "Relabel Topic" tabs to make your first edit!
            """)
        else:
            audit_df = pd.DataFrame(audit)
            
            # Show summary stats
            a1, a2, a3 = st.columns(3)
            with a1:
                merge_count = len(audit_df[audit_df["action_type"] == "merge"]) if "action_type" in audit_df.columns else 0
                st.metric("🔗 Total Merges", merge_count)
            with a2:
                relabel_count = len(audit_df[audit_df["action_type"] == "relabel"]) if "action_type" in audit_df.columns else 0
                st.metric("🏷️ Total Relabels", relabel_count)
            with a3:
                st.metric("📋 Total Actions", len(audit_df))
            
            st.divider()
            
            # Show table
            st.markdown("#### 📊 Audit History Table")
            st.dataframe(audit_df, use_container_width=True, hide_index=True)

            # Timeline visualization
            if "timestamp" in audit_df.columns and "action_type" in audit_df.columns:
                st.divider()
                st.markdown("### 📅 Action Timeline")
                st.caption("Chronological view of all model interventions.")
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
        st.warning(f"""
        ⚠️ **Could not load audit log**
        
        Error: {str(e)}
        
        **Troubleshooting:**
        - Ensure the FastAPI backend is running (`python -m src.api.main`)
        - Check that the `outputs/audit/` directory exists
        - Try performing a merge or relabel action to initialize the audit log
        """)

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

