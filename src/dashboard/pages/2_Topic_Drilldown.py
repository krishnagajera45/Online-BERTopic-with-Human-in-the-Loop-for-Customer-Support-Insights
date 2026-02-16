"""Topic Drill-down — Deep dive into a single topic with examples and metrics."""
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.utils.api_client import APIClient
from src.dashboard.components.theme import (
    inject_custom_css, page_header, metric_card, status_badge, render_footer,
)

st.set_page_config(page_title="Topic Drill-down", page_icon="🔍", layout="wide")
inject_custom_css()

if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()
api = st.session_state.api_client

page_header(
    "Topic Drill-down",
    "Select any topic to inspect its keywords, example documents, and per-topic metrics.",
    "🔍",
)

try:
    topics = api.get_topics()
    if not topics:
        st.warning("No topics found — train a model first.")
        st.stop()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# ── Selector ──────────────────────────────────────────────────────────────────
topic_map = {f"Topic {t['topic_id']}: {t['custom_label']}  ({t['count']} docs)": t["topic_id"] for t in topics}
selected_label = st.selectbox("Select a topic to explore:", list(topic_map.keys()))
selected_id = topic_map[selected_label]

try:
    topic = api.get_topic_details(selected_id)
except Exception as e:
    st.error(f"Could not load topic details: {e}")
    st.stop()

st.divider()

# ── Overview Cards ────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    metric_card("🆔", topic["topic_id"], "Topic ID")
with c2:
    metric_card("📄", f"{topic['count']:,}", "Documents")
with c3:
    window = f"{topic.get('window_start', '?')}  →  {topic.get('window_end', '?')}"
    metric_card("📅", window, "Time Window")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_info, tab_examples = st.tabs(["ℹ️ Info & Keywords", "📝 Examples"])

# ── INFO ──────────────────────────────────────────────────────────────────────
with tab_info:
    i1, i2 = st.columns([1, 1])
    with i1:
        st.markdown("#### Label")
        st.info(f"**{topic.get('custom_label', 'N/A')}**")
        if topic.get("gpt_summary"):
            st.markdown("#### GPT Summary")
            st.success(topic["gpt_summary"])

    with i2:
        st.markdown("#### Top Keywords")
        keywords = topic.get("top_words", [])[:20]
        if keywords:
            # Show as styled keyword tags
            tags_html = " ".join(
                f'<span class="badge badge-info" style="margin:2px;font-size:0.85rem;">{w}</span>'
                for w in keywords
            )
            st.markdown(tags_html, unsafe_allow_html=True)

        # Keyword bar chart
        if keywords:
            st.markdown("")
            kw_df = pd.DataFrame({"keyword": keywords, "rank": list(range(1, len(keywords) + 1))})
            kw_df["weight"] = [1 / r for r in kw_df["rank"]]
            fig_kw = px.bar(
                kw_df, y="keyword", x="weight", orientation="h",
                color="weight", color_continuous_scale="Tealgrn",
            )
            fig_kw.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis=dict(autorange="reversed"),
                showlegend=False, coloraxis_showscale=False,
                height=max(250, len(keywords) * 22),
            )
            st.plotly_chart(fig_kw, width='stretch')

# ── EXAMPLES ──────────────────────────────────────────────────────────────────
with tab_examples:
    num_examples = st.slider("Examples to show:", 5, 50, 10)
    try:
        examples = api.get_topic_examples(selected_id, limit=num_examples)
        if examples:
            for i, ex in enumerate(examples, 1):
                conf = ex.get("confidence", 0)
                if conf > 0.7:
                    badge = status_badge(f"{conf:.0%}", "success")
                elif conf > 0.4:
                    badge = status_badge(f"{conf:.0%}", "medium")
                else:
                    badge = status_badge(f"{conf:.0%}", "high")

                # Get document text (prefer original text, fallback to cleaned)
                doc_text = ex.get('text') or ex.get('text_cleaned') or "[Text not available]"
                
                with st.expander(
                    f"Example {i}  —  Doc {ex.get('doc_id', '?')}  |  Confidence: {conf:.1%}",
                    expanded=(i <= 3)  # Auto-expand first 3 examples
                ):
                    st.markdown(f"**Batch:** {ex.get('batch_id', '—')}  ·  **Timestamp:** {ex.get('timestamp', '—')}  ·  Confidence: {badge}", unsafe_allow_html=True)
                    st.divider()
                    st.markdown("##### 📄 Document Content")
                    st.markdown(f"> {doc_text}")
        else:
            st.info("No example documents found for this topic.")
    except Exception as e:
        st.warning(f"Could not load examples: {e}")

render_footer()

