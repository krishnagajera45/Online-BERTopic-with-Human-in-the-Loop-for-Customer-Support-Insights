"""Topic Inference — predict topics for new customer support messages."""
import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.utils.api_client import APIClient
from src.dashboard.components.theme import (
    inject_custom_css, page_header, metric_card, status_badge, render_footer,
)

st.set_page_config(page_title="Inference", page_icon="🔮", layout="wide")
inject_custom_css()

if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()
api = st.session_state.api_client

page_header(
    "Topic Inference",
    "Type or paste a customer support message and let the model predict its topic in real time.",
    "🔮",
)

# ── Example library ──────────────────────────────────────────────────────────
examples = {
    "💰 Billing Issue": "I was charged twice for my subscription this month. Can you help me get a refund?",
    "📱 App Crash": "The app keeps crashing when I try to upload photos. I've tried restarting but it doesn't help.",
    "🔑 Account Access": "I forgot my password and the reset email isn't arriving. How can I access my account?",
    "📦 Shipping": "Where is my package? The tracking number says it's been in transit for a week.",
    "❓ Product Q": "Does this model come in blue? I can only find it in black on the website.",
    "📶 Connectivity": "My internet has been down for two hours. The router lights are all off.",
}

# Layout
input_col, result_col = st.columns([1, 1], gap="large")

with input_col:
    st.markdown("### 💬 Input")

    example_choice = st.selectbox(
        "Pick an example or type your own:",
        options=["✍️ Custom"] + list(examples.keys()),
    )
    default_text = "" if example_choice == "✍️ Custom" else examples[example_choice]

    input_text = st.text_area(
        "Customer support message:",
        value=default_text,
        height=180,
        placeholder="Type or paste a message here…",
    )

    bc1, bc2 = st.columns(2)
    with bc1:
        predict = st.button("🔮 Predict Topic", type="primary", width='stretch')
    with bc2:
        if st.button("🗑️ Clear", width='stretch'):
            st.rerun()

with result_col:
    st.markdown("### 📊 Result")

    if predict:
        if not input_text.strip():
            st.warning("⚠️ Please enter some text first!")
        else:
            with st.spinner("Analysing…"):
                try:
                    result = api.infer_topic(input_text)

                    # Confidence gauge
                    conf = result["confidence"]
                    if conf > 0.7:
                        conf_color, conf_label, conf_level = "#00B894", "High", "success"
                    elif conf > 0.4:
                        conf_color, conf_label, conf_level = "#FDCB6E", "Medium", "medium"
                    else:
                        conf_color, conf_label, conf_level = "#E17055", "Low", "high"

                    # KPI cards
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        metric_card("🆔", result["topic_id"], "Topic ID")
                    with r2:
                        metric_card("📊", f"{conf:.0%}", "Confidence")
                    with r3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-icon">🎯</div>
                            <div class="metric-value" style="-webkit-text-fill-color:{conf_color};">{conf_label}</div>
                            <div class="metric-label">Confidence Level</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Topic label
                    st.markdown(f"#### Predicted Topic")
                    st.info(f"**{result['topic_label']}**")

                    # Keywords as tags
                    st.markdown("#### Topic Keywords")
                    tags = " ".join(
                        f'<span class="badge badge-info" style="margin:2px;font-size:0.85rem;">{w}</span>'
                        for w in result.get("top_words", [])
                    )
                    st.markdown(tags, unsafe_allow_html=True)

                    # Confidence gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=conf * 100,
                        title={"text": "Confidence", "font": {"size": 16, "color": "#DFE6E9"}},
                        number={"suffix": "%", "font": {"color": conf_color}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#636E72"},
                            "bar": {"color": conf_color},
                            "bgcolor": "#1A1D23",
                            "borderwidth": 2,
                            "bordercolor": "#2D3142",
                            "steps": [
                                {"range": [0, 40], "color": "rgba(225,112,85,0.15)"},
                                {"range": [40, 70], "color": "rgba(253,203,110,0.15)"},
                                {"range": [70, 100], "color": "rgba(0,184,148,0.15)"},
                            ],
                        },
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=220,
                        margin=dict(t=40, b=10),
                    )
                    st.plotly_chart(fig, width='stretch')

                    # Interpretation
                    st.markdown("#### 💡 Interpretation")
                    if conf > 0.7:
                        st.success("The model is **highly confident**. The text strongly aligns with this topic's characteristics.")
                    elif conf > 0.4:
                        st.warning("**Moderate confidence** — the text may contain elements of multiple topics.")
                    else:
                        st.error("**Low confidence** — the text may be ambiguous, novel, or not fit existing topics well.")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    st.info("Ensure the FastAPI backend is running and a model is trained.")
    else:
        st.markdown("""
        <div class="info-card">
            <h3>How it works</h3>
            <p>
            <strong>1.</strong> Text is cleaned (URLs, mentions, emojis removed).<br/>
            <strong>2.</strong> Sentence-BERT encodes the text into a 384-dim vector.<br/>
            <strong>3.</strong> The vector is compared to existing topic representations.<br/>
            <strong>4.</strong> A confidence score measures alignment with the predicted topic.
            </p>
        </div>
        <div class="info-card">
            <h3>Use Cases</h3>
            <p>
            • Auto-categorize incoming support tickets<br/>
            • Route messages to the right team<br/>
            • Identify trending topics in real time<br/>
            • Monitor topic distribution changes
            </p>
        </div>
        """, unsafe_allow_html=True)

render_footer()

