"""Inference Page - Predict topic for new text."""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.utils.api_client import APIClient

st.set_page_config(page_title="Inference", page_icon="🔮", layout="wide")

# Initialize API client
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

api = st.session_state.api_client

st.title("🔮 Topic Inference")
st.markdown("Predict topics for new customer support messages")

st.divider()

# Example texts
st.subheader("Try an Example")

examples = {
    "Billing Issue": "I was charged twice for my subscription this month. Can you help me get a refund?",
    "Technical Support": "The app keeps crashing when I try to upload photos. I've tried restarting but it doesn't help.",
    "Account Access": "I forgot my password and the reset email isn't arriving. How can I access my account?",
    "Shipping Inquiry": "Where is my package? The tracking number says it's been in transit for a week.",
    "Product Question": "Does this model come in blue? I can only find it in black on the website."
}

example_choice = st.selectbox(
    "Select an example or write your own below:",
    options=["Custom"] + list(examples.keys())
)

if example_choice == "Custom":
    default_text = ""
else:
    default_text = examples[example_choice]

st.divider()

# Text input
st.subheader("Input Text")

input_text = st.text_area(
    "Enter customer support message:",
    value=default_text,
    height=150,
    placeholder="Type or paste a customer support message here..."
)

# Predict button
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    predict_button = st.button("🔮 Predict Topic", type="primary", use_container_width=True)

with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

if predict_button:
    if not input_text.strip():
        st.warning("⚠️ Please enter some text first!")
    else:
        with st.spinner("Analyzing text..."):
            try:
                result = api.infer_topic(input_text)
                
                st.success("✅ Prediction Complete!")
                
                st.divider()
                
                # Display results
                st.subheader("Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Topic ID",
                        value=result['topic_id']
                    )
                
                with col2:
                    st.metric(
                        label="Confidence",
                        value=f"{result['confidence']:.1%}"
                    )
                
                with col3:
                    confidence = result['confidence']
                    if confidence > 0.7:
                        confidence_label = "High"
                        confidence_color = "🟢"
                    elif confidence > 0.4:
                        confidence_label = "Medium"
                        confidence_color = "🟠"
                    else:
                        confidence_label = "Low"
                        confidence_color = "🔴"
                    
                    st.metric(
                        label="Confidence Level",
                        value=f"{confidence_color} {confidence_label}"
                    )
                
                # Topic label
                st.subheader("Predicted Topic")
                st.info(f"**{result['topic_label']}**")
                
                # Top keywords
                st.subheader("Topic Keywords")
                keywords = ', '.join(result['top_words'])
                st.write(keywords)
                
                # Interpretation
                st.divider()
                st.subheader("💡 Interpretation")
                
                if result['confidence'] > 0.7:
                    st.success(
                        "The model is highly confident about this prediction. "
                        "The input text strongly aligns with this topic's characteristics."
                    )
                elif result['confidence'] > 0.4:
                    st.warning(
                        "The model has moderate confidence. "
                        "The text may contain elements of multiple topics or be atypical."
                    )
                else:
                    st.error(
                        "Low confidence prediction. "
                        "The text may be ambiguous, contain novel content, or not fit well into existing topics."
                    )
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                st.info("Make sure the FastAPI backend is running and a model has been trained.")

st.divider()

# Information
with st.expander("ℹ️ How does this work?"):
    st.markdown("""
    **Topic Inference Process:**
    
    1. **Text Preprocessing**: Your input is cleaned (URLs removed, lowercase, etc.)
    2. **Embedding**: The text is converted to a dense vector using sentence transformers
    3. **Topic Assignment**: The embedding is compared to existing topic representations
    4. **Confidence Score**: Based on the similarity to the assigned topic
    
    **Interpreting Confidence:**
    - **High (>70%)**: Strong match to topic characteristics
    - **Medium (40-70%)**: Reasonable match but some ambiguity
    - **Low (<40%)**: Weak match, may be novel or ambiguous content
    
    **Use Cases:**
    - Auto-categorize incoming support tickets
    - Route messages to appropriate teams
    - Identify trending topics in real-time
    - Monitor topic distribution changes
    """)

