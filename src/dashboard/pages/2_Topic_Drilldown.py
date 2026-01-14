"""Topic Drill-down Page - Deep dive into specific topics."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.utils.api_client import APIClient

st.set_page_config(page_title="Topic Drill-down", page_icon="🔍", layout="wide")

# Initialize API client
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

api = st.session_state.api_client

st.title("🔍 Topic Drill-down")
st.markdown("Deep dive into specific topics with examples and metrics")

st.divider()

try:
    # Get topics
    topics = api.get_topics()
    
    if not topics:
        st.warning("No topics found. Train a model first.")
        st.stop()
    
    # Topic selector
    topic_options = {f"Topic {t['topic_id']}: {t['custom_label']}": t['topic_id'] for t in topics}
    selected_topic_label = st.selectbox("Select a topic to explore:", list(topic_options.keys()))
    selected_topic_id = topic_options[selected_topic_label]
    
    # Get topic details
    topic = api.get_topic_details(selected_topic_id)
    
    st.divider()
    
    # Topic overview
    st.subheader(f"Topic {topic['topic_id']}: {topic['custom_label']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Document Count", topic['count'])
    
    with col2:
        st.metric("Batch ID", topic['batch_id'])
    
    with col3:
        window = f"{topic['window_start']} to {topic['window_end']}"
        st.metric("Time Window", window)
    
    # Top keywords
    st.subheader("Top Keywords")
    keywords_str = ', '.join(topic['top_words'][:20])
    st.info(keywords_str)
    
    st.divider()
    
    # Example documents
    st.subheader("Example Documents")
    
    num_examples = st.slider("Number of examples to show:", 5, 50, 10)
    
    try:
        examples = api.get_topic_examples(selected_topic_id, limit=num_examples)
        
        if examples:
            for i, example in enumerate(examples, 1):
                with st.expander(f"Example {i} - Doc ID: {example.get('doc_id', 'N/A')} (Confidence: {example.get('confidence', 0):.2%})"):
                    st.write(f"**Timestamp:** {example.get('timestamp', 'N/A')}")
                    st.write(f"**Batch:** {example.get('batch_id', 'N/A')}")
                    # Note: We don't have the actual text in assignments, 
                    # would need to join with original data
                    st.info("Full text would be displayed here in production")
        else:
            st.info("No examples found for this topic.")
    
    except Exception as e:
        st.warning(f"Could not load examples: {e}")
    
    st.divider()
    
    # Drift metrics (if available)
    st.subheader("Drift Metrics")
    
    st.info("Drift metrics for individual topics will be displayed here when available")
    st.caption("This would show centroid shift, keyword divergence, etc.")

except Exception as e:
    st.error(f"Error: {e}")

