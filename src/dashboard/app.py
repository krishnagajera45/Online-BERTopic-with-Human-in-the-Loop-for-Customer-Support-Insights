"""
Main Streamlit Dashboard for TwCS Topic Modeling.

This is the home page that provides an overview and navigation.
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dashboard.utils.api_client import APIClient

st.set_page_config(
    page_title="TwCS Topic Modeling Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize API client
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

api = st.session_state.api_client

# Header
st.title("📊 TwCS Topic Modeling Dashboard")
st.markdown("""
Welcome to the **Twitter Customer Support (TwCS) Online Topic Modeling System**.

This dashboard provides real-time insights into customer support conversations using BERTopic.
""")

st.divider()

# Overview metrics
col1, col2, col3 = st.columns(3)

try:
    # Get current statistics
    topics = api.get_topics()
    
    with col1:
        st.metric(
            label="📌 Total Topics",
            value=len(topics),
            help="Number of discovered topics"
        )
    
    with col2:
        total_docs = sum([t.get('count', 0) for t in topics])
        st.metric(
            label="📄 Documents Processed",
            value=f"{total_docs:,}",
            help="Total documents processed"
        )
    
    with col3:
        # Get pipeline status
        try:
            status = api.get_pipeline_status()
            st.metric(
                label="⚙️ Pipeline Status",
                value=status.get('status', 'Unknown').upper(),
                help="Current pipeline status"
            )
        except:
            st.metric(
                label="⚙️ Pipeline Status",
                value="N/A"
            )

except Exception as e:
    st.error(f"⚠️ Could not connect to API: {e}")
    st.info("Make sure the FastAPI backend is running on http://localhost:8000")
    st.code("python -m src.api.main", language="bash")
    st.stop()

st.divider()

# Feature overview
st.subheader("🚀 Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📊 Topics & Trends
    - View all discovered topics
    - Analyze topic trends over time
    - See topic keywords and sizes
    
    ### 🔍 Topic Drill-down
    - Deep dive into specific topics
    - View example conversations
    - Analyze topic drift metrics
    """)

with col2:
    st.markdown("""
    ### ⚠️ Drift Alerts
    - Monitor topic evolution
    - Get alerts on significant changes
    - Track prevalence shifts
    
    ### ✏️ HITL Editor
    - Merge similar topics
    - Relabel topics with custom names
    - Audit trail of all changes
    """)

st.markdown("""
### 🔮 Inference
- Predict topic for new text
- Get confidence scores
- See top keywords for predicted topic
""")

st.divider()

# Quick stats
st.subheader("📈 Quick Statistics")

if topics:
    # Top 5 topics by size
    st.markdown("**Top 5 Topics by Size:**")
    
    sorted_topics = sorted(topics, key=lambda x: x.get('count', 0), reverse=True)[:5]
    
    for i, topic in enumerate(sorted_topics, 1):
        with st.expander(f"#{i}  {topic.get('custom_label', f'Topic {topic.get('topic_id')}')} - {topic.get('count', 0)} docs"):
            st.write(f"**Keywords:** {', '.join(topic.get('top_words', [])[:5])}")
            st.write(f"**Batch:** {topic.get('batch_id', 'N/A')}")
            st.write(f"**Window:** {topic.get('window_start', 'N/A')} to {topic.get('window_end', 'N/A')}")

st.divider()

# Getting started
st.subheader("🎯 Getting Started")
st.markdown("""
1. **Check Topics & Trends** to see an overview of all topics
2. **View Drift Alerts** to monitor changes in topic distribution
3. **Use HITL Editor** to refine topics by merging or relabeling
4. **Run Inference** to predict topics for new customer support messages
""")

# Footer
st.divider()
st.caption("TwCS Online Topic Modeling System v0.1.0 | Powered by BERTopic & Streamlit")

