"""Topics & Trends Page - View all topics and their trends over time."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.utils.api_client import APIClient

st.set_page_config(page_title="Topics & Trends", page_icon="📊", layout="wide")

# Initialize API client
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

api = st.session_state.api_client

st.title("📊 Topics & Trends")
st.markdown("View all discovered topics and analyze their trends over time")

st.divider()

try:
    # Fetch topics
    topics = api.get_topics()
    
    if not topics:
        st.warning("No topics found. Run the ETL pipeline to train a model first.")
        st.stop()
    
    # Topics table
    st.subheader("Current Topics")
    
    topics_df = pd.DataFrame(topics)
    
    # Format for display
    display_df = topics_df[[
        'topic_id',
        'custom_label',
        'top_words',
        'count',
        'batch_id',
        'window_start',
        'window_end'
    ]].copy()
    
    display_df['top_words'] = display_df['top_words'].apply(lambda x: ', '.join(x[:5]))
    display_df.columns = ['Topic ID', 'Label', 'Top Keywords', 'Count', 'Batch', 'Window Start', 'Window End']
    
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True
    )
    
    st.divider()
    
    # Topic size distribution
    st.subheader("Topic Size Distribution")
    
    fig_dist = px.bar(
        topics_df.sort_values('count', ascending=False),
        x='topic_id',
        y='count',
        title="Documents per Topic",
        labels={'topic_id': 'Topic ID', 'count': 'Number of Documents'},
        hover_data=['custom_label']
    )
    fig_dist.update_traces(marker_color='lightblue')
    st.plotly_chart(fig_dist, width='stretch')
    
    st.divider()
    
    # Trends over time
    st.subheader("Topic Trends Over Time")
    
    try:
        trends = api.get_trends()
        
        if trends:
            trends_df = pd.DataFrame(trends)
            
            # Create line chart
            fig_trends = px.line(
                trends_df,
                x='batch_id',
                y='count',
                color='topic_id',
                title="Topic Counts by Batch",
                labels={'batch_id': 'Batch', 'count': 'Count', 'topic_id': 'Topic ID'}
            )
            st.plotly_chart(fig_trends, width='stretch')
            
            # Topic selector for detailed view
            st.subheader("Topic-Specific Trend")
            
            selected_topic = st.selectbox(
                "Select a topic to view its trend:",
                options=topics_df['topic_id'].tolist(),
                format_func=lambda x: f"Topic {x}: {topics_df[topics_df['topic_id']==x]['custom_label'].iloc[0]}"
            )
            
            if selected_topic is not None:
                topic_trends = trends_df[trends_df['topic_id'] == selected_topic]
                
                fig_single = px.bar(
                    topic_trends,
                    x='batch_id',
                    y='count',
                    title=f"Trend for Topic {selected_topic}",
                    labels={'batch_id': 'Batch', 'count': 'Count'}
                )
                fig_single.update_traces(marker_color='teal')
                st.plotly_chart(fig_single, width='stretch')
        else:
            st.info("No trend data available yet. Process more batches to see trends.")
    
    except Exception as e:
        st.warning(f"Could not load trends: {e}")
    
    st.divider()
    
    # Topic keywords cloud
    st.subheader("Topic Keywords")
    
    num_cols = 3
    rows = (len(topics) + num_cols - 1) // num_cols
    
    for row in range(rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            topic_idx = row * num_cols + col_idx
            if topic_idx < len(topics):
                topic = topics[topic_idx]
                with cols[col_idx]:
                    st.markdown(f"**Topic {topic['topic_id']}: {topic['custom_label']}**")
                    st.caption(f"Count: {topic['count']}")
                    keywords = ', '.join(topic['top_words'][:10])
                    st.text(keywords)

except Exception as e:
    st.error(f"Error loading topics: {e}")
    st.info("Make sure the FastAPI backend is running and a model has been trained.")

