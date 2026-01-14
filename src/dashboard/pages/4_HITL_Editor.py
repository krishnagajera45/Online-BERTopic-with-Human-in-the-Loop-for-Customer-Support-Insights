"""HITL Editor Page - Human-in-the-loop topic manipulation."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.utils.api_client import APIClient

st.set_page_config(page_title="HITL Editor", page_icon="✏️", layout="wide")

# Initialize API client
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

api = st.session_state.api_client

st.title("✏️ Human-in-the-Loop Topic Editor")
st.markdown("Refine topics by merging, splitting, or relabeling")

st.divider()

# Action selector
action = st.radio(
    "Select an action:",
    options=["Merge Topics", "Relabel Topic", "View Audit Log"],
    horizontal=True
)

st.divider()

try:
    topics = api.get_topics()
    
    if not topics:
        st.warning("No topics found. Train a model first.")
        st.stop()
    
    if action == "Merge Topics":
        st.subheader("🔗 Merge Topics")
        st.markdown("Combine multiple similar topics into one.")
        
        # Topic selector
        topic_options = {f"Topic {t['topic_id']}: {t['custom_label']} ({t['count']} docs)": t['topic_id'] 
                        for t in topics}
        
        selected_topics = st.multiselect(
            "Select topics to merge:",
            options=list(topic_options.keys())
        )
        
        if len(selected_topics) < 2:
            st.info("👆 Select at least 2 topics to merge")
        else:
            topic_ids = [topic_options[t] for t in selected_topics]
            
            st.write(f"**Selected Topic IDs:** {topic_ids}")
            
            # New label
            new_label = st.text_input(
                "New label for merged topic:",
                placeholder="e.g., 'Billing and Payment Issues'"
            )
            
            # Note
            note = st.text_area(
                "Note (optional):",
                placeholder="Reason for merging these topics..."
            )
            
            # Merge button
            if st.button("🔗 Merge Topics", type="primary"):
                if new_label:
                    try:
                        result = api.merge_topics(topic_ids, new_label, note)
                        st.success(f"✅ {result['message']}")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error merging topics: {e}")
                else:
                    st.warning("Please provide a new label for the merged topic")
    
    elif action == "Relabel Topic":
        st.subheader("🏷️ Relabel Topic")
        st.markdown("Change the custom label for a topic.")
        
        # Topic selector
        topic_options = {f"Topic {t['topic_id']}: {t['custom_label']}": t['topic_id'] 
                        for t in topics}
        
        selected_topic = st.selectbox(
            "Select topic to relabel:",
            options=list(topic_options.keys())
        )
        
        topic_id = topic_options[selected_topic]
        
        # Show current label and keywords
        topic_details = next(t for t in topics if t['topic_id'] == topic_id)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Label:**")
            st.info(topic_details['custom_label'])
        
        with col2:
            st.write("**Top Keywords:**")
            st.info(', '.join(topic_details['top_words'][:5]))
        
        # New label
        new_label = st.text_input(
            "New label:",
            placeholder="Enter new label for this topic"
        )
        
        # Note
        note = st.text_area(
            "Note (optional):",
            placeholder="Reason for relabeling..."
        )
        
        # Relabel button
        if st.button("🏷️ Update Label", type="primary"):
            if new_label:
                try:
                    result = api.relabel_topic(topic_id, new_label, note)
                    st.success(f"✅ {result['message']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error relabeling topic: {e}")
            else:
                st.warning("Please enter a new label")
    
    elif action == "View Audit Log":
        st.subheader("📋 Audit Log")
        st.markdown("History of all HITL actions")
        
        try:
            audit_log = api.get_audit_log(limit=100)
            
            if not audit_log:
                st.info("No audit log entries yet. Perform some HITL actions first.")
            else:
                audit_df = pd.DataFrame(audit_log)
                
                st.write(f"**Total Actions:** {len(audit_df)}")
                
                # Display log
                for _, entry in audit_df.iterrows():
                    with st.expander(
                        f"{entry['action_type'].upper()} - {entry.get('timestamp', 'N/A')}"
                    ):
                        st.write(f"**Action:** {entry['action_type']}")
                        st.write(f"**Old Topics:** {entry.get('old_topics', 'N/A')}")
                        st.write(f"**New Topics:** {entry.get('new_topics', 'N/A')}")
                        st.write(f"**Timestamp:** {entry.get('timestamp', 'N/A')}")
                        
                        if entry.get('user_note'):
                            st.write(f"**Note:** {entry['user_note']}")
        
        except Exception as e:
            st.error(f"Error loading audit log: {e}")

except Exception as e:
    st.error(f"Error: {e}")

