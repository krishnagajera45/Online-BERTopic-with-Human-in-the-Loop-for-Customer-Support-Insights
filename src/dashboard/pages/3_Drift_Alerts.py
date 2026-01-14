"""Drift Alerts Page - Monitor topic drift and view alerts."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.utils.api_client import APIClient

st.set_page_config(page_title="Drift Alerts", page_icon="⚠️", layout="wide")

# Initialize API client
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

api = st.session_state.api_client

st.title("⚠️ Drift Alerts")
st.markdown("Monitor topic evolution and drift detection alerts")

st.divider()

try:
    # Get alerts
    limit = st.slider("Number of alerts to show:", 10, 100, 50)
    alerts = api.get_alerts(limit=limit)
    
    if not alerts:
        st.success("✅ No drift alerts. All topics are stable!")
        st.info("Drift alerts will appear here when significant topic changes are detected.")
        st.stop()
    
    # Convert to DataFrame
    alerts_df = pd.DataFrame(alerts)
    
    # Summary metrics
    st.subheader("Alert Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Alerts", len(alerts))
    
    with col2:
        high_severity = len(alerts_df[alerts_df['severity'] == 'high'])
        st.metric("High Severity", high_severity, delta="⚠️" if high_severity > 0 else None)
    
    with col3:
        medium_severity = len(alerts_df[alerts_df['severity'] == 'medium'])
        st.metric("Medium Severity", medium_severity)
    
    st.divider()
    
    # Filter by severity
    st.subheader("Filter Alerts")
    
    severity_filter = st.multiselect(
        "Filter by severity:",
        options=['high', 'medium', 'low'],
        default=['high', 'medium']
    )
    
    filtered_alerts = alerts_df[alerts_df['severity'].isin(severity_filter)]
    
    # Display alerts
    st.subheader(f"Alerts ({len(filtered_alerts)})")
    
    # Severity color mapping
    severity_colors = {
        'high': '🔴',
        'medium': '🟠',
        'low': '🟢'
    }
    
    for _, alert in filtered_alerts.iterrows():
        severity_icon = severity_colors.get(alert['severity'], '⚪')
        
        with st.expander(
            f"{severity_icon} {alert['severity'].upper()} - {alert['reason']} (Alert ID: {alert['alert_id']})"
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Topic ID:** {alert['topic_id']}")
                st.write(f"**Window Start:** {alert['window_start']}")
                st.write(f"**Created:** {alert.get('created_at', 'N/A')}")
            
            with col2:
                st.write(f"**Severity:** {alert['severity'].upper()}")
                st.write(f"**Reason:** {alert['reason']}")
            
            # Show metrics if available
            if alert.get('metrics_json'):
                st.write("**Metrics:**")
                st.code(alert['metrics_json'], language='json')
    
    st.divider()
    
    # Alert statistics
    st.subheader("Alert Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Alerts by severity
        severity_counts = alerts_df['severity'].value_counts()
        st.bar_chart(severity_counts)
        st.caption("Alerts by Severity")
    
    with col2:
        # Alerts by reason
        reason_counts = alerts_df['reason'].value_counts().head(5)
        st.bar_chart(reason_counts)
        st.caption("Top 5 Alert Reasons")

except Exception as e:
    st.error(f"Error loading alerts: {e}")
    st.info("Make sure the backend is running and drift detection has been performed.")

