import streamlit as st
import logging
import torch

logger = logging.getLogger("EarthObservationAI")

# Decision threshold constants
BALANCED_THRESHOLD = 0.376
HIGH_RECALL_THRESHOLD = 0.50
HIGH_PRECISION_THRESHOLD = 0.30

def render_sidebar(processing_time=None):
    st.sidebar.markdown("## Configuration")
    
    model_choice = st.sidebar.selectbox(
        "Model Checkpoint",
        ["Block Split ResNet-18 (Recommended)", "Naive Split ResNet-18"]
    )
    
    model_filename = "resnet18_block_final.pt" if "Block" in model_choice else "resnet18_naive_final.pt"
    
    threshold_mode = st.sidebar.radio(
        "Operating Point",
        ["Balanced (Optimal ROC)", "High Recall (Sensitive)", "High Precision (Confident)"],
        help="Selects the decision boundary for temporal change detection. Balanced uses Youden's J statistic (threshold=0.376)."
    )
    
    if threshold_mode == "High Recall (Sensitive)":
        threshold_value = HIGH_RECALL_THRESHOLD
    elif threshold_mode == "High Precision (Confident)":
        threshold_value = HIGH_PRECISION_THRESHOLD
    else:
        threshold_value = BALANCED_THRESHOLD
        
    logger.info(f"Sidebar configuration updated: checkpoint={model_filename}, mode={threshold_mode}, threshold={threshold_value:.3f}")
        
    st.sidebar.caption(f"Active Threshold Boundary: **{threshold_value:.3f}**")
    
    dashboard_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Single-Tile Pair Analysis", "Regional 5x5 Grid Analysis"]
    )
    

    
    return {
        "model_filename": model_filename,
        "threshold_value": threshold_value,
        "dashboard_mode": dashboard_mode
    }
