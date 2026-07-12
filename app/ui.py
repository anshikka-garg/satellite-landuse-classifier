import streamlit as st
import numpy as np
from src.dataset import EUROSAT_CLASSES

def render_header():
    st.title("Satellite Land-Use Classifier & Change Detector")

def show_summary_card(status_color, status_text, similarity, threshold_value, decision_text):
    st.subheader("Analysis Summary")
    message = f"**{status_text}** ({decision_text}) | Cosine Similarity: **{similarity:.4f}** (Threshold: **{threshold_value:.3f}**)"
    if "No" in status_text:
        st.success(message)
    else:
        st.error(message)

def show_similarity_card(status_color, similarity, threshold_value):
    st.markdown(f"**Cosine Similarity Score**: **{similarity:.4f}** (Threshold: **{threshold_value:.3f}**)")
    st.progress(max(0.0, min(1.0, float(similarity))))

def show_prediction_card(label, pred_class, conf):
    class_name = EUROSAT_CLASSES[pred_class]
    st.subheader(label)
    st.markdown(f"**Predicted Land Use**: **{class_name}**")
    st.markdown(f"**Confidence**: **{conf*100:.1f}%**")
    st.progress(float(conf))

def show_regional_metrics(num_changes, sim_matrix):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Flagged Changed Cells", f"{num_changes} / 25")
    with col2:
        st.metric("Average Cosine Similarity", f"{np.mean(sim_matrix):.3f}")
