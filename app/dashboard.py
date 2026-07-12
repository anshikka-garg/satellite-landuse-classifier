import streamlit as st
import numpy as np
import torch
import os
import time
import logging
from PIL import Image
from torchvision import transforms

from prediction import analyze_tile_pair
from ui import show_summary_card, show_similarity_card, show_prediction_card, show_regional_metrics
from src.gradcam import overlay_heatmap
from src.change_detection import simulate_region_pairs, plot_region_change_heatmap, EmbeddingExtractor

logger = logging.getLogger("EarthObservationAI")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def run_single_tile_analysis(model, eurosat, threshold_value, device):
    st.header("Single-Tile Pair Analysis")
    st.write("Compare Before and After satellite tiles to classify land-use, evaluate visual explanation maps, and extract embeddings.")

    input_type = st.radio("Image Source", ["Preloaded Sample Pairs", "Upload Custom Images"], horizontal=True)

    t1_img, t2_img = None, None

    if input_type == "Preloaded Sample Pairs":
        sample_options = {
            "Annual Crop → Industrial (Change)": (25472, 4989),
            "Forest → Forest (Seasonal Variation)": (16384, 16385),
            "Sea/Lake → Residential (Change)": (21185, 25747),
            "Pasture → Pasture (No Change)": (7239, 7240),
            "River → Highway (Change)": (2409, 2449)
        }
        selected_sample = st.selectbox("Sample Pair Selection", list(sample_options.keys()))
        idx1, idx2 = sample_options[selected_sample]
        logger.info(f"Preloaded sample pair selected: key='{selected_sample}' (indices={idx1}, {idx2})")
        t1_img = eurosat[idx1][0]
        t2_img = eurosat[idx2][0]
    else:
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            st.markdown("### Before Image (T1)")
            uploaded_file1 = st.file_uploader("Upload Before (T1)", type=["png", "jpg", "jpeg", "tif"], label_visibility="collapsed")
            if uploaded_file1:
                logger.info(f"Custom before image uploaded: name='{uploaded_file1.name}', size={uploaded_file1.size} bytes")
                t1_img = Image.open(uploaded_file1).convert("RGB")
        with col_up2:
            st.markdown("### After Image (T2)")
            uploaded_file2 = st.file_uploader("Upload After (T2)", type=["png", "jpg", "jpeg", "tif"], label_visibility="collapsed")
            if uploaded_file2:
                logger.info(f"Custom after image uploaded: name='{uploaded_file2.name}', size={uploaded_file2.size} bytes")
                t2_img = Image.open(uploaded_file2).convert("RGB")

    if t1_img is not None and t2_img is not None:
        t1_img_resized = t1_img.resize((64, 64))
        t2_img_resized = t2_img.resize((64, 64))

        t1_tensor = eval_transform(t1_img_resized).unsqueeze(0)
        t2_tensor = eval_transform(t2_img_resized).unsqueeze(0)

        # Profile processing time
        t_start = time.time()
        # Execute analysis logic
        with st.spinner("Extracting embeddings and generating explanations..."):
            results = analyze_tile_pair(model, t1_tensor, t2_tensor, threshold_value)
        t_end = time.time()
        
        processing_time = t_end - t_start
        st.session_state.processing_time = processing_time
        logger.info(f"Single-tile inference completed in {processing_time:.4f}s")

        # Set up UI parameters
        if results["has_changed"]:
            status_color = "#ef4444"
            status_text = "Change Detected"
            decision_text = "Flag for Review"
        else:
            status_color = "#10b981"
            status_text = "No Significant Change"
            decision_text = "Stable / Approved"

        # Render summary dashboard cards
        show_summary_card(status_color, status_text, results["similarity"], threshold_value, decision_text)
        show_similarity_card(status_color, results["similarity"], threshold_value)

        col1, col2 = st.columns(2)

        with col1:
            show_prediction_card("Before (T1)", results["pred_class1"], results["conf1"])
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.markdown("<div style='font-size: 0.78rem; color: #9ca3af; margin-bottom: 4px; font-weight: 500;'>Original Image</div>", unsafe_allow_html=True)
                st.image(t1_img_resized, use_column_width=True)
            with col_img2:
                st.markdown("<div style='font-size: 0.78rem; color: #9ca3af; margin-bottom: 4px; font-weight: 500;'>Grad-CAM Overlay</div>", unsafe_allow_html=True)
                overlay1 = overlay_heatmap(np.array(t1_img_resized), results["heatmap1"], alpha=0.45)
                st.image(overlay1, use_column_width=True)

        with col2:
            show_prediction_card("After (T2)", results["pred_class2"], results["conf2"])
            col_img3, col_img4 = st.columns(2)
            with col_img3:
                st.markdown("<div style='font-size: 0.78rem; color: #9ca3af; margin-bottom: 4px; font-weight: 500;'>Original Image</div>", unsafe_allow_html=True)
                st.image(t2_img_resized, use_column_width=True)
            with col_img4:
                st.markdown("<div style='font-size: 0.78rem; color: #9ca3af; margin-bottom: 4px; font-weight: 500;'>Grad-CAM Overlay</div>", unsafe_allow_html=True)
                overlay2 = overlay_heatmap(np.array(t2_img_resized), results["heatmap2"], alpha=0.45)
                st.image(overlay2, use_column_width=True)


def run_regional_analysis(model, eurosat, manifest, threshold_value, device):
    st.header("Regional 5x5 Grid Analysis")
    st.write("Analyze simulated spatial block regions representing T1 and T2 time periods. The dashboard generates similarity heatmaps and highlights changed cells.")

    if manifest is None:
        logger.error("Failed to run regional analysis: split manifest file not loaded.")
        st.error("Split manifest split_manifest.npz not found under data/processed/. Cannot load simulated grids.")
    else:
        block_test_idxs = manifest["block_test"]
        
        region_idx = st.selectbox(
            "Simulated Region Pair Selection",
            [1, 2, 3, 4, 5],
            format_func=lambda x: f"Region Pair {x}"
        )

        @st.cache_data
        def get_simulated_regions(_dataset, idxs):
            logger.info("Generating regional grid splits...")
            return simulate_region_pairs(_dataset, idxs, num_regions=5, grid_size=5, change_prob=0.4, seed=123)

        sim_regions = get_simulated_regions(eurosat, block_test_idxs)
        selected_region = sim_regions[region_idx - 1]

        extractor = EmbeddingExtractor(model, device)

        logger.info(f"Running grid similarity comparisons for region index={region_idx} with threshold={threshold_value:.3f}...")
        
        # Profile processing time
        t_start = time.time()
        with st.spinner("Analyzing regional embedding grids..."):
            fig, sim_matrix, pred_change = plot_region_change_heatmap(eurosat, selected_region, extractor, threshold_value)
            num_changes = np.sum(pred_change)
        t_end = time.time()
        
        processing_time = t_end - t_start
        st.session_state.processing_time = processing_time
        logger.info(f"Regional grid analysis completed in {processing_time:.4f}s")
            
        show_regional_metrics(num_changes, sim_matrix)
        st.pyplot(fig)
