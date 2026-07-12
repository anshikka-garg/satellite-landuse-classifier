import os
import torch
import numpy as np
import streamlit as st
import logging
from src.model import TransferLearningModel
from src.change_detection import EmbeddingExtractor, compute_cosine_similarity
from src.gradcam import GradCAM

logger = logging.getLogger("EarthObservationAI")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_transfer_learning_model(filename):
    model_path = os.path.join("models", filename)
    if not os.path.exists(model_path):
        logger.error(f"Checkpoint file not found: {model_path}")
        return None
    
    logger.info(f"Loading transfer learning model checkpoint: {filename}...")
    model = TransferLearningModel(num_classes=10, backbone_name="resnet18").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Model checkpoint {filename} successfully loaded on device: {device}.")
    return model

def analyze_tile_pair(model, t1_tensor, t2_tensor, threshold_value):
    logger.info("Initializing embedding extractor and Grad-CAM layers...")
    extractor = EmbeddingExtractor(model, device)
    gradcam_target = model.backbone.layer4[1].conv2
    cam_model = GradCAM(model, gradcam_target)
    
    # Extract Embeddings and Cosine Similarity
    t1_emb = extractor.get_embedding(t1_tensor)
    t2_emb = extractor.get_embedding(t2_tensor)
    similarity = compute_cosine_similarity(t1_emb, t2_emb)[0]
    has_changed = similarity < threshold_value
    
    # Classifications and Heatmaps
    heatmap1, conf1 = cam_model.generate_heatmap(t1_tensor)
    heatmap2, conf2 = cam_model.generate_heatmap(t2_tensor)
    
    pred_class1 = model(t1_tensor.to(device)).argmax(dim=1).item()
    pred_class2 = model(t2_tensor.to(device)).argmax(dim=1).item()
    
    logger.info(f"Tile pair analysis complete: similarity={similarity:.4f}, change_detected={has_changed}")
    return {
        "similarity": similarity,
        "has_changed": has_changed,
        "pred_class1": pred_class1,
        "conf1": conf1,
        "heatmap1": heatmap1,
        "pred_class2": pred_class2,
        "conf2": conf2,
        "heatmap2": heatmap2
    }
