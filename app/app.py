import os
import sys
import numpy as np
import streamlit as st
import logging

# Configure logger first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EarthObservationAI")

# Resolve project root dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(project_root)
sys.path.append(project_root)

from src.dataset import load_eurasat
from styles import inject_premium_styles
from sidebar import render_sidebar
from prediction import load_transfer_learning_model, device
from ui import render_header
from dashboard import run_single_tile_analysis, run_regional_analysis

# Page config
st.set_page_config(
    page_title="Satellite Land-Use Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject styles
inject_premium_styles()

@st.cache_resource
def load_eurosat_dataset():
    logger.info("Initializing EuroSAT local dataset loader...")
    return load_eurasat(root=os.path.join("data", "raw"), download=False)

@st.cache_resource
def load_models_and_manifest():
    manifest_path = os.path.join("data", "processed", "split_manifest.npz")
    logger.info(f"Loading split manifest file: {manifest_path}")
    return np.load(manifest_path) if os.path.exists(manifest_path) else None

# Load global resources
eurosat = load_eurosat_dataset()
manifest = load_models_and_manifest()

# Render title and details
render_header()

# Render sidebar configurations
config = render_sidebar(st.session_state.get("processing_time", None))

# Load chosen model checkpoint
model = load_transfer_learning_model(config["model_filename"])

if model is not None:
    logger.info(f"Dashboard running mode: {config['dashboard_mode']} (threshold: {config['threshold_value']:.3f})")
    if config["dashboard_mode"] == "Single-Tile Pair Analysis":
        run_single_tile_analysis(model, eurosat, config["threshold_value"], device)
    else:
        run_regional_analysis(model, eurosat, manifest, config["threshold_value"], device)
else:
    logger.warning("Model checkpoint could not be loaded. Dashboard is inactive.")
    st.warning("Please train the transfer learning model to enable dashboard analysis.")
