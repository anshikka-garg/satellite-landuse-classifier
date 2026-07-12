# Satellite Image Land-Use Classifier & Temporal Change Detector

🚀 **Live Demo**
🌐 Try the application here:
[Streamlit App Link](https://satellite-landuse-classifier.streamlit.app/) *(Link to be updated)*

Experience the Satellite Land-Use Classifier and Temporal Change Detector without any local setup. Upload satellite image tiles, perform single-tile analysis with Grad-CAM explainability, and run regional change mapping in real-time.

---

## Project Summary

**Satellite Image Land-Use Classifier & Temporal Change Detector** is an AI-powered computer vision system designed for earth observation and environmental monitoring. Instead of manually inspecting satellite imagery over large areas or time spans, users can leverage this application to automatically classify land use, explain model decisions visually, and track temporal land-cover changes.

The system features:
1. **Transfer Learning Pipeline:** Uses a fine-tuned ResNet-18 model trained on the EuroSAT dataset to classify tiles into 10 distinct land-use categories (e.g., forest, agriculture, residential, highway).
2. **Temporal Change Detection:** Employs embedding-based cosine similarity with ROC-optimal thresholding to identify land-cover modifications between pairs of images captured at different times.
3. **Explainable AI (XAI):** Integrates Grad-CAM to highlight the specific visual regions driving the model's classification decisions, offering full transparency.
4. **Interactive Dashboard:** Built with Streamlit, enabling users to perform single-tile inspection, analyze regional grids, and toggle confidence thresholds dynamically.

*Developed as a capstone project for the **Celebal Technologies Data Science Internship** (Celebal Excellence Internship Program, May–July 2026).*
