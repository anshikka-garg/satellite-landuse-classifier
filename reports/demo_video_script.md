# Demo Video Script (3-Minute Walkthrough)

This script is structured to fit into a **3-minute** presentation of the Satellite Landuse Classifier & Temporal Change Detector. Use a screen recorder (like OBS, Loom, or Zoom) to capture the workspace and Streamlit dashboard.

---

## Part 1: Introduction & Architecture (0:00 - 0:30)

### Visuals
* Start with the **VS Code editor** showing the project directory structure.
* Briefly hover over `src/`, `notebooks/`, `models/`, `data/`, and `reports/`.

### Speaking Script
> "Hello! Today I will demonstrate our Satellite Land-Use Classifier and Temporal Change Detector.
> Our project is structured into three main phases: (1) training a robust spatial classifier, (2) detecting changes over time using deep feature embeddings, and (3) validating performance under out-of-domain conditions.
> Here is our codebase, structured cleanly with data pipelines, transfer learning models, and evaluation modules."

---

## Part 2: Spatial Leakage & Training (0:30 - 1:15)

### Visuals
* Open `notebooks/01_data_pipeline.ipynb` showing the block split visualization.
* Open `reports/final_report.pdf` (Page 3) showing the performance metrics table.

### Speaking Script
> "Geospatial data is highly autocorrelated—adjacent tiles are nearly identical. A standard random split causes severe information leakage, resulting in an overoptimistic validation F1 of 0.9840.
> To measure true generalization, we implemented a Spatially Blocked Split, assigning entire blocks to either train or validation. 
> Fine-tuning a pre-trained ResNet-18 model on this split yields a robust Macro-F1 score of 0.9634. This 2.1% performance drop represents the true generalizability gap, protecting us from model failures during deployment."

---

## Part 3: Temporal Change Detection (1:15 - 2:00)

### Visuals
* Open `notebooks/04_change_detection.ipynb`.
* Show the ROC curve and Cosine Similarity Distribution plots.

### Speaking Script
> "For temporal change detection, we remove the final classification head of our block-split model to extract 512-dimensional embeddings. 
> To detect land-use changes between T1 and T2 time-steps, we calculate the Cosine Similarity between their embeddings.
> By testing on 1,000 simulated tile pairs, our ROC analysis shows an AUC of 0.9418. Using Youden's J statistic, we selected the optimal change threshold of 0.376. This threshold successfully distinguishes actual land-cover changes from seasonal variations and lighting noise."

---

## Part 4: Streamlit Dashboard Demo (2:00 - 2:45)

### Visuals
* Switch to the **Streamlit Dashboard** running in your local browser (`http://localhost:8501`).
* **Interaction 1**: Single-Tile Analysis. Select "Sample 1: Annual Crop -> Industrial (Change)". Show the change alert and the GradCAM overlays showing model focus.
* **Interaction 2**: Switch to "Regional 5x5 Grid Analysis". Select "Region Pair 1". Scroll down to show the 5x5 visual grids, the similarity heatmap, and the green/red predicted change map.

### Speaking Script
> "Now let's see the interactive Streamlit dashboard. 
> In Single-Tile Analysis, users can compare preloaded samples or upload custom images. For example, comparing this crop tile to an industrial site correctly triggers the Land-Use Change alert. The GradCAM overlays show exactly where the model is focusing.
> In Regional 5x5 Grid Analysis, we process simulated multi-temporal grid pairs. The dashboard extracts embedding grids, computes similarities, and maps exactly which cells have changed, highlighting them in red."

---

## Part 5: Out-of-Domain Generalization & Conclusion (2:45 - 3:00)

### Visuals
* Show the `reports/top_5_errors_notebook.png` image or open `notebooks/05_evaluation.ipynb` showing the top-5 errors.

### Speaking Script
> "Finally, we tested generalization on the out-of-domain UC Merced aerial dataset, reaching 76.0% accuracy. We isolated the top 5 most confident errors and documented hypotheses—such as spatial similarity and class overlap—to guide future improvements like multi-spectral band training. 
> Thank you!"
