# Demo Video — Walkthrough Script

**File:** `demo\_video.mp4`
**Duration:** \~3 minutes 32 seconds
**Format:** Silent screen recording of the live Streamlit dashboard
(hosted on Streamlit Community Cloud)

This document describes what is shown in the video, minute by minute, for
anyone reviewing the recording without narration.

\---

## 0:00 – 1:10 — Single-Tile Pair Analysis

* Dashboard loaded with the **Block Split ResNet-18 (Recommended)** checkpoint.
* A preloaded sample pair is selected from the dropdown
(e.g. *"Annual Crop → Industrial (Change)"*).
* For each tile (Before / After), the dashboard displays:

  * The predicted land-use class and confidence score.
  * The original 64×64 tile alongside its Grad-CAM attention overlay,
showing which pixels most influenced the prediction.
* The **Analysis Summary** banner reports the cosine similarity score
between the two tiles' embeddings and flags whether a change was
detected relative to the active threshold.

## 1:10 – 2:20 — Operating Point Toggle (Bonus B)

* The sidebar's **Operating Point** control is switched between the three
available modes:

  * **Balanced (Optimal ROC)** — threshold 0.376, derived from the
Youden's J statistic on the ROC curve (AUC = 0.9418).
  * **High Recall (Sensitive)** — threshold 0.50, flags more tiles as
changed (favors catching every real change).
  * **High Precision (Confident)** — threshold 0.30, flags fewer tiles
(favors avoiding false alarms).
* The change-detected banner and the Active Threshold Boundary caption
update live as each operating point is selected, demonstrating that the
decision boundary is configurable rather than hardcoded.

## 2:20 – 3:32 — Regional 5×5 Grid Analysis

* Analysis Mode switched to **Regional 5×5 Grid Analysis**.
* A simulated region pair is selected from the dropdown.
* The dashboard renders the four-panel view:

  1. T1 (Before) 5×5 tile grid
  2. T2 (After) 5×5 tile grid
  3. Cosine similarity heatmap across all 25 cell pairs
  4. Change map (green = correctly unchanged, red = correctly flagged
change, yellow = disagreement between the ground-truth simulation
label and the model's threshold decision)
* Summary metrics (total flagged cells, average cosine similarity) are
shown above the figure.

\---

## Notes for reviewers

* The recording is silent by design; all UI labels are self-descriptive
(see headings in each panel).
* The dashboard is deployed on Streamlit Community Cloud using a
pre-cached demo subset of EuroSAT tiles (rather than the full 27,000-tile
dataset) to stay within cloud storage limits — this is documented in
`src/dataset.py`. Running `streamlit run app/app.py` locally with the
full dataset (see root `README.md` setup instructions) enables the same
features against the complete data.
* Live app: see the Streamlit Cloud link in the root `README.md`.

