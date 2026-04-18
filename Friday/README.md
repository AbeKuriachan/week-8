# Friday Assignment - Week 08

This project addresses the Friday assignment covering Transfer Learning and Clinical Triage logic. The assignment has been structured cleanly to separate modular source code from the main execution layers.

### Project Structure
- `src/data_loader.py`: Dataset formulation.
- `src/models.py`: PyTorch ImageNet builder architectures (ResNet18).
- `src/evaluator.py`: Analysis logic (Grad-CAM sim, per-class evaluation, triage thresholds).
- `src/me1_prep.py`: Concept notes for ME1.
- `friday_assignment.ipynb`: The main notebook combining the logic and exporting the output.

### Requirements
- **Python Version:** 3.9+ 
- Packages Needed: see `requirements.txt`

### Note on Data
Due to the absence of the raw `.png`/`.jpg` chest X-Rays in the repository data folder at the time of construction, synthetic noisy images were populated matching the metadata `image_id`s. This ensures the notebook compiles robustly end-to-end. Swap the images in `/data/images` with real ones at any time!
