# Thursday Assignment - Week 08

This project addresses the Thursday assignment covering RNNs and Sequential Data. The assignment has been structured cleanly to separate modular source code from the main execution layers.

### Project Structure
- `src/data_prep.py`: Data cleaning and feature extraction logic. Includes robust datetime parsing for `chat_logs` and sequential temporal splits for `stock_prices`.
- `src/models_stock.py`: Implements PyTorch LSTM for predicting next-day close price, and an Autoregressive baseline for comparison.
- `src/models_churn.py`: Churn prediction leveraging Tabular Data, alongside cost optimization analysis to compute outreach thresholds.
- `src/bptt.py`: Hand-built Backpropagation Through Time to demonstrate vanishing gradient phenomena analytically.
- `thursday_assignment.ipynb`: The main notebook combining the logic and exporting the output.

### Requirements
- **Python Version:** 3.9+ 
- Packages Needed: see `requirements.txt`
  - `pandas`
  - `numpy`
  - `torch`
  - `scikit-learn`
  - `nbformat`

### How To Run
The TA can simply execute the notebook `thursday_assignment.ipynb` sequentially. It automatically imports its methods from the `src/` directory.

> **Note:** If `thursday_assignment.ipynb` gets deleted, you can regenerate it dynamically by running:
> `python generate_notebook.py`
