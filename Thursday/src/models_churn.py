import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
import pandas as pd

class TabularChurnModel:
    """
    Sub-step 4: Tabular model aggregating chat interactions.
    Hypothesis: Tabular aggregated features (like max sentiment drop, total interactions) 
    capture the churn risk well without the overhead of sequence modeling.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X_test):
        # Return probability of class 1 (churn)
        return self.model.predict_proba(X_test)[:, 1]

def evaluate_churn_cost_model(y_true, y_probs, cost_tp=5, cost_fp=10, cost_fn=50):
    """
    Sub-step 5: Cost Model for Outreach.
    cost_tp: Cost of contacting a customer who was correctly identified as churning (e.g., incentive cost)
    cost_fp: Cost of contacting a non-churner (e.g., wasted incentive)
    cost_fn: Cost of missing a churner (e.g., Lost Lifetime Value)
    cost_tn: 0 (No contact, no churn)
    Note: Threshold optimization logic.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    best_threshold = 0.5
    min_cost = float('inf')
    
    costs_at_thresholds = []
    
    for t in np.arange(0.01, 1.0, 0.05):
        preds = (y_probs >= t).astype(int)
        
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        
        # Calculate cost
        total_cost = (tp * cost_tp) + (fp * cost_fp) + (fn * cost_fn)
        
        costs_at_thresholds.append((t, total_cost, tp + fp)) # tp+fp is number of contacts
        
        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = t

    # Find the data point for best threshold
    _, best_cost, num_contacted = next(item for item in costs_at_thresholds if item[0] == best_threshold)
    
    results = {
        'best_threshold': best_threshold,
        'min_cost': min_cost,
        'num_contacted_per_month': num_contacted,
        'roc_auc': roc_auc_score(y_true, y_probs)
    }
    
    return results

def get_risk_list(customer_ids, y_probs, threshold):
    """ Return ranked risk list of customers. """
    results = pd.DataFrame({'customer_id': customer_ids, 'churn_prob': y_probs})
    results = results[results['churn_prob'] >= threshold].sort_values(by='churn_prob', ascending=False)
    return results
