import pandas as pd
import numpy as np

def prepare_stock_sequences(df, ticker, window_size=20, train_ratio=0.8):
    """
    Sub-step 1: Prepare sequence dataset for next-day close prediction.
    """
    df_tkr = df[df['ticker'] == ticker].sort_values('date').copy()
    data_close = df_tkr['close'].values.reshape(-1, 1)

    min_val = np.min(data_close)
    max_val = np.max(data_close)
    data_scaled = (data_close - min_val) / (max_val - min_val)

    X, y = [], []
    for i in range(len(data_scaled) - window_size):
        X.append(data_scaled[i : i + window_size])
        y.append(data_scaled[i + window_size])
        
    X = np.array(X)
    y = np.array(y)

    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, (min_val, max_val), df_tkr

def clean_chat_logs(df):
    """
    Sub-step 2: Clean chat timestamps and prepare for EDA / Modeling
    """
    df_clean = df.copy()
    
    def robust_parse(ts_str):
        try:
            return pd.to_datetime(ts_str)
        except Exception:
            return pd.NaT

    try:
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], format='mixed')
    except Exception:
        df_clean['timestamp'] = df_clean['timestamp'].apply(robust_parse)
        
    df_clean = df_clean.dropna(subset=['timestamp'])
    # Sort chronologically
    df_clean = df_clean.sort_values(by=['timestamp'])
    
    return df_clean

def extract_churn_features(df):
    """
    Aggregate features for Tabular Model.
    Since there is no customer_id, each interaction (chat_id) represents a prediction instance.
    """
    tabular_df = df.copy()
    
    # We must explicitly encode anything that is a string, like sentiment, intent, status etc.
    features_to_encode = ['customer_sentiment', 'primary_intent', 'resolution_status', 'product_tier']
    for f in features_to_encode:
        if f in tabular_df.columns:
            tabular_df = pd.get_dummies(tabular_df, columns=[f], drop_first=True)
            
    cols_to_drop = ['timestamp']
    for c in cols_to_drop:
        if c in tabular_df.columns:
            tabular_df = tabular_df.drop(columns=[c])
            
    return tabular_df
