import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
import numpy as np

class StockLSTM(nn.Module):
    """
    Sub-step 3: LSTM for Next-Day Stock Return Prediction.
    Takes input of shape (batch, seq_len, 1) and outputs (batch, 1).
    """
    def __init__(self, hidden_dim=32, num_layers=1):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x is (batch, seq, features)
        out, (hn, cn) = self.lstm(x)
        # out[:, -1, :] gets the hidden state of the last timestep
        out = self.fc(out[:, -1, :])
        return out

def train_stock_lstm(model, X_train, y_train, epochs=20, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()
    
    return model

def evaluate_stock_lstm(model, X_test, y_test, scaler):
    """ Returns MAE in the original scale. """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_t).numpy().flatten()
    
    min_val, max_val = scaler
    
    preds_orig = preds * (max_val - min_val) + min_val
    y_test_orig = y_test * (max_val - min_val) + min_val
    
    mae = np.mean(np.abs(preds_orig - y_test_orig))
    return mae, preds_orig, y_test_orig

class AutoregressiveBaseline:
    """
    Sub-step 6: Autoregressive Baseline using simple Linear Regression
    Flatten the window to serve as features for predicting the next step.
    """
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X_train, y_train):
        # X_train is (samples, seq_len, 1)
        X_flat = X_train.reshape((X_train.shape[0], -1))
        self.model.fit(X_flat, y_train)

    def predict(self, X_test):
        X_flat = X_test.reshape((X_test.shape[0], -1))
        return self.model.predict(X_flat)

def evaluate_autoregressive(model, X_test, y_test, scaler):
    preds = model.predict(X_test)
    min_val, max_val = scaler
    
    preds_orig = preds * (max_val - min_val) + min_val
    y_test_orig = y_test * (max_val - min_val) + min_val
    
    mae = np.mean(np.abs(preds_orig - y_test_orig))
    return mae, preds_orig, y_test_orig
