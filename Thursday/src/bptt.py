import torch
import torch.nn as nn
import numpy as np

class SimpleRNN(nn.Module):
    """ Used purely for Autograd comparison """
    def __init__(self, input_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

def manual_bptt_single_layer(X, W_xh, W_hh, b_h, W_hy, b_y, y_true):
    """
    Sub-step 7: Manual BPTT for a single-layer RNN.
    X: shape (seq_len, input_size)
    """
    seq_len, input_size = X.shape
    hidden_size = W_hh.shape[0]
    
    # Forward pass
    h_states = np.zeros((seq_len + 1, hidden_size))
    
    for t in range(seq_len):
        x_t = X[t].reshape(-1, 1) # (input_size, 1)
        h_prev = h_states[t - 1] if t > 0 else np.zeros((hidden_size, 1))
        if len(h_prev.shape) == 1:
            h_prev = h_prev.reshape(-1, 1)
        
        # h_t = tanh(W_xh * x_t + W_hh * h_prev + b_h)
        h_t_raw = np.dot(W_xh, x_t) + np.dot(W_hh, h_prev) + b_h
        h_states[t] = np.tanh(h_t_raw).flatten()
        
    h_last = h_states[seq_len - 1].reshape(-1, 1)
    y_pred = np.dot(W_hy, h_last) + b_y
    
    # Loss: MSE = 0.5 * (y_pred - y_true)^2  -> gradient is y_pred - y_true
    dy = y_pred - y_true
    
    # Gradients for output layer
    dW_hy = np.dot(dy, h_last.T)
    db_y = dy
    
    # Backward pass through time
    dh_next = np.dot(W_hy.T, dy)
    
    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    
    # Store gradient norms to observe vanishing gradient
    grad_norms = []
    
    for t in reversed(range(seq_len)):
        h_t = h_states[t].reshape(-1, 1)
        x_t = X[t].reshape(-1, 1)
        
        # dtanh / dx = 1 - tanh(x)^2
        # dh_raw = dh_next * (1 - h_t^2)
        dh_raw = dh_next * (1 - h_t**2)
        
        dW_xh += np.dot(dh_raw, x_t.T)
        
        if t > 0:
            h_prev = h_states[t-1].reshape(-1, 1)
            dW_hh += np.dot(dh_raw, h_prev.T)
            
        dh_next = np.dot(W_hh.T, dh_raw)
        grad_norms.append(np.linalg.norm(dh_raw))
        
    # Reverse to represent t=0 to t=seq-1
    grad_norms.reverse()
    
    return dW_xh, dW_hh, dW_hy, grad_norms

def run_vanishing_gradient_demo():
    """ Runs sequence length 5 vs 50 and outputs gradient norms. """
    input_size = 1
    hidden_size = 4
    
    np.random.seed(42)
    # small weights to exacerbate vanishing gradient intentionally
    W_xh = np.random.randn(hidden_size, input_size) * 0.1
    W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
    b_h = np.zeros((hidden_size, 1))
    W_hy = np.random.randn(1, hidden_size)
    b_y = np.zeros((1, 1))
    
    X_5 = np.random.randn(5, 1)
    X_50 = np.random.randn(50, 1)
    y_true = np.array([[1.0]])
    
    _, _, _, norms_5 = manual_bptt_single_layer(X_5, W_xh, W_hh, b_h, W_hy, b_y, y_true)
    _, _, _, norms_50 = manual_bptt_single_layer(X_50, W_xh, W_hh, b_h, W_hy, b_y, y_true)
    
    return norms_5, norms_50
