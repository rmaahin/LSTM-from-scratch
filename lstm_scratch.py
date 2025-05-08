import numpy as np
import os

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        """
        Initialize LSTM cell parameters
        
        Args:
            input_size: Dimension of input vector
            hidden_size: Dimension of hidden state
        """
        # Initialize weights and biases for gates
        # Weights for input gate
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bi = np.zeros((hidden_size, 1))
        
        # Weights for forget gate
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))
        
        # Weights for output gate
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bo = np.zeros((hidden_size, 1))
        
        # Weights for cell state
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bc = np.zeros((hidden_size, 1))
        
        self.hidden_size = hidden_size
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """Derivative of tanh function"""
        return 1 - np.power(x, 2)
    
    def forward_step(self, xt, h_prev, c_prev):
        """
        Forward pass for a single timestep
        
        Args:
            xt: Input at current timestep (input_size, 1)
            h_prev: Previous hidden state (hidden_size, 1)
            c_prev: Previous cell state (hidden_size, 1)
            
        Returns:
            h_next: Next hidden state
            c_next: Next cell state
            cache: Values needed for backward pass
        """
        # Concatenate input and previous hidden state
        concat = np.vstack((h_prev, xt))
        
        # Input gate
        i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
        
        # Forget gate
        f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
        
        # Output gate
        o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        
        # Candidate cell state
        c_tilde = self.tanh(np.dot(self.Wc, concat) + self.bc)
        
        # Cell state
        c_next = f * c_prev + i * c_tilde
        
        # Hidden state
        h_next = o * self.tanh(c_next)
        
        # Store values needed for backward pass
        cache = (i, f, o, c_tilde, c_next, h_prev, c_prev, xt, concat)
        
        return h_next, c_next, cache
    
    def backward_step(self, dh_next, dc_next, cache):
        """
        Backward pass for a single timestep
        
        Args:
            dh_next: Gradient of loss with respect to next hidden state
            dc_next: Gradient of loss with respect to next cell state
            cache: Values from forward pass
            
        Returns:
            dx: Gradient of loss with respect to input
            dh_prev: Gradient of loss with respect to previous hidden state
            dc_prev: Gradient of loss with respect to previous cell state
            dW: Dictionary containing gradients of all weights and biases
        """
        # Unpack cache
        i, f, o, c_tilde, c_next, h_prev, c_prev, xt, concat = cache
        
        # Initialize gradients
        dW = {
            'Wi': np.zeros_like(self.Wi),
            'bi': np.zeros_like(self.bi),
            'Wf': np.zeros_like(self.Wf),
            'bf': np.zeros_like(self.bf),
            'Wo': np.zeros_like(self.Wo),
            'bo': np.zeros_like(self.bo),
            'Wc': np.zeros_like(self.Wc),
            'bc': np.zeros_like(self.bc)
        }
        
        # Gradient of loss with respect to output gate
        do = self.tanh(c_next) * dh_next
        do_input = self.sigmoid_derivative(o) * do
        dW['Wo'] += np.dot(do_input, concat.T)
        dW['bo'] += do_input
        
        # Gradient of loss with respect to cell state
        dc = o * self.tanh_derivative(self.tanh(c_next)) * dh_next + dc_next
        
        # Gradient of loss with respect to input gate
        di = c_tilde * dc
        di_input = self.sigmoid_derivative(i) * di
        dW['Wi'] += np.dot(di_input, concat.T)
        dW['bi'] += di_input
        
        # Gradient of loss with respect to forget gate
        df = c_prev * dc
        df_input = self.sigmoid_derivative(f) * df
        dW['Wf'] += np.dot(df_input, concat.T)
        dW['bf'] += df_input
        
        # Gradient of loss with respect to candidate cell state
        dc_tilde = i * dc
        dc_tilde_input = self.tanh_derivative(c_tilde) * dc_tilde
        dW['Wc'] += np.dot(dc_tilde_input, concat.T)
        dW['bc'] += dc_tilde_input
        
        # Gradient of loss with respect to concatenated input
        dconcat = (np.dot(self.Wo.T, do_input) +
                  np.dot(self.Wi.T, di_input) +
                  np.dot(self.Wf.T, df_input) +
                  np.dot(self.Wc.T, dc_tilde_input))
        
        # Split gradient for previous hidden state and input
        dh_prev = dconcat[:self.hidden_size]
        dx = dconcat[self.hidden_size:]
        
        # Gradient of loss with respect to previous cell state
        dc_prev = f * dc
        
        return dx, dh_prev, dc_prev, dW

class LSTM:
    def __init__(self, input_size, hidden_size, output_size=2):
        """
        Initialize LSTM network
        
        Args:
            input_size: Dimension of input vector
            hidden_size: Dimension of hidden state
            output_size: Dimension of output (default=2 for x,y coordinates)
        """
        self.cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Output layer weights
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))
        
    def forward(self, x, h0=None, c0=None):
        """
        Forward pass for entire sequence
        
        Args:
            x: Input sequence (sequence_length, input_size, batch_size)
            h0: Initial hidden state (hidden_size, batch_size)
            c0: Initial cell state (hidden_size, batch_size)
            
        Returns:
            h: Hidden states for all timesteps
            c: Cell states for all timesteps
            y: Output predictions
            cache: Values needed for backward pass
        """
        # Get dimensions
        seq_len, input_size, batch_size = x.shape
        
        # Initialize hidden and cell states if not provided
        if h0 is None:
            h0 = np.zeros((self.hidden_size, batch_size))
        if c0 is None:
            c0 = np.zeros((self.hidden_size, batch_size))
            
        # Initialize output arrays
        h = np.zeros((seq_len, self.hidden_size, batch_size))
        c = np.zeros((seq_len, self.hidden_size, batch_size))
        y = np.zeros((seq_len, self.output_size, batch_size))
        
        # Initialize cache list
        cache = []
        
        # Process each timestep
        h_prev = h0
        c_prev = c0
        
        for t in range(seq_len):
            # Get current input
            xt = x[t]
            
            # Forward step
            h_next, c_next, step_cache = self.cell.forward_step(xt, h_prev, c_prev)
            
            # Compute output
            y[t] = np.dot(self.Wy, h_next) + self.by
            
            # Store outputs
            h[t] = h_next
            c[t] = c_next
            
            # Update previous states
            h_prev = h_next
            c_prev = c_next
            
            # Store cache
            cache.append(step_cache)
            
        return h, c, y, cache
    
    def backward(self, x, y, y_pred, cache):
        """
        Backward pass through time
        
        Args:
            x: Input sequence (sequence_length, input_size, batch_size)
            y: True output sequence (sequence_length, output_size, batch_size)
            y_pred: Predicted output sequence (sequence_length, output_size, batch_size)
            cache: Values from forward pass
            
        Returns:
            gradients: Dictionary containing gradients of all parameters
        """
        seq_len, input_size, batch_size = x.shape
        
        # Initialize gradients
        gradients = {
            'Wy': np.zeros_like(self.Wy),
            'by': np.zeros_like(self.by),
            'Wi': np.zeros_like(self.cell.Wi),
            'bi': np.zeros_like(self.cell.bi),
            'Wf': np.zeros_like(self.cell.Wf),
            'bf': np.zeros_like(self.cell.bf),
            'Wo': np.zeros_like(self.cell.Wo),
            'bo': np.zeros_like(self.cell.bo),
            'Wc': np.zeros_like(self.cell.Wc),
            'bc': np.zeros_like(self.cell.bc)
        }
        
        # Initialize gradients for hidden and cell states
        dh_next = np.zeros((self.hidden_size, batch_size))
        dc_next = np.zeros((self.hidden_size, batch_size))
        
        # Backward pass through time
        for t in reversed(range(seq_len)):
            # Gradient of loss with respect to output
            dy = y_pred[t] - y[t]
            
            # Gradient of loss with respect to output layer weights
            gradients['Wy'] += np.dot(dy, cache[t][4].T)  # cache[t][4] is h_next
            gradients['by'] += np.sum(dy, axis=1, keepdims=True)
            
            # Gradient of loss with respect to hidden state
            dh = np.dot(self.Wy.T, dy) + dh_next
            
            # Backward step
            dx, dh_next, dc_next, dW = self.cell.backward_step(dh, dc_next, cache[t])
            
            # Accumulate gradients
            for key in dW:
                gradients[key] += dW[key]
        
        return gradients
    
    def update_parameters(self, gradients, learning_rate):
        """
        Update network parameters using gradients
        
        Args:
            gradients: Dictionary containing gradients of all parameters
            learning_rate: Learning rate for gradient descent
        """
        # Update output layer weights
        self.Wy -= learning_rate * gradients['Wy']
        self.by -= learning_rate * gradients['by']
        
        # Update LSTM cell weights
        self.cell.Wi -= learning_rate * gradients['Wi']
        self.cell.bi -= learning_rate * gradients['bi']
        self.cell.Wf -= learning_rate * gradients['Wf']
        self.cell.bf -= learning_rate * gradients['bf']
        self.cell.Wo -= learning_rate * gradients['Wo']
        self.cell.bo -= learning_rate * gradients['bo']
        self.cell.Wc -= learning_rate * gradients['Wc']
        self.cell.bc -= learning_rate * gradients['bc']
    
    def compute_loss(self, y_pred, y_true):
        """Compute mean squared error loss"""
        return np.mean((y_pred - y_true) ** 2)
    
    def compute_rmse(self, y_pred, y_true):
        """Compute root mean squared error (RMSE)"""
        return np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    def train_step(self, x, y_true, learning_rate=0.01):
        """
        Perform one training step
        
        Args:
            x: Input sequence (sequence_length, input_size, batch_size)
            y_true: True output sequence (sequence_length, output_size, batch_size)
            learning_rate: Learning rate for gradient descent
            
        Returns:
            loss: Mean squared error loss
        """
        # Forward pass
        _, _, y_pred, cache = self.forward(x)
        
        # Compute loss
        loss = self.compute_loss(y_pred, y_true)
        
        # Backward pass
        gradients = self.backward(x, y_true, y_pred, cache)
        
        # Update parameters
        self.update_parameters(gradients, learning_rate)
        
        return loss

def train_model(model, data_loader, num_epochs=10, batch_size=32, learning_rate=0.01,
                lr_patience=5, early_stopping_patience=10, save_path='training_plots'):
    """
    Train the LSTM model with learning rate scheduling, early stopping, and model checkpointing
    
    Args:
        model: LSTM model instance
        data_loader: CarDataLoader instance
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        lr_patience: Patience for learning rate scheduling
        early_stopping_patience: Patience for early stopping
        save_path: Path to save training plots and model checkpoints
    """
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_data()
    
    # Initialize learning rate scheduler, early stopping, and model checkpoint
    lr_scheduler = LearningRateScheduler(learning_rate, patience=lr_patience)
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    visualizer = TrainingVisualizer()
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(save_path, 'best_model.pkl'),
        monitor='val_loss',
        mode='min'
    )
    
    # Create directory for saving plots and checkpoints
    os.makedirs(save_path, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        total_loss = 0
        num_batches = len(X_train) // batch_size
        
        for i in range(num_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            x_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            # Reshape for LSTM input
            x_batch = x_batch.transpose(1, 2, 0)  # (seq_len, input_size, batch_size)
            y_batch = y_batch.transpose(1, 2, 0)  # (seq_len, output_size, batch_size)
            
            # Training step
            loss = model.train_step(x_batch, y_batch, learning_rate)
            total_loss += loss
            
        # Print epoch statistics
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
        
        # Evaluate on validation set
        x_val = X_val.transpose(1, 2, 0)
        y_val = y_val.transpose(1, 2, 0)
        _, _, y_pred_val, _ = model.forward(x_val)
        val_loss = model.compute_loss(y_pred_val, y_val)
        val_rmse = model.compute_rmse(y_pred_val, y_val)
        print(f"Validation Loss: {val_loss:.6f}, Validation RMSE: {val_rmse:.6f}")
        
        # Save model if it's the best so far
        checkpoint.save_model(model, val_loss, epoch)
        
        # Check early stopping
        if early_stopping.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load the best model for final evaluation
    checkpoint.load_model(model)
    
    # Plot training metrics
    visualizer.plot_metrics(os.path.join(save_path, 'training_metrics.png'))
    
    # Evaluate on test set and plot predictions
    x_test = X_test.transpose(1, 2, 0)
    y_test = y_test.transpose(1, 2, 0)
    _, _, y_pred_test, _ = model.forward(x_test)
    test_loss = model.compute_loss(y_pred_test, y_test)
    test_rmse = model.compute_rmse(y_pred_test, y_test)
    print(f"\nTest Loss: {test_loss:.6f}, Test RMSE: {test_rmse:.6f}")
    
    # Plot predictions
    visualizer.plot_predictions(y_test.transpose(1, 0, 2), 
                              y_pred_test.transpose(1, 0, 2),
                              os.path.join(save_path, 'predictions.png'))

if __name__ == "__main__":
    # Example usage
    from data_loader import CarDataLoader
    from training_utils import (LearningRateScheduler, EarlyStopping, 
                              TrainingVisualizer, ModelCheckpoint)
    
    # Initialize data loader
    data_loader = CarDataLoader("car_data")
    
    # Initialize LSTM model
    input_size = 12  # Number of input features
    hidden_size = 64
    output_size = 2  # x,y coordinates
    
    model = LSTM(input_size, hidden_size, output_size)
    
    # Train the model
    train_model(model, data_loader, 
                num_epochs=50,  # Increased epochs since we have early stopping
                batch_size=32,
                learning_rate=0.01,
                lr_patience=5,
                early_stopping_patience=10,
                save_path='training_plots') 