import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CarDataLoader:
    def __init__(self, data_dir, sequence_length=62, prediction_steps=5):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing the car data files
            sequence_length: Number of time steps in input sequence
            prediction_steps: Number of future steps to predict
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def load_data(self):
        """Load and preprocess all data files"""
        all_inputs = []
        all_outputs = []
        
        # Check if directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory {self.data_dir} does not exist")
        
        # Get list of files
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        if not files:
            raise ValueError(f"No .csv files found in {self.data_dir}")
        
        print(f"Found {len(files)} data files")
        
        # Load all files
        for filename in files:
            try:
                file_path = os.path.join(self.data_dir, filename)
                # Read CSV file
                data = pd.read_csv(file_path).values
                
                # Check data shape
                if data.shape[0] < self.sequence_length + self.prediction_steps:
                    print(f"Warning: File {filename} has insufficient time steps. Skipping.")
                    continue
                
                # Input: sequence_length time steps, only x,y coordinates
                x = data[:self.sequence_length, :2]  # Only take first 2 features (x,y)
                # Output: next prediction_steps time steps, x,y coordinates
                y = data[self.sequence_length:self.sequence_length + self.prediction_steps, :2]
                
                all_inputs.append(x)
                all_outputs.append(y)
                
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue
        
        if not all_inputs:
            raise ValueError("No valid data files were processed")
        
        print(f"Successfully processed {len(all_inputs)} files")
        
        # Convert to numpy arrays
        X = np.array(all_inputs)  # (num_samples, sequence_length, 2)
        y = np.array(all_outputs) # (num_samples, prediction_steps, 2)
        
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")
        
        # Normalize the input data
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = self.input_scaler.fit_transform(X_reshaped)
        X = X_normalized.reshape(X.shape)
        
        # Normalize the target data
        y_reshaped = y.reshape(-1, y.shape[-1])
        y_normalized = self.target_scaler.fit_transform(y_reshaped)
        y = y_normalized.reshape(y.shape)
        
        # Split into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Reshape for LSTM input (sequence_length, input_dim, batch_size)
        X_train = np.transpose(X_train, (1, 2, 0))
        X_val = np.transpose(X_val, (1, 2, 0))
        X_test = np.transpose(X_test, (1, 2, 0))
        
        # Reshape targets (prediction_steps, output_dim, batch_size)
        y_train = np.transpose(y_train, (1, 2, 0))
        y_val = np.transpose(y_val, (1, 2, 0))
        y_test = np.transpose(y_test, (1, 2, 0))
        
        print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
        print(f"Validation shapes: X={X_val.shape}, y={y_val.shape}")
        print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def inverse_transform_predictions(self, y_pred):
        """
        Convert normalized predictions back to original scale
        
        Args:
            y_pred: Normalized predictions (prediction_steps, output_dim, batch_size)
            
        Returns:
            Original scale predictions
        """
        # Reshape to 2D array
        y_pred_reshaped = y_pred.transpose(1, 0, 2).reshape(-1, 2)
        # Inverse transform
        y_pred_original = self.target_scaler.inverse_transform(y_pred_reshaped)
        # Reshape back to original shape
        return y_pred_original.reshape(2, -1, y_pred.shape[2]).transpose(1, 0, 2) 