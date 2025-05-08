import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CarDataLoader:
    def __init__(self, data_dir, sequence_length=67, prediction_steps=5):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing the car data files
            sequence_length: Number of time steps in each sequence
            prediction_steps: Number of future steps to predict
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.scaler = StandardScaler()
        
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
                if data.shape[0] < self.sequence_length:
                    print(f"Warning: File {filename} has insufficient time steps. Skipping.")
                    continue
                
                # Input: first 62 time steps, all 12 features
                x = data[:62, :]
                # Output: next 5 time steps, first 2 features (x, y)
                y = data[62:67, :2]
                
                # Check shapes
                if x.shape != (62, 12):
                    print(f"Warning: File {filename} has incorrect input shape {x.shape}. Expected (62, 12). Skipping.")
                    continue
                if y.shape != (5, 2):
                    print(f"Warning: File {filename} has incorrect output shape {y.shape}. Expected (5, 2). Skipping.")
                    continue
                
                all_inputs.append(x)
                all_outputs.append(y)
                
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue
        
        if not all_inputs:
            raise ValueError("No valid data files were processed")
        
        print(f"Successfully processed {len(all_inputs)} files")
        
        # Convert to numpy arrays
        X = np.array(all_inputs)  # (num_samples, 62, 12)
        y = np.array(all_outputs) # (num_samples, 5, 2)
        
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")
        
        # Normalize the input data
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X = X_normalized.reshape(X.shape)
        
        # Split into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def inverse_transform(self, data):
        """Convert normalized data back to original scale"""
        return self.scaler.inverse_transform(data) 