import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import pickle
import os

class LearningRateScheduler:
    def __init__(self, initial_lr: float, decay_factor: float = 0.1, patience: int = 5):
        """
        Initialize learning rate scheduler
        
        Args:
            initial_lr: Initial learning rate
            decay_factor: Factor to multiply learning rate by when reducing
            patience: Number of epochs to wait before reducing learning rate
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.decay_factor = decay_factor
        self.patience = patience
        self.best_loss = float('inf')
        self.wait = 0
        
    def step(self, current_loss: float) -> float:
        """
        Update learning rate based on validation loss
        
        Args:
            current_loss: Current validation loss
            
        Returns:
            Updated learning rate
        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.current_lr *= self.decay_factor
                self.wait = 0
                print(f"Reducing learning rate to {self.current_lr}")
        
        return self.current_lr

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in validation loss to be considered as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        
    def step(self, current_loss: float) -> bool:
        """
        Check if training should be stopped
        
        Args:
            current_loss: Current validation loss
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False

class TrainingVisualizer:
    def __init__(self):
        """Initialize training visualizer"""
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_rmses: List[float] = []
        self.val_rmses: List[float] = []
        self.learning_rates: List[float] = []
        
    def update(self, train_loss: float, val_loss: float, 
              train_rmse: float, val_rmse: float, lr: float):
        """
        Update training metrics
        
        Args:
            train_loss: Training loss
            val_loss: Validation loss
            train_rmse: Training RMSE
            val_rmse: Validation RMSE
            lr: Current learning rate
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_rmses.append(train_rmse)
        self.val_rmses.append(val_rmse)
        self.learning_rates.append(lr)
        
    def plot_metrics(self, save_path: str = None):
        """
        Plot training metrics
        
        Args:
            save_path: Path to save the plot (optional)
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot RMSE
        ax2.plot(self.train_rmses, label='Training RMSE')
        ax2.plot(self.val_rmses, label='Validation RMSE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Training and Validation RMSE')
        ax2.legend()
        ax2.grid(True)
        
        # Plot learning rate
        ax3.plot(self.learning_rates)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        save_path: str = None):
        """
        Plot true vs predicted trajectories
        
        Args:
            y_true: True trajectories
            y_pred: Predicted trajectories
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 6))
        
        # Plot first few samples
        for i in range(min(5, y_true.shape[0])):
            plt.plot(y_true[i, :, 0], y_true[i, :, 1], 'b-', label='True' if i == 0 else None)
            plt.plot(y_pred[i, :, 0], y_pred[i, :, 1], 'r--', label='Predicted' if i == 0 else None)
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('True vs Predicted Trajectories')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

class ModelCheckpoint:
    def __init__(self, filepath: str, monitor: str = 'val_loss', mode: str = 'min'):
        """
        Initialize model checkpoint
        
        Args:
            filepath: Path to save the model
            monitor: Metric to monitor ('val_loss' or 'val_rmse')
            mode: 'min' for loss, 'max' for accuracy
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
    def save_model(self, model, current_value: float, epoch: int):
        """
        Save model if it's the best so far
        
        Args:
            model: LSTM model to save
            current_value: Current value of monitored metric
            epoch: Current epoch number
            
        Returns:
            bool: True if model was saved, False otherwise
        """
        if self.mode == 'min':
            is_better = current_value < self.best_value
        else:
            is_better = current_value > self.best_value
            
        if is_better:
            self.best_value = current_value
            self.best_epoch = epoch
            
            # Save model state
            model_state = {
                'Wy': model.Wy,
                'by': model.by,
                'cell': {
                    'Wi': model.cell.Wi,
                    'bi': model.cell.bi,
                    'Wf': model.cell.Wf,
                    'bf': model.cell.bf,
                    'Wo': model.cell.Wo,
                    'bo': model.cell.bo,
                    'Wc': model.cell.Wc,
                    'bc': model.cell.bc
                }
            }
            
            with open(self.filepath, 'wb') as f:
                pickle.dump(model_state, f)
            
            print(f"Model saved at epoch {epoch} with {self.monitor}: {current_value:.6f}")
            return True
            
        return False
    
    def load_model(self, model):
        """
        Load the best model
        
        Args:
            model: LSTM model to load weights into
        """
        if os.path.exists(self.filepath):
            with open(self.filepath, 'rb') as f:
                model_state = pickle.load(f)
            
            # Load weights
            model.Wy = model_state['Wy']
            model.by = model_state['by']
            model.cell.Wi = model_state['cell']['Wi']
            model.cell.bi = model_state['cell']['bi']
            model.cell.Wf = model_state['cell']['Wf']
            model.cell.bf = model_state['cell']['bf']
            model.cell.Wo = model_state['cell']['Wo']
            model.cell.bo = model_state['cell']['bo']
            model.cell.Wc = model_state['cell']['Wc']
            model.cell.bc = model_state['cell']['bc']
            
            print(f"Loaded model from epoch {self.best_epoch} with {self.monitor}: {self.best_value:.6f}")
        else:
            print("No saved model found") 