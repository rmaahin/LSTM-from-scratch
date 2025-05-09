import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. LSTM CELL FROM SCRATCH
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate weights
        self.W_ii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # Forget gate weights
        self.W_if = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # Output gate weights
        self.W_io = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        # Cell candidate weights
        self.W_ig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        # Uniform initialization
        for name, param in self.named_parameters():
            if 'W_' in name:
                nn.init.uniform_(param, -0.1, 0.1)
            elif 'b_f' in name:
                nn.init.ones_(param)  # Forget gate bias to 1
            elif 'b_' in name:
                nn.init.zeros_(param)

    def forward(self, x, h_prev, c_prev):
        # Gate computations with separate input and hidden weights
        i_t = torch.sigmoid(x @ self.W_ii + h_prev @ self.W_hi + self.b_i)
        f_t = torch.sigmoid(x @ self.W_if + h_prev @ self.W_hf + self.b_f)
        o_t = torch.sigmoid(x @ self.W_io + h_prev @ self.W_ho + self.b_o)
        g_t = torch.tanh(x @ self.W_ig + h_prev @ self.W_hg + self.b_g)

        # Cell and hidden state updates
        c_next = f_t * c_prev + i_t * g_t
        h_next = o_t * torch.tanh(c_next)

        return h_next, c_next

# 2. SEQUENCE WRAPPER
class StackedLSTMSequenceModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2, dropout: float = 0.3):
        super(StackedLSTMSequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm_layers = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers - 1)
        ])

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, future_steps=5):
        batch_size, seq_len, _ = input_seq.size()
        h_t = [torch.zeros(batch_size, self.hidden_size, device=input_seq.device) for _ in range(self.num_layers)]
        c_t = [torch.zeros(batch_size, self.hidden_size, device=input_seq.device) for _ in range(self.num_layers)]

        # Pass input sequence through LSTM
        for t in range(seq_len):
            x = input_seq[:, t, :]
            for i, lstm in enumerate(self.lstm_layers):
                h_t[i], c_t[i] = lstm(x, h_t[i], c_t[i])
                if i < self.num_layers - 1:
                    x = self.dropout_layers[i](h_t[i])
                else:
                    x = h_t[i]

        # Predict future steps autoregressively
        outputs = []
        x = input_seq[:, -1, :]  # Start with last input
        for _ in range(future_steps):
            for i, lstm in enumerate(self.lstm_layers):
                h_t[i], c_t[i] = lstm(x, h_t[i], c_t[i])
                if i < self.num_layers - 1:
                    x = self.dropout_layers[i](h_t[i])
                else:
                    x = h_t[i]
            out = self.output_layer(x)
            outputs.append(out.unsqueeze(1))

            # Use predicted (x, y) and fill rest of features with last input
            x = torch.zeros_like(input_seq[:, 0, :])
            x[:, :2] = out
            if input_seq.shape[2] > 2:
                x[:, 2:] = input_seq[:, -1, 2:]  # keep same other features

        return torch.cat(outputs, dim=1)


# 3. DATASET LOADER
class CarTrajectoryDataset(Dataset):
    def __init__(self, file_paths, input_steps=62, pred_steps=5, input_scaler=None, target_scaler=None, feature_indices=None):
        self.data = []
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.feature_indices = feature_indices

        for path in file_paths:
            df = pd.read_csv(path).dropna().values
            if df.shape[0] < input_steps + pred_steps:
                continue

            if self.feature_indices:
                input_seq = df[:input_steps, self.feature_indices]
            else:
                input_seq = df[:input_steps]  # All 12 features by default

            target_seq = df[input_steps:input_steps + pred_steps, :2]  # (x, y)
            self.data.append((input_seq, target_seq))

        if self.input_scaler:
            all_inputs = np.vstack([x for x, _ in self.data])
            self.input_scaler.fit(all_inputs)
            self.data = [(self.input_scaler.transform(x), y) for x, y in self.data]

        if self.target_scaler:
            all_targets = np.vstack([y for _, y in self.data])
            self.target_scaler.fit(all_targets)
            self.data = [(x, self.target_scaler.transform(y)) for x, y in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 4. TRAINING + EVALUATION
def train(model, train_loader, val_loader, num_features, distance_threshold, num_epochs=20, lr=0.001, device='cuda', patience=5, scheduler=None, optimizer=None):
    model = model.to(device)
    criterion = nn.MSELoss()

    train_rmse_log = []
    val_rmse_log = []

    best_val_rmse = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output_seq = model(x_batch, future_steps=5)
            loss = criterion(output_seq, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_rmse = np.sqrt(np.mean(train_losses))
        val_rmse = evaluate(model, val_loader, distance_threshold, device)

        train_rmse_log.append(avg_train_rmse)
        val_rmse_log.append(val_rmse)

        print(f"Train RMSE: {avg_train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

        # Scheduler step
        if scheduler:
            scheduler.step(val_rmse)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict()
            patience_counter = 0
            print("New best model found. Saving...")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
    save_path = f"train_val_rmse_using_{num_features}_features.png"
    # Save RMSE curve
    plt.figure()
    plt.plot(train_rmse_log, label='Train RMSE')
    plt.plot(val_rmse_log, label='Val RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training & Validation RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def evaluate(model, dataloader, distance_threshold, device='cuda'):
    model.eval()
    rmse_list = []
    correct = 0
    total = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output_seq = model(x_batch, future_steps=5)
            loss = criterion(output_seq, y_batch)
            rmse = torch.sqrt(loss)
            rmse_list.append(rmse.item())

            # Flatten batch and time dims for distance calculation
            preds_flat = output_seq.reshape(-1, 2).cpu().numpy()
            targets_flat = y_batch.reshape(-1, 2).cpu().numpy()

            distances = np.linalg.norm(preds_flat - targets_flat, axis=1)
            correct += np.sum(distances < distance_threshold)
            total += len(distances)

    avg_rmse = np.mean(rmse_list)
    accuracy = correct / total if total > 0 else 0.0

    print(f"Eval RMSE: {avg_rmse:.4f}")
    print(f"% of predictions where distance between predicted and true (x, y) < {distance_threshold} unit): {accuracy*100:.2f}%")
    return avg_rmse

def test_and_collect_predictions(model, dataloader, device='cuda'):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            output_seq = model(x_batch, future_steps=5)
            all_preds.append(output_seq.cpu())
            all_targets.append(y_batch)

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    return preds, targets

def plot_prediction(preds, targets, sample_idx, num_features):
    save_path = f"sample_prediction_using_{num_features}_features.png"
    pred = preds[sample_idx]
    true = targets[sample_idx]
    plt.figure(figsize=(6, 6))
    plt.plot(true[:, 0], true[:, 1], marker='o', label='True', linestyle='-')
    plt.plot(pred[:, 0], pred[:, 1], marker='x', label='Predicted', linestyle='--')
    plt.title(f"Sample {sample_idx} Prediction")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(save_path)
    plt.close()

def plot_xy_over_time(preds, targets, num_features, sample_idx=0):
    """
    Plot predicted vs true X and Y coordinates over time for a single sample.
    """
    time_steps = np.arange(preds.shape[1])
    pred_x = preds[sample_idx, :, 0]
    pred_y = preds[sample_idx, :, 1]
    true_x = targets[sample_idx, :, 0]
    true_y = targets[sample_idx, :, 1]
    save_path = f"xy_over_time_using_{num_features}_features.png"
    plt.figure(figsize=(10, 6))

    # X over time
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, true_x, 'b-o', label='True X')
    plt.plot(time_steps, pred_x, 'r--x', label='Pred X')
    plt.ylabel("X Coordinate")
    plt.title(f"Trajectory Over Time (Sample {sample_idx})")
    plt.legend()
    plt.grid(True)

    # Y over time
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, true_y, 'b-o', label='True Y')
    plt.plot(time_steps, pred_y, 'r--x', label='Pred Y')
    plt.xlabel("Timestep")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using the following device: ", device)

    feature_names = [
        "Local_X", "Local_Y", "v_Vel", "v_Acc", "Space_Headway", "dis_cen",
        "i_l", "i_r", "i_f", "dis_l", "dis_r", "dis_f"
    ]
    distance_threshold = 5.0
    # 1. Load file paths
    data_dir = '../car_data'  # <-- Replace with actual path if needed
    file_list = sorted(glob.glob(os.path.join(data_dir, '*.csv')))

    # 2. Initialize scalers
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Choose input features: specify column indices (0 to 11)
    # Example: [0, 1] for just (x, y); None for all 12 features
    feature_indices = [0, 1] # e.g., [0, 1, 2, 3] for x, y, velocity, acceleration

    # 3. Create dataset and scale both inputs and targets
    full_dataset = CarTrajectoryDataset(
        file_list,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        feature_indices=feature_indices
    )
    if feature_indices is None:
        used_features = feature_names
    else:
        used_features = [feature_names[i] for i in feature_indices]

    print(f"Using input features ({len(used_features)}): {used_features}")

    input_dim = len(feature_indices) if feature_indices is not None else 12

    # 4. Split into train/val/test
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

    # 5. Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)

    # 6. Initialize model
    model = StackedLSTMSequenceModel(
        input_size=input_dim,
        hidden_size=64,
        output_size=2,
        num_layers=3,
        dropout=0.3
    )
    num_features = len(used_features)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    # 7. Train model
    train(
        model,
        train_loader,
        val_loader,
        num_features,
        distance_threshold,
        num_epochs=5,
        lr=0.001,
        device=device,
        patience=7,
        scheduler=scheduler,
        optimizer=optimizer
    )

    # 8. Test and get predictions (in normalized space)
    preds_scaled, targets_scaled = test_and_collect_predictions(model, test_loader, device=device)

    # 9. Inverse transform to original scale
    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 2)).reshape(preds_scaled.shape)
    targets = target_scaler.inverse_transform(targets_scaled.reshape(-1, 2)).reshape(targets_scaled.shape)
    
    # 10. Plot predictions
    sample_idx = 0
    plot_prediction(preds, targets, sample_idx, num_features)

    # 11. Compute and log metrics
    true_rmse = np.sqrt(np.mean((preds - targets) ** 2))
    distances = np.linalg.norm(preds.reshape(-1, 2) - targets.reshape(-1, 2), axis=1)
    accuracy = np.mean(distances < distance_threshold) * 100

    print(f"True Test RMSE (original scale): {true_rmse:.4f}")
    print(f"Test Accuracy (within {distance_threshold} m): {accuracy:.2f}%")
    
    log_file = f"evaluation_results_using_{num_features}.txt"
    # 11. Write metrics to file
    with open(log_file, "w") as f:
        f.write("Evaluation Metrics:\n")
        f.write(f"Test RMSE (original scale): {true_rmse:.4f}\n")
        f.write(f"Test Accuracy (within 5m): {accuracy:.2f}%\n")

    plot_xy_over_time(preds, targets, num_features, sample_idx=0)

if __name__ == '__main__':
    main()