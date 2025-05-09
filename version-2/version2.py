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
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size

        self.W_f = nn.Parameter(torch.randn(concat_size, hidden_size) * 0.1)
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        self.W_i = nn.Parameter(torch.randn(concat_size, hidden_size) * 0.1)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        self.W_c = nn.Parameter(torch.randn(concat_size, hidden_size) * 0.1)
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        self.W_o = nn.Parameter(torch.randn(concat_size, hidden_size) * 0.1)
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev, c_prev):
        combined = torch.cat([x_t, h_prev], dim=1)
        f_t = torch.sigmoid(combined @ self.W_f + self.b_f)
        i_t = torch.sigmoid(combined @ self.W_i + self.b_i)
        c_tilde = torch.tanh(combined @ self.W_c + self.b_c)
        c_next = f_t * c_prev + i_t * c_tilde
        o_t = torch.sigmoid(combined @ self.W_o + self.b_o)
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

    def forward(self, input_seq):
        batch_size, seq_len, _ = input_seq.size()
        h_t = [torch.zeros(batch_size, self.hidden_size, device=input_seq.device) for _ in range(self.num_layers)]
        c_t = [torch.zeros(batch_size, self.hidden_size, device=input_seq.device) for _ in range(self.num_layers)]

        outputs = []

        for t in range(seq_len):
            x = input_seq[:, t, :]
            for i, lstm in enumerate(self.lstm_layers):
                h_t[i], c_t[i] = lstm(x, h_t[i], c_t[i])
                if i < self.num_layers - 1:  # Apply dropout to all but last
                    x = self.dropout_layers[i](h_t[i])
                else:
                    x = h_t[i]
            output = self.output_layer(x)
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1)

# 3. DATASET LOADER
class CarTrajectoryDataset(Dataset):
    def __init__(self, file_paths, input_steps=62, pred_steps=5, input_scaler=None, target_scaler=None):
        self.data = []
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

        for path in file_paths:
            df = pd.read_csv(path).dropna().values  # Handles headers
            if df.shape[0] < input_steps + pred_steps:
                continue
            input_seq = df[:input_steps]
            target_seq = df[input_steps:input_steps + pred_steps, :2]  # (x, y) only
            self.data.append((input_seq, target_seq))

        # Apply scaling if scalers provided
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
def train(model, train_loader, val_loader, num_epochs=20, lr=0.001, device='cuda'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_rmse_log = []
    val_rmse_log = []

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output_seq = model(x_batch)
            pred_next_5 = output_seq[:, -5:]
            loss = criterion(pred_next_5, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_rmse = np.sqrt(np.mean(train_losses))
        val_rmse = evaluate(model, val_loader, device)

        train_rmse_log.append(avg_train_rmse)
        val_rmse_log.append(val_rmse)

        print(f"Train RMSE: {avg_train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

    # Save training curve
    plt.figure()
    plt.plot(train_rmse_log, label='Train RMSE')
    plt.plot(val_rmse_log, label='Val RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training & Validation RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig("train_val_rmse.png")
    plt.close()

def evaluate(model, dataloader, device='cuda'):
    model.eval()
    rmse_list = []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output_seq = model(x_batch)
            pred_next_5 = output_seq[:, -5:]
            loss = criterion(pred_next_5, y_batch)
            rmse = torch.sqrt(loss)
            rmse_list.append(rmse.item())
    return np.mean(rmse_list)

def test_and_collect_predictions(model, dataloader, device='cuda'):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            output_seq = model(x_batch)
            pred_next_5 = output_seq[:, -5:]
            all_preds.append(pred_next_5.cpu())
            all_targets.append(y_batch)

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    # test_rmse = np.sqrt(np.mean((preds - targets) ** 2))
    # print(f"Test RMSE: {test_rmse:.4f}")
    return preds, targets

def plot_prediction(preds, targets, sample_idx=0, save_path="sample_prediction.png"):
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

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using the following device: ", device)
    
    # 1. Load file paths
    data_dir = '../car_data'  # <-- Replace with actual path if needed
    file_list = sorted(glob.glob(os.path.join(data_dir, '*.csv')))

    # 2. Initialize scalers
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # 3. Create dataset and scale both inputs and targets
    full_dataset = CarTrajectoryDataset(
        file_list,
        input_scaler=input_scaler,
        target_scaler=target_scaler
    )

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
        input_size=12,
        hidden_size=64,
        output_size=2,
        num_layers=3,       # Try 3 layers
        dropout=0.3         # Apply dropout between them
    )

    # 7. Train model
    train(model, train_loader, val_loader, num_epochs=20, lr=0.001, device=device)

    # 8. Test and get predictions (in normalized space)
    preds_scaled, targets_scaled = test_and_collect_predictions(model, test_loader, device=device)

    # 9. Inverse transform to original scale
    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 2)).reshape(preds_scaled.shape)
    targets = target_scaler.inverse_transform(targets_scaled.reshape(-1, 2)).reshape(targets_scaled.shape)

    true_rmse = np.sqrt(np.mean((preds - targets) ** 2))
    print(f"True Test RMSE (original scale): {true_rmse:.4f}")
    
    # 10. Plot predictions
    plot_prediction(preds, targets, sample_idx=0)
  # Plot the first sample

if __name__ == '__main__':
    main()