import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------
# 1. Load CSV and basic preprocessing
# ---------------------------------------
# Adjust the path here if your CSV has a different name
csv_path = "can_data.csv"
df = pd.read_csv(csv_path)

# Use the actual column names from the CSV header
payload_cols = ["Engine Coolant Temperature [Â°C]", "Intake Manifold Absolute Pressure [kPa]", "Engine RPM [RPM]", "Vehicle Speed Sensor [km/h]", "Intake Air Temperature [Â°C]", "Air Flow Rate from Mass Flow Sensor [g/s]", "Absolute Throttle Position [%]", "Ambient Air Temperature [Â°C]", "Accelerator Pedal Position D [%]", "Accelerator Pedal Position E [%]"]

# Extract payload as a NumPy array of shape [N_rows, num_payload_cols]
data_array = df[payload_cols].fillna(0).values.astype(np.float32)

# Normalize each payload byte to the range [0,1]
data_min = data_array.min(axis=0)
data_max = data_array.max(axis=0)
data_norm = (data_array - data_min) / (data_max - data_min + 1e-6)

# Check for NaN or inf values in the normalized data
if np.isnan(data_norm).any() or np.isinf(data_norm).any():
    print("Warning: Input data contains NaN or inf values after normalization.")

# ---------------------------------------
# 2. Create sliding-window sequences
# ---------------------------------------
def create_sequences(data, seq_length=50):
    """
    data: NumPy array of shape [N_samples, num_features]
    seq_length: how many timesteps per sequence
    returns: NumPy array of shape [N_sequences, seq_length, num_features]
    """
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i : i + seq_length]
        sequences.append(seq)
    return np.stack(sequences)

seq_length = 50  # 50 timesteps per window; you can tweak this
all_sequences = create_sequences(data_norm, seq_length)

# Split into 80% train, 20% test
split_idx = int(0.8 * len(all_sequences))
train_seqs = all_sequences[:split_idx]
test_seqs = all_sequences[split_idx:]

print(f"Total sequences: {len(all_sequences)}")
print(f"Train sequences: {len(train_seqs)}, Test sequences: {len(test_seqs)}")

# ---------------------------------------
# 3. PyTorch Dataset & DataLoader
# ---------------------------------------
class CANSequenceDataset(Dataset):
    def __init__(self, sequences):
        # sequences: NumPy array [N, seq_length, num_features]
        self.sequences = torch.from_numpy(sequences)  # convert to torch.Tensor

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Return a single sequence: shape [seq_length, num_features]
        return self.sequences[idx]

batch_size = 64
train_dataset = CANSequenceDataset(train_seqs)
test_dataset = CANSequenceDataset(test_seqs)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------------------------------
# 4. LSTM Autoencoder Definition
# ---------------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, num_layers=2):
        super().__init__()
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        # Bottleneck
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        """
        x: [batch_size, seq_length, input_dim]
        returns: reconstruction [batch_size, seq_length, input_dim]
        """
        enc_out, _ = self.encoder_lstm(x)          # enc_out: [B, seq_length, hidden_dim]
        h_final = enc_out[:, -1, :]                # final hidden state: [B, hidden_dim]
        z = self.fc_enc(h_final)                   # latent vector: [B, latent_dim]
        h_dec = self.fc_dec(z).unsqueeze(1)        # [B, 1, hidden_dim]
        h_dec_rep = h_dec.repeat(1, x.size(1), 1)   # [B, seq_length, hidden_dim]
        dec_out, _ = self.decoder_lstm(h_dec_rep)  # [B, seq_length, input_dim]
        return dec_out

# Instantiate the model
input_dim = train_seqs.shape[-1]  # number of payload features (e.g., 8)
model = LSTMAutoencoder(input_dim)

# ---------------------------------------
# 5. Training Loop (MSE on reconstruction)
# ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)                  # [B, seq_length, input_dim]
        optimizer.zero_grad()
        recon = model(batch)                      # [B, seq_length, input_dim]
        loss = criterion(recon, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * batch.size(0)

    epoch_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.6f}")

# ---------------------------------------
# 6. Basic Testing & Reconstruction Error
# ---------------------------------------
model.eval()
recon_errors = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        recon = model(batch)
        # Compute MSE per sequence (averaged across timesteps & features)
        batch_err = torch.mean((recon - batch) ** 2, dim=(1, 2))  # [B]
        recon_errors.append(batch_err.cpu().numpy())

recon_errors = np.concatenate(recon_errors)
print(f"Example reconstruction errors (first 10): {recon_errors[:10]}")

# Compute a simple threshold on the test set: mean + 4*std
mean_err = recon_errors.mean()
std_err  = recon_errors.std()
threshold = mean_err + 4 * std_err
print(f"Threshold for anomalies = {threshold:.6f}")

import torch

# 1) Save the trained model
torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim": input_dim,
    "seq_length": seq_length
}, "model_checkpoint.pth")

# 2) Save normalization parameters (data_min, data_max)
np.savez("norm_params.npz", data_min=data_min, data_max=data_max)

# 3) Save threshold as a single scalar
torch.save(torch.tensor(threshold), "threshold.pt")

print("Saved model_checkpoint.pth, norm_params.npz, and threshold.pt")

# Flag any sequence whose error exceeds threshold
anomaly_flags = recon_errors > threshold
print(f"Flagged anomalies in test set: {anomaly_flags.sum()} / {len(recon_errors)}")


