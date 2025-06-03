import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. Redefine the exact same LSTM autoencoder class
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, num_layers=2):
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=input_dim, num_layers=num_layers, batch_first=True
        )

    def forward(self, x):
        enc_out, _ = self.encoder_lstm(x)           # [B, seq_len, hidden_dim]
        h_final = enc_out[:, -1, :]                 # [B, hidden_dim]
        z = self.fc_enc(h_final)                    # [B, latent_dim]
        h_dec = self.fc_dec(z).unsqueeze(1)         # [B, 1, hidden_dim]
        h_dec_rep = h_dec.repeat(1, x.size(1), 1)   # [B, seq_len, hidden_dim]
        dec_out, _ = self.decoder_lstm(h_dec_rep)   # [B, seq_len, input_dim]
        return dec_out

# 2. Load normalization params + threshold
norm_data = np.load("norm_params.npz")
data_min = norm_data["data_min"]
data_max = norm_data["data_max"]
threshold = torch.load("threshold.pt").item()

# 3. Load trained model weights
checkpoint = torch.load("model_checkpoint.pth", map_location="cpu")
input_dim = checkpoint["input_dim"]
seq_length = checkpoint["seq_length"]
model = LSTMAutoencoder(input_dim)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 4. Load new CAN CSV
new_csv = "new_can_data.csv" 
df = pd.read_csv(new_csv)

payload_cols = ["Engine Coolant Temperature [Â°C]", "Intake Manifold Absolute Pressure [kPa]", "Engine RPM [RPM]", "Vehicle Speed Sensor [km/h]", "Intake Air Temperature [Â°C]", "Air Flow Rate from Mass Flow Sensor [g/s]", "Absolute Throttle Position [%]", "Ambient Air Temperature [Â°C]", "Accelerator Pedal Position D [%]", "Accelerator Pedal Position E [%]"]

data_array = df[payload_cols].fillna(0).values.astype(np.float32)
# Apply same normalization
data_norm = (data_array - data_min) / (data_max - data_min + 1e-6)

# 5. Create sliding-window sequences
def create_sequences(data, seq_length):
    seqs = []
    for i in range(len(data) - seq_length + 1):
        seqs.append(data[i : i + seq_length])
    return np.stack(seqs)

all_sequences = create_sequences(data_norm, seq_length)
print(f"Generated {len(all_sequences)} windows from new CSV.")

#  Wrap in a Dataset + DataLoader
class CANSeqDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.from_numpy(sequences).float()
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx]

dataset = CANSeqDataset(all_sequences)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# 6. Compute reconstruction errors
errors = []
with torch.no_grad():
    for batch in loader:
        recon = model(batch)  # [B, seq_len, input_dim]
        batch_err = torch.mean((recon - batch) ** 2, dim=(1, 2))  # [B]
        errors.append(batch_err.cpu().numpy())

errors = np.concatenate(errors)
print(f"Sample reconstruction errors (first 10): {errors[:10]}")

# 7. Flag anomalies
anomaly_flags = errors > threshold
num_anomalies = int(np.sum(anomaly_flags))
print(f"Detected {num_anomalies} anomalous windows out of {len(errors)} total.")

# (Optional) Print row index & timestamp for each flagged window
flagged_idxs = np.where(anomaly_flags)[0]
for idx in flagged_idxs:
    start_row = idx
    timestamp = df.iloc[start_row]["timestamp"] if "timestamp" in df.columns else "N/A"
    print(f"Anomaly window at row {start_row}, timestamp {timestamp}")
