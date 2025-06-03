# server_main.py

import os
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------
# 1. LSTM autoencoder
# ---------------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, num_layers=2):
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        enc_out, _ = self.encoder_lstm(x)           # [B, seq_len, hidden_dim]
        h_final = enc_out[:, -1, :]                 # [B, hidden_dim]
        z = self.fc_enc(h_final)                    # [B, latent_dim]
        h_dec = self.fc_dec(z).unsqueeze(1)         # [B, 1, hidden_dim]
        h_dec_rep = h_dec.repeat(1, x.size(1), 1)   # [B, seq_len, hidden_dim]
        dec_out, _ = self.decoder_lstm(h_dec_rep)   # [B, seq_len, input_dim]
        return dec_out

# ---------------------------------------
# 2. On startup: load model, norm params, threshold
# ---------------------------------------
@app.on_event("startup")
def load_model_and_params():
    # Path to your files (adjust if these live in subfolders)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ML_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../ml'))
    model_path = os.path.join(ML_DIR, "model_checkpoint.pth")
    norm_path = os.path.join(ML_DIR, "norm_params.npz")
    threshold_path = os.path.join(ML_DIR, "threshold.pt")
    
    if not os.path.exists(model_path) or not os.path.exists(norm_path) or not os.path.exists(threshold_path):
        raise RuntimeError("Missing one of: model_checkpoint.pth, norm_params.npz, threshold.pt")
    
    # 2.a) Load normalization parameters
    global data_min, data_max, seq_length, model, threshold
    norm_data = np.load(norm_path)
    data_min = norm_data["data_min"]
    data_max = norm_data["data_max"]
    
    # 2.b) Load threshold
    threshold = torch.load(threshold_path, weights_only=True).item()
    
    # 2.c) Load model checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    input_dim = checkpoint["input_dim"]
    seq_length = checkpoint["seq_length"]
    
    # Recreate model architecture & load weights
    model_local = LSTMAutoencoder(input_dim)
    model_local.load_state_dict(checkpoint["model_state_dict"])
    model_local.eval()
    
    # Keep model in global scope so endpoint can use it
    model = model_local

# ---------------------------------------
# 3. Helper: create sliding windows of length seq_length
# ---------------------------------------
def create_sequences(data: np.ndarray, seq_length: int) -> np.ndarray:
    """
    data: NumPy array of shape [N_rows, num_features]
    seq_length: integer (same as used in training)
    returns: NumPy array [N_windows, seq_length, num_features]
    """
    seqs = []
    for i in range(len(data) - seq_length + 1):
        seqs.append(data[i : i + seq_length])
    return np.stack(seqs)

# 3.a) Wrap into a PyTorch Dataset (so we can batch‐inference)
class CANSeqDataset(Dataset):
    def __init__(self, sequences: np.ndarray):
        # sequences: [N_windows, seq_length, num_features]
        self.tensor = torch.from_numpy(sequences).float()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        # returns a single window of shape [seq_length, num_features]
        return self.tensor[idx]

# ---------------------------------------
# 4. POST /evaluate endpoint
#    - Expects an uploaded CSV file (field name "file")
#    - Returns a JSON listing anomaly windows (indices + optional timestamp)
# ---------------------------------------
@app.post("/evaluate")
async def evaluate_new_can(file: UploadFile = File(...)):
    # 4.a) Check extension
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files accepted")

    # 4.b) Read CSV into a pandas DataFrame
    try:
        # Read into memory; for very large files you might stream instead
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")

    # 4.c) Extract payload columns (must match training columns)
    payload_cols = [
        "Engine Coolant Temperature [Â°C]",
        "Intake Manifold Absolute Pressure [kPa]",
        "Engine RPM [RPM]",
        "Vehicle Speed Sensor [km/h]",
        "Intake Air Temperature [Â°C]",
        "Air Flow Rate from Mass Flow Sensor [g/s]",
        "Absolute Throttle Position [%]",
        "Ambient Air Temperature [Â°C]",
        "Accelerator Pedal Position D [%]",
        "Accelerator Pedal Position E [%]"
    ]
    missing = [col for col in payload_cols if col not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns in uploaded CSV: {missing}")

    data_array = df[payload_cols].fillna(0).values.astype(np.float32)
    # 4.d) Apply same normalization: (x - min) / (max - min)
    data_norm = (data_array - data_min) / (data_max - data_min + 1e-6)

    # 4.e) Create sliding-window sequences
    sequences = create_sequences(data_norm, seq_length)
    if len(sequences) == 0:
        raise HTTPException(status_code=400, detail="CSV too short—no sequences of length seq_length")

    # 4.f) Inference in batches
    dataset = CANSeqDataset(sequences)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    recon_errors = []
    with torch.no_grad():
        for batch in loader:
            # `[batch_size, seq_length, input_dim]`
            rec = model(batch)
            # MSE over (seq_length × input_dim) for each window in the batch
            errs = torch.mean((rec - batch) ** 2, dim=(1, 2))  # shape: [batch_size]
            recon_errors.append(errs.cpu().numpy())

    recon_errors = np.concatenate(recon_errors)  # shape [num_windows]
    # 4.g) Flag anomalies
    anomaly_flags = recon_errors > threshold
    num_windows = len(anomaly_flags)
    num_anoms = int(anomaly_flags.sum())

    # 4.h) Build a JSON-friendly list of flagged indices (and timestamps if present)
    flagged_indices = np.where(anomaly_flags)[0].tolist()
    response_list = []
    for idx in flagged_indices:
        # Each idx corresponds to: window covers rows [idx .. idx+seq_length-1]
        # We can report the starting row and (if available) its timestamp
        row_start = idx
        ts = df.iloc[row_start]["timestamp"] if "timestamp" in df.columns else None
        response_list.append({"window_index": idx, "start_row": row_start, "timestamp": ts})

    return JSONResponse(
        {
            "total_windows": num_windows,
            "num_anomalies": num_anoms,
            "anomalies": response_list
        }
    )
