import torch
import torch.nn as nn

class CANLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, num_layers=2):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        enc_out, _ = self.encoder_lstm(x)             # [B, T, hidden_dim]
        h_final = enc_out[:, -1, :]                    # [B, hidden_dim]
        z = self.fc_enc(h_final)                       # [B, latent_dim]
        h_dec = self.fc_dec(z).unsqueeze(1)            # [B, 1, hidden_dim]
        h_dec_rep = h_dec.repeat(1, x.size(1), 1)      # [B, T, hidden_dim]
        dec_out, _ = self.decoder_lstm(h_dec_rep)      # [B, T, input_dim]
        return dec_out


    model = CANLSTMAutoencoder(input_dim=F)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for batch in train_loader:    # train_loader yields only “normal” windows
            optimizer.zero_grad()
            recon = model(batch)      # [B, T, F]
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Train Loss: {total_loss/len(train_loader)}")

    # Compute reconstruction errors on a held-out normal validation set
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in val_norm_loader:
            recon = model(batch)
            batch_err = ((recon - batch)**2).mean(dim=[1,2])  # MSE per window
            errors.append(batch_err.cpu().numpy())
    errors = np.concatenate(errors)
    mu, sigma = errors.mean(), errors.std()
    threshold = mu + 3*sigma
    print("Anomaly Threshold:", threshold)
