import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Matrix Dataset Loader
class SpatialTemporalDataset(Dataset):
    def __init__(self, csv_file, scaler=None, seq_length=6):
        df = pd.read_csv(csv_file)
        
        # Drop datetime for the neural network
        cols_to_drop = ['datetime', 'datetime_shifted']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        # Ensure all remaining columns are numeric
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
        self.feature_names = df.columns.tolist()
        
        # 1. Identify the 30 traffic columns dynamically
        self.traffic_cols = [col for col in df.columns if 'Node_' in col]
        
        # 2. LOG TRANSFORM only the traffic columns
        df[self.traffic_cols] = np.log1p(df[self.traffic_cols])
        
        # 3. Scale the data
        if scaler is None:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(df.values)
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(df.values)
            
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Input: All 34 features for the last 'seq_length' hours
        x = self.data[idx : idx + self.seq_length]
        
        # Target: The next hour. We only want to predict the first 31 columns 
        # (30 Nodes Traffic + 1 Carbon Intensity). We don't need the AI to predict the time.
        y = self.data[idx + self.seq_length, :31] 
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 2. The Multi-Variate LSTM Architecture
class CityWideLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=256, num_layers=3, output_size=31):
        super(CityWideLSTM, self).__init__()
        # 2 layers are better here to prevent overfitting on the 7-day dataset
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 3. Training Loop
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Initialize Dataset (Seq length of 6 hours is optimal for a 7-day dataset)
    train_ds = SpatialTemporalDataset('matrix_train.csv', seq_length=6)
    
    # Pass the train scaler to the validation dataset to ensure consistency
    val_ds = SpatialTemporalDataset('matrix_val.csv', scaler=train_ds.scaler, seq_length=6)
    
    # Save the scaler for testing and NS-3 inference later
    joblib.dump(train_ds.scaler, 'matrix_scaler.pkl')
    
    # Smaller batch size (16) because we have fewer total rows now
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = CityWideLSTM().to(device)
    
    # Huber Loss is robust to traffic spikes
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    print("Starting Spatial-Temporal Matrix Training...")
    epochs = 60
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b)
            
            # Calculate loss (equally across all 30 nodes and carbon)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                val_pred = model(x_v)
                val_loss += criterion(val_pred, y_v).item()
        
        if (epoch + 1) % 5 == 0:
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # Save the final matrix-capable model
    torch.save(model.state_dict(), 'citywide_lstm.pth')
    print("Model saved as 'citywide_lstm.pth'")
    print("Scaler saved as 'matrix_scaler.pkl'")

if __name__ == "__main__":
    train()