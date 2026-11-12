import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- UPDATED: Now takes actuals_rescaled as an argument ---
def generate_presentation_traces(predictions_rescaled, actuals_rescaled, plot_timestamps):
    print("\n--- Generating Ground-Truth MOO Traces ---")
    
    num_nodes = 30
    
    # 1. Define the 4 Clusters and their Solar Multipliers
    clusters = [0]*8 + [1]*8 + [2]*8 + [3]*6
    solar_scales = {0: 1.2, 1: 0.5, 2: 0.2, 3: 0.0}
    
    # 2. Hardware Power Model
    P_idle = 100.0  # Watts just to stay on
    alpha = 0.1     # Watts per packet of traffic
    peak_solar = 250.0 # Base solar max at noon (Watts)
    
    trace_rows = []
    
    for t in range(len(predictions_rescaled)):
        dt = pd.to_datetime(plot_timestamps[t])
        hour = dt.hour
        
        # The AI's City-Wide Carbon Prediction for this hour
        grid_ci = predictions_rescaled[t, 30] 
        
        # Mathematical Solar Curve (Peaks at 12:00 PM)
        if 8 <= hour <= 16:
            base_solar = np.sin((hour - 8) * np.pi / 8) * peak_solar
        else:
            base_solar = 0.0
            
        row = {
            'Timestamp': plot_timestamps[t],
            'Grid_Carbon_Intensity': round(grid_ci, 2)
        }
        
        for n in range(num_nodes):
            # --- THE DIGITAL TWIN SPLIT ---
            # AI makes decisions based on PREDICTED traffic
            predicted_traffic = max(0, predictions_rescaled[t, n])
            
            # The physical world operates on ACTUAL traffic (from matrix_test.csv)
            actual_traffic = max(0, actuals_rescaled[t, n])
            
            cluster_id = clusters[n]
            
            # --- PHYSICS CALCULATION (Using AI Predictions for logic) ---
            node_solar_w = base_solar * solar_scales[cluster_id]
            node_demand_w = P_idle + (predicted_traffic * alpha)
            net_watts = max(0, node_demand_w - node_solar_w)
            
            # --- MOO DECISION ENGINE ---
            if net_watts == 0 or predicted_traffic > 640:
                state = 0
                action = "Active (Max QoS)"
            elif predicted_traffic < 215 and grid_ci > 385:
                state = 2
                action = "Deep Sleep (Rerouting)"
                # In deep sleep, idle power drops to 10W
                net_watts = max(0, 10.0 - node_solar_w) 
            else:
                state = 1
                action = "Light Sleep (Buffering)"
                # In light sleep, idle power drops to 50W
                net_watts = max(0, 50.0 + (predicted_traffic * alpha) - node_solar_w)
                
            carbon_emitted = (net_watts / 1000.0) * grid_ci 
                
            # --- SAVE TO ROW (NS-3 will simulate the ACTUAL traffic!) ---
            row[f'Node_{n}_State'] = state
            row[f'Node_{n}_Traffic'] = round(actual_traffic, 1) # <--- GROUND TRUTH
            row[f'Node_{n}_Net_Watts'] = round(net_watts, 2)
            row[f'Node_{n}_Carbon_gCO2'] = round(carbon_emitted, 2)
            row[f'Node_{n}_Action'] = action
            
        trace_rows.append(row)
        
    # Export to CSV
    df_trace = pd.DataFrame(trace_rows)
    df_trace.to_csv('ns3_moo_traces.csv', index=False)
    print(f"Success! Highly-detailed trace file generated with {len(df_trace)} hours.")
    print("Look for 'ns3_moo_traces.csv' in your project folder.")


# 1. Architecture (Must match exactly with your training script)
class CityWideLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=256, num_layers=3, output_size=31):
        super(CityWideLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def test_citywide_model():
    # --- 1. Load and Clean Data ---
    df = pd.read_csv('matrix_test.csv')
    seq_length = 6
    
    plot_timestamps = pd.to_datetime(df['datetime']).values[seq_length:]
    
    cols_to_drop = ['datetime', 'datetime_shifted']
    df_numeric = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    traffic_cols = [col for col in df_numeric.columns if 'Node_' in col]
    
    # --- 2. Feature Engineering (Log Transform) ---
    df_numeric[traffic_cols] = np.log1p(df_numeric[traffic_cols])
    
    # --- 3. Scaling & Model Loading ---
    scaler = joblib.load('matrix_scaler.pkl')
    scaled_data = scaler.transform(df_numeric.values)
    
    model = CityWideLSTM(input_size=34, hidden_size=256, num_layers=3)
    model.load_state_dict(torch.load('citywide_lstm.pth', map_location=torch.device('cpu')))
    model.eval()
    
    predictions = []
    actuals = []
    
    # --- 4. Sliding Window Inference ---
    print("Running Proactive Inference on Test Set...")
    with torch.no_grad():
        for i in range(len(scaled_data) - seq_length):
            x = torch.tensor(scaled_data[i : i + seq_length], dtype=torch.float32).unsqueeze(0)
            y_pred = model(x)
            
            predictions.append(y_pred.squeeze().numpy())
            actuals.append(scaled_data[i + seq_length, :31])

    # --- 5. Reverse Scaling & Log ---
    pred_full = np.zeros((len(predictions), 34))
    pred_full[:, :31] = np.array(predictions)
    predictions_rescaled = scaler.inverse_transform(pred_full)[:, :31]
    
    act_full = np.zeros((len(actuals), 34))
    act_full[:, :31] = np.array(actuals)
    actuals_rescaled = scaler.inverse_transform(act_full)[:, :31]
    
    predictions_rescaled[:, :30] = np.expm1(predictions_rescaled[:, :30])
    actuals_rescaled[:, :30] = np.expm1(actuals_rescaled[:, :30])
    
    # --- 6. Error Metrics ---
    epsilon = 1.0 
    mape_traffic = np.mean(np.abs(actuals_rescaled[:, :30] - predictions_rescaled[:, :30]) / (actuals_rescaled[:, :30] + epsilon)) * 100
    mape_carbon = np.mean(np.abs(actuals_rescaled[:, 30] - predictions_rescaled[:, 30]) / (actuals_rescaled[:, 30] + epsilon)) * 100

    print(f"\n--- Multi-Node Evaluation Results (42-Day Dataset) ---")
    print(f"Overall Traffic MAPE (Average of 30 Nodes): {mape_traffic:.2f}%")
    print(f"Carbon Intensity MAPE: {mape_carbon:.2f}%")
    
    # --- 7. Plotting (Objective 1 Visualization) ---
    plot_hours = min(72, len(predictions_rescaled)) 
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    ax1.plot(plot_timestamps[:plot_hours], actuals_rescaled[:plot_hours, 0], 
             label='Actual Traffic (Node 0)', color='#1f77b4', linewidth=2)
    ax1.plot(plot_timestamps[:plot_hours], predictions_rescaled[:plot_hours, 0], 
             label='AI Predicted Traffic', color='#ff7f0e', linestyle='--', linewidth=2)
    ax1.set_title("Urban Network Traffic Forecasting (30-Node Proactive Model)", fontsize=14)
    ax1.set_ylabel("Traffic Volume")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(plot_timestamps[:plot_hours], actuals_rescaled[:plot_hours, 30], 
             label='Actual Grid Carbon', color='#2ca02c', linewidth=2)
    ax2.plot(plot_timestamps[:plot_hours], predictions_rescaled[:plot_hours, 30], 
             label='AI Predicted Carbon', color='#d62728', linestyle='--', linewidth=2)
    ax2.set_title("Grid Carbon Intensity Forecasting (Temporal Optimization Source)", fontsize=14)
    ax2.set_ylabel("gCO2eq/kWh")
    ax2.set_xlabel("Date/Time")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('citywide_forecast_result_2s.png')
    print("\nGraph saved as 'citywide_forecast_results_2s.png'")
    
    # --- UPDATED FUNCTION CALL ---
    generate_presentation_traces(predictions_rescaled, actuals_rescaled, plot_timestamps)

if __name__ == "__main__":
    test_citywide_model()