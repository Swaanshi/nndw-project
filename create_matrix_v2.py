import pandas as pd
import numpy as np
import os
import glob

def build_42day_matrix_fast():
    # 1. Load Carbon Data
    carbon_file = 'snapshots_2026-02-10_IT-NO-2024-hourly.csv'
    carbon_df = pd.read_csv(carbon_file)
    carbon_df['datetime'] = pd.to_datetime(carbon_df['Datetime (UTC)']).astype('datetime64[ns]')
    carbon_df['datetime_shifted'] = carbon_df['datetime'] - pd.DateOffset(years=11)
    carbon_df = carbon_df.sort_values('datetime_shifted').drop_duplicates('datetime_shifted')
    # 2. Get Files
    data_folder = 'dataverse_files'
    all_files = glob.glob(os.path.join(data_folder, "*.txt"))
    if not all_files:
        print("No files found!")
        return

    print(f"Processing {len(all_files)} files. Please wait...")
    
    # Harvard columns: square_id, time, sms_in, sms_out, call_in, call_out, internet
    # We only want: square_id (0), time (1), internet (6)
    df_list = []
    for i, file in enumerate(all_files):
        print(f"[{i+1}/{len(all_files)}] Reading: {os.path.basename(file)}")
        try:
            # Using usecols makes it MUCH faster and uses less RAM
            temp_df = pd.read_csv(
                file, 
                sep='\t', 
                header=None, 
                names=['square_id', 'time', 'internet'],
                usecols=[0, 1, 6],
                engine='c'
            )
            
            # Fast datetime conversion
            temp_df['datetime'] = pd.to_datetime(temp_df['time'], unit='ms').dt.floor('h').astype('datetime64[ns]')
            df_list.append(temp_df[['datetime', 'square_id', 'internet']])
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")
        
    print("Combining days...")
    traffic_df = pd.concat(df_list, ignore_index=True)
    
    # 3. Find Top 30
    top_30 = traffic_df.groupby('square_id')['internet'].sum().nlargest(30).index
    traffic_filtered = traffic_df[traffic_df['square_id'].isin(top_30)]

    # 4. Pivot
    print("Pivoting...")
    traffic_matrix = traffic_filtered.pivot_table(
        index='datetime', 
        columns='square_id', 
        values='internet', 
        aggfunc='sum'
    ).fillna(0)
    
    traffic_matrix.columns = [f"Node_{int(col)}_Traffic" for col in traffic_matrix.columns]
    
    # 5. Merge & Split
    print("Merging with Carbon data...")
    traffic_matrix_reset = traffic_matrix.reset_index()
    traffic_matrix_reset['datetime'] = traffic_matrix_reset['datetime'].astype('datetime64[ns]')
    traffic_matrix_reset = traffic_matrix_reset.sort_values('datetime') # Sort left as well

    merged_matrix = pd.merge_asof(
        traffic_matrix_reset,
        carbon_df[['datetime_shifted', 'Carbon intensity gCO₂eq/kWh (Life cycle)']],
        left_on='datetime',
        right_on='datetime_shifted',
        direction='nearest'
    ).dropna()
    
    merged_matrix.rename(columns={'Carbon intensity gCO₂eq/kWh (Life cycle)': 'Carbon_Intensity'}, inplace=True)
    
    # Time Features
    merged_matrix['hour_sin'] = np.sin(2 * np.pi * merged_matrix['datetime'].dt.hour / 24)
    merged_matrix['hour_cos'] = np.cos(2 * np.pi * merged_matrix['datetime'].dt.hour / 24)
    merged_matrix['day_of_week'] = merged_matrix['datetime'].dt.dayofweek

    # Split
    train_idx = int(len(merged_matrix) * 0.70)
    val_idx = int(len(merged_matrix) * 0.85)

    merged_matrix.iloc[:train_idx].to_csv('matrix_train.csv', index=False)
    merged_matrix.iloc[train_idx:val_idx].to_csv('matrix_val.csv', index=False)
    merged_matrix.iloc[val_idx:].to_csv('matrix_test.csv', index=False)

    print(f"Done! Final shape: {merged_matrix.shape}")

if __name__ == "__main__":
    build_42day_matrix_fast()