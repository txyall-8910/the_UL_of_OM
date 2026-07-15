"""
Description: Commuting Flow Extraction from Sparse Trajectory Data
This script processes individual-level spatiotemporal records (e.g., Call Detail Records)
to infer primary anchor locations (Home and Work) and aggregate them into
Origin-Destination (OD) commuting flows.
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# 1. PARAMETER INITIALIZATION
# =============================================================================
INPUT_FILE = 'data/SET2_P01.csv'
OUTPUT_DIR = 'results'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'commuting_flows.csv')

# --- Thresholds for Anchor Location Inference ---
# Adjusted thresholds designed to accommodate data sparsity (e.g., event-driven CDR data).
MAX_DUR = 6.0          # Maximum allowable duration (hours) for a single stay to prevent overestimation from missing data.
MIN_HOME_HOURS = 15.0  # Minimum cumulative hours required during nighttime hours over the observation period.
MIN_WORK_HOURS = 10.0  # Minimum cumulative hours required during daytime hours over the observation period.
MIN_HOME_DAYS = 4      # Minimum number of distinct days observed at the home location.
MIN_WORK_DAYS = 3      # Minimum number of distinct days observed at the work location.
MIN_DOMINANCE = 0.6    # Minimum spatial dominance ratio (duration at primary location / total duration in the time window).

# =============================================================================
# 2. DATA INGESTION AND PREPROCESSING
# =============================================================================
print("1. Loading and preprocessing data...")
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Error: Input file not found at {INPUT_FILE}")

# Load dataset and parse temporal information
df = pd.read_csv(INPUT_FILE)
df['time'] = pd.to_datetime(df['time'])

# Sort chronologically per user to reconstruct individual trajectories
df = df.sort_values(['User ID', 'time'])

# Isolate weekday mobility by filtering out weekends (Saturday=5, Sunday=6)
# This ensures the inference captures routine commuting behaviors.
df['weekday'] = df['time'].dt.weekday
df = df[~df['weekday'].isin([5, 6])].copy()

# Extract temporal features for subsequent time-window filtering
df['hour'] = df['time'].dt.hour
df['date'] = df['time'].dt.date

# =============================================================================
# 3. STAY DURATION ESTIMATION
# =============================================================================
print("2. Estimating stay durations...")
# Calculate the time interval to the next recorded event for each user
df['next_time'] = df.groupby('User ID')['time'].shift(-1)
df['duration_hours'] = (df['next_time'] - df['time']).dt.total_seconds() / 3600

# Handle the last record of each user and cap extreme durations caused by data gaps
df['duration_hours'] = df['duration_hours'].fillna(0.5)
df['duration_hours'] = df['duration_hours'].clip(upper=MAX_DUR)

# =============================================================================
# 4. TEMPORAL WINDOW DEFINITION
# =============================================================================
# Define diurnal periods corresponding to typical resting and working hours
mask_home = (df['hour'] >= 21) | (df['hour'] < 7)  # Nighttime: 21:00 to 07:00
mask_work = (df['hour'] >= 9) & (df['hour'] < 17)  # Daytime: 09:00 to 17:00

# =============================================================================
# 5. HOME LOCATION INFERENCE
# =============================================================================
print("3. Inferring primary Home locations...")
# Aggregate temporal presence per user per location during nighttime
home_stats = df[mask_home].groupby(['User ID', 'base station ID']).agg({
    'duration_hours': 'sum',
    'date': 'nunique'
}).reset_index()
home_stats.rename(columns={'date': 'days_count'}, inplace=True)

# Calculate total nighttime duration per user to compute spatial dominance
user_total_home = home_stats.groupby('User ID')['duration_hours'].sum().rename('total_home_hours')
home_stats = home_stats.merge(user_total_home, on='User ID')

# Apply heuristic filters to ensure statistical robustness
home_stats = home_stats[home_stats['total_home_hours'] >= MIN_HOME_HOURS]
home_stats = home_stats[home_stats['days_count'] >= MIN_HOME_DAYS]
home_stats = home_stats[home_stats['duration_hours'] >= (home_stats['total_home_hours'] * MIN_DOMINANCE)]

# Extract the most dominant location as the inferred Home
home_stats = home_stats.sort_values('duration_hours', ascending=False).drop_duplicates('User ID')
home_stats = home_stats[['User ID', 'base station ID']].rename(columns={'base station ID': 'home'})
print(f"   -> Successfully inferred Home locations for {len(home_stats)} users.")

# =============================================================================
# 6. WORK LOCATION INFERENCE
# =============================================================================
print("4. Inferring primary Work locations...")
# Merge inferred home locations to exclude them from work location candidates
work_df = df[mask_work].merge(home_stats[['User ID', 'home']], on='User ID', how='left')
work_df = work_df[work_df['base station ID'] != work_df['home']]

# Aggregate temporal presence per user per location during daytime
work_stats = work_df.groupby(['User ID', 'base station ID']).agg({
    'duration_hours': 'sum',
    'date': 'nunique'
}).reset_index()
work_stats.rename(columns={'date': 'days_count'}, inplace=True)

# Calculate total daytime duration per user to compute spatial dominance
user_total_work = work_stats.groupby('User ID')['duration_hours'].sum().rename('total_work_hours')
work_stats = work_stats.merge(user_total_work, on='User ID')

# Apply heuristic filters
work_stats = work_stats[work_stats['total_work_hours'] >= MIN_WORK_HOURS]
work_stats = work_stats[work_stats['days_count'] >= MIN_WORK_DAYS]
work_stats = work_stats[work_stats['duration_hours'] >= (work_stats['total_work_hours'] * MIN_DOMINANCE)]

# Extract the most dominant location as the inferred Work
work_stats = work_stats.sort_values('duration_hours', ascending=False).drop_duplicates('User ID')
work_stats = work_stats[['User ID', 'base station ID']].rename(columns={'base station ID': 'work'})
print(f"   -> Successfully inferred Work locations for {len(work_stats)} users.")

# =============================================================================
# 7. ORIGIN-DESTINATION (OD) FLOW AGGREGATION
# =============================================================================
print("5. Generating Origin-Destination (OD) matrices...")
# Intersect users with both valid Home and Work locations
final_df = home_stats.merge(work_stats, on='User ID', how='inner')

# Exclude telecommuters or individuals living and working in the same spatial unit
final_df = final_df[final_df['home'] != final_df['work']]

# Aggregate individual trajectories into macroscopic commuting flows
commuting_flows = final_df.groupby(['home', 'work']).size().reset_index(name='commuters')

print(f"   -> Total valid OD pairs extracted: {len(commuting_flows)}")

# =============================================================================
# 8. DATA EXPORT
# =============================================================================
if len(commuting_flows) > 0:
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"   -> Created output directory: {OUTPUT_DIR}")

    # Export the aggregated flow network to a CSV file
    commuting_flows.to_csv(OUTPUT_FILE, index=False)
    print(f"SUCCESS: Commuting flow matrix successfully saved to {OUTPUT_FILE}")
else:
    print("WARNING: No valid commuting flows extracted based on the current threshold configurations. Output file was NOT generated.")