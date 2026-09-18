import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Starting advanced fraud detection analysis with risk parameters...")

# Ensure transaction_date is datetime type and sort by user and date
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df = df.sort_values(by=['user_id', 'transaction_date']).reset_index(drop=True)

# Calculate user-specific historical average amount, frequency, and unique locations/devices
user_historical_metrics = df.groupby('user_id').agg(
    historical_avg_amount=('amount', 'mean'),
    historical_num_transactions=('user_id', 'count'),
    first_transaction_date=('transaction_date', 'min'),
    last_transaction_date=('transaction_date', 'max'),
    historical_unique_locations=('location_id', lambda x: set(x.unique())),
    historical_unique_devices=('device_id', lambda x: set(x.unique()))
).reset_index()

# Calculate overall average weekly transactions for comparison
days_in_dataset = (df['transaction_date'].max() - df['transaction_date'].min()).days
if days_in_dataset > 0:
    user_historical_metrics['historical_avg_weekly_transactions'] = \
        (user_historical_metrics['historical_num_transactions'] / days_in_dataset) * 7
else:
    user_historical_metrics['historical_avg_weekly_transactions'] = 0

# Define 'recent' activity window (e.g., last 1 week)
latest_date = df['transaction_date'].max()
recent_transactions_window_start = latest_date - pd.Timedelta(weeks=1)

recent_df = df[df['transaction_date'] >= recent_transactions_window_start]

user_recent_activity = recent_df.groupby('user_id').agg(
    recent_num_transactions=('amount', 'count'),
    recent_max_amount=('amount', 'max'),
    recent_sum_amount=('amount', 'sum'), # Also useful for frequency assessment
    recent_unique_locations=('location_id', lambda x: set(x.unique())),
    recent_unique_devices=('device_id', lambda x: set(x.unique()))
).reset_index()

# Merge historical metrics with recent activity
user_detection_df = user_historical_metrics.merge(user_recent_activity, on='user_id', how='left').fillna(0)

# Fill NaN for set columns with empty sets (for users with no recent activity)
for col in ['historical_unique_locations', 'historical_unique_devices', 'recent_unique_locations', 'recent_unique_devices']:
    if col in user_detection_df.columns:
        user_detection_df[col] = user_detection_df[col].apply(lambda x: x if isinstance(x, set) else set())

# --- Risk Parameter Calculation ---

# Initialize risk score
user_detection_df['risk_score'] = 0.0

# Parameters for risk increment
AMOUNT_DEVIATION_FACTOR = 10  # Points for each unit of amount multiplier above 1
FREQUENCY_DEVIATION_FACTOR = 15 # Points for each unit of frequency multiplier above 1
NEW_LOCATION_RISK_POINTS = 25 # Points for a new location
BLOCK_THRESHOLD = 100
NEW_DEVICE_RISK_POINTS = BLOCK_THRESHOLD   # Points for a new device, set to BLOCK_THRESHOLD for immediate block
SHARED_DEVICE_RISK_POINTS = 40 # Points for a user transacting on a risky shared device
SHARED_DEVICE_HIGH_PAYMENT_FACTOR = 2.0 # Multiplier for overall average amount to define 'high payment'

WARNING_THRESHOLD = 50

# Calculate amount-based risk
user_detection_df['amount_multiplier'] = user_detection_df.apply(
    lambda row: row['recent_max_amount'] / row['historical_avg_amount'] if row['historical_avg_amount'] > 0 else 0,
    axis=1
)
user_detection_df['amount_risk_points'] = np.maximum(0, (user_detection_df['amount_multiplier'] - 1) * AMOUNT_DEVIATION_FACTOR)

# Calculate frequency-based risk
user_detection_df['frequency_multiplier'] = user_detection_df.apply(
    lambda row: row['recent_num_transactions'] / row['historical_avg_weekly_transactions'] if row['historical_avg_weekly_transactions'] > 0 else 0,
    axis=1
)
user_detection_df['frequency_risk_points'] = np.maximum(0, (user_detection_df['frequency_multiplier'] - 1) * FREQUENCY_DEVIATION_FACTOR)

# Calculate new location risk
user_detection_df['new_location_risk_points'] = user_detection_df.apply(
    lambda row: NEW_LOCATION_RISK_POINTS if (row['recent_unique_locations'] - row['historical_unique_locations']) else 0,
    axis=1
)

# Calculate new device risk
user_detection_df['new_device_risk_points'] = user_detection_df.apply(
    lambda row: NEW_DEVICE_RISK_POINTS if (row['recent_unique_devices'] - row['historical_unique_devices']) else 0,
    axis=1
)

# Calculate shared device risk (multiple users on same device with high payments)
# Calculate overall average amount for thresholding within this cell
average_amount_overall = df["amount"].mean()

# 1. Identify devices that are shared AND have high recent payments
recent_device_summary = recent_df.groupby('device_id').agg(
    num_users_on_device=('user_id', 'nunique'),
    max_amount_on_device=('amount', 'max')
).reset_index()

risky_shared_devices = recent_device_summary[
    (recent_device_summary['num_users_on_device'] > 1) &
    (recent_device_summary['max_amount_on_device'] > (average_amount_overall * SHARED_DEVICE_HIGH_PAYMENT_FACTOR))
]

# 2. Get the user_ids involved in transactions on these risky shared devices
users_on_risky_shared_devices = recent_df[recent_df['device_id'].isin(risky_shared_devices['device_id'])]['user_id'].unique()

# Initialize shared_device_risk_points column to 0
user_detection_df['shared_device_risk_points'] = 0

# Assign risk points to users found on risky shared devices
user_detection_df.loc[user_detection_df['user_id'].isin(users_on_risky_shared_devices), 'shared_device_risk_points'] = SHARED_DEVICE_RISK_POINTS

# Total risk score
user_detection_df['risk_score'] = (
    user_detection_df['amount_risk_points'] +
    user_detection_df['frequency_risk_points'] +
    user_detection_df['new_location_risk_points'] +
    user_detection_df['new_device_risk_points'] +
    user_detection_df['shared_device_risk_points'] # NEW ADDITION
)

print("\nUser risk parameters calculated (first 5 rows with significant risk):")
display(user_detection_df[user_detection_df['risk_score'] > 0].sort_values(by='risk_score', ascending=False).head())

# --- Apply Risk Thresholds --- 

high_risk_users = user_detection_df[user_detection_df['risk_score'] >= BLOCK_THRESHOLD]
warning_users = user_detection_df[
    (user_detection_df['risk_score'] >= WARNING_THRESHOLD) &
    (user_detection_df['risk_score'] < BLOCK_THRESHOLD)
]

if not high_risk_users.empty:
    print(f"\n--- HIGH RISK USERS - Temporary Block Recommended ({len(high_risk_users)} found) ---")
    print("These users have a risk score above the block threshold:")
    display(high_risk_users[[
        'user_id', 'risk_score', 'historical_avg_amount', 'recent_max_amount',
        'historical_avg_weekly_transactions', 'recent_num_transactions',
        'new_location_risk_points', 'new_device_risk_points',
        'shared_device_risk_points' # NEW ADDITION
    ]].head())
    print(f"Action: Block these users until risk reduces (score >={BLOCK_THRESHOLD}).")
    for index, row in high_risk_users.head().iterrows():
        if row['new_device_risk_points'] >= BLOCK_THRESHOLD:
            print(f"  Note for User {int(row['user_id'])}: Block triggered due to detection of a new device.")

if not warning_users.empty:
    print(f"\n--- WARNING USERS - Monitoring Recommended ({len(warning_users)} found) ---")
    print("These users have a risk score above the warning threshold but below the block threshold:")
    display(warning_users[[
        'user_id', 'risk_score', 'historical_avg_amount', 'recent_max_amount',
        'historical_avg_weekly_transactions', 'recent_num_transactions',
        'new_location_risk_points', 'new_device_risk_points',
        'shared_device_risk_points' # NEW ADDITION
    ]].head())
    print(f"Action: Monitor these users closely and investigate recent activity (score >={WARNING_THRESHOLD} and < {BLOCK_THRESHOLD}).")

if high_risk_users.empty and warning_users.empty:
    print("\nNo users currently flagged for warning or blocking based on the defined risk thresholds.")

print("\nAdvanced fraud detection analysis complete. Note: In a live system, risk scores would typically decay over time if no further anomalous activity is detected.")
