import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Initial Data Generation ---
# Simulate an initial dataset for a fintech project with multiple users and transactions
initial_num_users = 1000  # Number of unique users in the initial set
initial_num_transactions = 50000  # Total number of transactions in the initial set

np.random.seed(42) # for reproducibility

# Generate user IDs for each transaction
initial_user_ids = np.random.randint(1, initial_num_users + 1, size=initial_num_transactions)

# Generate transaction amounts (e.g., mean of 500, std dev of 200), ensuring amounts are positive
initial_amounts = np.maximum(0, np.random.normal(loc=500, scale=200, size=initial_num_transactions))

# Generate transaction dates over a period (e.g., last year)
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
time_delta_seconds = (end_date - start_date).total_seconds()
initial_random_dates = [start_date + timedelta(seconds=np.random.randint(0, time_delta_seconds)) for _ in range(initial_num_transactions)]

# Generate behavioral context: location, device_id, recipient_id
num_locations = 50 # Number of distinct locations
num_devices = 100  # Number of distinct devices
num_recipients = 2000 # Number of distinct recipients

# Simulate location_ids (some locations might be more frequent)
initial_location_ids = np.random.choice(np.arange(1, num_locations + 1), size=initial_num_transactions, p=np.random.dirichlet(np.ones(num_locations)*0.5))

# Simulate device_ids (users might use a few common devices, or new ones)
initial_device_ids = np.random.choice(np.arange(1, num_devices + 1), size=initial_num_transactions, p=np.random.dirichlet(np.ones(num_devices)*0.3))

# Simulate recipient_ids (can be different from user_ids)
initial_recipient_ids = np.random.randint(1, num_recipients + 1, size=initial_num_transactions)

initial_data = {
    "user_id": initial_user_ids,
    "amount": initial_amounts,
    "transaction_date": initial_random_dates,
    "location_id": initial_location_ids,
    "device_id": initial_device_ids,
    "recipient_id": initial_recipient_ids
}

df = pd.DataFrame(initial_data)

# Sort by user_id and transaction_date for better readability/analysis
df = df.sort_values(by=['user_id', 'transaction_date']).reset_index(drop=True)

print("--- Initial Dataset Summary ---")
print(f"Initial number of unique users: {df['user_id'].nunique()}")
print(f"Initial total number of transactions: {len(df)}")
print(f"Initial number of unique locations: {df['location_id'].nunique()}")
print(f"Initial number of unique devices: {df['device_id'].nunique()}")
print(f"Initial number of unique recipients: {df['recipient_id'].nunique()}")


# --- Adding More Users and Transactions ---

# Define parameters for adding new data
num_additional_users = 500 # Number of new users to add
num_transactions_per_additional_user = 20 # Number of transactions for each new user

# Determine the next available user ID
max_existing_user_id = df['user_id'].max()

# Generate new user IDs, ensuring they are distinct from existing ones
new_user_ids_list = np.arange(max_existing_user_id + 1, max_existing_user_id + 1 + num_additional_users)

# Create transactions for new users
additional_data_list = []
for user_id in new_user_ids_list:
    # Generate amounts for each new user's transactions
    new_amounts = np.maximum(0, np.random.normal(loc=550, scale=180, size=num_transactions_per_additional_user))

    # Generate dates for each new user's transactions (can be same or different range)
    new_random_dates = [start_date + timedelta(seconds=np.random.randint(0, time_delta_seconds)) for _ in range(num_transactions_per_additional_user)]

    # Generate behavioral context for new users
    new_location_ids = np.random.choice(np.arange(1, num_locations + 1), size=num_transactions_per_additional_user, p=np.random.dirichlet(np.ones(num_locations)*0.5))
    new_device_ids = np.random.choice(np.arange(1, num_devices + 1), size=num_transactions_per_additional_user, p=np.random.dirichlet(np.ones(num_devices)*0.3))
    new_recipient_ids = np.random.randint(1, num_recipients + 1, size=num_transactions_per_additional_user)
    
    for i in range(num_transactions_per_additional_user):
        additional_data_list.append({
            "user_id": user_id,
            "amount": new_amounts[i],
            "transaction_date": new_random_dates[i],
            "location_id": new_location_ids[i],
            "device_id": new_device_ids[i],
            "recipient_id": new_recipient_ids[i]
        })

# Convert to DataFrame and concatenate with the existing one
new_transactions_df = pd.DataFrame(additional_data_list)
df = pd.concat([df, new_transactions_df], ignore_index=True)

# Re-sort the combined DataFrame
df = df.sort_values(by=['user_id', 'transaction_date']).reset_index(drop=True)

print("\n--- Combined Dataset Summary ---")
print(f"Total number of unique users in dataset: {df['user_id'].nunique()}")
print(f"Total number of transactions in dataset: {len(df)}")
print(f"Total number of unique locations: {df['location_id'].nunique()}")
print(f"Total number of unique devices: {df['device_id'].nunique()}")
print(f"Total number of unique recipients: {df['recipient_id'].nunique()}")

# Update overall average for the combined dataset
average_amount_overall = df["amount"].mean()
print(f"Overall average transaction amount: {average_amount_overall:.2f}")

# --- Risk Assessment Logic (for a single hypothetical transaction) ---
# This is a separate illustrative example for checking a single transaction's risk
new_transaction_to_check = 5000 # A specific transaction amount to evaluate for risk
print(f"\nSpecific new transaction amount to evaluate for risk: {new_transaction_to_check}")

if new_transaction_to_check > average_amount_overall * 3:
    print("Risk for hypothetical new transaction: HIGH")
    print("Reason: Transaction amount is unusually high compared to the overall average.")
else:
    print("Risk for hypothetical new transaction: LOW")

print("\nFirst 5 rows of the expanded dataset:")
entry = int(input("enter the number of rows to display"))
