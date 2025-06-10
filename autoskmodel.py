import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import autosklearn.classification

# Load dataset
df = pd.read_csv('v3_cleaned_ktmb_data.csv')

# Feature selection
features = ['hour', 'day_of_week', 'is_holiday', 'hour_bin', 'is_weekend',
            'hour_is_weekend', 'hour_is_holiday', 'station_pair_encoded',
            'origin_encoded', 'destination_encoded']

df_sample = df.sample(n=100000, random_state=42)  # Sample 10% of the data for faster processing
X = df_sample[features]
y = df_sample['peak_label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create AutoSklearn model 
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=3600,
    per_run_time_limit=360,
    memory_limit=8192,
    n_jobs=1, 
    ensemble_kwargs={'ensemble_size': 50},
    include={
        'classifier': [
            'random_forest',
            'decision_tree',
            'k_nearest_neighbors',
            'liblinear_svc',
        ]
    },
    seed=42
)

# Fit the model
automl.fit(X_train_scaled, y_train)

# Predict
y_pred = automl.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

# Print model details
print("\nLeaderboard of the top models AutoSklearn chose:")
print(automl.show_models())
print("\n=== AutoSklearn Leaderboard ===")
print(automl.leaderboard(detailed=True))

# Print general statistics
print("\nAutoSklearn statistics:")
print(automl.sprint_statistics())

# Print best accuracy
print("\nBest Model Accuracy: {:.2f}".format(acc))