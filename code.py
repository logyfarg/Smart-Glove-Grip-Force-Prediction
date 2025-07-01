# Smart Glove Grip Force Prediction
# Generating synthetic data + training Random Forest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt 

# 1. Generate synthetic dataset
np.random.seed(42)

n_samples = 1000
sensor_1 = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
sensor_2 = np.random.normal(loc=5.0, scale=2.0, size=n_samples)
sensor_3 = np.random.normal(loc=10.0, scale=3.0, size=n_samples)
sensor_4 = np.random.normal(loc=15.0, scale=4.0, size=n_samples)
sensor_5 = np.random.normal(loc=20.0, scale=5.0, size=n_samples)

# Synthetic target with some noise
grip_force = (
    2.5 * sensor_1
    + 1.2 * sensor_2
    + 0.8 * sensor_3
    + 1.5 * sensor_4
    + 0.5 * sensor_5
    + np.random.normal(0, 2, n_samples)
)

# Create DataFrame
data = pd.DataFrame({
    "Sensor1": sensor_1,
    "Sensor2": sensor_2,
    "Sensor3": sensor_3,
    "Sensor4": sensor_4,
    "Sensor5": sensor_5,
    "GripForce": grip_force
})

# 2. Split data into features and target
X = data.drop("GripForce", axis=1)
y = data["GripForce"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Make predictions
predictions = model.predict(X_test_scaled)

# 7. Evaluate
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"\n✅ R² Score: {r2:.4f}")
print(f"✅ Mean Squared Error: {mse:.2f}")

# 8. Plot actual vs. predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.7, color='b', label="Predictions")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    lw=2,
    label="Perfect Prediction"
)
plt.xlabel("Actual Grip Force")
plt.ylabel("Predicted Grip Force")
plt.title("Smart Glove - Actual vs. Predicted Grip Force")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grip_force_prediction_plot.png")
plt.show()
