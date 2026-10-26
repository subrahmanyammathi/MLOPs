import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load processed dataset
df = pd.read_csv(r"D:\MlOPs\processed_student_data.csv")

# Separate features and target
X = df.drop(columns=["G3"])
y = df["G3"]

# Encode categorical variables
X = pd.get_dummies(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Create the model folder beside train.py and save the model
model_dir = Path(__file__).resolve().parent / "model"
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "deployment.pkl"
joblib.dump(model, model_path)

print(f"Model training completed. Saved to: {model_path}")