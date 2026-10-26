import joblib
import pandas as pd
from pathlib import Path

script_dir = Path(__file__).resolve().parent

# Load dataset and model
df = pd.read_csv(script_dir.parent / "processed_student_data.csv")
model = joblib.load(script_dir / "model" / "deployment.pkl")

# Prepare features
X = df.drop(columns=["G3"])
X = pd.get_dummies(X)
X = X.reindex(columns=model.feature_names_in_, fill_value=0)

# Predict the first row
prediction = model.predict(X.iloc[[0]])

print("Prediction:", round(prediction[0], 2))
print("Actual:", df.iloc[0]["G3"])