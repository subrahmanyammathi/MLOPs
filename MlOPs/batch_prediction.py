import joblib
import pandas as pd

df = pd.read_csv(r"D:\MlOPs\processed_student_data.csv")
print("Dataset loaded successfully")

X = df.drop(columns=["G3"])
X = pd.get_dummies(X)

model = joblib.load(
    r"C:\Users\ABHISHEK KUMAR\Downloads\tuned_random_forest.pkl"
)
print("Deployment model loaded successfully")

expected_columns = model.feature_names_in_
X = X.reindex(columns=expected_columns, fill_value=0)

df["Predicted_G3"] = model.predict(X).round(2)
df.to_csv(r"D:\MlOPs\batch_predictions.csv", index=False)

print(df[["G3", "Predicted_G3"]].head())
print("Total students:", len(df))
print(r"Predictions saved to D:\MlOPs\batch_predictions.csv")