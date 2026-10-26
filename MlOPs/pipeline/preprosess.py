import pandas as pd

# Load dataset
df = pd.read_csv('D:\MlOPs\student-por.csv', sep=',')

# Save processed dataset
df.to_csv('../processed_student_data.csv', index=False)

print("Data Processing Completed")