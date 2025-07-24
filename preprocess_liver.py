import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Load data
data = pd.read_csv('data/liver.csv')

# Encode categorical columns
if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Drop rows with missing values
data = data.dropna()

# Feature/target split
X = data.drop('Dataset', axis=1)
y = data['Dataset'].apply(lambda x: 1 if x == 1 else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (only numeric columns)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
os.makedirs('data/processed', exist_ok=True)
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('data/processed/liver_X_train.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('data/processed/liver_X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('data/processed/liver_y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('data/processed/liver_y_test.csv', index=False)

# Save scaler
joblib.dump(scaler, 'models/liver_scaler.joblib')

print('Liver disease data preprocessed and saved.') 