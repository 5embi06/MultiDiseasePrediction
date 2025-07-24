import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Load data
data = pd.read_csv('data/heart.csv')

# Ensure all columns are numeric (convert if needed)
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
data = data.dropna()

# Feature/target split
X = data.drop('target', axis=1)
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
os.makedirs('data/processed', exist_ok=True)
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('data/processed/heart_X_train.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('data/processed/heart_X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('data/processed/heart_y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('data/processed/heart_y_test.csv', index=False)

# Save scaler
joblib.dump(scaler, 'models/heart_scaler.joblib')

print('Heart disease data preprocessed and saved.') 