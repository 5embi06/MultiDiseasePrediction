import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load data
data = pd.read_csv('data/kidney.csv')

data = data.dropna()

# Assume 'classification' is the target column (adjust if needed)
X = data.drop('classification', axis=1)
y = data['classification'].apply(lambda x: 1 if x == 'ckd' else 0)  # Binary encoding

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
os.makedirs('data/processed', exist_ok=True)
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('data/processed/kidney_X_train.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('data/processed/kidney_X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('data/processed/kidney_y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('data/processed/kidney_y_test.csv', index=False)

# Save scaler
import joblib
joblib.dump(scaler, 'models/kidney_scaler.joblib')

print('Kidney disease data preprocessed and saved.') 