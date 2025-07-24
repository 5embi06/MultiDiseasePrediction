import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os

# Load processed data
X_train = pd.read_csv('data/processed/kidney_X_train.csv')
X_test = pd.read_csv('data/processed/kidney_X_test.csv')
y_train = pd.read_csv('data/processed/kidney_y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/kidney_y_test.csv').values.ravel()

models = {
    'logreg': LogisticRegression(max_iter=1000, random_state=42),
    'rf': RandomForestClassifier(n_estimators=100, random_state=42),
    'xgb': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

os.makedirs('models', exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'\nModel: {name}')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    joblib.dump(model, f'models/kidney_{name}.joblib')

print('Kidney disease models trained and saved.') 