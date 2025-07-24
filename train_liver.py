import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os

# Load processed data
X_train = pd.read_csv('data/processed/liver_X_train.csv')
X_test = pd.read_csv('data/processed/liver_X_test.csv')
y_train = pd.read_csv('data/processed/liver_y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/liver_y_test.csv').values.ravel()

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
    # Use macro average for multiclass/binary
    print('Precision:', precision_score(y_test, y_pred, average='macro'))
    print('Recall:', recall_score(y_test, y_pred, average='macro'))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    joblib.dump(model, f'models/liver_{name}.joblib')

print('Liver disease models trained and saved.') 