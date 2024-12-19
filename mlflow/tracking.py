import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score  # <-- Corrected import

# Load data
data = pd.read_csv(r"C:\Users\kiran\Documents\SPIT\SEM I\FDS\Experiement No. - 09\data\Cancer_Data.csv")

# Map diagnosis labels
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target
X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Log model with an example input and conda environment specification
example_input = X_train[0:1]

conda_env = {
    'channels': ['defaults'],
    'dependencies': [
        'python=3.8',
        'scikit-learn=1.0',
        'cloudpickle=2.0.0',
        {
            'pip': ['mlflow']
        }
    ]
}

with mlflow.start_run():
    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))  # <-- Corrected usage
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    mlflow.sklearn.log_model(model, "model", conda_env=conda_env, input_example=example_input)

# Save model
joblib.dump(model, "model.pkl")

print("Model has been logged and saved.")
