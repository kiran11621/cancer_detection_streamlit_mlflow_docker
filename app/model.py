import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import mlflow
import joblib

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

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# Log metrics with MLflow
mlflow.log_param("model", "Logistic Regression")
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("roc_auc", roc_auc)

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(scaler, 'scaler.pkl')

# Print evaluation metrics
print("Accuracy:", accuracy)
print("ROC AUC:", roc_auc)
print(classification_report(y_test, y_pred))
