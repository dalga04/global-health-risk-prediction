import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ======================
# Load Data
# ======================
data1 = pd.read_csv('Global Health Statistics.csv')
data2 = pd.read_csv('health_lifestyle_dataset.csv')

# ======================
# Encoding
# ======================
le = LabelEncoder()
data2_encoded = data2.copy()

categorical_cols = ['gender', 'smoker', 'alcohol', 'family_history']

for col in categorical_cols:
    data2_encoded[col] = le.fit_transform(data2_encoded[col].astype(str))

# ======================
# Features & Target
# ======================
X = data2_encoded.drop(columns=['id', 'disease_risk'])
y = data2_encoded['disease_risk']

# ======================
# Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Model
# ======================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ======================
# Prediction
# ======================
y_pred = model.predict(X_test)

# ======================
# Evaluation
# ======================
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

# ======================
# Confusion Matrix
# ======================
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ======================
# Feature Importance
# ======================
plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', color='teal')
plt.title('Feature Importances')
plt.show()
