# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('heart.csv')

# Separate features (X) and label (y)
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Split data into train (60%), validation (20%), and test (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y, test_size=0.4, random_state=42)

# Split X_temp and y_temp into validation (20%) and test (20%)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)
y_val, y_test = train_test_split(y_temp, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Check the sizes of the splits
print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
y_val_pred_logreg = logreg.predict(X_val_scaled)
logreg_val_accuracy = accuracy_score(y_val, y_val_pred_logreg)
print(f"Logistic Regression Validation Accuracy: {logreg_val_accuracy}")

# Print detailed classification report (Precision, Recall, F1-score) for Logistic Regression on validation set
print("\nLogistic Regression Classification Report (Validation Set):")
print(classification_report(y_val, y_val_pred_logreg))

# Confusion Matrix for Logistic Regression (Validation Set)
cm_logreg_val = confusion_matrix(y_val, y_val_pred_logreg)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_logreg_val, annot=True, fmt="d", cmap='Blues')
plt.title("Logistic Regression Confusion Matrix (Validation Set)")
plt.xlabel('Predicted labels')
plt.ylabel('Actual labels')
plt.show()

# Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train_scaled, y_train)
y_val_pred_tree = tree.predict(X_val_scaled)
tree_val_accuracy = accuracy_score(y_val, y_val_pred_tree)
print(f"Decision Tree Validation Accuracy: {tree_val_accuracy}")

# Print detailed classification report (Precision, Recall, F1-score) for Decision Tree on validation set
print("\nDecision Tree Classification Report (Validation Set):")
print(classification_report(y_val, y_val_pred_tree))

# Confusion Matrix for Decision Tree (Validation Set)
cm_tree_val = confusion_matrix(y_val, y_val_pred_tree)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_tree_val, annot=True, fmt="d", cmap='Blues')
plt.title("Decision Tree Confusion Matrix (Validation Set)")
plt.xlabel('Predicted labels')
plt.ylabel('Actual labels')
plt.show()

# Based on validation accuracy, we choose Logistic Regression (assuming it performs better)
# Testing the final model (Logistic Regression) on the test set
y_test_pred_logreg = logreg.predict(X_test_scaled)
logreg_test_accuracy = accuracy_score(y_test, y_test_pred_logreg)
print(f"Logistic Regression Test Accuracy: {logreg_test_accuracy}")

# Print detailed classification report (Precision, Recall, F1-score) for Logistic Regression on test set
print("\nLogistic Regression Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred_logreg))

# Confusion Matrix for Logistic Regression (Test Set)
cm_logreg_test = confusion_matrix(y_test, y_test_pred_logreg)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_logreg_test, annot=True, fmt="d", cmap='Blues')
plt.title("Logistic Regression Confusion Matrix (Test Set)")
plt.xlabel('Predicted labels')
plt.ylabel('Actual labels')
plt.show()