# ===============================================
# Task 1: Decision Tree Classifier - Bank Marketing
# Predict whether a customer will purchase a product/service
# Using the Bank Marketing Dataset (local file)
# ===============================================

# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Load dataset from local file
# Load dataset with correct separator
data = pd.read_csv("bank.csv", sep=';')  # sep=';' is important
print("Dataset loaded successfully!\n")
print(data.head())
print(data.info())

# Step 3: Explore the dataset
print("=== First 5 rows of dataset ===")
print(data.head())
print("\n=== Dataset Info ===")
print(data.info())
print("\n=== Target Distribution ===")
print(data['y'].value_counts())  # 'y' is the target variable

# Step 4: Encode categorical variables
categorical_cols = data.select_dtypes(include='object').columns
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

print("\n=== Dataset after encoding categorical features ===")
print(data.head())

# Step 5: Split data into features (X) and target (y)
X = data.drop('y', axis=1)  # All columns except 'y'
y = data['y']               # Target column

# Step 6: Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Initialize and train Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42, max_depth=5)  # Limit depth to prevent overfitting
clf.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = clf.predict(X_test)

# Step 9: Evaluate the model
print("\n=== Model Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['No','Yes'], filled=True, rounded=True)
plt.title("Decision Tree for Bank Marketing Dataset")
plt.show()
