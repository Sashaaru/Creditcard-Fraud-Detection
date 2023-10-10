# Creditcard-Fraud-Detection
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.
**Importing Libraries**
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

**#Loading the Dataset**
data = pd.read_csv("C:\\Users\\rameshr\\PycharmProjects\\Credit card Fraud\\creditcard.csv")
This line reads the credit card transaction data from a CSV file and stores it in a Pandas DataFrame called data.

**Data Preprocessing**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

 **Handling Imbalanced Data**
X = data.drop('Class', axis=1)
y = data['Class']
oversampler = SMOTE(sampling_strategy=0.5)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

X contains the features, excluding the 'Class' column.
y contains the target variable, 'Class' (fraud or not).
SMOTE is used to oversample the minority class (fraud) while preserving the majority class.

**Splitting the Data**
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

The dataset is split into training and testing sets using train_test_split.
X_train and y_train contain the training data.
X_test and y_test contain the testing data.

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Feature scaling is performed again, but this time only on the training and testing sets separately. It ensures that scaling is based only on the training data to prevent data leakage.

**PCA Dimensionality Reduction**

n_components = 10
pca = PCA(n_components=n_components)

PCA is applied to reduce the dimensionality of the dataset to 10 principal components.
X_pca contains the transformed data after PCA.

**Logistic Regression Model**

# Create a Logistic Regression model
logistic_model = LogisticRegression()

# Train the Logistic Regression model on the PCA-transformed training data
logistic_model.fit(X_train_pca, y_train)

# Make predictions on the PCA-transformed testing data
y_pred_logistic = logistic_model.predict(X_test_pca)


**Explanation:**

Logistic Regression is a linear classification model that estimates the probability that a given input belongs to a particular class.
In this code, a Logistic Regression model is created and initialized.
The model is trained using the PCA-transformed training data (X_train_pca and y_train) using the fit method.
After training, the model is used to make predictions on the PCA-transformed testing data (X_test_pca), and the predictions are stored in y_pred_logistic.
**

**Support Vector Classifier (SVC) Model**
# Create an SVM (Support Vector Classifier) model
svc_model = SVC()

# Train the SVC model on the PCA-transformed training data
svc_model.fit(X_train_pca, y_train)

# Make predictions on the PCA-transformed testing data
y_pred_svc = svc_model.predict(X_test_pca)

**Explanation:

Support Vector Classifier (SVC) is a powerful classification algorithm that finds a hyperplane that best separates data into different classes while maximizing the margin between the classes.
In this code, an SVC model is created and initialized.
The model is trained using the PCA-transformed training data (X_train_pca and y_train) using the fit method.
Predictions are made on the PCA-transformed testing data (X_test_pca) using the predict method, and the predictions are stored in y_pred_svc.**

**Decision Tree Classifier Model**

# Create a Decision Tree Classifier model
tree_model = DecisionTreeClassifier()

# Train the Decision Tree Classifier model on the PCA-transformed training data
tree_model.fit(X_train_pca, y_train)

# Make predictions on the PCA-transformed testing data
y_pred_tree = tree_model.predict(X_test_pca)

Explanation:

Decision Tree Classifier is a tree-based classification algorithm that partitions the data into subsets based on the values of features and makes decisions at each node.
In this code, a Decision Tree Classifier model is created and initialized.
The model is trained using the PCA-transformed training data (X_train_pca and y_train) using the fit method.
Predictions are made on the PCA-transformed testing data (X_test_pca) using the predict method, and the predictions are stored in y_pred_tree.
