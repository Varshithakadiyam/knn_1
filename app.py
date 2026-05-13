import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Page title
st.title("KNN Classifier - Iris Dataset")

st.write("This app predicts the Iris flower type using K-Nearest Neighbors (KNN).")

# Load dataset
data = load_iris()

X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create KNN model
knn_classifier = KNeighborsClassifier(
    n_neighbors=3,
    metric='euclidean',
    p=2,
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    n_jobs=-1
)

# Train model
knn_classifier.fit(X_train, y_train)

# Predictions
y_pred = knn_classifier.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy
st.subheader("Model Accuracy")
st.write(f"Accuracy of KNN classifier: {accuracy:.2f}")

# User input
st.subheader("Enter Flower Measurements")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width", 2.0, 5.0, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width", 0.1, 3.0, 0.2)

# Create input dataframe
input_data = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=data.feature_names
)

# Prediction button
if st.button("Predict Flower Type"):
    prediction = knn_classifier.predict(input_data)
    predicted_class = data.target_names[prediction][0]

    st.success(f"Predicted Flower Type: {predicted_class}")