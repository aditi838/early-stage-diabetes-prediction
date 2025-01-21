import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
data_url = "diabetes_data_upload.csv"
data = pd.read_csv(data_url)

# Preprocessing
labelencoder = LabelEncoder()
for column in data.columns:
    data[column] = labelencoder.fit_transform(data[column])

X = data.drop(columns=['class'])
Y = data['class']

# Train a RandomForest model
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1000)
model = RandomForestClassifier(random_state=1000)
model.fit(x_train, y_train)

# Define feature names for user input
feature_mapping = {
    'Gender': "What is your gender?",
    'Age': "What is your age?",
    'Polyuria': "Do you experience frequent urination?",
    'Polydipsia': "Do you often feel excessively thirsty?",
    'sudden weight loss': "Have you had sudden weight loss recently?",
    'weakness': "Do you often feel weak or tired?",
    'Polyphagia': "Do you feel excessively hungry?",
    'Genital thrush': "Do you experience itching or infection in the genital area?",
    'visual blurring': "Do you have blurred vision?",
    'Itching': "Do you often feel itchy?",
    'Irritability': "Do you experience frequent mood swings or irritability?",
    'delayed healing': "Do your wounds take longer than usual to heal?",
    'partial paresis': "Do you feel partial numbness or weakness in muscles?",
    'muscle stiffness': "Do you experience stiffness in your muscles?",
    'Alopecia': "Do you have unusual hair loss?",
    'Obesity': "Are you considered overweight or obese?"
}

# Streamlit app
st.title("Diabetes Prediction App")
st.write("Answer the following questions to predict your risk of diabetes. All inputs are required.")

# Collect user input for all features
user_input = {}
for feature, question in feature_mapping.items():
    if feature == 'Age':
        user_input[feature] = st.slider(question, 1, 120, 25)  # Slider for age
    elif feature == 'Gender':
        user_input[feature] = 1 if st.radio(question, ["Male", "Female"]) == "Male" else 0  # Gender as Male/Female
    else:
        user_input[feature] = 1 if st.radio(question, ["Yes", "No"]) == "Yes" else 0

# Convert user input to DataFrame
input_data = pd.DataFrame([user_input])
input_data = input_data[X.columns]  # Reorder columns to match the training data

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_prob = model.predict_proba(input_data)[0][1]  # Probability of diabetes
    result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes Detected"

    # Display result
    st.subheader(f"Prediction: {result}")
    st.write(f"Probability of Diabetes: {prediction_prob:.2%}")

    # Display actionable insights
    if prediction_prob > 0.8:
        risk_level = "High Risk"
        st.warning("High risk of diabetes detected. Please consult a healthcare professional.")
    elif prediction_prob > 0.65:
        st.success("Medium risk of diabetes. Maintain a healthy lifestyle!")
    elif prediction_prob > 0.4:
        st.success("Low risk of diabetes. Maintain a healthy lifestyle!")
    else:
        st.success("Very Low risk of diabetes. Maintain a healthy lifestyle!")

    # Optional: Feature importance explanation
    st.write("### Feature Importance")
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(feature_importance)

# Display model evaluation metrics
if st.checkbox("Show Model Evaluation Metrics"):
    y_pred = model.predict(x_test)
    st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
    st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
