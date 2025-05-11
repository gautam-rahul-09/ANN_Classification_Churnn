import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('my_model.keras')

# Load the encoder and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', min_value=18, max_value=100, value=30)
balance = st.number_input('Balance', min_value=0.0, value=1000.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
tenure = st.number_input('Tenure', min_value=0, value=5)
num_products = st.number_input('Number of Products', min_value=1, max_value=4, value=2)
has_cr_card = int(st.checkbox('Has Credit Card'))
is_active_member = int(st.checkbox('Is Active Member'))

# Prepare the input data
input_data = pd.DataFrame({
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Balance': [balance],
    'CreditScore': [credit_score],
    'EstimatedSalary': [estimated_salary],
    'Tenure': [tenure],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

# One-hot encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(onehot_encoder_geo.feature_names_in_)
)

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

# Output result
if prediction_prob > 0.5:
    st.write('🔴 The customer is **likely to churn**.')
else:
    st.write('🟢 The customer is **not likely to churn**.')
