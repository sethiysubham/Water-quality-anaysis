import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title and description of the app
st.title('Water Quality Analysis')
st.write('This app analyzes water quality using machine learning.')

# Load the dataset
@st.cache_data
def load_data():
    # Replace 'water_quality.csv' with the path to your dataset
    data = pd.read_csv('water_potbility.csv')
    return data

data = load_data()

# Show the raw dataset
st.subheader('Raw Data')
st.write(data.head())

# Features and Labels
X = data.drop('Quality', axis=1)  # Assuming 'Quality' is the target variable
y = data['Quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")

# Input new water parameters for prediction
st.subheader('Predict Water Quality')
st.write('Enter water quality parameters to predict the quality.')

# Dynamically generate input fields based on dataset features
input_data = {}
for feature in X.columns:
    input_data[feature] = st.number_input(f'Enter {feature}', value=float(data[feature].mean()))

# Convert input data into a dataframe
input_df = pd.DataFrame([input_data])

# Predict quality for the new input
if st.button('Predict'):
    prediction = clf.predict(input_df)
    st.write(f'The predicted water quality is: {prediction[0]}')

# Optionally, show feature importances
st.subheader('Feature Importance')
importances = clf.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
st.bar_chart(importance_df.set_index('Feature'))