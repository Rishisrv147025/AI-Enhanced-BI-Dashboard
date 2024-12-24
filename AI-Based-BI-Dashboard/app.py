import streamlit as st
import pandas as pd
from data_processing import load_data, handle_missing_data, normalize_numerical_columns
from visualizations import univariate_analysis, bivariate_analysis, multivariate_analysis
from model_building import train_model
from query_processing import execute_query
from utils import get_operations_list

# Set up Streamlit components
st.title('Machine Learning with Data Analysis and Forecasting')

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx", "json"])

# Data upload and processing
if uploaded_file:
    data = load_data(uploaded_file)
    
    if data is not None:
        st.write("Data Preview:", data.head())

        # Handle missing values
        st.subheader('Handle Missing Data')
        data = handle_missing_data(data)
        st.write("Data after handling missing values:", data.head())

        # Normalize the data
        st.subheader('Normalize Data')
        data = normalize_numerical_columns(data)
        st.write("Normalized Data:", data.head())

        # Data analysis options
        analysis_type = st.selectbox("Select Analysis Type", ["Univariate", "Bivariate", "Multivariate"])

        if analysis_type == "Univariate":
            univariate_analysis(data)
        elif analysis_type == "Bivariate":
            bivariate_analysis(data)
        elif analysis_type == "Multivariate":
            multivariate_analysis(data)

        # Model Training
        st.subheader('Train Model')
        if st.button("Train Model"):
            model_results = train_model(uploaded_file)
            st.write("Model Training Complete:", model_results)

        # Query Execution
        st.subheader('Execute Query on Data')
        user_query = st.text_input("Enter Query:")
        if user_query:
            result, action = execute_query(user_query, data)
            st.write(f"Action Performed: {action}")
            st.write("Query Result:", result)
