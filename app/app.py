import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    # Try multiple possible paths for the model and scaler
    possible_paths = [
        ('../models', '../models'),  # If app.py is in app/ folder (Streamlit Cloud)
        ('models', 'models'),  # If run from root
        ('.', '.'),  # If files are in same directory
        ('./models', './models'),  # Explicit relative path
    ]
    
    model = None
    scaler = None
    
    # Debug: Show current working directory
    st.info(f"🔍 Searching for model files. Current directory: {os.getcwd()}")
    st.info(f"📂 Directory contents: {os.listdir('.')}")
    
    # Check if models folder exists
    if os.path.exists('../models'):
        st.info(f"✅ Found '../models' directory. Contents: {os.listdir('../models')}")
    if os.path.exists('models'):
        st.info(f"✅ Found 'models' directory. Contents: {os.listdir('models')}")
    
    for model_dir, scaler_dir in possible_paths:
        model_path = os.path.join(model_dir, 'random_forest_model.pkl')
        scaler_path = os.path.join(scaler_dir, 'scaler.pkl')
        
        st.info(f"🔍 Trying path: {model_path}")
        
        if os.path.exists(model_path):
            try:
                # Try loading with joblib first (recommended for sklearn)
                model = joblib.load(model_path)
                st.success(f"✅ Model loaded successfully from: {model_path}")
                
                # Try to load scaler
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    st.success(f"✅ Scaler loaded successfully from: {scaler_path}")
                else:
                    st.warning(f"⚠️ Scaler not found at: {scaler_path}")
                    st.warning("Predictions may be inaccurate without the scaler!")
                
                return model, scaler
            except Exception as e:
                st.error(f"❌ Error loading with joblib: {str(e)}")
                # Fallback to pickle if joblib fails
                try:
                    with open(model_path, 'rb') as file:
                        model = pickle.load(file)
                    st.success(f"✅ Model loaded successfully from: {model_path}")
                    
                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as file:
                            scaler = pickle.load(file)
                        st.success(f"✅ Scaler loaded successfully from: {scaler_path}")
                    else:
                        st.warning(f"⚠️ Scaler not found at: {scaler_path}")
                    
                    return model, scaler
                except Exception as e2:
                    st.error(f"❌ Error loading with pickle: {str(e2)}")
                    continue
    
    st.error("❌ Model file not found. Please ensure 'random_forest_model.pkl' is in the 'models' folder.")
    st.error("❌ Scaler file not found. Please ensure 'scaler.pkl' is in the 'models' folder.")
    st.info(f"📁 Current working directory: {os.getcwd()}")
    st.info("💡 Tip: Run 'streamlit run app.py' from the project root directory")
    return None, None

model, scaler = load_model_and_scaler()

# Title and description
st.title("💳 Credit Card Fraud Detection System")
st.markdown("---")
st.write("Enter transaction details below to check for potential fraud.")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Details")
    amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, step=0.01)
    trans_hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)
    trans_day = st.slider("Day of Month (1-31)", 1, 31, 15)
    trans_weekday = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)
    trans_month = st.slider("Month (1-12)", 1, 12, 6)
    
with col2:
    st.subheader("Customer & Location Details")
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    city_pop = st.number_input("City Population", min_value=0.0, value=50000.0, step=1000.0)
    age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
    merchant_enc = st.number_input("Merchant Code", min_value=-1.0, value=0.0, step=0.000001, format="%.6f")
    job_enc = st.number_input("Job Code", min_value=-1.0, value=0.0, step=0.000001, format="%.6f")
    state_enc = st.number_input("State Code", min_value=-1.0, value=0.0, step=0.000001, format="%.6f")

# Category selection
st.subheader("Transaction Category")
st.write("Select the transaction category (only one should be 1, rest should be 0)")

col3, col4, col5 = st.columns(3)

with col3:
    category_food_dining = st.checkbox("Food & Dining", value=False)
    category_gas_transport = st.checkbox("Gas & Transport", value=False)
    category_grocery_net = st.checkbox("Grocery (Online)", value=False)
    category_grocery_pos = st.checkbox("Grocery (POS)", value=False)
    category_health_fitness = st.checkbox("Health & Fitness", value=False)

with col4:
    category_home = st.checkbox("Home", value=False)
    category_kids_pets = st.checkbox("Kids & Pets", value=False)
    category_misc_net = st.checkbox("Miscellaneous (Online)", value=False)
    category_misc_pos = st.checkbox("Miscellaneous (POS)", value=False)

with col5:
    category_personal_care = st.checkbox("Personal Care", value=False)
    category_shopping_net = st.checkbox("Shopping (Online)", value=False)
    category_shopping_pos = st.checkbox("Shopping (POS)", value=False)
    category_travel = st.checkbox("Travel", value=False)

st.markdown("---")

# Predict button
if st.button("🔍 Check for Fraud", type="primary"):
    if model is not None:
        # Prepare input data
        input_data = pd.DataFrame({
            'amt': [amt],
            'gender': [gender],
            'city_pop': [city_pop],
            'trans_hour': [trans_hour],
            'trans_day': [trans_day],
            'trans_weekday': [trans_weekday],
            'trans_month': [trans_month],
            'age': [age],
            'merchant_enc': [merchant_enc],
            'job_enc': [job_enc],
            'category_food_dining': [int(category_food_dining)],
            'category_gas_transport': [int(category_gas_transport)],
            'category_grocery_net': [int(category_grocery_net)],
            'category_grocery_pos': [int(category_grocery_pos)],
            'category_health_fitness': [int(category_health_fitness)],
            'category_home': [int(category_home)],
            'category_kids_pets': [int(category_kids_pets)],
            'category_misc_net': [int(category_misc_net)],
            'category_misc_pos': [int(category_misc_pos)],
            'category_personal_care': [int(category_personal_care)],
            'category_shopping_net': [int(category_shopping_net)],
            'category_shopping_pos': [int(category_shopping_pos)],
            'category_travel': [int(category_travel)],
            'state_enc': [state_enc]
        })
        
        # Apply scaling to numerical features if scaler is available
        if scaler is not None:
            num_cols = ['amt', 'city_pop', 'age', 'merchant_enc', 'job_enc',
                       'trans_hour', 'trans_day', 'trans_weekday', 'trans_month']
            input_data[num_cols] = scaler.transform(input_data[num_cols])
        else:
            st.warning("⚠️ Scaler not loaded. Predictions may be inaccurate!")
        
        # Make prediction
        try:
            # Get probability scores
            proba = model.predict_proba(input_data)[0]
            fraud_probability = proba[1]
            
            # Apply threshold of 0.2
            prediction = 1 if fraud_probability >= 0.2 else 0
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                if prediction == 1:
                    st.error("⚠️ FRAUD DETECTED")
                    st.metric("Fraud Probability", f"{fraud_probability*100:.2f}%")
                else:
                    st.success("✅ LEGITIMATE TRANSACTION")
                    st.metric("Fraud Probability", f"{fraud_probability*100:.2f}%")
            
            with col_result2:
                st.metric("Legitimate Probability", f"{(1-fraud_probability)*100:.2f}%")
                st.write(f"**Threshold:** 0.2 (20%)")
                
            # Show probability bar
            st.progress(fraud_probability)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    else:
        st.error("Model not loaded. Cannot make predictions.")

# Footer
st.markdown("---")
st.markdown("**Note:** This model uses a threshold of 0.2 for fraud detection to minimize false negatives.")