
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# Set page config
st.set_page_config(
    page_title="Smart City Traffic Forecast",
    page_icon="🚦",
    layout="wide"
)

# App title
st.title("🚦 Smart City Traffic Forecasting")
st.markdown("Predict traffic volumes using machine learning for smarter urban mobility.")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Section", 
    ["Home", "Live Predictions", "Model Performance", "About"])

# Load model function
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_traffic_model.pkl')
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        return model, scaler_X, scaler_y
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None

# Load performance data
@st.cache_data
def load_performance():
    try:
        with open('model_performance.json', 'r') as f:
            return json.load(f)
    except:
        return None

# Home page
if app_mode == "Home":
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Project Details")
        st.markdown("""
        - **Objective**: Short-term traffic volume forecasting
        - **Models**: Linear Regression, Random Forest, Gradient Boosting
        - **Technology**: Python, Scikit-learn, Streamlit
        - **Use Case**: Smart city traffic management
        """)
    
    with col2:
        st.subheader("Key Features")
        st.markdown("""
        - Real-time traffic predictions
        - Multiple ML model comparison
        - Interactive parameter tuning
        - Production-ready deployment
        """)
    
    # Show best model performance
    performance_data = load_performance()
    if performance_data:
        best_model = max(performance_data.items(), key=lambda x: x[1]['R2'])
        st.success(f"**Best Model**: {best_model[0]} | **R² Score**: {best_model[1]['R2']:.4f}")

# Live Predictions page
elif app_mode == "Live Predictions":
    st.header("Live Traffic Predictions")
    
    model, scaler_X, scaler_y = load_model()
    
    if model and scaler_X and scaler_y:
        st.subheader("Enter Traffic Conditions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hour = st.slider("Hour of Day", 0, 23, 8)
            speed = st.slider("Vehicle Speed (km/h)", 0, 120, 50)
            occupancy = st.slider("Road Occupancy", 0.0, 1.0, 0.6)
        
        with col2:
            temperature = st.slider("Temperature (°C)", -10, 40, 20)
            traffic_1h_ago = st.number_input("Traffic Volume 1 Hour Ago", min_value=0, value=1000)
            is_rush_hour = st.checkbox("Rush Hour Period", value=False)
        
        if st.button("Predict Traffic Volume", type="primary"):
            with st.spinner("Calculating prediction..."):
                try:
                    # Create features for prediction
                    hour_sin = np.sin(2 * np.pi * hour / 24)
                    hour_cos = np.cos(2 * np.pi * hour / 24)
                    
                    features = np.array([[
                        speed, occupancy, temperature,
                        hour_sin, hour_cos, 0, 0,
                        0, 1 if is_rush_hour else 0, 0,
                        traffic_1h_ago, 0, 0, 0, traffic_1h_ago,
                        traffic_1h_ago, 0, traffic_1h_ago
                    ]])
                    
                    # Make prediction
                    features_scaled = scaler_X.transform(features)
                    prediction_scaled = model.predict(features_scaled)
                    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
                    
                    # Display results
                    st.success(f"**Predicted Traffic Volume**: {prediction:.0f} vehicles/hour")
                    
                    # Traffic assessment
                    if prediction < 800:
                        st.info("🟢 Light Traffic - Good conditions")
                    elif prediction < 1200:
                        st.info("🟡 Moderate Traffic - Normal conditions")
                    else:
                        st.warning("🔴 Heavy Traffic - Congested conditions")
                        
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    # Fallback calculation
                    fallback_pred = 800 + (hour * 15) + ((60 - speed) * 3) + (traffic_1h_ago * 0.2)
                    if is_rush_hour:
                        fallback_pred += 200
                    st.info(f"Estimated traffic: {fallback_pred:.0f} vehicles/hour")
    else:
        st.error("Models not loaded properly")

# Model Performance page
elif app_mode == "Model Performance":
    st.header("Model Performance Analysis")
    
    performance_data = load_performance()
    
    if performance_data:
        # Display performance metrics
        perf_df = pd.DataFrame(performance_data).T
        perf_df = perf_df.reset_index().rename(columns={'index': 'Model'})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Metrics")
            st.dataframe(perf_df, use_container_width=True)
            
            best_idx = perf_df['R2'].idxmax()
            best_model = perf_df.loc[best_idx]
            st.success(f"**Best Model**: {best_model['Model']} | **R²**: {best_model['R2']:.4f}")
        
        with col2:
            st.subheader("Performance Comparison")
            metric = st.selectbox("Select Metric", ['RMSE', 'MAE', 'R2'])
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(perf_df['Model'], perf_df[metric], color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
            
            if metric == 'R2':
                ax.set_ylabel('R² Score (Higher is Better)')
            else:
                ax.set_ylabel(f'{metric} (Lower is Better)')
            
            ax.set_title(f'Model {metric} Comparison')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', 
                       ha='center', va='bottom')
            
            st.pyplot(fig)

# About page
elif app_mode == "About":
    st.header("About This Project")
    
    st.markdown("""
    ### Smart City Traffic Forecasting System
    
    **Project Overview**:
    This application demonstrates machine learning for short-term traffic volume 
    forecasting in smart city environments.
    
    **Technical Stack**:
    - Python, Scikit-learn, Streamlit
    - Multiple ML models compared
    - Production-ready deployment
    
    **Use Cases**:
    - Urban traffic management
    - Route planning and optimization
    - Smart city infrastructure
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Smart City AI Project | v1.0")
