import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from streamlit.logger import get_logger

from config import (
    FASTAPI_BACKEND_ENDPOINT, 
    FASTAPI_WINE_MODEL_LOCATION,
    WINE_FEATURES,
    QUALITY_LABELS,
    APP_CONFIG
)

LOGGER = get_logger(__name__)

def check_backend_status():
    """Check if FastAPI backend is running"""
    try:
        backend_request = requests.get(FASTAPI_BACKEND_ENDPOINT)
        if backend_request.status_code == 200:
            st.success("🟢 Backend API Online")
            return True
        else:
            st.warning("🟡 Backend Connection Issues")
            return False
    except requests.ConnectionError as ce:
        LOGGER.error(f"Backend connection error: {ce}")
        st.error("🔴 Backend API Offline")
        return False

def create_sidebar():
    """Create sidebar with wine feature inputs"""
    with st.sidebar:
        st.header("🍷 Wine Quality Predictor")
        
        # Backend status check
        backend_status = check_backend_status()
        
        st.divider()
        st.subheader("Configure Wine Properties")
        
        # Create feature input widgets
        feature_values = {}
        
        # Group features for better organization
        st.markdown("**Acidity Measures**")
        for feature in ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'pH']:
            config = WINE_FEATURES[feature]
            feature_values[feature] = st.slider(
                label=feature.replace('_', ' ').title(),
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                step=config["step"],
                help=config["help"],
                format="%.3f" if feature in ['volatile_acidity', 'citric_acid'] else "%.2f"
            )
        
        st.markdown("**Sugar & Chemical Content**")
        for feature in ['residual_sugar', 'chlorides', 'sulphates']:
            config = WINE_FEATURES[feature]
            feature_values[feature] = st.slider(
                label=feature.replace('_', ' ').title(),
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                step=config["step"],
                help=config["help"],
                format="%.3f"
            )
        
        st.markdown("**Physical Properties**")
        for feature in ['density', 'alcohol']:
            config = WINE_FEATURES[feature]
            feature_values[feature] = st.slider(
                label=feature.replace('_', ' ').title(),
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                step=config["step"],
                help=config["help"],
                format="%.5f" if feature == 'density' else "%.1f"
            )
        
        st.markdown("**Sulfur Dioxide (Preservative)**")
        for feature in ['free_sulfur_dioxide', 'total_sulfur_dioxide']:
            config = WINE_FEATURES[feature]
            feature_values[feature] = st.slider(
                label=feature.replace('_', ' ').title(),
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                step=config["step"],
                help=config["help"],
                format="%.0f"
            )
        
        st.divider()
        
        # File upload option
        st.subheader("📁 Upload Wine Data")
        uploaded_file = st.file_uploader(
            "Upload JSON file with wine properties",
            type=['json'],
            help="Upload a JSON file containing wine features for batch prediction"
        )
        
        if uploaded_file:
            st.write("**File Preview:**")
            try:
                file_data = json.load(uploaded_file)
                st.json(file_data)
                st.session_state["uploaded_data"] = file_data
                st.session_state["IS_FILE_UPLOADED"] = True
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
                st.session_state["IS_FILE_UPLOADED"] = False
        else:
            st.session_state["IS_FILE_UPLOADED"] = False
        
        st.divider()
        
        # Prediction buttons
        predict_button = st.button(
            "🔮 Predict Quality", 
            type="primary",
            use_container_width=True,
            disabled=not backend_status
        )
        
        batch_predict_button = st.button(
            "📊 Batch Predict",
            use_container_width=True,
            disabled=not backend_status or not st.session_state.get("IS_FILE_UPLOADED", False)
        )
        
        return feature_values, predict_button, batch_predict_button


def create_main_content():
    """Create main content area"""
    st.title("🍷 Wine Quality Prediction Dashboard")
    st.markdown("---")
    
    # Info section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "Random Forest")
    
    with col2:
        st.metric("Features", "11 Chemical Properties")
    
    with col3:
        st.metric("Quality Range", "3-8 (Poor to Good)")
    
    st.markdown("---")
    
    # Instructions
    with st.expander("ℹ️ How to Use This App"):
        st.markdown("""
        **Single Prediction:**
        1. Adjust wine properties using the sidebar sliders
        2. Click "Predict Quality" to get quality score
        
        **Batch Prediction:**
        1. Upload a JSON file with wine data
        2. Click "Batch Predict" for multiple predictions
        
        **Wine Quality Scale:**
        - 3-4: Poor Quality
        - 5-6: Average Quality  
        - 7-8: Good Quality
        """)
    
    return st.empty()  # Placeholder for results


def make_prediction(feature_values):
    """Make single wine quality prediction"""
    
    # Prepare data for API
    wine_data = {
        "fixed_acidity": feature_values["fixed_acidity"],
        "volatile_acidity": feature_values["volatile_acidity"], 
        "citric_acid": feature_values["citric_acid"],
        "residual_sugar": feature_values["residual_sugar"],
        "chlorides": feature_values["chlorides"],
        "free_sulfur_dioxide": feature_values["free_sulfur_dioxide"],
        "total_sulfur_dioxide": feature_values["total_sulfur_dioxide"],
        "density": feature_values["density"],
        "pH": feature_values["pH"],
        "sulphates": feature_values["sulphates"],
        "alcohol": feature_values["alcohol"]
    }
    
    try:
        with st.spinner('🔄 Analyzing wine properties...'):
            response = requests.post(
                f'{FASTAPI_BACKEND_ENDPOINT}/predict',
                json=wine_data,
                headers={'Content-Type': 'application/json'}
            )
        
        if response.status_code == 200:
            result = response.json()
            quality_score = result.get("response", 0)
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Predicted Quality Score",
                    value=f"{quality_score + 3}",  # Convert back to original scale
                    help="Wine quality on scale of 3-8"
                )
            
            with col2:
                quality_label = QUALITY_LABELS.get(quality_score, "Unknown")
                st.metric(
                    label="Quality Category", 
                    value=quality_label
                )
            
            # Feature importance visualization
            st.subheader("🎯 Your Wine Profile")
            
            # Create a simple feature display
            feature_df = pd.DataFrame([
                {"Property": k.replace('_', ' ').title(), "Value": v} 
                for k, v in wine_data.items()
            ])
            
            st.dataframe(feature_df, use_container_width=True)
            
        else:
            st.error(f"❌ Prediction failed. Server responded with status: {response.status_code}")
            
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        LOGGER.error(f"Prediction error: {e}")

def make_batch_prediction(file_data):
    """Make batch predictions from uploaded file"""
    try:
        with st.spinner('🔄 Processing batch predictions...'):
            response = requests.post(
                f'{FASTAPI_BACKEND_ENDPOINT}/predict_batch',
                json=file_data,
                headers={'Content-Type': 'application/json'}
            )
        
        if response.status_code == 200:
            results = response.json()
            
            st.success("✅ Batch Prediction Complete!")
            
            # Process and display results
            if isinstance(results, list):
                # Convert to DataFrame for better visualization
                batch_results = []
                for i, result in enumerate(results):
                    quality_score = result.get("response", 0)
                    batch_results.append({
                        "Sample": i + 1,
                        "Predicted Quality": quality_score + 3,
                        "Category": QUALITY_LABELS.get(quality_score, "Unknown")
                    })
                
                results_df = pd.DataFrame(batch_results)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Samples", len(results))
                
                with col2:
                    avg_quality = np.mean([r["Predicted Quality"] for r in batch_results])
                    st.metric("Average Quality", f"{avg_quality:.2f}")
                
                with col3:
                    good_wines = len([r for r in batch_results if r["Predicted Quality"] >= 7])
                    st.metric("Good Quality Wines", f"{good_wines}")
            
        else:
            st.error(f"❌ Batch prediction failed. Status: {response.status_code}")
            
    except Exception as e:
        st.error(f"❌ Error during batch prediction: {str(e)}")
        LOGGER.error(f"Batch prediction error: {e}")




##  Controller sort of;
    ## First configure page (set_page_config), then create UI(create_sidebar, create_main_content), 
    ## then handle button clicks (make_prediction, make_batch_prediction).

def main():
    """Main application function"""

    st.set_page_config(
        page_title=APP_CONFIG["page_title"],
        page_icon=APP_CONFIG["page_icon"],
        layout=APP_CONFIG["layout"]
    )
    
    # UI components
    feature_values, predict_button, batch_predict_button = create_sidebar()
    result_container = create_main_content()
    
    # predictions
    with result_container.container():
        if predict_button:
            if FASTAPI_WINE_MODEL_LOCATION.is_file():
                make_prediction(feature_values)
            else:
                st.error("❌ Wine quality model not found! Please run train.py first.")
                LOGGER.warning("Model file not found")
        
        elif batch_predict_button and st.session_state.get("IS_FILE_UPLOADED", False):
            uploaded_data = st.session_state.get("uploaded_data", {})
            make_batch_prediction(uploaded_data)

if __name__ == "__main__":
    main()