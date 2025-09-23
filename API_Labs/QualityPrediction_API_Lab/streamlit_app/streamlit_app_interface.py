"""
Wine Quality Streamlit App - Function Overview
Quick reference for streamlit_app.py functions and their purposes.

SYSTEM FUNCTIONS
==================
check_backend_status() → bool
    Pings FastAPI server, shows green/red status in UI

UI CREATION FUNCTIONS  
========================
create_sidebar() → (features_dict, predict_btn, batch_btn)
    Creates all 11 wine sliders + buttons + file upload

create_main_content() → empty_container
    Makes title, metrics, instructions area

PREDICTION FUNCTIONS
=======================
make_prediction(feature_values) → None
    Single wine prediction: takes slider values → calls API → shows result

make_batch_prediction(file_data) → None  
    Multiple wines: takes JSON file → calls API → shows results table

MAIN CONTROLLER
==================
main() → None
    Orchestrates everything: setup page → create UI → handle button clicks

---
Read this first, then dive into streamlit_app.py implementation.
"""