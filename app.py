import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from modules.prediction import load_model, predict_disease
from modules.utils import get_disease_info, get_symptoms_dict


# Page configuration
st.set_page_config(
    page_title="Health Diagnostic System",
    page_icon="🏥",
    layout="wide"
)

def main():
    # Header
    st.title("Health Diagnostic System")
    st.markdown("""
    This application helps diagnose diseases based on symptoms. Please select your symptoms below.
    """)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses machine learning to predict diseases based on symptoms. "
        "It provides recommendations for precautions, medications, workouts, and diets."
    )
    
    # Load symptoms dictionary
    symptoms_dict = get_symptoms_dict()
    symptoms_list = list(symptoms_dict.keys())
    
    # Multi-select for symptoms
    st.subheader("Select Your Symptoms")
    selected_symptoms = st.multiselect(
        "Choose all that apply:",
        options=symptoms_list,
        help="Select multiple symptoms you are experiencing"
    )
   
    model_choice="Random Forest"
    # Prediction section
    if st.button("Analyze Symptoms"):
        if len(selected_symptoms) < 1:
            st.warning("Please select at least one symptom for diagnosis.")
        else:
            with st.spinner("Analyzing your symptoms..."):
                # Make prediction
                model = load_model(model_choice)
                predicted_disease, confidence = predict_disease(selected_symptoms, model)
                
                # Get disease info
                disease_info = get_disease_info(predicted_disease)
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.success(f"### Predicted Condition: {predicted_disease}")
                    st.markdown("### Description")
                    st.write(disease_info['description'])
                
                with col2:
                    st.markdown("### Selected Symptoms")
                    for symptom in selected_symptoms:
                        st.write(f"• {symptom}")
                
                # Display recommendations in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["Precautions", "Medications", "Diet", "Exercise"])
                
                with tab1:
                    st.subheader("Recommended Precautions")
                    for i, precaution in enumerate(disease_info['precautions'], 1):
                        if isinstance(precaution, str) and precaution.strip():  # Check if valid string
                            st.write(f"{i}. {precaution}")
                
                with tab2:
                    st.subheader("Recommended Medications")
                    st.warning("⚠️ Consult with a healthcare professional before taking any medications.")
                    for i, medication in enumerate(disease_info['medications'], 1):
                        if isinstance(medication, str) and medication.strip():  # Check if valid string
                            st.write(f"{i}. {medication}")
                
                with tab3:
                    st.subheader("Recommended Diet")
                    for i, diet in enumerate(disease_info['diets'], 1):
                        if isinstance(diet, str) and diet.strip():  # Check if valid string
                            st.write(f"{i}. {diet}")
                
                with tab4:
                    st.subheader("Recommended Exercises")
                    for i, workout in enumerate(disease_info['workouts'], 1):
                        if isinstance(workout, str) and workout.strip():  # Check if valid string
                            st.write(f"{i}. {workout}")
    
    st.markdown("---")
    st.caption(
        "**Disclaimer**: This application is for educational purposes only and is not a substitute for professional "
        "medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified "
        "health provider with any questions you may have regarding a medical condition."
    )

if __name__ == "__main__":
    main()
