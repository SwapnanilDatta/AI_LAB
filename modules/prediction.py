import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Model paths
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Dictionary of available models
MODEL_DICT = {
    "Random Forest": f"{MODEL_DIR}/random_forest_model.pkl",
    "SVC": f"{MODEL_DIR}/svc_model.pkl",
    "KNeighbors": f"{MODEL_DIR}/knn_model.pkl",
    "Gradient Boosting": f"{MODEL_DIR}/gradient_boosting_model.pkl",
    "MultinomialNB": f"{MODEL_DIR}/multinomial_nb_model.pkl"
}

# Disease mapping
DISEASES_LIST = {
    'Fungal infection': 'Fungal infection', 
    'Allergy': 'Allergy', 
    'GERD': 'GERD',
    'Chronic cholestasis': 'Chronic cholestasis',
    'Drug Reaction': 'Drug Reaction',
    'Peptic ulcer disease': 'Peptic ulcer disease',
    'AIDS': 'AIDS',
    'Diabetes': 'Diabetes',
    'Gastroenteritis': 'Gastroenteritis',
    'Bronchial Asthma': 'Bronchial Asthma',
    'Hypertension': 'Hypertension',
    'Migraine': 'Migraine',
    'Cervical spondylosis': 'Cervical spondylosis',
    'Paralysis (brain hemorrhage)': 'Paralysis (brain hemorrhage)',
    'Jaundice': 'Jaundice',
    'Malaria': 'Malaria',
    'Chicken pox': 'Chicken pox',
    'Dengue': 'Dengue',
    'Typhoid': 'Typhoid',
    'hepatitis A': 'hepatitis A',
    'Hepatitis B': 'Hepatitis B',
    'Hepatitis C': 'Hepatitis C',
    'Hepatitis D': 'Hepatitis D',
    'Hepatitis E': 'Hepatitis E',
    'Alcoholic hepatitis': 'Alcoholic hepatitis',
    'Tuberculosis': 'Tuberculosis',
    'Common Cold': 'Common Cold',
    'Pneumonia': 'Pneumonia',
    'Dimorphic hemorrhoids(piles)': 'Dimorphic hemorrhoids(piles)',
    'Heart attack': 'Heart attack',
    'Varicose veins': 'Varicose veins',
    'Hypothyroidism': 'Hypothyroidism',
    'Hyperthyroidism': 'Hyperthyroidism',
    'Hypoglycemia': 'Hypoglycemia',
    'Osteoarthritis': 'Osteoarthritis',
    'Arthritis': 'Arthritis',
    '(vertigo) Paroxysmal Positional Vertigo': '(vertigo) Paroxysmal Positional Vertigo',
    'Acne': 'Acne',
    'Urinary tract infection': 'Urinary tract infection',
    'Psoriasis': 'Psoriasis',
    'Impetigo': 'Impetigo'
}

def load_model(model_name="Random Forest"):
    """
    Load a trained model from file or train if not available
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        model: Trained model
    """
    model_path = MODEL_DICT.get(model_name, MODEL_DICT["Random Forest"])
    
    # Check if model exists
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    else:
        # Train a new model if requested model doesn't exist
        model = train_new_model(model_name)
        
    return model

def train_new_model(model_name):
    """
    Train a new model and save it to disk
    
    Args:
        model_name (str): Name of the model to train
        
    Returns:
        model: Trained model
    """
    # Load training data
    df = pd.read_csv('new/Training.csv')
    
    # Preprocess data
    X = df.drop('prognosis', axis=1)
    y = df['prognosis']
    
    # Create model based on name
    if model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42, n_estimators=100,max_depth=10,min_samples_leaf=1,min_samples_split=5)
    elif model_name == "SVC":
        model = SVC(kernel='linear', probability=True)
    elif model_name == "KNeighbors":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42, n_estimators=100)
    elif model_name == "MultinomialNB":
        model = MultinomialNB()
    else:
        # Default to Random Forest
        model = RandomForestClassifier(random_state=42)
    
    # Train model
    model.fit(X, y)
    
    # Save model
    model_path = MODEL_DICT[model_name]
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
        
    return model

def predict_disease(patient_symptoms, model):
    """
    Predict disease based on symptoms using the provided model
    
    Args:
        patient_symptoms (list): List of patient symptoms
        model: Trained model
        
    Returns:
        tuple: (predicted_disease, confidence)
    """
    from modules.utils import get_symptoms_dict
    
    # Get symptoms dictionary
    symptoms_dict = get_symptoms_dict()
    
    # Create input vector
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    
    # Get prediction and probability
    prediction = model.predict([input_vector])[0]  # This will now return the disease name directly
    
    # Get confidence if model supports predict_proba
    try:
        proba = model.predict_proba([input_vector])[0]
        disease_idx = list(model.classes_).index(prediction)
        confidence = proba[disease_idx] * 100
    except:
        confidence = 95.0  # Default confidence if model doesn't support probabilities
    
    # Use the prediction directly since it's now the disease name
    predicted_disease = DISEASES_LIST[prediction]
    
    return predicted_disease, confidence