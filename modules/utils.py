import pandas as pd
import os
import ast

# Update DATA_DIR to point to your new folder
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'new')

def load_dataframes():
    """Load all necessary dataframes from the new folder"""
    try:
        description = pd.read_csv(os.path.join(DATA_DIR, 'description.csv'))
        precautions = pd.read_csv(os.path.join(DATA_DIR, 'precautions_df.csv'))
        medications = pd.read_csv(os.path.join(DATA_DIR, 'medications.csv'))
        diets = pd.read_csv(os.path.join(DATA_DIR, 'diets.csv'))
        workout = pd.read_csv(os.path.join(DATA_DIR, 'workout_df.csv'))
        symptoms = pd.read_csv(os.path.join(DATA_DIR, 'symtoms_df.csv'))
        training = pd.read_csv(os.path.join(DATA_DIR, 'Training.csv'))
        
        return {
            'description': description,
            'precautions': precautions,
            'medications': medications,
            'diets': diets,
            'workout': workout,
            'symptoms': symptoms,
            'training': training
        }
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def get_disease_info(predicted_disease):
    """Get all information about a disease"""
    dfs = load_dataframes()
    
    try:
        # Get description
        desc = dfs['description'][dfs['description']['Disease'] == predicted_disease]['Description'].iloc[0]
        
        # Get precautions - using the correct column names from precautions_df.csv
        pre_row = dfs['precautions'][dfs['precautions']['Disease'] == predicted_disease].iloc[0]
        precautions = [pre_row['Precaution_1'], pre_row['Precaution_2'], 
                      pre_row['Precaution_3'], pre_row['Precaution_4']]
        
        # Get workout suggestions from workout_df.csv
        workout_data = dfs['workout'][dfs['workout']['disease'] == predicted_disease]['workout'].tolist()
        
        # Get medications and parse the string list
        med_row = dfs['medications'][dfs['medications']['Disease'] == predicted_disease]['Medication'].iloc[0]
        try:
            medications = ast.literal_eval(med_row)
        except:
            medications = []
        
        # Get diets and parse the string list
        diet_row = dfs['diets'][dfs['diets']['Disease'] == predicted_disease]['Diet'].iloc[0]
        try:
            diets = ast.literal_eval(diet_row)
        except:
            diets = []
        
        return {
            'description': desc,
            'precautions': [p for p in precautions if pd.notna(p) and str(p).strip()],
            'workouts': [w for w in workout_data if pd.notna(w) and str(w).strip()],
            'medications': [m for m in medications if pd.notna(m) and str(m).strip()],
            'diets': [d for d in diets if pd.notna(d) and str(d).strip()]
        }
    except Exception as e:
        print(f"Error getting disease info: {str(e)}")
        return {
            'description': "Information not available",
            'precautions': [],
            'workouts': [],
            'medications': [],
            'diets': []
        }

def get_symptoms_dict():
    """Get the dictionary mapping symptoms to indices"""
    df = load_dataframes()['training']
    symptoms = df.columns.tolist()
    symptoms.remove('prognosis')  # Remove target column
    return {symptom: idx for idx, symptom in enumerate(symptoms)}