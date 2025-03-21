import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

def load_Loan_Data():
    df = pd.read_csv("Loan_Data.csv", nrows=1)  # Juste une ligne pour récupérer les colonnes
    return df

data = load_Loan_Data()
features = list(data.columns)

mlflow.set_tracking_uri("http://localhost:5001")
model_uri = "runs:/0289c4d20ba6480f869f3f90b7d80ebf/model"  

model = mlflow.sklearn.load_model(model_uri)


st.title(" Prédiction de risque avec Logistic Regression")
st.write("Remplissez les valeurs des caractéristiques pour obtenir une prédiction:")

# Créer un formulaire avec les features du dataset
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(feature, value=0.0)

# Quand l'utilisateur clique sur prédire
if st.button("Prédire"):
    # Convertir en DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Prédiction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    
    # Affichage du résultat
    st.write(f"### 🟢 Résultat : {'Danger' if prediction == 1 else 'Bon'} (Probabilité : {proba:.2f})")

