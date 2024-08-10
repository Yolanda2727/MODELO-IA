
import streamlit as st
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
import torch

# Título de la aplicación
st.title("Aplicación de Clasificación de Casos Clínicos")

# Descripción del caso clínico como entrada de texto
case_description = st.text_area("Descripción del Caso Clínico:")

# Función para preprocesar texto con BERT
def preprocess_text_with_bert(text):
    tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
    model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
    return embedding[:128]  # Reducimos la dimensión del vector a 128

# Modelo de clasificación (Logistic Regression)
model = LogisticRegression()

# Datos de ejemplo para el entrenamiento
X_data = [
    "Paciente de 75 años con cáncer avanzado y dolor severo. El paciente rechaza quimioterapia adicional.",
    "Paciente de 50 años con insuficiencia renal crónica. Familia solicita cuidados paliativos.",
    "Paciente de 30 años con traumatismo craneoencefálico severo. Estado vegetativo permanente.",
    "Paciente de 80 años con demencia avanzada. Decisiones sobre intervención quirúrgica menor.",
    "Paciente de 60 años con ELA. Consideración de ventilación mecánica invasiva.",
    "Paciente de 40 años con leucemia. Solicitud de trasplante de médula ósea experimental.",
    "Paciente de 85 años con enfermedad cardíaca severa. Consideración de retiro de soporte vital.",
    "Paciente de 65 años con complicaciones postoperatorias. Familia solicita reanimación agresiva.",
    "Paciente de 55 años con cáncer de páncreas. Solicitud de tratamiento experimental.",
    "Paciente de 45 años con daño hepático severo por consumo de alcohol. Consideración de trasplante hepático."
]
y_data = [0, 1, 0, 0, 0, 1, 0, 0, 1, 0]  # Etiquetas de ejemplo

# Preprocesamiento de los textos de entrenamiento
X_transformed = np.array([preprocess_text_with_bert(text) for text in X_data])

# Entrenamiento del modelo
model.fit(X_transformed, y_data)

# Clasificación cuando se presiona el botón
if st.button("Clasificar"):
    if case_description:
        # Preprocesar el texto ingresado
        case_vector = preprocess_text_with_bert(case_description).reshape(1, -1)
        # Predecir la clase
        prediction = model.predict(case_vector)
        # Mostrar el resultado
        st.write(f"Clasificación del caso: {'Caso para intervención' if prediction[0] == 1 else 'Caso para cuidados paliativos'}")
    else:
        st.write("Por favor, ingrese una descripción del caso clínico.")
