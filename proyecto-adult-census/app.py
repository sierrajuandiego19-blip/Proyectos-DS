import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Predicción de Ingresos - Censo Adultos", layout="wide")

# Título y descripción
st.title("🚀 Clasificador de Ingresos (UCI Adult Census)")
st.markdown("""
Esta aplicación utiliza un modelo de **Machine Learning (Random Forest)** entrenado con un **Pipeline de Scikit-Learn** para predecir si una persona gana más de 50,000 USD anuales basándose en datos del censo.
""")

# Cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load('best_model.joblib')

model = load_model()

# Sidebar para entradas del usuario
st.sidebar.header("Entrada de Datos")

def user_input_features():
    age = st.sidebar.slider("Edad", 17, 90, 30)
    workclass = st.sidebar.selectbox("Clase de Trabajo", 
                                   ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                                    'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    fnlwgt = st.sidebar.number_input("fnlwgt (Peso Final)", value=180000)
    education = st.sidebar.selectbox("Educación", 
                                   ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 
                                    'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', 
                                    '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
    education_num = st.sidebar.slider("Años de Educación", 1, 16, 10)
    marital_status = st.sidebar.selectbox("Estado Civil", 
                                        ['Married-civ-spouse', 'Divorced', 'Never-married', 
                                         'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    occupation = st.sidebar.selectbox("Ocupación", 
                                    ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                                     'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
                                     'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
                                     'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    relationship = st.sidebar.selectbox("Relación", 
                                      ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    race = st.sidebar.selectbox("Raza", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    sex = st.sidebar.radio("Sexo", ['Male', 'Female'])
    capital_gain = st.sidebar.number_input("Ganancia de Capital", value=0)
    capital_loss = st.sidebar.number_input("Pérdida de Capital", value=0)
    hours_per_week = st.sidebar.slider("Horas por Semana", 1, 99, 40)
    native_country = st.sidebar.selectbox("País de Origen", 
                                        ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Iran', 'Portugal', 'Nicaragua', 'Peru', 'Greece', 'France', 'Ecuador', 'Ireland', 'Hong', 'Cambodia', 'Trinadad&Tobago', 'Thailand', 'Laos', 'Pound', 'Hungary', 'Scotland', 'Honduras', 'Holand-Netherlands'])

    data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'education-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

# Mostrar entradas
st.subheader("Datos Ingresados")
st.write(df_input)

# Predicción
if st.button("Predecir"):
    prediction = model.predict(df_input)
    prediction_proba = model.predict_proba(df_input)

    st.subheader("Resultado de la Predicción")
    res = ">50K" if prediction[0] == ">50K" else "<=50K"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Categoría Predicha", res)
    with col2:
        st.metric("Probabilidad (>50K)", f"{prediction_proba[0][1]*100:.2f}%")

    if res == ">50K":
        st.success("La persona probablemente gana más de 50,000 USD anuales.")
    else:
        st.warning("La persona probablemente gana 50,000 USD o menos anuales.")

# Información del Pipeline
st.divider()
st.subheader("Arquitectura del Pipeline")
st.info("""
Este proyecto implementa:
1. **SimpleImputer**: Manejo de valores faltantes.
2. **StandardScaler**: Normalización de variables numéricas.
3. **OneHotEncoder**: Transformación de variables categóricas.
4. **ColumnTransformer**: Integración de transformaciones heterogéneas.
5. **Random Forest**: Modelo final de clasificación.
""")
