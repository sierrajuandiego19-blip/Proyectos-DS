# Predicción de Ingresos: Adult Census Income Project 🚀

Este repositorio contiene un proyecto completo de **Data Science** diseñado para un entorno universitario. El objetivo principal es predecir si una persona tiene ingresos superiores a $50,000 USD anuales basándose en datos del censo de 1994 (UCI Adult Census Dataset).

## 📋 Características del Proyecto

Este proyecto destaca por la implementación de un flujo de trabajo profesional utilizando:
- **Scikit-Learn Pipelines**: Para un procesamiento de datos reproducible y limpio.
- **ColumnTransformer**: Manejo simultáneo de variables numéricas y categóricas.
- **Preprocesamiento Avanzado**:
  - Imputación de valores faltantes (SimpleImputer).
  - Escalado de datos (StandardScaler).
  - Codificación de variables categóricas (OneHotEncoder).
  - Reducción de dimensionalidad (PCA).
- **Modelos de Machine Learning**: Comparación entre Regresión Logística y Random Forest.
- **Interfaz de Usuario**: Aplicación interactiva desarrollada con **Streamlit**.

## 📁 Estructura del Repositorio

- `eda_script.py`: Script para el Análisis Exploratorio de Datos y generación de visualizaciones.
- `model_training.py`: Implementación del pipeline de preprocesamiento y entrenamiento de modelos.
- `app.py`: Código de la aplicación web Streamlit.
- `best_model.joblib`: El modelo Random Forest entrenado y listo para producción.
- `informe_proyecto_datascience.md`: Informe técnico detallado del proyecto.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.
- `income_distribution.png` & `correlation_heatmap.png`: Gráficos generados durante el EDA.

## 🚀 Cómo Ejecutar el Proyecto

1. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar el Entrenamiento (opcional)**:
   ```bash
   python model_training.py
   ```

3. **Lanzar la App de Streamlit**:
   ```bash
   streamlit run app.py
   ```

## 📊 Resultados

El modelo final (**Random Forest**) alcanzó una precisión aproximada del **85%** en el conjunto de prueba, demostrando ser robusto para manejar la complejidad de los datos censales.

## 📚 Referencias

- Dataset: [UCI Machine Learning Repository - Adult](https://archive.ics.uci.edu/dataset/2/adult)
- Herramientas: Scikit-Learn, Pandas, Streamlit, Matplotlib, Seaborn.
