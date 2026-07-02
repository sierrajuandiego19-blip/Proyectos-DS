import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

def perform_eda():
    # 1. Cargar el dataset
    print("Cargando dataset Adult Census Income...")
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets

    # Unir para EDA
    df = pd.concat([X, y], axis=1)

    # 2. Información básica
    print("\n--- Información General ---")
    print(df.info())
    
    # 3. Manejo de valores "?" (común en este dataset)
    df.replace('?', np.nan, inplace=True)
    
    print("\n--- Valores Faltantes ---")
    print(df.isnull().sum())

    # 4. Estadísticas descriptivas
    print("\n--- Estadísticas Descriptivas (Numéricas) ---")
    print(df.describe())

    # 5. Visualización - Distribución de la variable objetivo
    plt.figure(figsize=(8, 5))
    sns.countplot(x='income', data=df)
    plt.title('Distribución de Ingresos (>50K vs <=50K)')
    plt.savefig('income_distribution.png')
    plt.close()

    # 6. Visualización - Correlación entre variables numéricas
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de Calor de Correlación')
    plt.savefig('correlation_heatmap.png')
    plt.close()

    print("\nEDA completado. Imágenes guardadas.")

if __name__ == "__main__":
    perform_eda()
