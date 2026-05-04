import pandas as pd
import numpy as np
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_models():
    # 1. Cargar datos
    print("Cargando datos...")
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets

    # Limpiar etiquetas de la variable objetivo (quitar puntos al final si existen)
    y = y.iloc[:, 0].str.strip().str.replace('.', '', regex=False)
    
    # Manejar "?" como NaN
    X = X.replace('?', np.nan)

    # 2. Definir columnas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # 3. Crear transformadores para el ColumnTransformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 4. ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 5. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Definir Pipelines para los modelos
    # Modelo 1: Regresión Logística con PCA
    pipe_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=50)), # Reducir a 50 componentes principales
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # Modelo 2: Random Forest
    pipe_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 7. Entrenar y Evaluar
    models = {'Logistic Regression': pipe_lr, 'Random Forest': pipe_rf}
    
    for name, pipe in models.items():
        print(f"\nEntrenando {name}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        print(f"Resultados para {name}:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # 8. Guardar el mejor modelo (Random Forest usualmente)
    print("\nGuardando modelo y preprocesador...")
    joblib.dump(pipe_rf, 'best_model.joblib')
    print("Modelo guardado como 'best_model.joblib'")

if __name__ == "__main__":
    train_models()
