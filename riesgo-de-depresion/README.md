## 🧠 Análisis Predictivo: Riesgo de Depresión por Uso de Redes Sociales

## Autores
## Pablo Lopera, Luciano Sanin, Juan Diego Sierra

## Descripción del Proyecto
Este proyecto de Machine Learning tiene como objetivo analizar cómo el uso de las redes sociales influye en la salud mental de los adolescentes, específicamente en el riesgo de depresión. Utiliza un dataset sintético pero realista para explorar patrones de comportamiento, hábitos diarios y factores psicológicos que pueden predecir el nivel de riesgo de depresión (bajo, medio, alto).
El análisis incluye un exhaustivo proceso de Exploración de Datos (EDA), preprocesamiento de variables y la implementación de diversos modelos de clasificación para identificar el más efectivo en la predicción del riesgo de depresión.

## Variable del modelo

El dataset utilizado, **Teen Mental Health Dataset**, fue obtenido de Kaggle [1]. Contiene 2500 registros con las siguientes características:

| Columna | Descripción |
|---|---|
| `age` | Edad del adolescente (13–19) |
| `gender` | Género (Male o Female) |
| `daily_social_media_hours` | Horas diarias dedicadas a redes sociales |
| `platform_usage` | Plataforma de red social principal (Instagram, TikTok, Both, Other) |
| `sleep_hours` | Horas promedio de sueño diario |
| `screen_time_before_sleep` | Uso del teléfono antes de dormir |
| `academic_performance` | Puntuación GPA (escala 2.0–4.0) |
| `physical_activity` | Horas de ejercicio diario |
| `social_interaction_level` | Nivel de interacción social (Low, Medium, High) |
| `stress_level` | Puntuación de estrés (1–10) |
| `anxiety_level` | Puntuación de ansiedad (1–10) |
| `depression_risk` | Variable objetivo: Riesgo de depresión (Low, Medium, High) |

# 📊**Conclusiones del EDA:**
*   Mayor uso de redes sociales → Mayor estrés y ansiedad.
*   Menos horas de sueño → Mayor estrés y riesgo de depresión.
*   Más actividad física → Reducción del estrés y la ansiedad.
*   Más tiempo de pantalla antes de dormir → Reducción de la calidad del sueño.
*   Alto estrés + pocas horas de sueño → Menor rendimiento académico.

## 📂 Estructura del Proyecto

```text
Py1/
├── Data/
│   ├── processed.csv
│   └── raw/
├── Modelos/
│   ├── label_encoder.pkl          # Para decodificar predicciones numéricas
│   ├── mejor_modelo.pkl           # Modelo entrenado listo para producción
│   └── standard_scaler.pkl        # Para escalar nuevos datos de entrada
├── Notebooks/
│   ├── 01_Carga_de_Datos_.ipynb
│   ├── 02_Analisis_Inicial__de__Variables.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Preprocesamiento_de_datos.ipynb
│   ├── 05_Modelamiento_Predictivo.ipynb
│   └── 06_Guardado_del_modelo.ipynb
├── .gitignore
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
└── uv.lock
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
└── uv.lock
```

# 🚀 **Como ejecutar el proyecto:**

# Opción A — Google Colab (recomendado)

1. Abre el notebook en [Google Colab](https://colab.research.google.com)
2. Cuando se pida, sube el archivo `Teen_Mental_Health_Dataset.csv`
3. Ejecuta todas las celdas con **Runtime → Run all**

# Opción B — VS Code local con uv

**Requisitos:** Python 3.10+, [uv](https://docs.astral.sh/uv/) instalado

```bash
# 1. Clona el repositorio
git clone https://github.com/sierrajuandiego19-blip/Py1.git
cd Py1

# 2. Crea el entorno virtual e instala dependencias
uv sync

# 3. Instala las dependencias del proyecto
uv add pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter

# 4. Abre los notebooks
uv run jupyter notebook
```

> ⚠️ Si usas VS Code local, cambia la celda de carga del CSV de Colab por:
> ```python
> df_raw = pd.read_csv('../Data/Teen_Mental_Health_Dataset.csv')
> ```


## 📦 Dependencias principales

| Librería | Uso |
|---|---|
| `pandas` | 3.0.2 |
| `numpy` | 2.4.4 |
| `plotly` | 6.7.0 |
| `matplotlib` | 3.10.9 |
| `scikit-learn` | 1.8.0 |
| `xgboost` | 3.2.0 |
| `joblib` | 1.5.3 |
| `scipy` | 1.17.1 |
| `statsmodels` |0.14.6 |

---

## 🏆Resultados y Conclusiones

El análisis reveló que `stress_level` y `anxiety_level` son las variables que mejor separan los grupos de riesgo de depresión. `sleep_hours` y `academic_performance` también muestran una relación inversa con el riesgo. En cuanto a las variables categóricas, el `social_interaction_level` es un predictor clave, donde una alta interacción social se asocia con un menor riesgo de depresión.

Se evaluaron varios modelos de clasificación, y los resultados se compararon en diferentes escenarios de preprocesamiento. El mejor modelo fue Random Forest, se seleccionó basándose en la métrica F1-weighted, que es más robusta para datasets desbalanceados.
- **Accuracy:** ~76-77%
- **F1-weighted:** ~0.77 (validación cruzada 5 folds)
- **Variable más predictiva:** `stress_level` y `anxiety_level`

## Referencias

[1] Suresh Beekhani. (2024). *Teen Social Media Usage & Mental Health*. Kaggle. Disponible en: [https://www.kaggle.com/datasets/sureshbeekhani/teen-social-media-usage-and-mental-health](https://www.kaggle.com/datasets/sureshbeekhani/teen-social-media-usage-and-mental-health)



