[🇺🇸 Read in English](#-rusty-bargain--used-car-valuation-with-machine-learning) | [🇲🇽 Leer en Español](#-rusty-bargain--tasación-de-autos-usados-con-machine-learning)

---

# 🚗 Rusty Bargain — Used Car Valuation with Machine Learning

> Comprehensive comparison of regression models (LightGBM, CatBoost, XGBoost) to estimate the market value of used cars, prioritizing an optimal balance between **predictive quality, training time, and prediction speed**.

🚀 **[Launch Interactive Streamlit App](https://rusty-bargain-car-price.streamlit.app/)** | 🌐 **[View Quarto Web Report](https://andagui-ds.github.io/rusty-bargain-car-price/)**

---

## 🎯 Business Problem

**Rusty Bargain** is developing an app to attract new customers by allowing them to quickly find out the market value of their car. To make the user experience viable in production, the team requires a model that meets three strict criteria:
1. **High Quality:** Minimize prediction error (RMSE).
2. **Prediction Speed:** Deliver fast inference inside the app to prevent user drop-off.
3. **Training Time:** Enable periodic model retraining without excessive computational costs.

## 📊 Data

* **Source:** `car_data.csv`
* **Initial Records:** 354,369 rows and 16 columns.
* **Data Cleaning:** Dropped irrelevant columns (such as dates and postal codes) and filtered out outliers (prices below €100, unrealistic registration years, 0 HP engine power).
* **Final Modeling Volume:** 305,018 clean records.
* **Key Features:** Brand, model, registration year, horsepower (HP), fuel type, transmission, etc.

## 🧠 Approach and Modeling

Evaluated 6 different algorithms, ranging from a linear baseline to industry-leading *Gradient Boosting* frameworks. Applied `OrdinalEncoder` for tree models without native categorical support, and `One-Hot Encoding` for models like XGBoost.

**Evaluated Models:**
* Linear Regression (Sanity check)
* Decision Tree
* Random Forest
* **LightGBM**
* CatBoost (with native categorical feature handling)
* XGBoost

## 🏆 Results and Performance (The Differentiator!)

This benchmark demonstrates why model selection goes beyond chasing the lowest error alone. Comparative results on the test set:

| Model | RMSE (€) | Training Time (s) | Prediction Time (s) |
| :--- | ---: | ---: | ---: |
| **LightGBM** 🏆 | **1,630.02** | **1.96** | 0.31 |
| Random Forest | 1,634.05 | 5.59 | 0.17 |
| CatBoost | 1,674.79 | 51.70 | 0.17 |
| XGBoost | 1,720.10 | 182.31 | 0.44 |
| Decision Tree | 1,911.18 | 0.57 | 0.01 |
| Linear Regression | 2,950.17 | 0.04 | 0.002 |

*(Note: Linear Regression confirms that the data lacks a linear relationship, justifying the use of tree-based ensembles).*

## ✅ Final Recommendation

**LightGBM is the undisputed winner for this project.**
* **Quality:** Achieved the lowest RMSE (€1,630.02), slightly outperforming Random Forest.
* **Speed:** Its training time (1.96 seconds) is **26x faster than CatBoost** and **90x faster than XGBoost**.
* **Business Conclusion:** LightGBM allows fast iteration, frequent retraining with incoming vehicle listings, and minimal latency inside the Rusty Bargain app. Although accurate, CatBoost and XGBoost are impractical for an agile pipeline at this data scale.

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, Scikit-learn, LightGBM, CatBoost, XGBoost, NumPy, Streamlit
* **Metrics:** RMSE (Root Mean Squared Error), Execution Time.

## 🔗 Full Analysis & Production App

* 📓 [Original Jupyter Notebook with full code and experimentation](notebooks/analisis-rusty-bargain.ipynb)
* 🚀 [Interactive Used Car Price Estimator (Streamlit)](https://rusty-bargain-car-price.streamlit.app/)

## ▶️ How to Run Locally

    pip install -r requirements.txt
    streamlit run app.py

---

# 🚗 Rusty Bargain — Tasación de Autos Usados con Machine Learning

> Comparación exhaustiva de modelos de regresión (LightGBM, CatBoost, XGBoost) para estimar el valor de mercado de autos usados, priorizando un equilibrio óptimo entre **calidad predictiva, tiempo de entrenamiento y velocidad de predicción**.

🚀 **[Abrir App Interactiva en Streamlit](https://rusty-bargain-car-price.streamlit.app/)** | 🌐 **[Ver Reporte Web en Quarto](https://andagui-ds.github.io/rusty-bargain-car-price/)**

---

## 🎯 Problema de Negocio

**Rusty Bargain** está desarrollando una aplicación para atraer nuevos clientes, permitiéndoles averiguar rápidamente el valor de mercado de su coche. Para que la experiencia sea viable en producción, el equipo necesita un modelo que cumpla con tres requisitos estrictos:
1. **Alta calidad:** Minimizar el error de predicción (RMSE).
2. **Velocidad de predicción:** Responder rápidamente en la app para no perder clientes.
3. **Tiempo de entrenamiento:** Permitir actualizaciones periódicas del modelo sin costes computacionales excesivos.

## 📊 Datos

* **Fuente:** `car_data.csv`
* **Registros iniciales:** 354,369 filas y 16 columnas.
* **Limpieza:** Se descartaron columnas irrelevantes (como fechas y códigos postales) y se filtraron valores atípicos (precios menores a 100 €, años de matriculación irreales, potencias de 0 CV).
* **Volumen final para modelado:** 305,018 registros limpios.
* **Variables clave:** Marca, modelo, año de matriculación, potencia (CV), tipo de combustible, caja de cambios, etc.

## 🧠 Enfoque y Modelado

Se evaluaron 6 algoritmos diferentes, desde un modelo lineal base hasta los algoritmos de *Gradient Boosting* más avanzados del mercado. Para los modelos que no soportan categóricas nativamente, se aplicó `OrdinalEncoder`, y para otros como XGBoost, `One-Hot Encoding`.

**Modelos evaluados:**
* Regresión Lineal (Prueba de cordura)
* Árbol de Decisión
* Random Forest
* **LightGBM**
* CatBoost (con manejo nativo de categóricas)
* XGBoost

## 🏆 Resultados y Rendimiento (¡El diferenciador!)

Aquí se demuestra por qué la elección del modelo va más allá de buscar el error más bajo. Tabla comparativa en el conjunto de prueba:

| Modelo | RMSE (€) | Tiempo de Entrenamiento (s) | Tiempo de Predicción (s) |
| :--- | ---: | ---: | ---: |
| **LightGBM** 🏆 | **1,630.02** | **1.96** | 0.31 |
| Random Forest | 1,634.05 | 5.59 | 0.17 |
| CatBoost | 1,674.79 | 51.70 | 0.17 |
| XGBoost | 1,720.10 | 182.31 | 0.44 |
| Árbol de Decisión | 1,911.18 | 0.57 | 0.01 |
| Regresión Lineal | 2,950.17 | 0.04 | 0.002 |

*(Nota: La Regresión Lineal confirma que los datos no tienen una relación lineal, justificando el uso de algoritmos basados en árboles)*.

## ✅ Recomendación Final

**LightGBM es el ganador indiscutible para este proyecto.**
* **Calidad:** Obtuvo el mejor RMSE (1,630.02 €), superando ligeramente a Random Forest.
* **Velocidad:** Su tiempo de entrenamiento (1.96 segundos) es **26 veces más rápido que CatBoost** y **90 veces más rápido que XGBoost**.
* **Conclusión de negocio:** LightGBM permite iterar rápidamente, reentrenar el modelo con nuevos datos constantemente y ofrecer una latencia mínima en la aplicación de Rusty Bargain. CatBoost y XGBoost, aunque precisos, resultan inviables para un pipeline ágil en este volumen de datos.

## 🛠️ Stack Técnico

* **Lenguajes:** Python
* **Librerías:** Pandas, Scikit-learn, LightGBM, CatBoost, XGBoost, NumPy, Streamlit
* **Métricas:** RMSE (Root Mean Squared Error), Tiempos de ejecución.

## 🔗 Ver el Análisis Completo y App en Vivo

* 📓 [Notebook original con todo el código y la experimentación](notebooks/analisis-rusty-bargain.ipynb)
* 🚀 [Tasador Interactivo de Autos Usados (Streamlit)](https://rusty-bargain-car-price.streamlit.app/)

## ▶️ Cómo ejecutar localmente

    pip install -r requirements.txt
    streamlit run app.py
