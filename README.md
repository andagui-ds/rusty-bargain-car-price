# 🚗 Rusty Bargain — Tasación de Autos Usados con Machine Learning

> Comparación exhaustiva de modelos de regresión (LightGBM, CatBoost, XGBoost) para estimar el valor de mercado de autos usados, priorizando un equilibrio óptimo entre **calidad predictiva, tiempo de entrenamiento y velocidad de predicción**.

## 🎯 Problema de Negocio

Rusty Bargain está desarrollando una aplicación para atraer nuevos clientes, permitiéndoles averiguar rápidamente el valor de mercado de su coche. Para que la experiencia sea viable en producción, el equipo necesita un modelo que cumpla con tres requisitos estrictos:
1. **Alta calidad:** Minimizar el error de predicción (RMSE).
2. **Velocidad de predicción:** Responder rápidamente en la app para no perder clientes.
3. **Tiempo de entrenamiento:** Permitir actualizaciones periódicas del modelo sin costes computacionales excesivos.

## 📊 Datos

- **Fuente:** `car_data.csv`
- **Registros iniciales:** 354,369 filas y 16 columnas.
- **Limpieza:** Se descartaron columnas irrelevantes (como fechas y códigos postales) y se filtraron valores atípicos (precios menores a 100 €, años de matriculación irreales, potencias de 0 CV).
- **Volumen final para modelado:** 305,018 registros limpios.
- **Variables clave:** Marca, modelo, año de matriculación, potencia (CV), tipo de combustible, caja de cambios, etc.

## 🧠 Enfoque y Modelado

Se evaluaron 6 algoritmos diferentes, desde un modelo lineal base hasta los algoritmos de *Gradient Boosting* más avanzados del mercado. Para los modelos que no soportan categóricas nativamente, se aplicó `OrdinalEncoder`, y para otros como XGBoost, `One-Hot Encoding`.

**Modelos evaluados:**
- Regresión Lineal (Prueba de cordura)
- Árbol de Decisión
- Random Forest
- **LightGBM**
- CatBoost (con manejo nativo de categóricas)
- XGBoost

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

*(Nota: La Regresión Lineal confirma que los datos no tienen una relación lineal, justificando el uso de algoritmos basados en árboles).*

## ✅ Recomendación Final

**LightGBM es el ganador indiscutible para este proyecto.**
- **Calidad:** Obtuvo el mejor RMSE (1,630.02 €), superando ligeramente a Random Forest.
- **Velocidad:** Su tiempo de entrenamiento (1.96 segundos) es **26 veces más rápido que CatBoost** y **90 veces más rápido que XGBoost**.
- **Conclusión de negocio:** LightGBM permite iterar rápidamente, reentrenar el modelo con nuevos datos constantemente y ofrecer una latencia mínima en la aplicación de Rusty Bargain. CatBoost y XGBoost, aunque precisos, resultan inviables para un pipeline ágil en este volumen de datos.

## 🛠️ Stack Técnico
- **Lenguajes:** Python
- **Librerías:** Pandas, Scikit-learn, LightGBM, CatBoost, XGBoost, NumPy
- **Métricas:** RMSE (Root Mean Squared Error), Tiempos de ejecución.

## 🔗 Ver el Análisis Completo
- 📓 [Notebook original con todo el código y la experimentación](notebooks/analisis-rusty-bargain.ipynb)
