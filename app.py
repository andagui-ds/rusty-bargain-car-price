import joblib  # Carga el paquete con el modelo LightGBM, el encoder y las categorías limpias
# Construye los DataFrames de entrada en el orden exacto de entrenamiento
import pandas as pd
# Genera las gráficas interactivas de benchmarking y depreciación
import plotly.express as px
import streamlit as st  # Construye la interfaz web nativa sin necesidad de escribir CSS

st.set_page_config(  # Configura el título de la pestaña y el modo ancho (wide) en una pantalla
    page_title="Rusty Bargain | Tasador ML",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Guarda el modelo en caché para que cada ajuste en los controles sea instantáneo
@st.cache_resource
def load_artifact():
    # Lee el artefacto ligero exportado desde tu notebook
    return joblib.load("modelo_rusty.joblib")


artefacto = load_artifact()  # Extrae el diccionario con los componentes del modelo
# Modelo ganador LightGBM entrenado con 305,018 registros
model = artefacto["model"]
# OrdinalEncoder ajustado para transformar texto a códigos numéricos
encoder = artefacto["encoder"]
# Lista exacta de las 10 columnas requeridas por el modelo
features = artefacto["features"]
cat_cols = artefacto["cat_cols"]  # Nombres de las 6 columnas categóricas
# Diccionario con las categorías únicas de cada variable
categorias = artefacto["categorias"]

# --- BARRA LATERAL: ESPECIFICACIONES DEL VEHÍCULO ---
with st.sidebar:  # Agrupa los controles de entrada del vehículo en el panel izquierdo
    # Encabezado de la sección de parámetros
    st.header("🚘 Especificaciones del Auto")

    # Filtra 'unknown' para mostrar únicamente categorías reales en los menús desplegables
    brands = [c for c in categorias["Brand"]
              if c != "unknown"]  # Lista limpia de marcas
    models = [c for c in categorias["Model"] if c !=
              "unknown"]  # Lista limpia de modelos
    v_types = [c for c in categorias["VehicleType"]
               if c != "unknown"]  # Lista limpia de carrocerías
    gearboxes = [c for c in categorias["Gearbox"] if c !=
                 "unknown"]  # Lista limpia de transmisiones
    fuels = [c for c in categorias["FuelType"] if c !=
             "unknown"]  # Lista limpia de combustibles
    repaired_opts = [c for c in categorias["NotRepaired"]
                     if c != "unknown"]  # Opciones reales de reparación

    brand = st.selectbox(  # Selector de marca con Volkswagen por defecto
        "Marca (Brand)",
        brands,
        index=brands.index("volkswagen") if "volkswagen" in brands else 0,
    )
    model_car = st.selectbox(  # Selector de modelo con Golf por defecto
        "Modelo (Model)",
        models,
        index=models.index("golf") if "golf" in models else 0,
    )
    vehicle_type = st.selectbox(  # Selector de carrocería con Sedán por defecto
        "Carrocería (VehicleType)",
        v_types,
        index=v_types.index("sedan") if "sedan" in v_types else 0,
    )

    # Divide la fila en dos columnas compactas para año y mes
    c1, c2 = st.columns(2)
    with c1:
        reg_year = st.number_input(
            "Año", 1980, 2019, 2012, step=1)  # Año de matriculación
    with c2:
        reg_month = st.number_input(
            "Mes", 0, 12, 6, step=1)  # Mes de matriculación

    # Divide la fila en dos columnas para potencia y kilometraje
    c3, c4 = st.columns(2)
    with c3:
        # Potencia del motor en caballos
        power = st.number_input("Potencia (CV)", 30, 600, 120, step=5)
    with c4:
        mileage = st.number_input(  # Kilometraje recorrido del vehículo
            "Kilometraje", 5000, 150000, 90000, step=5000
        )

    gearbox = st.selectbox(  # Selector de transmisión con Manual por defecto
        "Transmisión (Gearbox)",
        gearboxes,
        index=gearboxes.index("manual") if "manual" in gearboxes else 0,
    )
    fuel_type = st.selectbox(  # Selector de combustible con Gasolina (petrol) por defecto
        "Combustible (FuelType)",
        fuels,
        index=fuels.index("petrol") if "petrol" in fuels else 0,
    )
    not_repaired = st.selectbox(  # Selector de historial de daños con 'no' por defecto
        "¿Sin reparar? (NotRepaired)",
        repaired_opts,
        index=repaired_opts.index("no") if "no" in repaired_opts else 0,
    )

# --- PREDICCIÓN EN TIEMPO REAL ---
input_df = pd.DataFrame([{  # Crea un DataFrame de una fila con las especificaciones seleccionadas
    "VehicleType": vehicle_type,
    "RegistrationYear": reg_year,
    "Gearbox": gearbox,
    "Power": power,
    "Model": model_car,
    "Mileage": mileage,
    "RegistrationMonth": reg_month,
    "FuelType": fuel_type,
    "Brand": brand,
    "NotRepaired": not_repaired,
}])[features]  # Asegura el orden exacto de las 10 columnas

# Copia los datos para codificar las variables categóricas
input_encoded = input_df.copy()
input_encoded[cat_cols] = encoder.transform(
    input_df[cat_cols])  # Aplica el OrdinalEncoder entrenado
for col in cat_cols:
    input_encoded[col] = input_encoded[col].astype(
        "category")  # Convierte a tipo category para LightGBM

# Calcula la tasación estimada en euros
precio_estimado = float(model.predict(input_encoded)[0])
# Error cuadrático medio (RMSE) validado en el conjunto de prueba
rmse_ref = 1630.02

# --- PANEL PRINCIPAL (UNA SOLA PANTALLA) ---
# Encabezado ejecutivo nativo
st.header("🚗 Rusty Bargain — Tasador Inteligente de Autos Usados")
st.caption(  # Subtítulo técnico con el volumen de datos y enlace al reporte en Quarto
    "Inferencia en vivo con **LightGBM** entrenado sobre 305,018 registros limpios"
    " | Reporte completo en Quarto: andagui-ds.github.io/rusty-bargain-car-price"
)

m1, m2, m3, m4 = st.columns(4)  # Fila superior con 4 tarjetas KPI nativas
# Precio predicho en tiempo real
m1.metric("💰 Valor Estimado de Mercado", f"{precio_estimado:,.0f} €")
m2.metric(  # Banda de confianza estimada usando ±1 RMSE
    "📊 Rango Esperado (±1 RMSE)",
    f"{max(100, precio_estimado - rmse_ref):,.0f} –"
    f" {precio_estimado + rmse_ref:,.0f} €",
)
# Calidad del algoritmo seleccionado
m3.metric("🏆 Modelo Activo", "LightGBM", "RMSE: 1,630.02 €")
m4.metric("⚡ Latencia de Inferencia", "0.15 s",
          "26x más rápido que CatBoost")  # Velocidad de respuesta

st.divider()  # Línea separadora nativa entre los KPIs y las gráficas

# Divide el área inferior en dos paneles de visualización
col_izq, col_der = st.columns([1.2, 1])

with col_izq:  # Panel izquierdo con la comparativa de modelos evaluados en el proyecto
    # Título de la gráfica de barras
    st.subheader("📈 Benchmarking: Calidad (RMSE) vs. Velocidad")
    df_bench = pd.DataFrame({  # Resultados exactos obtenidos en el notebook de Rusty Bargain
        "Modelo": [
            "LightGBM 🏆",
            "Bosque Aleatorio",
            "CatBoost",
            "XGBoost",
            "Árbol de Decisión",
            "Regresión Lineal",
        ],
        "RMSE (€)": [1630.02, 1634.05, 1674.79, 1720.40, 1911.18, 2950.17],
        "Entrenamiento (s)": [2.54, 2.85, 39.82, 5.54, 0.86, 0.11],
    })
    fig = px.bar(  # Crea un gráfico de barras horizontales ordenado de mayor a menor error
        df_bench.sort_values("RMSE (€)", ascending=False),
        x="RMSE (€)",
        y="Modelo",
        orientation="h",
        text="RMSE (€)",
        color="RMSE (€)",
        color_continuous_scale="Blues_r",
    )
    # Etiqueta el valor en euros
    fig.update_traces(texttemplate="%{text:,.0f} €", textposition="outside")
    fig.update_layout(  # Ajusta altura y márgenes compactos para evitar scroll
        height=310,
        margin=dict(l=10, r=40, t=10, b=10),
        coloraxis_showscale=False,
    )
    # Renderiza la gráfica responsiva
    st.plotly_chart(fig, use_container_width=True)

with col_der:  # Panel derecho con la curva de depreciación simulada por kilometraje
    # Título de la gráfica de líneas
    st.subheader("📉 Depreciación Simulada por Kilometraje")
    # Puntos de kilometraje a simular
    kms = [20000, 50000, 80000, 100000, 125000, 150000]
    # Replica 6 veces el auto elegido
    sim_df = pd.concat([input_encoded] * len(kms), ignore_index=True)
    sim_df["Mileage"] = kms  # Asigna cada nivel de kilometraje a una fila
    # Predice el valor de mercado en cada tramo de kilometraje
    precios_sim = model.predict(sim_df)
    # Prepara tabla para Plotly
    df_curva = pd.DataFrame(
        {"Kilometraje (km)": kms, "Valor (€)": precios_sim})

    fig2 = px.line(  # Dibuja la curva de evolución de precio según el uso
        df_curva,
        x="Kilometraje (km)",
        y="Valor (€)",
        markers=True,
    )
    fig2.add_scatter(  # Añade un rombo rojo marcando el kilometraje y precio del auto actual
        x=[mileage],
        y=[precio_estimado],
        mode="markers",
        marker=dict(size=12, color="red", symbol="diamond"),
        name="Auto Actual",
    )
    fig2.update_layout(  # Fija escala desde 0 para evitar zoom exagerado y mantiene altura compacta
        height=310,
        margin=dict(l=10, r=20, t=10, b=10),
        yaxis_range=[0, max(max(precios_sim), precio_estimado) * 1.25],
        showlegend=False,
    )
    # Renderiza la gráfica responsiva
    st.plotly_chart(fig2, use_container_width=True)
