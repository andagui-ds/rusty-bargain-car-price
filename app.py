import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Rusty Bargain | Tasador ML",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifact():
    return joblib.load("modelo_rusty.joblib")


artefacto = load_artifact()
model = artefacto["model"]
encoder = artefacto["encoder"]
features = artefacto["features"]
cat_cols = artefacto["cat_cols"]
categorias = artefacto["categorias"]

# --- BARRA LATERAL: ESPECIFICACIONES DEL VEHÍCULO ---
with st.sidebar:
    st.header("🚘 Especificaciones del Auto")
    brand = st.selectbox(
        "Marca (Brand)",
        categorias["Brand"],
        index=categorias["Brand"].index("volkswagen")
        if "volkswagen" in categorias["Brand"]
        else 0,
    )
    model_car = st.selectbox(
        "Modelo (Model)",
        categorias["Model"],
        index=categorias["Model"].index("golf")
        if "golf" in categorias["Model"]
        else 0,
    )
    vehicle_type = st.selectbox(
        "Carrocería (VehicleType)", categorias["VehicleType"]
    )

    c1, c2 = st.columns(2)
    with c1:
        reg_year = st.number_input("Año", 1980, 2019, 2012, step=1)
    with c2:
        reg_month = st.number_input("Mes", 0, 12, 6, step=1)

    c3, c4 = st.columns(2)
    with c3:
        power = st.number_input("Potencia (CV)", 30, 600, 120, step=5)
    with c4:
        mileage = st.number_input(
            "Kilometraje", 5000, 150000, 90000, step=5000
        )

    gearbox = st.selectbox("Transmisión (Gearbox)", categorias["Gearbox"])
    fuel_type = st.selectbox("Combustible (FuelType)", categorias["FuelType"])
    not_repaired = st.selectbox(
        "¿Sin reparar? (NotRepaired)", categorias["NotRepaired"]
    )

# --- PREDICCIÓN EN TIEMPO REAL ---
input_df = pd.DataFrame([{
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
}])[features]

input_encoded = input_df.copy()
input_encoded[cat_cols] = encoder.transform(input_df[cat_cols])
for col in cat_cols:
    input_encoded[col] = input_encoded[col].astype("category")

precio_estimado = float(model.predict(input_encoded)[0])
rmse_ref = 1630.02

# --- PANEL PRINCIPAL (UNA SOLA PANTALLA) ---
st.title("🚗 Rusty Bargain — Tasador Inteligente de Autos Usados")
st.caption(
    "Inferencia en vivo con **LightGBM** entrenado sobre 305,018 registros limpios"
    " | Reporte completo en Quarto: andagui-ds.github.io/rusty-bargain-car-price"
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("💰 Valor Estimado de Mercado", f"{precio_estimado:,.0f} €")
m2.metric(
    "📊 Rango Esperado (±1 RMSE)",
    f"{max(100, precio_estimado - rmse_ref):,.0f} –"
    f" {precio_estimado + rmse_ref:,.0f} €",
)
m3.metric("🏆 Modelo Activo", "LightGBM", "RMSE: 1,630.02 €")
m4.metric("⚡ Latencia de Inferencia", "0.15 s", "26x más rápido que CatBoost")

st.divider()

col_izq, col_der = st.columns([1.2, 1])

with col_izq:
    st.subheader("📈 Benchmarking: Calidad (RMSE) vs. Velocidad")
    df_bench = pd.DataFrame({
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
    fig = px.bar(
        df_bench.sort_values("RMSE (€)", ascending=False),
        x="RMSE (€)",
        y="Modelo",
        orientation="h",
        text="RMSE (€)",
        color="RMSE (€)",
        color_continuous_scale="Blues_r",
    )
    fig.update_traces(texttemplate="%{text:,.0f} €", textposition="outside")
    fig.update_layout(
        height=310,
        margin=dict(l=10, r=40, t=10, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_der:
    st.subheader("📉 Depreciación Simulada por Kilometraje")
    kms = [20000, 50000, 80000, 100000, 125000, 150000]
    sim_df = pd.concat([input_encoded] * len(kms), ignore_index=True)
    sim_df["Mileage"] = kms
    precios_sim = model.predict(sim_df)
    df_curva = pd.DataFrame(
        {"Kilometraje (km)": kms, "Valor (€)": precios_sim})
    fig2 = px.line(
        df_curva,
        x="Kilometraje (km)",
        y="Valor (€)",
        markers=True,
    )
    fig2.update_layout(height=310, margin=dict(l=10, r=20, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)
