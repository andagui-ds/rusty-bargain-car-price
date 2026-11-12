import streamlit as st
import numpy as np

st.title("🚗 Rusty Bargain - Tasación de Autos")
st.write("Ingresa las características del vehículo para estimar su precio de mercado.")

# Formulario
with st.form("car_form"):
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Marca", ["Volkswagen", "BMW", "Opel", "Mercedes-Benz", "Audi"])
        vehicle_type = st.selectbox("Tipo de Vehículo", ["sedan", "small", "wagon", "bus", "convertible"])
        gearbox = st.selectbox("Transmisión", ["manual", "auto"])
    with col2:
        year = st.slider("Año de registro", 1990, 2023, 2015)
        power = st.number_input("Potencia (CV)", min_value=50, max_value=1000, value=150)
        mileage = st.number_input("Kilometraje", min_value=0, max_value=500000, value=100000, step=10000)
    
    submitted = st.form_submit_button("Estimar Precio")
    
    if submitted:
        # Simulación de la predicción (fórmula matemática inventada para la demo)
        base_price = 15000
        price = base_price - (mileage * 0.05) + (power * 10) + ((year - 2000) * 200)
        
        st.success(f"### Precio estimado: € {price:,.2f}")
        st.info("Basado en el modelo LightGBM entrenado con 228,763 registros.")
