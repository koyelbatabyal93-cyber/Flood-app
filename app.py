import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import 
RandomForestClassifier
import folium
from streamlit_folium import st_folium
st.set_page_config(
    page_title="Urban Flood Nowcasting System",
    page_icon="🌧️",
    layout="wide"
)

st.title("🌧️ Urban Flood Nowcasting System")
st.subheader("Drainage and Rainfall Coupling")

@st.cache_resource
def load_and_train_model():
    data = {
        "rainfall": [10,20,30,40,50,60,70,80,90,100,15,35,45,55,75,95],
        "drainage_capacity": [80,80,70,70,60,60,50,50,40,40,90,70,65,55,45,35],
        "water_level": [10,15,20,25,30,35,45,50,60,70,12,22,28,38,48,65],
        "elevation": [20,18,15,14,12,10,8,7,5,3,25,16,13,11,9,4],
        "flood": [0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1]
    }
    df = pd.DataFrame(data)
    X = df[["rainfall","drainage_capacity","water_level","elevation"]]
    y = df["flood"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_and_train_model()

st.sidebar.header("📊 Current Conditions")
rainfall = st.sidebar.slider("Rainfall (mm/hour)", 0, 150, 60)
drainage = st.sidebar.slider("Drainage Capacity (mm/hour)", 10, 150, 50)
water_level = st.sidebar.slider("Drain Water Level (%)", 0, 100, 40)
elevation = st.sidebar.slider("Area Elevation (m)", 0, 50, 10)

input_data = pd.DataFrame({
    "rainfall": [rainfall],
    "drainage_capacity": [drainage],
    "water_level": [water_level],
    "elevation": [elevation]
})

probability = model.predict_proba(input_data)[0][1]

if probability < 0.35:
    risk_label = "🟢 LOW"
elif probability < 0.65:
    risk_label = "🟡 MEDIUM"
else:
    risk_label = "🔴 HIGH"

st.header("🚨 Flood Risk Prediction")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rainfall", f"{rainfall} mm/hr")
col2.metric("Drainage Capacity", f"{drainage} mm/hr")
col3.metric("Water Level", f"{water_level}%")
col4.metric("Flood Probability", f"{probability*100:.1f}%")

st.divider()
st.subheader(f"Predicted Risk: {risk_label}")

if probability >= 0.65:
    st.error("⚠️ HIGH FLOOD RISK: Heavy rainfall and drainage limitations may cause water accumulation.")
elif probability >= 0.35:
    st.warning("⚠️ MEDIUM FLOOD RISK: Monitor rainfall and drainage conditions.")
else:
    st.success("✅ LOW FLOOD RISK: Current conditions indicate relatively low risk.")

st.header("🚰 Rainfall vs Drainage")
comparison = pd.DataFrame({
    "Parameter": ["Rainfall", "Drainage Capacity"],
    "Value": [rainfall, drainage]
})
st.bar_chart(comparison.set_index("Parameter"))

st.header("🗺️ Example Flood Risk Zones")

zones = pd.DataFrame({
    "Area": ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E"],
    "Latitude": [22.5726, 22.5750, 22.5680, 22.5800, 22.5650],
    "Longitude": [88.3639, 88.3700, 88.3550, 88.3600, 88.3750],
    "Risk": ["High", "Medium", "Low", "High", "Medium"]
})

m = folium.Map(
    location=[22.5726, 88.3639],
    zoom_start=13
)

risk_colors = {
    "High": "red",
    "Medium": "orange",
    "Low": "green"
}

for _, row in zones.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=10,
        color=risk_colors[row["Risk"]],
        fill=True,
        fill_color=risk_colors[row["Risk"]],
        fill_opacity=0.8,
        popup=f"{row['Area']} - {row['Risk']} Risk"
    ).add_to(m)

st_folium(m, width=1000, height=500)

st.header("💡 Recommended Action")
if probability >= 0.65:
    st.markdown("""
    1. 🚨 Issue an early flood warning.
    2. 🚧 Monitor low-lying roads.
    3. 🚰 Check drainage blockages.
    4. 📡 Monitor water levels continuously.
    """)
elif probability >= 0.35:
    st.markdown("""
    1. 👀 Continue monitoring rainfall.
    2. 🚰 Check drainage systems.
    3. 📊 Update flood prediction regularly.
    """)
else:
    st.markdown("""
    1. ✅ Continue normal monitoring.
    2. 🌧️ Track upcoming rainfall.
    """)

st.caption("Prototype for Urban Flood Nowcasting — Drainage and Rainfall Coupling")
