import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
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

st.header("🗺️ Select Flood Location")

default_lat = 22.5726
default_lon = 88.3639

m = folium.Map(
    location=[default_lat, default_lon],
    zoom_start=13
)

map_data = st_folium(
    m,
    width=1000,
    height=500
)

if map_data and map_data["last_clicked"]:

    selected_lat = map_data["last_clicked"]["lat"]
    selected_lon = map_data["last_clicked"]["lng"]

    st.success("📍 Location Selected")

    st.write(f"**Latitude:** {selected_lat:.6f}")
    st.write(f"**Longitude:** {selected_lon:.6f}")

    st.header("📊 Flood Prediction Result")

    st.write(
        f"📍 **Selected Location:** "
        f"{selected_lat:.6f}, {selected_lon:.6f}"
    )

    st.write(f"🌊 **Flood Risk:** {probability * 100:.2f}%")

    if probability >= 0.65:
        st.error("🚨 HIGH FLOOD RISK")
    elif probability >= 0.35:
        st.warning("⚠️ MEDIUM FLOOD RISK")
    else:
        st.success("✅ LOW FLOOD RISK")

else:
    st.info("👆 Please click on the map to select a location.")

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
