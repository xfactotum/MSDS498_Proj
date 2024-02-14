import streamlit as st

st.title("Nutrition Goals")
Cal_PDV = st.slider("Calories:", min_value=0, max_value=100, value=100)
Fat_PDV = st.slider("Total Fat:", min_value=0, max_value=100, value=100)
Carb_PDV = st.slider("Carbohydrates:", min_value=0, max_value=100, value=100)
Sod_PDV = st.slider("Sodium:", min_value=0, max_value=100, value=100)

if Cal_PDV >= 100:
    Cal_PDV = "MAX"
if Fat_PDV >= 100:
    Fat_PDV = "MAX"
if Carb_PDV >= 100:
    Carb_PDV = "MAX"
if Sod_PDV >= 100:
    Sod_PDV = "MAX"

st.write(f"Calorie Range: 0 - {Cal_PDV}")
st.write(f"Total Fat Range: 0 - {Fat_PDV}")
st.write(f"Carbohydrate Range: 0 - {Carb_PDV}")
st.write(f"Sodium Range: 0 - {Sod_PDV}")