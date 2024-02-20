import os
import streamlit as st  # web development
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # Dashboard Charts


st.set_page_config(
    page_title="Sous Chef Data Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

st.title(":bar_chart: Sous Chef Data Dashboard")

#Loading Data into Frame
#
df = pd.read_parquet("../data/processed/final_recipe_sample.parquet")


placeholder = st.empty()
avg_ingredients = np.mean(df['n_ingredients'])
avg_time = np.mean(df['minutes'])

with placeholder.container():
    # Summary Metrics
    kpi1, kpi2 = st.columns(2)

    kpi1.metric(label="Avg Ingredients", value=round(avg_ingredients))
    kpi2.metric(label="Avg Time", value=round(avg_time))

df





