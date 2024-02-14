import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load a data frame
recipes = pd.read_parquet("../data/processed/final_recipe_sample.parquet")

#Split the nutrition facts into individual columns
recipes[['calories','total_fat','sugar','sodium','protein','sat_fat','carbs']] = \
              recipes['nutrition'].str.split(',',n=7, expand=True)
recipes['calories'] = recipes['calories'].map(lambda x: x.lstrip('['))
recipes['carbs'] = recipes['carbs'].map(lambda x: x.rstrip(']'))
recipes[['calories','total_fat','sugar','sodium','protein','sat_fat','carbs']] = \
recipes[['calories','total_fat','sugar','sodium','protein','sat_fat','carbs']].apply(pd.to_numeric)
#Remove unecessary columns
recipes = recipes.drop(columns=['contributor_id','submitted','nutrition'])



st.dataframe(recipes, use_container_width=True)