import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import math

st.title("Explore the Recipes!")
st.text("Use the interactive filters below to refine your search.")
my_df = st.empty()

# Load a data frame
recipes = pd.read_parquet("../data/processed/final_recipe_sample.parquet")

#Split the nutrition facts into individual columns
recipes[['Calories','Total_Fat','Sugar','Sodium','Protein','Sat_Fat','Carbs']] = \
              recipes['nutrition'].str.split(',',n=7, expand=True)
recipes['Calories'] = recipes['Calories'].map(lambda x: x.lstrip('['))
recipes['Carbs'] = recipes['Carbs'].map(lambda x: x.rstrip(']'))
recipes[['Calories','Total_Fat','Sugar','Sodium','Protein','Sat_Fat','Carbs']] = \
recipes[['Calories','Total_Fat','Sugar','Sodium','Protein','Sat_Fat','Carbs']].apply(pd.to_numeric)
#Remove unecessary columns
recipes = recipes.drop(columns=['contributor_id','submitted','nutrition'])

#Data Cleaning
recipes = recipes[recipes['minutes'] < 240]

#Histogram of Time to Prep
fig1, ax1 = plt.subplots(figsize=(12,4))
N, bins, patches = ax1.hist(recipes['minutes'], bins=24, color = 'lightgray',
                            edgecolor='white', linewidth=1)
ax1.set_xlabel('Time to Prepare (min)')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram of Time to Prepare Recipe')

with st.expander("Time to Prep"):
    uiTime = st.slider("Time to Prepare (min):", min_value=0, max_value=240, value=240)
    hist_threshold = math.ceil(uiTime/10)
    for i in range(0, hist_threshold):
        patches[i].set_facecolor('orange')
    st.pyplot(fig1)

#Box Plots of Nutrition Values
bp = recipes[['Total_Fat','Sat_Fat','Sodium','Carbs','Sugar','Protein']]
fig2, ax2 = plt.subplots(figsize=(12,6))
ax2 = sns.boxplot(data = bp, palette = "YlOrBr", showfliers = False)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=60,fontsize=15)
ax2.set_ylabel('Percent Daily Value', fontsize = 15)

with st.expander("Nutrition"):
    with st.container():
        st.pyplot(fig2)

    col1, col2 = st.columns(2)
    with col1:
        uiTotal_Fat = st.slider("Total Fat:", min_value=0, max_value=100, value=(0,100))
        uiSat_Fat = st.slider("Saturated Fat:", min_value=0, max_value=100, value=(0, 100))
        uiSodium = st.slider("Sodium:", min_value=0, max_value=100, value=(0, 100))
    with col2:
        uiCarbs = st.slider("Carbohydrates:", min_value=0, max_value=100, value=(0, 100))
        uiSugar = st.slider("Sugar:", min_value=0, max_value=100, value=(0, 100))
        uiProtein = st.slider("Protein:", min_value=0, max_value=100, value=(0, 100))

#Create Dataframe of Ingredients
ingredients =pd.DataFrame({'Ingredient' : recipes['ingredients'].str.split(',').explode(),
                           'Exclude' : False})
#Remove the special characters from the original strings
ingredients['Ingredient'] = ingredients['Ingredient'].map(lambda x: x.lstrip("[' ").rstrip(" ']"))
ingredients['Ingredient'] = ingredients['Ingredient'].map(lambda x: x.lstrip('"').rstrip('"'))
ingredients = ingredients.drop_duplicates(subset=['Ingredient'])
ingredients = ingredients.reset_index(drop=True)

with st.expander("Ingredients"):
    uiIngred = st.data_editor(ingredients, use_container_width=True, hide_index=True,
                              disabled=['Ingredient'])
#Create Dataframe of Tags
tags = pd.DataFrame({'Tag' : recipes['tags'].str.split(',').explode(),
                     'Include' : False})
#Remove the special characters from the original strings
tags['Tag'] = tags['Tag'].map(lambda x: x.lstrip("[' ").rstrip(" ']"))
tags['Tag'] = tags['Tag'].map(lambda x: x.lstrip('"').rstrip('"'))
tags = tags.drop_duplicates(subset=['Tag'])
tags = tags.reset_index(drop=True)

with st.expander("Tags"):
    uiTags = st.data_editor(tags, use_container_width=True, hide_index=True,
                              disabled=['Tag'])


selected = recipes[['name','description','tags','minutes','Calories','Total_Fat','Sat_Fat',
                    'Sodium','Sugar','Carbs','Protein','n_steps','steps','n_ingredients',
                    'ingredients',]]

#Reduce Data based on User's Time to Prep Input
selected = selected[selected['minutes'] <= uiTime]

#Reduce Data based on User's Nutritional Input
selected = selected[selected['Total_Fat'] >= uiTotal_Fat[0]]
selected = selected[selected['Sat_Fat'] >= uiSat_Fat[0]]
selected = selected[selected['Sodium'] >= uiSodium[0]]
selected = selected[selected['Carbs'] >= uiCarbs[0]]
selected = selected[selected['Sugar'] >= uiSugar[0]]
selected = selected[selected['Protein'] >= uiProtein[0]]
if uiTotal_Fat[1] != 100:
    selected = selected[selected['Total_Fat'] <= uiTotal_Fat[1]]
if uiSat_Fat[1] != 100:
    selected = selected[selected['Sat_Fat'] <= uiSat_Fat[1]]
if uiSodium[1] != 100:
    selected = selected[selected['Sodium'] <= uiSodium[1]]
if uiCarbs[1] != 100:
    selected = selected[selected['Carbs'] <= uiCarbs[1]]
if uiSugar[1] != 100:
    selected = selected[selected['Sugar'] <= uiSugar[1]]
if uiProtein[1] != 100:
    selected = selected[selected['Protein'] <= uiProtein[1]]

#Reduce Data based on User's Ingredient Input
uiIngred = uiIngred[uiIngred['Exclude'] == True]
if not uiIngred.empty:
    selected = selected[~selected['ingredients'].str.contains('|'.join(uiIngred['Ingredient']))]

#Reduce Data based on User's Tag Input
uiTags = uiTags[uiTags['Include'] == True]
if not uiTags.empty:
    selected = selected[selected['tags'].str.contains('|'.join(uiTags['Tag']))]

selected = selected.rename(columns={'name':'Recipe_Name',
                                    'description':'Description',
                                    'tags':'Tags',
                                    'minutes':'Time_to_Prep',
                                    'Total_Fat':'Total_Fat_PDV',
                                    'Sat_Fat':'Sat_Fat_PDV',
                                    'Sodium':'Sodium_PDV',
                                    'Sugar':'Sugar_PDV',
                                    'Carbs':'Carbohydrates_PDV',
                                    'Protein':'Protein_PDV',
                                    'n_steps':'Number_of_Steps',
                                    'steps':'Steps',
                                    'n_ingredients':'Number_of_Ingredients',
                                    'ingredients':'Ingredients'})
my_df.dataframe(selected, hide_index = True)