import pandas as pd
import streamlit as st

@st.cache
def load_data(filename):
    keep_columns =["iyear", "imonth", "iday", "country_txt", "region_txt","latitude","longitude"]
    df = pd.read_csv(filename,encoding="latin-1")  
    df = df[keep_columns]
    return df

df = load_data("global_terror.csv")
st.title("Global Terrorism Exploration APP")

#Create slider filter for panel exploration
col_filters, col_viz = st.beta_columns([1,3])
with col_filters:
    st.slider("Year", 0, 100, 50)
with col_viz:
    st.slider("Test", 0, 100, 50)

#Test purposes
st.write(df)
