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

region = st.selectbox("Choose a Region", df["region_txt"].unique())

#Create sliders to filter for panel exploration
col_filters, col_viz = st.beta_columns([1,3])
with col_filters:
    y_min = min(df["iyear"])
    y_max = max(df["iyear"])
    year = st.slider("Choose a Year:", y_min, y_max, (y_min,y_max))

    in_region=df["region_txt"] == region
    country= st.selectbox("Choose a Country", options = df.loc[in_region,"country_txt"].unique())


#Testing year slider choosed
st.subheader(f"You have selected year: {year}! and {country}")

#Test purposes
st.write(df)
