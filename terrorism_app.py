import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objs as go
from plotly.offline import iplot

@st.cache
def load_data(filename):
    keep_columns =[
        "iyear", 
        "imonth", 
        "iday", 
        "country_txt",
        "city",
        "region_txt",
        "latitude",
        "longitude", 
        "provstate"
        ]

    df = pd.read_csv(filename,encoding="latin-1")  
    df = df[keep_columns]
    return df

def line_attacks_over_time(df,country):

    country_filter = df["country_txt"] == country
    filtered = df[country_filter]
    data = filtered["iyear"].value_counts().to_frame().reset_index()
    data.columns = ["Year", "Number of Attacks"]
    data.sort_values(by = "Year", inplace = True)

    country_trace = go.Scatter(
        x = data["Year"],
        y = data["Number of Attacks"],
        mode = "lines+markers",
        marker = dict(color="#F2A154")
    )

    Layout = dict(
        margin=dict(l=0,r=0,b=0,t=0),
        width = 800,
        height = 200,
    )
    
    fig = go.Figure(
        data= country_trace,
        layout=go.Layout( Layout)
    )

    return fig

def hist_attacks_over_time(df,country):
    country_filter=df["country_txt"]==country
    filtered = df[country_filter]

    country_trace = go.Histogram(
        x = filtered["iyear"],
        marker = dict(color = "#F2A154")
    )

    Layout = dict(
        margin=dict(l=0,r=0,b=0,t=0),
        width = 500,
        height = 250,
    )
    
    fig = go.Figure(
        data= country_trace,
        layout=go.Layout( Layout)
    )

    return fig

def pie_most_dangerous_cities (df,country):
    country_filter = df["country_txt"] == country
    filtered = df[country_filter]

    bad_cities = filtered["city"].value_counts().to_frame().reset_index()
    bad_cities.columns = ["City", "Number of Attacks"]
    total_attacks = bad_cities["Number of Attacks"].sum()
    try:
        thresh = bad_cities["Number of Attacks"].iloc[9]
    except:
        thresh = 0

    big_bad_cities = bad_cities[bad_cities["Number of Attacks"]> thresh]
    other = pd.Series({"City":"Other",
    "Number of Attacks":total_attacks - big_bad_cities["Number of Attacks"].sum()})

    if other["Number of Attacks"] !=0:
        big_bad_cities=big_bad_cities.append(other, ignore_index=True)

    Data = dict(
        values = big_bad_cities["Number of Attacks"],
        labels = big_bad_cities["City"],
        type = "pie",
        hole = 0.3
    )
    Layout = dict(
        margin=dict(l=0,r=0,b=0,t=0),
        width = 400,
        height = 400,
    )
    
    fig = go.Figure(
        data= Data,
        layout=go.Layout( Layout)
    )


    return fig

df = load_data("Terrorism_clean_dataset.csv")

st.title("Global Terrorism Exploration APP")

#SideBar Region Selector
regions = df["region_txt"].unique().tolist()
regions.insert(0,"All Regions")
region = st.sidebar.selectbox("Choose a Region",options =  regions)
if region == "All Regions":
    in_region = df["region_txt"] == df["region_txt"]
else:
    in_region = df["region_txt"] == region

#SIDEBAR COUNTRY SELECTOR
countries = df.loc[in_region, "country_txt"].unique()
country= st.sidebar.selectbox("Choose a Country", options = countries)

is_country = df["country_txt"] == country

st.title(f"Terrorism Brief:{country}")

col_info, col_viz = st.beta_columns ([1,1.6])

with col_info:
    #Most Attacked City
    worst_city = df.loc[is_country,"city"].value_counts().head(1).index[0]
    city_in_state = df[df["city"] == worst_city].iloc[0]["provstate"]
    st.subheader("Most-Attacked City:")
    if type(city_in_state) != float:
        st.markdown(f"*{worst_city}, {city_in_state}*")
    else:
        st.markdown(f"*{worst_city}*")

    #Most Attacked Year
    worst_year = df.loc[is_country,"iyear"].value_counts().head(1).index[0]
    num_attacks_worst_year = df.loc[is_country,"iyear"].value_counts().iloc[0]
    st.subheader("Year of Most Attacks: ")
    st.markdown(f"*{worst_year} with {num_attacks_worst_year} attacks*")

    #Best Year
    best_year = df.loc[is_country,"iyear"].value_counts().tail(1).index[0]
    num_attacks_best_year = df.loc[is_country,"iyear"].value_counts().tail(1).values[0]
    st.subheader("Year of Fewest Attacks:")
    st.markdown(f"*{best_year} with {num_attacks_best_year} attacks*")

with col_viz:
    plot_type = st.selectbox("Choose a visualization:", options = ["Histogram: Attacks Over Time", "Pie: Most Dangerous Cities"])

    if plot_type == "Histogram: Attacks Over Time":
        st.plotly_chart(hist_attacks_over_time(df,country),width=300 , height=400, margin=dict(l=0, r=0, b=0, t=0),autosize=False,)
    elif plot_type == "Pie: Most Dangerous Cities":
        st.plotly_chart(pie_most_dangerous_cities(df,country), width=300 , height=400, margin=dict(l=0, r=0, b=0, t=0),autosize=False,)

st.subheader(f"{country}: Nationwide Attacks over Time")
st.plotly_chart(line_attacks_over_time(df,country))

map_data = df.dropna(axis=0, subset=["latitude", "longitude"])
in_lat = (df["latitude"] >=-90) & (df["latitude"] <=90)
in_lon = (df["longitude"] >=-180) & (df["longitude"]<=180)
map_data=map_data[in_lat & in_lon & is_country]
st.subheader(f"{country}: Map representation with all_attacks (1970-2017)")
st.map(map_data)