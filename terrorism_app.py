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
        "provstate",
        "targtype1_txt",
        "attacktype1_txt",
        "gname",
        "nkill",
        "nwound",
        ]

    df = pd.read_csv(filename,encoding="latin-1")  
    df = df[keep_columns]
    return df

def world_line_attacks_over_time(df):

    data = df["iyear"].value_counts().to_frame().reset_index()
    data.columns = ["Year", "Number of Attacks"]
    data.sort_values(by = "Year", inplace = True)

    country_trace = go.Scatter(
        x = data["Year"],
        y = data["Number of Attacks"],
        mode = "lines+markers",
        marker = dict(color="red")
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

def region_line_attacks_over_time(df,country):

    country_filter = df["country_txt"] == country
    filtered = df[country_filter]
    region = filtered.iloc[0]['region_txt']
    region_filter = df["region_txt"] == region
    filtered = df[region_filter]
    data = filtered["iyear"].value_counts().to_frame().reset_index()
    data.columns = ["Year", "Number of Attacks"]
    data.sort_values(by = "Year", inplace = True)

    country_trace = go.Scatter(
        x = data["Year"],
        y = data["Number of Attacks"],
        mode = "lines+markers",
        marker = dict(color="red")
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

    return fig, region



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
        marker = dict(color="red")
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
        thresh = bad_cities["Number of Attacks"].iloc[5]
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
        hole = 0.3,
        showlegend=False
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

def pie_most_attacked_targets (df,country):
    country_filter = df["country_txt"] == country
    filtered = df[country_filter]

    attacked_targets = filtered["targtype1_txt"].value_counts().to_frame().reset_index()
    attacked_targets.columns = ["Target", "Number of Attacks"]
    total_attacks = attacked_targets["Number of Attacks"].sum()

    try:
        thresh = attacked_targets["Number of Attacks"].iloc[6]
    except:
        thresh = 0

    main_target = attacked_targets[attacked_targets["Number of Attacks"]> thresh]
    other = pd.Series({"Target":"Other",
    "Number of Attacks":total_attacks - attacked_targets["Number of Attacks"].sum()})

    if other["Number of Attacks"] !=0:
        main_target= main_target.append(other, ignore_index=True)

    Data = dict(
        values = main_target["Number of Attacks"],
        labels = main_target["Target"],
        type = "pie",
        hole = 0.3,
        showlegend=False
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

def pie_most_freq_type_attack (df,country):
    country_filter = df["country_txt"] == country
    filtered = df[country_filter]

    attack_type = filtered["attacktype1_txt"].value_counts().to_frame().reset_index()
    attack_type.columns = ["Attack Type", "Number of Attacks"]
    total_attacks = attack_type["Number of Attacks"].sum()

    try:
        thresh = attack_type["Number of Attacks"].iloc[5]
    except:
        thresh = 0

    main_type_attack = attack_type[attack_type["Number of Attacks"]> thresh]
    other = pd.Series({"Attack Type":"Other",
    "Number of Attacks":total_attacks - attack_type["Number of Attacks"].sum()})

    if other["Number of Attacks"] !=0:
        main_type_attack= main_type_attack.append(other, ignore_index=True)

    Data = dict(
        values = main_type_attack["Number of Attacks"],
        labels = main_type_attack["Attack Type"],
        type = "pie",
        hole = 0.3,
        showlegend=False
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

def pie_most_active_groups (df,country):
    country_filter = df["country_txt"] == country
    filtered = df[country_filter]

    terror_groups = filtered["gname"].value_counts().to_frame().reset_index()
    terror_groups.columns = ["Terrorist Group", "Number of Attacks"]
    total_attacks = terror_groups["Number of Attacks"].sum()

    try:
        thresh = terror_groups["Number of Attacks"].iloc[5]
    except:
        thresh = 0

    main_terror_groups = terror_groups[terror_groups["Number of Attacks"]> thresh]
    other = pd.Series({"Terrorist Group":"Other",
    "Number of Attacks":total_attacks - terror_groups["Number of Attacks"].sum()})

    if other["Number of Attacks"] !=0:
        main_terror_groups= main_terror_groups.append(other, ignore_index=True)

    Data = dict(
        values = main_terror_groups["Number of Attacks"],
        labels = main_terror_groups["Terrorist Group"],
        type = "pie",
        hole = 0.3,
        showlegend=False
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

st.title(f"Terrorism Brief: {country}")

col_info, col_viz = st.beta_columns ([1,1.8])

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

isyear=df["iyear"].unique().tolist()
isyear.insert(0,"All history")

with col_viz:
    plot_type = st.selectbox("Choose a visualization:", options = ["Histogram: Attacks Over Time", "Pie: Most Dangerous Cities","Pie: Most Attacked Targets","Pie: Most Frequent Type of Attack", "Pie: Main Terrorist Groups"])

    if plot_type == "Histogram: Attacks Over Time":
        st.plotly_chart(hist_attacks_over_time(df,country),width=300 , height=400, margin=dict(l=0, r=0, b=0, t=0),autosize=False,)

    elif plot_type == "Pie: Most Dangerous Cities":
        st.plotly_chart(pie_most_dangerous_cities(df,country), width=300 , height=400, margin=dict(l=0, r=0, b=0, t=0),autosize=False,)
        worst_cities = df.loc[is_country,"city"].value_counts().head(5).rename_axis('City').reset_index(name='Number of Attacks')
        st.subheader("Most-Dangerous Cities:")
        st.dataframe(worst_cities)

    elif plot_type == "Pie: Most Attacked Targets":
        st.plotly_chart(pie_most_attacked_targets(df,country),width=300 , height=400, margin=dict(l=0, r=0, b=0, t=0),autosize=False,)
        worst_targets = df.loc[is_country,"targtype1_txt"].value_counts().head(5).rename_axis('Target Type').reset_index(name='Number of Attacks')
        st.subheader("Most Attacked Targets: ")
        st.dataframe(worst_targets)

    elif plot_type == "Pie: Most Frequent Type of Attack":
        st.plotly_chart(pie_most_freq_type_attack(df,country),width=300 , height=400, margin=dict(l=0, r=0, b=0, t=0),autosize=False,)
        freq_attack = df.loc[is_country,"attacktype1_txt"].value_counts().head(5).rename_axis('Attack Type').reset_index(name='Number of Attacks')
        st.subheader("Most frequent type of attack: ")
        st.dataframe(freq_attack)

    elif plot_type == "Pie: Main Terrorist Groups":
        st.plotly_chart(pie_most_active_groups(df,country),width=300 , height=400, margin=dict(l=0, r=0, b=0, t=0),autosize=False,)
        worst_groups = df.loc[is_country,"gname"].value_counts().head(5).rename_axis('Group Name').reset_index(name='Number of Attacks')
        st.subheader("Most Active Terrorist Groups: ")
        st.dataframe(worst_groups)

st.subheader(f"{country}: Nationwide Attacks over Time")
st.markdown(f"The following chart represents the total attacks per year from 1970 till 2017 in {country}")
st.plotly_chart(line_attacks_over_time(df,country))


figure, region1 = region_line_attacks_over_time(df,country)
st.subheader(f"{region1}: Attacks over Time")
st.markdown(f"The following chart represents the total attacks per year from 1970 till 2017 accross: {region1} region, the goal is to have as a reference the Terrorist activity in the region to see if {country} have a terrorist activity inusual regarding the region, if is a local problem, or a regional problem. ")
st.plotly_chart(figure)


st.subheader("Worldwide Attacks over Time")
st.markdown(f"The following chart represents the total attacks per year from 1970 till 2017 Worldwide")
st.plotly_chart(world_line_attacks_over_time(df))



st.header(f"Exploring data: {country}")

y_min =min(df["iyear"])
y_max =max(df["iyear"])
y1,y2 = st.sidebar.slider("Choose a range:", y_min, y_max, (y_min, y_max))
in_year_range = df["iyear"].isin(range(y1,y2+1))

st.subheader(f"{country}: Map representation with all_attacks between ({y1} - {y2})")
st.markdown(f"With regards to the period choosen, in the map is represented the attacks distributed accross the territory:")


map_data = df.dropna(axis=0, subset=["latitude", "longitude"])
in_lat = (df["latitude"] >=-90) & (df["latitude"] <=90)
in_lon = (df["longitude"] >=-180) & (df["longitude"]<=180)
map_data=map_data[in_lat & in_lon & is_country & in_year_range]

st.map(map_data)
df_filtered = df[in_year_range & is_country]
worst_groups_filtered = df_filtered.groupby(["gname"]).agg(Total_attacks=("nkill","count"), Total_Killed=("nkill","sum"), Total_Wound=("nwound","sum")).reset_index()
st.subheader(f"Actived Terrorist Groups between {y1} and {y2}: ")
st.markdown(f"With regards to the period choosen, in the following table shows the Actived Terrorist groups in {country} between {y1} and {y2}, with the total attacks committed, number of people killed and wound.")
st.dataframe(worst_groups_filtered)

worst_groups_filtered = df_filtered.groupby(["attacktype1_txt"]).agg(Total_attacks=("nkill","count"), Total_Killed=("nkill","sum"), Total_Wound=("nwound","sum")).reset_index()
st.subheader(f"Terrorist Attacks type used between {y1} and {y2}: ")
st.markdown(f"With regards to the period choosen, in the following table shows the type of terroris attacks in {country} between {y1} and {y2}, with the total attacks committed, number of people killed and wound.")
st.dataframe(worst_groups_filtered)
