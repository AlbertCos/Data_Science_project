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


# TODO: Add "all regions option"****

regions = df["region_txt"].unique().tolist()
regions.insert(0,"All Regions")
# st.write(type(regions))
# st.write(regions)

region = st.selectbox("Choose a Region", regions)

#Create sliders to filter for panel exploration
col_filters, col_viz = st.beta_columns([1,3])
with col_filters:
    y_min = min(df["iyear"])
    y_max = max(df["iyear"])

    y1,y2 = st.slider("Choose a Year:", y_min, y_max, (y_min,y_max))

    in_region = df["region_txt"] == df["region_txt"]
    if region != "All Regions":
        in_region=df["region_txt"] == region


    #TODO: ADD "all countries option" ****
    countries = df.loc[in_region, "country_txt"].unique().tolist()
    countries.insert(0,"All countries") 

    country= st.selectbox("Choose a Country", options = countries)
    is_country = df["country_txt"] == country
    in_year_range = df["iyear"].isin(range(y1,y2+1))

with col_viz:
    chart_type = st.selectbox("Select type of viz: ", options = ["Line", "Histogram","Map"])

    if chart_type == "Line":
        filtered = df.copy()
        filtered = filtered[is_country & in_year_range]
        filtered.index = filtered["iyear"] 
        line_data = filtered["iyear"].value_counts().to_frame().reset_index()
        line_data.columns = ["year","count"]
        line_data = line_data.sort_values(by = "year")
        line_data.index = line_data["year"]
        line_data.drop(columns = ["year"],inplace = True)
 
        st.line_chart(line_data, use_container_width = True)

    elif chart_type == "Map":
        st.write("There is a type error for now, to correct")
        st.map(df)
    elif chart_type =="Histogram":
        filtered = df.copy()
        filtered = filtered[is_country & in_year_range]
        filtered.index = filtered["iyear"] 
        line_data = filtered["iyear"].value_counts().to_frame().reset_index()
        line_data.columns = ["year","count"]
        line_data = line_data.sort_values(by = "year")
        line_data.index = line_data["year"]
        line_data.drop(columns = ["year"],inplace = True)
 
        st.bar_chart(line_data, use_container_width = True)


#filtereding year slider choosed
st.subheader(f"You have selected year: {y1} and {y2} and {country}")

#filtered purposes
st.write(df)
