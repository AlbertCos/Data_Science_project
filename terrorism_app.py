import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objs as go
from plotly.offline import iplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from geopy.geocoders import Nominatim




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
        "country",
        "region",
        "suicide",
        "attacktype1",
        "targtype1",
        "weaptype1",
        "weaptype1_txt",
        "success",
        "eventid",
        "natlty1",
        "natlty1_txt",
        "extended",
        "specificity",
        "vicinity",
        "crit1",
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
        width = 700,
        height = 250,
        xaxis=dict(title="Year"), 
        yaxis=dict(title="Number of attacks"),
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
        width = 700,
        height = 250,
        xaxis=dict(title="Year"), 
        yaxis=dict(title="Number of attacks"),
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
        width = 700,
        height = 250, 
        xaxis=dict(title="Year"), 
        yaxis=dict(title="Number of attacks"),
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
        margin=dict(l=20,r=0,b=0,t=0),
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
        margin=dict(l=20,r=0,b=0,t=0),
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
        margin=dict(l=20,r=0,b=0,t=0),
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
        margin=dict(l=20,r=0,b=0,t=0),
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

st.markdown("")
st.header(f"Terrorism Brief: {country}")
st.markdown("")
st.markdown("")

col_info, col_viz = st.beta_columns ([1,1.3])

##Brief part column information
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

    #Most Mortal Year
    df_filtered = df[is_country]
    mortal_year_filtered = df_filtered.groupby(["iyear"]).agg(Total_attacks=("nkill","count"), Total_Killed=("nkill","sum"), Total_Wound=("nwound","sum")).sort_values(by=["Total_Killed"],ascending=False).reset_index()
    year_mortal = mortal_year_filtered.iloc[0]["iyear"]
    num_kill_worst_year =  mortal_year_filtered.iloc[0]['Total_Killed']
    st.subheader("Year of Most people killed: ")
    st.markdown(f"*{year_mortal} with {num_kill_worst_year} people killed*")

    #Most Dangerous Terrorist Group
    terrorist_group_filtered = df_filtered.groupby(["gname"]).agg(Total_attacks=("nkill","count"), Total_Killed=("nkill","sum"), Total_Wound=("nwound","sum")).sort_values(by=["Total_Killed"],ascending=False).reset_index()
    worst_group = terrorist_group_filtered.iloc[0]["gname"]
    if worst_group == "Unknown":
        worst_group = terrorist_group_filtered.iloc[1]["gname"]
        num_kill_t_worst_year =  terrorist_group_filtered.iloc[1]['Total_Killed']
    else:
        num_kill_t_worst_year =  terrorist_group_filtered.iloc[0]['Total_Killed']

    st.subheader("The Most dangerous Terrorist Group: ")
    st.markdown(f"**{worst_group}** *with* **{num_kill_t_worst_year} people killed** *in total during all the period (1970-2017)*")


isyear=df["iyear"].unique().tolist()
isyear.insert(0,"All history")

## Brief part, plot selection option
with col_viz:
    plot_type = st.selectbox("Choose a visualization:", options = ["Pie: Most Dangerous Cities","Pie: Most Attacked Targets","Pie: Most Frequent Type of Attack", "Pie: Main Terrorist Groups"])

    if plot_type == "Pie: Most Dangerous Cities":
        st.plotly_chart(pie_most_dangerous_cities(df,country), width=300 , height=400, margin=dict(l=0, r=0, b=0, t=0),autosize=False,)
        worst_cities = df.loc[is_country,"city"].value_counts().head(5).rename_axis('City').reset_index(name='Total Attacks')
        st.subheader("Most-Dangerous Cities:")
        st.dataframe(worst_cities)

    elif plot_type == "Pie: Most Attacked Targets":
        st.plotly_chart(pie_most_attacked_targets(df,country),width=300 , height=400, margin=dict(l=0, r=0, b=0, t=0),autosize=False,)
        worst_targets = df.loc[is_country,"targtype1_txt"].value_counts().head(5).rename_axis('Target Type').reset_index(name='Total Attacks')
        st.subheader("Most Attacked Targets: ")
        st.dataframe(worst_targets)

    elif plot_type == "Pie: Most Frequent Type of Attack":
        st.plotly_chart(pie_most_freq_type_attack(df,country),width=300 , height=400, margin=dict(l=0, r=0, b=0, t=0),autosize=False,)
        freq_attack = df.loc[is_country,"attacktype1_txt"].value_counts().head(5).rename_axis('Attack Type').reset_index(name='Total Attacks')
        st.subheader("Most frequent type of attack: ")
        st.dataframe(freq_attack)

    elif plot_type == "Pie: Main Terrorist Groups":
        st.plotly_chart(pie_most_active_groups(df,country),width=300 , height=400, margin=dict(l=0, r=0, b=0, t=0),autosize=False,)
        worst_groups = df.loc[is_country,"gname"].value_counts().head(5).rename_axis('Group Name').reset_index(name='Total Attacks')
        st.subheader("Most Active Terrorist Groups: ")
        st.dataframe(worst_groups)

st.markdown("")
st.subheader(f"{country}: Nationwide Attacks over Time")
st.markdown(f"The following chart represents the **total attacks per year from 1970 till 2017** in **{country}**")
st.markdown("")
st.plotly_chart(line_attacks_over_time(df,country))


figure, region1 = region_line_attacks_over_time(df,country)
st.markdown("")
st.subheader(f"{region1}: Attacks over Time")
st.markdown(f"The following chart represents the **total attacks per year from 1970 till 2017** accross **{region1} region**, the goal is to have as a reference to compare with, in order to see if {country} have a terrorist activity inusual regarding the region, if is a local problem, or a regional problem. ")
st.markdown("")
st.plotly_chart(figure)

st.markdown("")
st.subheader("Worldwide Attacks over Time")
st.markdown(f"The following chart represents the **total attacks per year from 1970 till 2017 Worldwide**")
st.markdown("")
st.plotly_chart(world_line_attacks_over_time(df))


st.markdown("")
st.markdown("")
st.header(f"Exploring data: {country}")
st.markdown("")
st.markdown("")
st.warning("Please, choose a range of time using the slidebar on the left Menu to explore the data below.")
st.markdown("")

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
terror_groups_filtered = df_filtered.groupby(["gname"]).agg(Total_attacks=("nkill","count"), Total_Killed=("nkill","sum"), Total_Wound=("nwound","sum")).reset_index()
terror_groups_filtered = terror_groups_filtered.rename(columns={"gname":"Group Name"})

###################################################
st.markdown("")
st.markdown("")
st.subheader(f"Terrorist Groups in {country} between {y1} and {y2}: ")
st.markdown(f"With regards to the period choosen, in the following table shows the terrorist groups active in {country} between {y1} and {y2}, showing the total attacks committed, number of people killed and wound by each group.")
st.dataframe(terror_groups_filtered)
st.markdown("")
st.markdown("")

###################################################
attacks_type_filtered = df_filtered.groupby(["attacktype1_txt"]).agg(Total_attacks=("nkill","count"), Total_Killed=("nkill","sum"), Total_Wound=("nwound","sum")).reset_index()
st.subheader(f"Terrorist Attack types in {country} between {y1} and {y2}: ")
st.markdown(f"With regards to the period choosen, in the following table shows the types of terrorist attacks in {country} between {y1} and {y2}, with the total attacks committed, number of people killed and wound by each type")

fig = go.Figure(data=[go.Histogram(x=df_filtered["attacktype1_txt"],marker = dict(color = "red"))], layout= dict (title=f"{country}: Terrorist attacks types  ({y1} - {y2})", xaxis=dict(title="Terrorist attack Types"), yaxis=dict(title="Total number of attacks")))
st.plotly_chart(fig,width=600 , height=400, margin=dict(l=0, r=0, b=0, t=0))
st.markdown("")
st.markdown("")

attacks_type_filtered= attacks_type_filtered.rename(columns={"attacktype1_txt":"Type of Attack"})
st.markdown("")
st.markdown("")
st.markdown("In the table below, the more detailed information:")
st.dataframe(attacks_type_filtered)
st.markdown("")
st.markdown("")

st.subheader(f"Terrorist Attack targets in {country} between {y1} and {y2}: ")
st.markdown(f"With regards to the period choosen, in the following table shows the terrorist attack targets in {country} between {y1} and {y2}, with the total attacks committed, number of people killed and wound by each type")


fig = go.Figure(data=[go.Histogram(x=df_filtered["targtype1_txt"],marker = dict(color = "red"))], layout= dict (title=f"{country}: Terrorist attacks targets ({y1} - {y2})", xaxis=dict(title="Terrorist attack targets"), yaxis=dict(title="Total number of attacks")))
st.plotly_chart(fig,width=600 , height=400, margin=dict(l=0, r=0, b=0, t=0))


attacks_targets_filtered = df_filtered.groupby(["targtype1_txt"]).agg(Total_attacks=("nkill","count"), Total_Killed=("nkill","sum"), Total_Wound=("nwound","sum")).reset_index()
attacks_targets_filtered= attacks_targets_filtered.rename(columns={"targtype1_txt":"Type of Target"})
st.markdown("")
st.markdown("")
st.markdown("In the table below, the more detailed information:")
st.dataframe(attacks_targets_filtered)
st.markdown("")
st.markdown("")

###************************MACHINE LEARNING**********************************###

st.header(f"{country}: Machine Learning, terrorism success attack prediction")
st.markdown("")
st.markdown("")

#Choosing Variables for feed our  model

dfnew = df[["imonth","iday", "success","attacktype1","targtype1","natlty1","weaptype1","nkill","nwound","region","latitude","longitude","specificity","vicinity","extended","suicide"]]

#Creating new variable to substitue nkill by a boolean 
dfnew['lab_kill'] = dfnew['nkill'].apply(lambda x: 1 if x > 0 else 0)
dfnew['lab_wound'] = dfnew['nwound'].apply(lambda x: 1 if x > 0 else 0)

dfnew = dfnew.dropna()
X = dfnew.drop(["nkill","nwound","success"], axis=1, inplace = False)
Y = dfnew["success"]

##Spliting data set in training and test
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

classifier = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)
##Predict Y with the x test
y_pred = classifier.predict(X_test)

## Calculation of accuracy score
ac=str(round(accuracy_score(y_test,y_pred)*100,2))

st.markdown(f"Using for the prediction **Random Forest Clasification Algorithm with accuracy: {ac} %**. Please proceed below, introducing the data:")




##Function to get the imputs for predict
def user_report(df, dfclean, region, country):

    geolocator = Nominatim(user_agent="my_user_agent")

    cities =pd.unique(df[df['country_txt'] == country ]['city'].sort_values()).tolist()
    city = st.selectbox("Choose a City",options =  cities)

    loc = geolocator.geocode(city+','+ country)

    imonthst = dfclean["imonth"].sort_values().unique().tolist()
    imonths = st.selectbox("Choose month:",options = imonthst)

    idayst = dfclean["iday"].sort_values().unique().tolist()
    idays = st.selectbox("Choose day:",options =  idayst)

    Attacktext = pd.unique(df[df['country_txt'] == country ]['attacktype1_txt'].sort_values()).tolist()
    Attacktext = st.selectbox("Choose the Type of Attack",options =  Attacktext)
    attacktype1=df.loc[df['attacktype1_txt'] == Attacktext, 'attacktype1'].iloc[0]

    Targettext = pd.unique(df[df['city'] == city ]['targtype1_txt'].sort_values()).tolist()
    Targettext= st.selectbox("Choose a Target",options =  Targettext)
    targettype1=df.loc[df['targtype1_txt'] == Targettext, 'targtype1'].iloc[0]

    
    Weapntext = pd.unique(df[df['country_txt'] == country ]['weaptype1_txt'].sort_values()).tolist()
    Weapntext = st.selectbox("Choose the main Weapon used in the attack",options =  Weapntext)
    weaptype = df.loc[df['weaptype1_txt'] == Weapntext, 'weaptype1'].iloc[0]
    
    
    nattext = pd.unique(df[df['country_txt'] == country ]['natlty1_txt'].sort_values()).tolist()
    nattext = st.selectbox("Choose the target nationality",options =  nattext)
    natlty1= df.loc[df['natlty1_txt'] == nattext, 'natlty1'].iloc[0]

    
    killed = dfclean["lab_kill"].sort_values().unique().tolist()
    killed = st.selectbox("Choose if there if would be people killed, (0, no killed, 1 people killed)",options =  killed)

    twound = dfclean["lab_wound"].sort_values().unique().tolist()
    twound = st.selectbox("Choose if there if would be people wound, (0, no wound, 1, people wound)",options =  twound)
    
    region = df.loc[df['region_txt'] == region, 'region'].iloc[0]
    latitude = loc.latitude
    longitude = loc.longitude

    Specificity = dfclean["specificity"].sort_values().unique().tolist()
    Specificity = st.selectbox("Choose the Specificity", options =  Specificity)
    specificity = Specificity
    st.markdown("**Specificity:** **1** = event occurred in city/village/town and lat/long is for that location, **2** = event occurred in city/village/town and no lat/long could be found, so coordinates are for centroid of smallest subnational administrative region identified, **3** = event did not occur in city/village/town, so coordinates are for centroid of smallest subnational administrative region identified, **4** = no 2nd order or smaller region could be identified, so coordinates are for center of 1st order administrative region,**5** = no 1st order administative region could be identified for the location of the attack, so latitude and longitude are unknown")

    Vicinity = dfclean["vicinity"].sort_values().unique().tolist()
    Vicinity = st.selectbox("Choose if Vicinity (0 - No, 1 - Yes)", (0,1))
    vicinity = Vicinity

    Extended = dfclean["extended"].sort_values().unique().tolist()
    Extended = st.selectbox("Choose if Extended attack (0 - No, 1 - Yes)", options =  Extended)
    extended = Extended

    Suicide = dfclean["suicide"].sort_values().unique().tolist()
    Suicide = st.selectbox("Choose if Suicide Attack (0 - No, 1 - Yes)", options =  Suicide)
    suicide = Suicide

    user_report = {

        "imonth":imonths,
        "iday":idays,
        "attacktype1":attacktype1,
        "targtype1":targettype1,
        "natlty1":natlty1,
        "weaptype1":weaptype,
        "region": region,
        "latitude": latitude,
        "longitude": longitude,
        "specificity": specificity,
        "vicinity":vicinity,
        "extended": extended,
        "suicide":suicide,
        "lab_kill":killed,
        "lab_wound":twound,
    }

    report_data = pd.DataFrame(user_report, index=[0])
    return report_data

user_data = user_report(df, dfnew, region1,country)

## Predict using the inputs from the user
user_result= classifier.predict(user_data)

## Result prediction


st.subheader("The algorithm predicts that the terrosit attack would:")

output=""
if user_result[0]==0:
    output = "Failed"
else:
    output = "Succeed"
st.header (output)


