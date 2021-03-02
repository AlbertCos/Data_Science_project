# Data Science project: Terrorism exploration and Success prediction app

![](https://github.com/AlbertCos/Data_Science_project/blob/master/dataset-cover.png)

## Description
Terrorism exploration and Success prediction app, give to the users the posibility to explore the Global Terrorism database from 1970 to 2019 through a webapp developed using Streamlit, exploring by country and choosing a range of time. Also, gives the posibility to the user to predict if a terrorist attack would be successful or not. For the prediction, the app uses Random Forest Classification algorithm, with accuracy of 92%.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Codebook Documentation of the Database: https://www.start.umd.edu/gtd/downloads/Codebook.pdf**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


**Success of a terrorist strike** is defined according to the tangible effects of the attack. **Success is not judged in terms of the larger goals of the perpetrators.** For example, a bomb that exploded in a building would be counted as a success even if it did not succeed in bringing the building down or inducing government repression.
The definition of a successful attack depends on the type of attack. Essentially, the key question is whether or not the attack type took place. If a case has multiple attack types, it is successful if any of the attack types are successful, with the exception of assassinations, which are only successful if the intended target is killed.
    1 = "Yes" The incident was successful.
    0 = "No" The incident was not successful
   
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Link to the app: https://terrorismapp.herokuapp.com/

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Features

### **Explore by country**
In the sidebar, the user could choose a country to explore the terrorist activity during the period of time chose.  The first part, shows a brief description about the city most dangerous according to the terrorist activity from 1970 till 2019 for the country selected, the targets most attacked, the most important terrorist group, the worst year in terms of attacks, the worst year in terms of number of people killed and 4 differents pie charts to choose.

![](https://github.com/AlbertCos/Data_Science_project/blob/master/moviegif4.gif)


### **Explore by range of years**
In the second part, we could explore selecting by range of years in the sidebar. We can see the attacks in the map for the period selected, the terrorist group active in the country during that period, terrorist attacks types and terrorist attack targets.

![](https://github.com/AlbertCos/Data_Science_project/blob/master/project2gif.gif)


### **Machine Learning: Random Forest Clasification for Success prediction**

With an accuracy of 91.82 %, the user could predict if the terrorist attack could be successful or not, by imputing the variables: City, month, day, type of attack, target, Weapon, target nationality, people killed, specificity, vicinity, extended attack, Suicide attack.

![](https://github.com/AlbertCos/Data_Science_project/blob/master/project3gif.gif)


## Installation
In the document "requirements.txt" you can find the packages installed to run the app.
The webapp is using streamlit and deployed using Heroku.

## Supporting documentation:
To analyse the procedure of data preprocessing, data cleaning, feature selection and conclusions, find the link to the Notebook below:
https://github.com/AlbertCos/Data_Science_project/blob/master/Global_Terrorism_exploration_and_prediction_app.ipynb


## Database
This application uses the database: Global Terrorism Database, more information here: https://www.start.umd.edu/data-tools/global-terrorism-database-gtd
