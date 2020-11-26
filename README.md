# How to predict babies?

## Abstract
Birthrate numbers are an important factor regarding the demographic activities within a country. It gives an important insight on the nation’s population’s “health” when defining new policies. These monthly statistics are usually released with a lag of 3 weeks in France. Another particular aspect of births are their seasonality: They are high in summer and low in winter. It would thus be interesting to find google trends indicators that allow us to predict correctly these rates. Another motivation for this study choice is the temporal aspect of google trends: It is well known that parents search  information about their upcoming babies throughout the whole pregnancy period.  We would then also need to study which time is the most interesting to choose our google trends: 3 months prior to birth? 9 6 months? And finally what would the categories be that could help us indicate upcoming newborns?<br>
We would directly use data issued from French authorities.



## Research questions
- What categories are the most useful to predict babies ?
- When is the chosen indicator best suited to predict these time-series?

## Proposed dataset

We will be using the statistical report on births in France released by INSEE (National institute for statistics and economical studies) for the latest month: 
https://www.insee.fr/fr/statistiques/serie/000436391#Tableau


The dataset is already treated and will be downloaded as a .csv file<br>
Each number represents the amount of babies born for the given month.<br>
A clarification is made on babies born in 2020, these numbers are temporary as Covid-19 may have disturbed the birth sampling.<br>
The data is distributed on a monthly format, statistics are released for each month the last week of each momtnh. <br>

Google trend data in switzerland with relevant categories:
https://trends.google.com/trends/explore?geo=CH THOMAS MODIF

We will focus on the period 2004-2020 because it's when google trend start to record data.

## Methods

The method we thought of is the following:

1) Extracting and enhancing our datasets 
2) Applying a classical autoregressive model to predict the time series
3) Search and select (maybe through ‘spike and slab’ regression) various google trends categories
4) Visualize our predictions
5) Focus on recession to answer second research question


## Proposed timeline

The work will be decomposed into 3 parts
- Base predictions and searching for the best indicators
- Apply our google trends model to check for best predictions
- Getting a focus on the best period for our trends

## Organization within the team

One person will focus on google, searching and understanding the best possible indicators that could apply to birth rates.
The second member, will focus on developping the best possible code to have a good prediction using different tools
The last member will focus on the vizualisation aspect of the research projet and time period 

## Questions for TA's
Is it important to select categories through a ‘spike and slab’ regression or can we just select relevant categories such as "Jobs" and "Welfare & Unemployment" ? MODIF THOMAS

