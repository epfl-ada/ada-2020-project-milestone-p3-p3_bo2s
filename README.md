# How to predict babies?

## Abstract
Birthrate numbers are an important factor regarding the demographic activities within a country. It gives an important insight on the nation’s population’s “health” when defining new policies. These monthly statistics are usually released with a lag of 3 weeks in France. Another particular aspect of births are their seasonality: They are high in summer and low in winter. It would thus be interesting to find google trends indicators that allow us to predict correctly these rates. Another motivation for this study choice is the temporal aspect of google trends: It is well known that parents search  information about their upcoming babies throughout the whole pregnancy period.  We would then also need to study which time is the most interesting to choose our google trends: 3 months prior to birth? 6 months? 9 months? And finally what would the google trend categories that could help us indicate upcoming newborns be?<br>
We would directly use data issued from French authorities.



## Research questions
- What categories are the most useful to predict babies ?
- When is the chosen indicator best suited to predict these time-series?

## Proposed dataset

We will be using the statistical report on births in France released by INSEE (National institute for statistics and economical studies) for the latest month: 
https://www.insee.fr/fr/statistiques/serie/000436391#Tableau


The dataset is already treated and contains all the past data. It will be downloaded as a .csv file<br>
Each number represents the amount of babies born for the given month.<br>
A clarification is made on babies born in 2020, these numbers are temporary.<br>
The data is distributed on a monthly format, statistics are released for each month 3 weeks after the end of the month. <br>

Google trend data in switzerland with relevant categories:
https://trends.google.com/trends/explore?geo=FR

We will focus on the period 2004-2020 because it's when google trend start to record data.

## Methods

The method we followed is:

1) Extracting and enhancing our datasets 
2) finding best categories by :
    - Global correlation
    - Local correlation
    - Global improvement
    - Local improvement
3) Predict births data
4) Check which cathegories are the bests

## Proposed timeline

The work will be decomposed into 3 parts
- extract data
- Apply our google trends models and check for best predictions
- visualize the data

## Organization within the team

Thomas : Extract data from google, concentrate on selecting categories by improvement and wordcloud visualisation.
Pierre : Concentrate on selecting categories by regression analysis and result plotting.
Kevin :  Writing up the report, checking markdown syntax and preparing the final presentation.


## Questions for TA's
Is it important to select categories through a ‘spike and slab’ regression or can we just select relevant categories such as "Family" ?

