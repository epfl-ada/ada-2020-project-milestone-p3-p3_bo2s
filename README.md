# Google trends as a indicator for unemployement

## Abstract
Unemployment numbers are an important factor regarding the economic activities within a country. It gives an important insight on the nation’s economy’s “health” when defining new monetary policies. For decision takers, it is thus important to have up to date data in order to fully analyze the situation and set new policies fast. However, in the case of Switzerland, statistics are released approximately are released approximately 2 weeks after the end of the month, our objective would be to use the tools learned from the paper to predict time series around 2 weeks before the end of the month . An interesting aspect would also be to confirm another statement issued from the paper: The use of trends data to predict works far better during recession times. As we are also studying an economic output, we could have a look on that and confirm the authors statement.
We would directly use data issued from the Swiss authorities.


## Research questions
- Are there good topics extracted from google trends to predict the best way possible unemployement rates in Switzerland?
- Is there an optimal amount of google trends indicators to use?
- How well does the use of google trends helps predicting time series during recession times?

## Proposed dataset

We will be using the statistical report on unemployment in Switzerland released by SECO for the latest month: https://www.seco.admin.ch/dam/seco/fr/dokumente/Publikationen_Dienstleistungen/Publikationen_Formulare/Arbeit/Arbeitslosenversicherung/Die%20Lage%20auf%20dem%20Arbeitsmarkt/arbeitsmarkt_2020/alz_10_2020.pdf.download.pdf/PRESSEDOK2010_F.pdf

We will focus especially on the table T10 which indicates unemployment numbers in Switzerland throughout the years.

As this dataframe is in a pdf format, we will have to extract the data using python built-in functions to create a usable dataset.

The data is distributed on a monthly and yearly format, statistics are released each month. 

Each number represents the amount of unemployed people registered to unemployment offices in Switzerland.

## Methods

The method we thought of is the following:

1) Extracting and enhancing our dataset 
2) Applying a classical autoregressive model to predict the time series
3) Search for various indicators through google trends to discover best predictors
4) Visualize our predictions
5) Focus on recession to answer second research question
5) Classify the indicators from best to worst

Steps 3 and 4 are iterative as we will search for the best possible indicators


## Proposed timeline

The work will be decomposed into 3 parts
- Base predictions and searching for the best indicators
- Apply our google trends model to check for best predictions
- Getting a focus on recession periods to see how well our model works

## Organization within the team

One person will focus on google, searching and understanding the best possible indicators that could apply to unemployment rates.
The second member, will focus on developping the best possible code to have a good prediction using different tools
The last member will focus on the vizualisation aspect of the research projet, especially for recession periods.

## Questions for TA's


