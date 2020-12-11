#!/usr/bin/env python
# coding: utf-8

# # TODO / Commentaire
#  - Est ce qu'on met en log les births ?
#  - 
#  
#  use of contraceptive method
#  abortion
#  impact of emmigration
#  children as workpower

# # Wesh les reufs bien ou quoi
# 
# Importing data:
#  - `valeurs_mensuelles.csv`: containing the number of birth in France for each month from 01-01-2004 to 01-10-2020.<br>
#  Columns:
#      - Période: month
#      - Démographie - Nombre de naissances vivantes - France métropolitaine: number of birth for the corresponding month
#      - Codes: Codes for explaining what type of value we have in our case we have "P" for 2020 data standing for "provisional" and "A" everywhere else standing for normale value. (see `caract.csv` for further details)
#  - `GTD.csv`: containing the query number of google trend data for different categories.<br>
#    This dataframe is created in the notebook `GTD_preprocessing.ipynb`.<br>
#  Columns:
#      - date: date of the query number
#      - category id : columns containing all the query number in France for this category.
#      
#  - `categories.csv`: containing all the ids and names of the google trend categories<br>
#    This dataframe is created in the notebook `GTD_preprocessing.ipynb`. The categories can be seen as a list with sub-categories here: https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories<br>
#  Columns:
#      - id: id of the diffrent category
#      - name: Name of this category

# In[1]:


#***Attention***  les donnees n'ont pas les memes tailles
#valeurs mensuelle = 202x1        01-01-2004  --> 01-10-2020 
#GTD               = 204×916      01-01-2004  --> 01-12-2020       916 categories       (donnée incomplete pour decembre)    (certaines categories ne donne pas de resultat exemple : category 42 "jazz" https://trends.google.com/trends/explore?cat=42&date=all&geo=FR )
# categories       = 1133x3                                       1133 categories       (il y a bien 1133 category differentes sur google trend mais dans GTD on a enlevé les category sans resultat  il en reste 916)


# In[2]:


# Clear all variables
#get_ipython().run_line_magic('reset', '-f')

# Importing librairies: numpy, matplotlib, pandas, statmodel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.regression.rolling import RollingOLS


# In[3]:





# ## Analyse des categories et selection
# 
# start_date, end_date: start and end date of the total period
# 
# 
# normalisation_GTD = 
#  - 0: no normlisation
#  - 1: total
#  - 2: par category
#  
# best_selection = 
#  - 1: par correlation
#  - 2: par improvement solo
#  - 3: par improvement solo on the last shift months ex 12 best improvements on the lasts 12 months
# 
#  
# Remove_seasonnality = 
#  - 0: no
#  - 1 : yes
#  
# nb_category_prediction: number of category used for prediction (1-20)

# In[2]:


start_date = '2015-01-01' # '2015-01-01'   #'2004-01-01'
end_date = '2020-10-01'

normalisation_GTD = 1
 
best_selection = 3
shift = [1,2,3,5,7]
 
Remove_seasonnality = 0
 
nb_category_prediction = 1

k = 23

# do not touch 
#

MAES=[]
for shifting in shift:
    months_shift = k+shifting
    print(shifting)



    # Importing birth data with selecting correct columns and changing columns names
    data_folder = './Data/'
    birth_df = pd.read_csv(data_folder +'valeurs_mensuelles.csv', sep= ';', index_col= 0, skiprows=2, usecols=[0,1],header=0, names=['Période','birth'])
    
    # Changing indexes to panda date
    birth_df.index = pd.to_datetime(birth_df.index)
    
    # Putting sales data in log scale as we want
    birth_df['birth'] = birth_df['birth'].apply(np.log)
    
    # Getting an overview
    birth_df = birth_df.sort_index()
    birth_df
    
    
    # In[4]:
    
    
    # Importing google trend data
    from numpy import linalg as LA
    GTD_df = pd.read_csv(data_folder +'GTD.csv',index_col= 0)
    
    # Getting an overview
    GTD_df = GTD_df.loc['2004-01-01':'2020-10-01']
    col = pd.to_numeric(GTD_df.columns)
    GTD_df
    
    
    # In[5]:
    
    
    # Importing google trend categories
    categories_df = pd.read_csv(data_folder +'categories.csv',index_col= 0)
    
    # Keeping only the elements of interest and getting an overview
    categories_df = categories_df[categories_df['id'].isin(col)]
    categories_df
    
    
    
    
    
    
    
    
    
    # ### Saisonnality
    # 
    
    # In[7]:
    
    
    from statsmodels.tsa.seasonal import STL
    
    if Remove_seasonnality == 1:
        birth = STL(birth_df).fit()
        birth_df['birth'] = birth_df['birth'] - birth.seasonal
        
    birth_df['birth'].plot()
    
    
    # ### Shifting columns
    
    # In[8]:
    
    
    nan = np.empty(6)
    nan[:] = 0
    birth_df['birth_6'] =  [*nan , *birth_df.birth[:-6].values] # Moving down 6 rows and putting 0 in the empty space
    nan = np.empty(12)
    nan[:] = 0
    birth_df['birth_12'] =  [*nan , *birth_df.birth[:-12].values] # Moving down 12 rows and putting 0 in the empty space 
    
    
    # ### Start date /end date
    
    # In[9]:
    
    
    birth_df = birth_df.loc[start_date:end_date]
    GTD_df = GTD_df.loc[start_date:end_date]
    
    
    # ### Normalisation GTD
    
    # In[10]:
    
    
    if normalisation_GTD == 1:
        mean_x = np.mean(GTD_df)
        std_x = np.std(GTD_df)
        x = GTD_df
        x = x - mean_x
        for i in range (len(std_x)):
            if std_x[i] == 0:
                std_x[i]=1
        x = x / std_x
        GTD_df = x
        
    elif normalisation_GTD == 2:
        print("TODO")
    
    
    GTD_df
    
    
    # ### Correlation
    
    
    
    # In[12]:
    
    
    #size of the rolling window = 17 = 4 mois ds le précédent paper, on prend 6 dans celui là comme en plus on a une saisonnalité de 6 mois
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5269518/
    def best_feature_improvement(k,start = start_date, end = end_date):
        '''
        Input: 
            -start: start date in format '2004-01-01'
            -end: end date in format '2020-10-01'
            k: rolling windows size, recommended size : 6
        '''
    
    
        improvement = []
    
        #Checking for each feature
        for i in GTD_df.columns:
    
            #Setting the dfs
            df = birth_df.copy()
              
    
            #Setting the feature of interest
            df[str("y0")] = GTD_df[str(i)]
            df = df.loc[start:end]
            df = df.reset_index(drop=True) #Resetting the index to [0-n] format
    
            #Basic reg
            res_base = RollingOLS.from_formula('birth ~ birth_6 + birth_12', data=df, window=k).fit() #We use our rolling windows function
            params = pd.DataFrame(res_base.params.shift(periods=1, axis=0)) #we shift the output parameters one row down in order to apply to the next mont (predict)
            params.columns = ['a0','a1','a2'] #Changing the parameters' columns names
            df = pd.concat([df, params], axis=1) #adding it to our dataframe
            df['predict_base'] = df.a0 + df.a1*df.birth_6 + df.a2*df.birth_12 #predicting the values for the next month
    
            #Trend reg with feature of interest
            res_trend = RollingOLS.from_formula('birth ~ birth_6 + birth_12 + y0', data=df, window=k).fit()#We use our rolling windows function
            params = pd.DataFrame(res_trend.params.shift(periods=1, axis=0)) #we shift the output parameters one row down in order to apply to the next mont (predict)
            params.columns = ['b0','b1','b2','b3']#Changing the parameters' columns names
            df = pd.concat([df, params], axis=1) #adding it to our dataframe
            df['predict_trend'] = df.b0 + df.b1*df.birth_6 + df.b2*df.birth_12 + df.b3 * df.y0   #predicting the values for the next month
    
            #Calculating MAE and Improvement
            mae_base = np.mean(abs(df.birth-df.predict_base))*100
            mae_trends = np.mean(abs(df.birth-df.predict_trend))*100
            improvement_overall = (mae_base-mae_trends)*100 /mae_base
            improvement.append(improvement_overall)
    
    
        #Our df well ordered
        categories_best = categories_df.reset_index(drop=True)
        categories_best['Improvement'] = pd.DataFrame(improvement)
        categories_best = categories_best.sort_values('Improvement', ascending=False)
        categories_best = categories_best.reset_index(drop=True)
        
        return categories_best
    
    
    
    # In[14]:
    
    
    if best_selection==3:
        #create correct columns names
        columns_name = []
        for i in range (nb_category_prediction):
            columns_name.append(str("id{}".format(i)))
        
        #make shift
        Date_shifted = birth_df.index.shift(-months_shift, freq ='MS') 
        Date_shifted_1 = birth_df.index.shift(-1, freq ='MS') 
        
        predict_gtd = pd.DataFrame(dtype='float')
        categories_best = pd.DataFrame()
        improvements = pd.DataFrame()
        #for each date
        for i in range(months_shift,len(birth_df)):
            #get correct time range
            start = Date_shifted[i]
            end = Date_shifted_1[i]
            #find best categories
            categories_best_temp = best_feature_improvement(k,start,end)
            #add it to df
            predict_gtd = predict_gtd.append(categories_best_temp.id[0:nb_category_prediction].transpose())
            categories_best = categories_best.append(categories_best_temp.name[0:nb_category_prediction].transpose())
            improvements = improvements.append(categories_best_temp.Improvement[0:nb_category_prediction].transpose())
            
            
        
        #create dataframe with correct indices and columns name
        predict_gtd.index = birth_df.index[months_shift:]
        predict_gtd.columns = columns_name
        
        categories_best.index = birth_df.index[months_shift:]
        improvements.index = birth_df.index[months_shift:]
        categories_best = pd.merge(categories_best,improvements,left_index = True,right_index = True)
    
    
    # 
    
    
    # In[16]:
    
    
    predict_gtd = predict_gtd.astype(int)
    
    
    
    # ### Final prediction
    
    # In[17]:
    
    
    def prediction(start,end,k,data,GTD,ids):
        df = data.copy()
        temp_GTD = GTD.copy()
        df = df.loc[start:end]
        temp_GTD = temp_GTD.loc[str(start-pd.DateOffset(months=1)):str(end)]
            
        #Features selection
        formula = str('birth ~ birth_6 + birth_12')
        columns_name = ['b0','b1','b2']
        for j,i in enumerate(ids):
            df["y%d"%j] = temp_GTD.loc[:,str(int(i))]
            formula = formula + ' + ' + str("y%d"%j)
            columns_name.append(str("b%d"%(j+ 3)))
            #GTD_df.loc[:,str(categories_best.id[i])]
    
        index_date = df.index #saving the index for later plotting
        df = df.reset_index(drop=True) #Resetting the index to [0-n] format
    
    
        res_base = RollingOLS.from_formula('birth ~ birth_6 + birth_12', data=df, window=k).fit() #We use our rolling windows function
        params = pd.DataFrame(res_base.params.shift(periods=1, axis=0)) #we shift the output parameters one row down in order to apply to the next mont (predict)
        params.columns = ['a0','a1','a2'] #Changing the parameters' columns names
        df = pd.concat([df, params], axis=1) #adding it to our dataframe
        df['predict_base'] = df.a0 + df.a1*df.birth_6 + df.a2*df.birth_12 #predicting the values for the next month
    
        res_trend = RollingOLS.from_formula(formula, data=df, window=k).fit()#We use our rolling windows function
        params = pd.DataFrame(res_trend.params.shift(periods=1, axis=0)) #we shift the output parameters one row down in order to apply to the next mont (predict)
        params.columns = columns_name#Changing the parameters' columns names
        df = pd.concat([df, params], axis=1) #adding it to our dataframe
        df['predict_trend'] = df.b0 + df.b1*df.birth_6 + df.b2*df.birth_12 
        for j,i in enumerate(ids):
            df['predict_trend'] = df['predict_trend'] + params.iloc[:,3+j]*  df.iloc[:,3+j]  #predicting the values for the next month
    
    
        #add date index again
        df.index = index_date
    
        #MAE + Improvement overall calculation between the base fit and base+trend fit
        mae_base = np.mean(abs(df.birth-df.predict_base))*100
        mae_trends = np.mean(abs(df.birth-df.predict_trend))*100
        improvement_overall = (mae_base-mae_trends)*100 /mae_base
        return df,improvement_overall,mae_base,mae_trends
    
    
    # In[18]:
    
    
    temp_df=pd.DataFrame(GTD_df, dtype='float')
    
    #Birth_df Data
    df = birth_df.copy()
    
    '''
    ids=[]
    for i in range (nb_category_prediction):
        ids.append(categories_best.id[i])
    [df,improvement_overall] = prediction(start_date,end_date,k,df,temp_df,ids)
    '''
    predict_base=[np.nan]*(months_shift+1)
    predict_trend=[np.nan]*(months_shift+1)
    for i in range(months_shift+1,len(birth_df)):
        j=birth_df.index[i]
        ids = predict_gtd.loc[j,:]
        [pred,improvement_overall,mae_base,mae_trends] = prediction(j -pd.DateOffset(months=months_shift),j,k,df,temp_df,ids)
        predict_base.append(pred['predict_base'][-1])
        predict_trend.append(pred['predict_trend'][-1])
    
    df['predict_base'] = predict_base
    df['predict_trend'] = predict_trend
    
    
    
    
    # In[20]:
    MAES.append([mae_base,mae_trends,improvement_overall])

MAEs = pd.DataFrame(MAES, columns=['MAE base','MAE trend','Improvement'], index = shift)



