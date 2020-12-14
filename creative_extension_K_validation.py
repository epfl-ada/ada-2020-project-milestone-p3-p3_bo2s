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


start_date = '2016-01-01' # '2015-01-01'   #'2004-01-01'
end_date = '2020-10-01'

normalisation_GTD = 1
 
 
Remove_seasonnality = 0
 

k = list(range(19,26))



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




# %%

def prediction_bestK(start,end,k,data):
    
    
    MAES = []
    for i in k:
        df = data.copy()
        df = df.loc[start:end]
        index_date = df.index #saving the index for later plotting
        df = df.reset_index(drop=True) #Resetting the index to [0-n] format
        res_base = RollingOLS.from_formula('birth ~ birth_6 + birth_12', data=df, window=i).fit() #We use our rolling windows function
        params = pd.DataFrame(res_base.params.shift(periods=1, axis=0)) #we shift the output parameters one row down in order to apply to the next mont (predict)
        params.columns = ['a0','a1','a2'] #Changing the parameters' columns names
        df = pd.concat([df, params], axis=1) #adding it to our dataframe
        df['predict_base'] = df.a0 + df.a1*df.birth_6 + df.a2*df.birth_12 #predicting the values for the next month
         #add date index again
        df.index = index_date
    
        #MAE + Improvement overall calculation between the base fit and base+trend fit
        mae_base = np.mean(abs(df.birth-df.predict_base))*100
        MAES.append(mae_base)
    return MAES
        
    


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

MAES = prediction_bestK(start_date,end_date,k,df)


MAES = pd.DataFrame([MAES],columns=[['MAE Base %d' % x for x in k] ])
print(MAES)



