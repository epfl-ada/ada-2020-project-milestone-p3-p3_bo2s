#!/usr/bin/env python
# coding: utf-8

# # TODO
#  - make seasonnal function (juste pour le bo geste)
#  - validation pour choisir juste la meilleur category et pas les n'th meilleures

# # 0. Importing librairies

# In[20]:


# Importing librairies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.regression.rolling import RollingOLS

# External librairies
# Google trend unofficial API
from pytrends.request import TrendReq


# # 1. Preprocessing all the data
# ## 1.1 Birth data

# In[2]:


# Folder with all the csv to export/import
data_folder = './Data/'

# Importing birth data with selecting correct columns and changing columns names
birth_df = pd.read_csv(data_folder +'valeurs_mensuelles.csv', sep= ';', index_col= 0, skiprows=2, usecols=[0,1],header=0, names=['Période','birth'])

# Changing indexes to panda date
birth_df.index = pd.to_datetime(birth_df.index)

# Putting sales data in log scale as we want
birth_df['birth'] = birth_df['birth'].apply(np.log)

# Getting an overview
birth_df = birth_df.sort_index()
birth_df


# ## 1.2 Google Trend Data
# ### 1.2.1 Find all Google Trend Data categories

# In[3]:


# Function definition we will use later for extracting data contained in a nested dictionaries with lists.
def extract(dict_in, list_out, list_out2,list_out3 ,list_out4 , parent,i,MinLevel):
    '''
    Recursive function extracting data contained in a nested dictionaries with lists.
    It extract:
     - all integers
     - entry with key 'name'
     - the level (parent/children) of the nested dictionary
     - the parent name of the previous dictionary
     
    The code was adapted from here : https://stackoverflow.com/questions/10569636/how-to-get-all-keysvalues-in-nested-dict-of-list-of-dicts-and-dicts/10569687
    
    INPUT:
        - dict_in: dictionary you want to extract from
        - list_out(1-4): list containing all information found by previous recursion (more details in OUTPUT)
        - parent:  name of last entry with key 'name' and level equal to Minlevel found by previous recursion
        - i: level of previous recursive function
        - Min level: Level at where we want to keep track of parent dictionary
        
    OUTPUT:
        - list_out: list containing all integers entry in dict_in
        - list_out2: list containing all entry entry with key 'name' in dict_in
        - list_out3: list containing all levels of dict_in
        - list_out4: list containing all parent name of dict_in
        - parent: name of last entry with key 'name' and level equal to Minlevel
        - i: level of current recursive function
    '''
    i=i+1                                             # Add one level to recursive function

                                                      # Reverse the dictionary order
    rdict_in = dict()                                 # Prepare a new  dictionary
    for k in reversed(dict_in):                       # For each entry in reversed ordered dictionary
        rdict_in[k] = dict_in[k]                          # Save it in new dictionary
    
    for key, value in rdict_in.items():               # for all entry in reversed dict_in
        if isinstance(value, dict):                       # If value itself is dictionary
            extract(value,
                    list_out, list_out2,
                    list_out3, list_out4,
                    parent, i,MinLevel)                       # Then do recursion and input this dictionary in dict_in
        elif isinstance(value, list):                     # Else if this value is a list
            for key2 in value:                                 # Then for each entry in this list
                extract(key2,
                        list_out, list_out2,
                        list_out3,list_out4,
                        parent,i,MinLevel)                         # Do recursion and input this entry in dict_in
        elif isinstance(value, int):                      # Else if this value is an integer
            list_out.append(value)                             # Add the value to list_out
        elif key=='name':                                 # Else if this value has key == 'name'
            list_out2.append(value)                            # Add this value to list_out2
            list_out3.append(i)                                # Add the level of this value in list_out3
            if i==MinLevel:                                    # If current level is equal to the level we want to keep track of
                parent = value                                     # Then add current value to the parent variable
                list_out4.append('nul')                            # Add the value 'nul' as parent of current value
            else:                                              # Else
                list_out4.append(parent)                           # Add the last parent as parent of current value
    return list_out,list_out2,list_out3,list_out4 , parent ,i # Return the two list to have valid recursion


# In[4]:


# Create a trend rquest object with language, timzone offset, number of retry if request fail, time factor to make each retry (wait 0.1s , 0.2s, 0.3s, ...)
pytrends = TrendReq(hl='US-US',tz=60, retries=10,backoff_factor=0.1,)

# Extract all google trend categories in a nested dictionary
categories_dictionary = pytrends.categories()

# Initailaze list for extracting data
categories_ids =[]
categories_names = []
level=[]
parent =[]
# Initialise level, and name for categories with no parents
i=0
init='nul'
# Selecting the level of categroies we want to keep as parent
MinLevel = 2
# Extracting the Categories in the nested dictionary using recursive function
[categories_ids,categories_names,level,parent,init,i] = extract(categories_dictionary,categories_ids,categories_names,level,parent , init,i,MinLevel)

# making a dataframe and drooping duplicates categories (ex: category Programmation id: 31 is a sub category of 'Computer Hardware' and 'Computer science')
categories_df = pd.DataFrame(zip(categories_ids,categories_names,level,parent),columns=['id','name','level','parent']).sort_values(['id','level'])
categories_df=categories_df.drop_duplicates(subset ="id")

# saving to a csv a getting an overview
categories_df.to_csv(data_folder + 'categories.csv')
categories_df


# ### 1.2.2 Extracting all Google Trend Data for each categories

# In[5]:


# Initializing a dataframe for Google Trend Data
GTD_df = pd.DataFrame()
# for each categories
for j,i in enumerate(categories_df['id']):
    # build request payload empty key word with france geolocation with the i'th category from 2004-01-01 to 2020-12-31
    kw_list = [""]
    pytrends.build_payload(kw_list, geo='FR', cat=i , timeframe='2004-01-01 2020-12-31')
    
    # getting google trend data
    temp = pytrends.interest_over_time()
    
    # if the return is not empty save data (may happen for small categories with not enough data the return is empty
    # Ex : category 42 "jazz" https://trends.google.com/trends/explore?cat=42&date=all&geo=FR)
    if not temp.empty:
        GTD_df[i]=temp.iloc[:,0]   
    
# getting an overview    
GTD_df


# An error is araising because pytrend is not unofficial pseudo API for google trend. Therefore some error are raising and they are not solved yet.<br>
# As we can read here: https://github.com/GeneralMills/pytrends/issues/413, the error 429 we are obtaining now is often a "Too many request" code but this error is arraising for random number of requests and has no definite solution.<br>
# Therefore, we made the preprocessing in another notebook by running it multiple time and made a full dataframe containing all the request that we will import in the next cells.

# In[7]:


# Importing google trend data preprocessed exactly the same in another notebook
GTD_df = pd.read_csv(data_folder +'GTD.csv',index_col= 0)
GTD_df.index = pd.to_datetime(GTD_df.index)
GTD_df


# # 2. Category Selection
# 
# ## 2.1. Creating all function needed for category selection

# In[18]:


#10 best features selection based on its correlation with the birth number
def bestFeatureCorrelation(date, k):
    '''
    date: date at which we want to determine the correlation
    k: size of the data we want to check the correlation on (size recommended: identical to the ORLS)
    '''    
    
    #Number of month in which we should look for correlation before choosing the best (size of the rolling ols)
    pearson = [] #df in which we store the pearson correlation factor
    kendall = [] #df in which we store the kendall correlation factor
    spearman = [] #df in which we store the spearman correlation factor
    
    date = single_date.strftime("%Y-%m-%d")
    end_date = datetime.strptime(date, '%Y-%m-%d')  
    start_date = end_date - relativedelta(months=+k)

    end_date = datetime.strftime(end_date, '%Y-%m-%d')
    start_date = datetime.strftime(start_date, '%Y-%m-%d')

    temp_df=pd.DataFrame(GTD_df.loc[start_date:end_date], dtype='float') #Taking the time index that are also in the birth_df
    birth_temp = pd.DataFrame(birth_df.loc[start_date:end_date], dtype='float')
    for i,j in enumerate(categories_df['id']):

        if str(j) in temp_df.columns:
            #Pearson correlation calculation
            pearsonTemp = np.abs(birth_temp.iloc[:,0].corr(temp_df[str(j)],method='pearson'))
            pearson.append(pearsonTemp)
            #Kendall correlation calculation
            kendallTemp = np.abs(birth_temp.iloc[:,0].corr(temp_df[str(j)],method='kendall'))       
            kendall.append(kendallTemp)
            #Spearman correlation calculation
            spearmanTemp = np.abs(birth_temp.iloc[:,0].corr(temp_df[str(j)],method='spearman'))
            spearman.append(spearmanTemp)

        else:
            #if the correlation could not be calculated -> put NaN
            pearson.append(np.nan)
            kendall.append(np.nan)
            spearman.append(np.nan)

    #Normalizing our correlation indicators in order to combine them and compare them
    pearsonNorm = 100 / np.nanmax(pearson)
    categories_df['Pearson'] = np.multiply(pearsonNorm, pearson)
    kendallNorm = 100 / np.nanmax(kendall)    
    categories_df['Kendall'] = np.multiply(kendallNorm, kendall)
    spearmanNorm = 100 / np.nanmax(spearman)
    categories_df['Spearman'] = np.multiply(spearmanNorm, spearman)

    #Crossing between our correlations to see which feature is the best
    cal = pd.DataFrame([categories_df["Spearman"], categories_df["Pearson"], categories_df["Kendall"]]).transpose()
    cal = cal.mean(axis=1)
    categories_df['Mean'] = cal

    #Displaying the 10 best
    categories_best = categories_df.sort_values('Mean', ascending=False).iloc[0:20,:].reset_index(drop=True)
    return  categories_best

# best improvement 
def best_feature_improvement(k,start = '2004-01-01', end = '2020-10-01'):
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
        df = df.loc[start:end]
        temp_GTD_df = GTD_df.loc[start:end]
        df["y0"] = temp_GTD_df[i]
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

def prediction(date,k,birth_df,GTD,ids):
    '''
    date: dates at which we want to predict
    k: size of the rolling windows
    df: birth df
    temp_GTD: Google Trends DF
    ids.: ids of category
    '''
    
    df = birth_df.copy()
    temp_GTD = GTD_df.copy()
    
    end = datetime.strptime(date, '%Y-%m-%d')  
    start = end - relativedelta(months=+k)
    end = datetime.strftime(end, '%Y-%m-%d')
    start = datetime.strftime(start, '%Y-%m-%d')
    
    df = df.loc[start:end]
    temp_GTD = temp_GTD.loc[start:end]
    
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
    return df,improvement_overall


#Function to return all the value in the date range
def dateRange(start_date, end_date,k):
    '''
    start_date: start date of the date range
    end_date: end date of the date range
    '''
    for n in range(k, rep):#GTD_df.shape[0]):
        yield start_date + relativedelta(months=n)

#Function to standardize data
def standardize(data):
    mean_x = np.mean(data)
    std_x = np.std(data)
    x = data
    x = x - mean_x
    for i in range (len(std_x)):
        if std_x[i] == 0:
            std_x[i]=1
    x = x / std_x
    return x

#Count frequency
def CountFrequency(my_list): 
  
    # Creating an empty dictionary  
    freq = {} 
    for item in my_list: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
    
    return freq

def make_date_best(best):
    temp = pd.DataFrame(index = birth_df.index)
    for i in range (number_of_best_cat):
        temp[str("id{}".format(best.index[i]))] = np.ones((len(birth_df),1)) * best.id[i]
    return temp


def best_feature_improvement_rolling(months_shift):
    columns_name = []
    for i in range (number_of_best_cat):
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
        predict_gtd = predict_gtd.append(categories_best_temp.id[0:number_of_best_cat].transpose())
        categories_best = categories_best.append(categories_best_temp.name[0:number_of_best_cat].transpose())
        improvements = improvements.append(categories_best_temp.Improvement[0:number_of_best_cat].transpose())
        print(birth_df.index[i])
        
    
    #create dataframe with correct indices and columns name
    predict_gtd.index = birth_df.index[months_shift:]
    predict_gtd.columns = columns_name
    
    categories_best.index = birth_df.index[months_shift:]
    improvements.index = birth_df.index[months_shift:]
    categories_best = pd.merge(categories_best,improvements,left_index = True,right_index = True)
    return categories_best

# ## 2.2 Category Selection
#  

# In[9]:

# Make columns name to numeric
col = pd.to_numeric(GTD_df.columns)

# Keeping only the elements of interest and getting an overview
categories_df = categories_df[categories_df['id'].isin(col)]

# standardize the Google Trend Data
GTD_df = standardize(GTD_df)

# Remove seasonnality 
# Make function

# Shifting columns to make Auto Regressive model
nan = np.empty(6)
nan[:] = 0
birth_df['birth_6'] =  [*nan , *birth_df.birth[:-6].values] # Moving down 6 rows and putting 0 in the empty space
nan = np.empty(12)
nan[:] = 0
birth_df['birth_12'] =  [*nan , *birth_df.birth[:-12].values] # Moving down 12 rows and putting 0 in the empty space

k = 23
shift = 1
number_of_best_cat =1

months_shift = k+shift
# ### 2.2.1 By correlation overall

# In[ ]:





# ### 2.2.2 By correlation with rolling window

# In[14]:



#Data for correlation
#start_date = '2004-01-01' 
#end_date = '2020-11-01'
#k = 18

#Date processing
start_date = datetime.strptime(start_date, '%Y-%m-%d')  
end_date = datetime.strptime(end_date, '%Y-%m-%d')
rep = len(pd.date_range(start_date,end_date, freq='M'))

#output dataframe for the best parameters
best = pd.DataFrame()

#Loop in which we find the best hyper parameters with correlation
for single_date in dateRange(start_date, end_date, k):
    date = single_date.strftime("%Y-%m-%d")
    corr = bestFeatureCorrelation(date, k)
    best[date]=corr.id[0:10]
    if single_date.month == 1:
        print(date)


# In[ ]:





# ### 2.2.3 By improvement overall

# In[21]:

# find best feature overall
best = best_feature_improvement(k)

# Make a datframe with these feature for each month
best = make_date_best(best)

# ### 2.2.4 By improvement with rolling window

# In[ ]:

best = best_feature_improvement_rolling(months_shift)



# ## 2.3 Best category overview

# In[16]:


best


# # 3. Prediction and visualization
# 
# ## 3.1 Prediction

# In[ ]:





# ## 3.2 Prediction plotting

# In[ ]:


#Function to show the plot from paper
def showPlot(df, improvement_overall):
    '''
    df: DataFrame with all the parameters to plot (birth; predict_base; predict_trend)
    improvement_overall: value of the improvement
    '''
    
    #Defining the overall parameters for the figure
    params = {'legend.fontsize': 20,
              'legend.handlelength': 3,
              'figure.figsize': (15,10),
              'axes.labelsize' : 15,
              'xtick.labelsize' : 15,
              'ytick.labelsize' : 15}
    plt.rcParams.update(params) #applying them

    #plotting each curve with specific parameters
    fig, ax = plt.subplots()
    ax.plot(df.birth, 'k', linewidth=2, label='Actual') #Thicker line for the real data
    ax.plot(df.predict_base, 'k--', linewidth=1,label='Base') #Doted line for the predicted curve with basic data
    ax.plot(df.predict_trend, 'k', linewidth=1, label='Trends') #Classic line for the predicted curve with basic + trend data

    #Defining figure title
    plt.suptitle('Births in France', fontsize=20)

    #Defining (x;y) labels
    plt.xlabel('Index')
    plt.ylabel('log(mvp)')

    #Plotting the legend
    plt.legend(loc="upper right")

    #Creating the box with the MAE improvements
    textstr = '\n'.join((
        r'MAE improvement',
        r'Overall = $%.1f$%%' % (improvement_overall, )))
    ax.text(0.013, 0.135, textstr, transform=ax.transAxes, fontsize=17.5,
            verticalalignment='top', bbox=dict(facecolor='none', edgecolor='black', pad=10))

    #Showing the plot
    plt.show
    return


# ## 3.3 Visualisation of categories

# In[ ]:




