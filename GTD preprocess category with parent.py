import pandas as pd

# External librairies
# Google trend unofficial API
from pytrends.request import TrendReq

data_folder = './Data/'

def extract(dict_in, list_out, list_out2,list_out3 ,list_out4 , parent,i):
    '''
    Recursive function extracting all integers and entry with key 'name' contained in a nested dictionary with list.
    code adapted from here : https://stackoverflow.com/questions/10569636/how-to-get-all-keysvalues-in-nested-dict-of-list-of-dicts-and-dicts/10569687
    
    INPUT:
        - dict_in: dictionary you want to extract from
    
    OUTPUT:
        - list_out: list containing all integers entry in dict_in
        - list_out2: list containing all entry entry with key 'name' in dict_in
    '''
    i=i+1
    
    rdict_in = dict()
    for k in reversed(dict_in):
        rdict_in[k] = dict_in[k]
    
    for key, value in rdict_in.items():               # for all entry in dict_in
        if isinstance(value, dict):                       # If value itself is dictionary
            extract(value, list_out, list_out2,list_out3,list_out4 , parent,i)                # Then do recursion and input this dictionary in dict_in
        elif isinstance(value, list):                     # Else if this value is a list
            for key2 in value:                                 # Then for each entry in this list
                extract(key2, list_out, list_out2,list_out3,list_out4 , parent,i)                  # Fo recursion and input this entry in dict_in
        elif isinstance(value, int):                      # Else if this value is an integer
            list_out.append(value)                             # add the value to list_out
        elif key=='name':                                 # Else if this value has key == 'name'
            list_out2.append(value)                            # add this value to list_out2
            list_out3.append(i)
            if i==3:
                parent = value
                list_out4.append('nul')
            else:
                list_out4.append(parent)
    return list_out,list_out2,list_out3,list_out4 , parent ,i                      # Return the two list to have valid recursion



# Create a trend rquest object with language, timzone offset, number of retry if request fail, time factor to make each retry (wait 0.1s , 0.2s, 0.3s, ...)
pytrends = TrendReq(hl='fr-FR',tz=60, retries=10,backoff_factor=0.1,)

# Extract all google trend categories in a nested dictionary
categories_dictionary = pytrends.categories()

# initailaze list for extracting data
categories_ids =[]
categories_names = []
level=[]
parent =[]
# Extracting all the data in nested dictionary with function
i=0
init='nul'
[categories_ids,categories_names,level,parent,init,i] = extract(categories_dictionary,categories_ids,categories_names,level,parent , init,i)

# making a dataframe and drooping duplicates categories (ex: category Programmtion id: 31 is a sub category of 'Computer Hardware' and 'COmputer science')
categories_df = pd.DataFrame(zip(categories_ids,categories_names,level,parent),columns=['id','name','level','parent']).sort_values(['id','level'])
categories_df=categories_df.drop_duplicates(subset ="id")

# saving to a csv a getting an overview
#categories_df.to_csv(data_folder + 'categoriesWparent.csv')
#categories_df