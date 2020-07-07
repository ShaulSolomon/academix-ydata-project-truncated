import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

'''
USED TO GET ONE-HOT ENCODING FOR CATEGORICAL VARIABLES

[x] - Institution (We can take the top 1000)
[x] - Country (We can take all that we see ~ 122) (The states are more equally distributed...)

'''

class CatFeat():
    
    def __init__(self,df):
        self.df = df
        self.freq_inst = self.set_most_frequent_institute()
        self.freq_country = self.set_all_countries()
        self.ohe_inst = self.set_ohe_inst()
        self.ohe_country = self.set_ohe_country()
    
    def set_most_frequent_institute(self):
        freq_inst = self.df.groupby('last_author_inst').size().sort_values()
        return freq_inst[freq_inst > 4].index
    
    def set_all_countries(self):
        return list(self.df.last_author_country.unique())
        
    def set_ohe_inst(self):
        return OneHotEncoder(handle_unknown = 'ignore').fit([[x] for x in self.freq_inst])
    
    def set_ohe_country(self):
        return OneHotEncoder(handle_unknown = 'ignore').fit([[str(x)] for x in self.freq_country])
    
    def get_ohe_inst(self,df_inst):
        return self.ohe_inst.transform(np.array(df_inst).reshape(-1,1)).toarray()
    
    def get_ohe_country(self,df_country):
        return self.ohe_country.transform(np.array(df_country).reshape(-1,1)).toarray()
    
    
        
        
        