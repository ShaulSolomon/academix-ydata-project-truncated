import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from utils import PROJECT_ROOT, DATA_PATH
import re
import py_4.get_mesh_vec as get_mesh_vec
import py_4.get_names_vec as get_names_vec
import py_4.get_cat_vec as get_cat_vet
import string
import py_4.get_co_authors_vec as get_co_authors_vec


class VAE_Features():
    
    def __init__(self,df_train,mesh_path_file=PROJECT_ROOT+"data/mesh_data/MeSHFeatureGeneratedByDeepWalk.csv"):
        self.df_train = df_train
        self.mesh_features = get_mesh_vec.MeshEmbeddings(mesh_path_file)
        self.mesh_features.set_mesh_freq(df_train.mesh.to_list())
        self.name_emb=get_names_vec.NameEmbeddings()
        self.cat_feats =  get_cat_vet.CatFeat(df_train)
        self.co_authors_emb=get_co_authors_vec.CoAuthorEmbeddings()

    def get_all_features(self,df):
        '''
        Goes through all the other functions and returns the full vector embedding for all of the features
        
            Currently implemented:
                [x] mesh embeddings 
                [x] num mesh terms 
                [x] coauthor embeddings 
                [x] # of coauthors
                [-] coauthor name char frequencies  [currently commented out]
                [x] OneHotEncoding for Institution [if appear at least 5 times]
                [x] OneHotEncoding for Country 
                [ ] Pub Year
                
        Currently experimenting with:
                [ ] mesh embeddings
                
        '''

        feat_mesh = self.get_mesh_features(df)
        feat_coauth = self.get_co_authors_features(df)
        feat_cat = self.get_cat_features(df)
        feat = np.hstack((feat_mesh,feat_coauth,feat_cat))
        
        # SANITY TEST
        # feat_name = self.get_names_features(df)
                
        feat = feat_mesh
        self.input_dims = feat.shape[1]

        return feat
        
    def get_mesh_features(self, df):
        '''
        Returns all the features related to mesh terms.
        
            - mesh embeddings (currently using weigted average based off inverted IDF)
            - number of mesh terms
        
        '''
        mesh_emb = self.mesh_features.get_feat_mesh(df.mesh.to_list())
        mesh_count = self.mesh_features.get_mesh_count(df).reshape(-1,1)
#         return np.hstack((mesh_emb,mesh_count))

        return mesh_emb
        
        


    def get_co_authors_features(self, df):
        '''
        Returns all the features related to co authors .
        
            - co_authors to vector        
        '''
        df['co_authors']=df.authors.apply( lambda x: [i['name'] for i in x] )
        res=[]
        count_coauth = []
        for au_list in df['co_authors'].to_list():
            res.extend(self.co_authors_emb.infer_vec([i for i in au_list]))
            count_coauth.append(len(au_list))

        name_vec = np.array(res).reshape(df.shape[0],-1)
        count_coauth = np.array(count_coauth).reshape(-1,1)
        #name_freq = self.get_freq_char(df.co_authors) 
        return np.hstack((name_vec,count_coauth))

    def get_names_features(self, df):
        '''
        Returns all the features related to names .
        
            - name to vectors       
        '''
        df['last_names']=df.last_author_name.apply(lambda x: x.split(',')[0])
        res=[]
        for name in df['last_names'].to_list():
            res.extend(self.name_emb.infer_vec([i for i in name]))
            
        return res

    
    def get_freq_char(self,df_coauth):
        all_freq_char = []
        dict_char_freq_base = {}
        for c in string.ascii_lowercase[:26]:
            dict_char_freq_base[c] = 0
        for name in df_coauth:
            txt = re.sub(", ","","".join(np.array(name).flatten())).lower()
            dict_char_freq = dict_char_freq_base.copy()
            for c in txt.lower():
                dict_char_freq[c] += 1
            all_freq_char.append(list(dict_char_freq.values()) / np.sum(list(dict_char_freq.values())))
        return np.array(all_freq_char)
    
    def get_cat_features(self,df):
        ohe_inst = self.cat_feats.get_ohe_inst(df.last_author_inst)
        ohe_country = self.cat_feats.get_ohe_country(df.last_author_country)
        return np.hstack((ohe_inst,ohe_country))
