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
from sklearn.preprocessing import StandardScaler

class VAE_Features():
    
<<<<<<< HEAD
    def __init__(self,df_train,scaling_flag = True):
=======
    def __init__(self,df_train,mesh_path_file=PROJECT_ROOT+"data/mesh_data/MeSHFeatureGeneratedByDeepWalk.csv",scaling_flag = True):
>>>>>>> new_branch
        self.df_train = df_train
        
        self.mesh_features = get_mesh_vec.MeshEmbeddings_own(PROJECT_ROOT+"data/mesh_data/ownmesh2vec.model")
#         self.mesh_features = get_mesh_vec.MeshEmbeddings(PROJECT_ROOT+"data/mesh_data/MeSHFeatureGeneratedByDeepWalk.csv")

        self.co_authors_emb=get_co_authors_vec.CoAuthorEmbeddings_own()
#         self.co_authors_emb=get_co_authors_vec.CoAuthorEmbeddings()

        
        self.mesh_features.set_mesh_freq(df_train.mesh.to_list())
        self.name_emb=get_names_vec.NameEmbeddings()
        self.cat_feats =  get_cat_vet.CatFeat(df_train)
<<<<<<< HEAD
=======
        self.co_authors_emb=get_co_authors_vec.CoAuthorEmbeddings()
>>>>>>> new_branch
        self.scaler = None
        # If we should scale the Data
        self.scaling_flag = scaling_flag

    def get_all_features(self,df):
        '''
        Goes through all the other functions and returns the full vector embedding for all of the features (added scaled)
        
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
<<<<<<< HEAD
        feat_coauth = self.get_own_co_authors_features(df)
#         feat_coauth = self.get_co_authors_features(df)
        feat_cat = self.get_cat_features(df)
        feat = np.hstack((feat_mesh,feat_coauth,feat_cat))
                
        
        if self.scaling_flag:
            if self.scaler is None:
                print("Defining new scaler")
                scaler = StandardScaler()
                feat = scaler.fit_transform(feat)
                self.scaler = scaler
            else:
                print("Using old scaler")
                feat = self.scaler.transform(feat)
            
=======
        feat_coauth = self.get_co_authors_features(df)
        feat_cat = self.get_cat_features(df)
        feat = np.hstack((feat_mesh,feat_coauth,feat_cat))
                
>>>>>>> new_branch
        self.input_dims = feat.shape[1]
        
        if self.scaling_flag:
            if self.scaler is None:
                print("Defining new scaler")
                scaler = StandardScaler()
                feat = scaler.fit_transform(feat)
                self.scaler = scaler
            else:
                print("Using old scaler")
                feat = self.scaler.transform(feat)

        return feat
        
    def get_mesh_features(self, df):
        '''
        Returns all the features related to mesh terms.
        
            - mesh embeddings (currently using weigted average based off inverted IDF)
            - number of mesh terms
        
        '''
        mesh_emb = self.mesh_features.get_feat_mesh(df.mesh.to_list())
        mesh_count = self.mesh_features.get_mesh_count(df).reshape(-1,1)
        return np.hstack((mesh_emb,mesh_count))


    def get_own_co_authors_features(self, df):
        '''
        Returns all the features related to co authors .
        
            - co_authors to vector        
        ''' 
        df['co_authors']=df.authors.apply(lambda x: [i['name'].lower().split(",") for i in x] )
        coauth_emb, count_coauth = self.co_authors_emb.get_feat_coauth(df.co_authors.to_list())
        return np.hstack((coauth_emb,count_coauth))
    
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
<<<<<<< HEAD


=======
>>>>>>> new_branch

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
