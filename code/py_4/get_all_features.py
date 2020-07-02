import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from utils import PROJECT_ROOT, DATA_PATH
import re
import py_4.get_mesh_vec as get_mesh_vec

class VAE_Features():
    
    def __init__(self,df_train,mesh_path_file=PROJECT_ROOT+"data/mesh_data/MeSHFeatureGeneratedByDeepWalk.csv"):
        self.df_train = df_train
        self.mesh_features = get_mesh_vec.MeshEmbeddings(mesh_path_file)
        self.mesh_features.set_mesh_freq(df_train.mesh.to_list())
    
        
    def get_all_features(self,df):
        '''
        Goes through all the other functions and returns the full vector embedding for all of the features
        
            Currently implemented:
                - mesh embeddings (get_mesh_features)
                - num mesh terms (get_mesh_features
        '''
        feat = self.get_mesh_features(df)
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
        return np.hstack((mesh_emb,mesh_count))