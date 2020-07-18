import py_4.feature_helper as feature_helper
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np
from os import makedirs, path
from utils import PROJECT_ROOT, DATA_PATH
from gensim.models import Word2Vec
 
class CoAuthorEmbeddings():
    """
    class for handeling co-author vector
    """
#     def __init__(self):
#         self.model_path=PROJECT_ROOT+"code/models/co_authors_epochs_2_vectorSize_64_window_2.model"
#         self.model = self.get_model()

#     def get_model(self):
#         if path.exists(self.model_path):
#             print(f"'{self.model_path}' already exits. Using existing model to re-generate results.")
#             model = Doc2Vec.load(self.model_path)
#         else:
#             print('error finding names model')
#         return model

#     def infer_vec(self, co_authors: list):
#         return self.model.infer_vector(co_authors)

    def __init__(self):
        self.model_path = PROJECT_ROOT + "data/owncoauth2vec.model"
        self.model =  Word2Vec.load(self.model_path)
        self.coauth_dict = {k : self.model[k] for k in self.model.wv.index2word}
        
    def get_feat_coauth(self, df_coauth):
        all_names = [[["_".join((b[0],b[-1].strip()[:1]))][0] for b in x] for x in df_coauth]
        lst = []
        total_authors = []
        for names in all_names:
            lst.append(self.get_coauth_emb(names).reshape(1,-1))
            total_authors.append(len(names))
        return np.array(lst).squeeze(), np.array(total_authors).reshape(-1,1) 
        
    def get_coauth_emb(self, co_authors: list) -> np.array:    
        coauth_emb = np.array([], dtype=np.float32).reshape(0,64)
        for name in co_authors:
            if name in self.coauth_dict:
                coauth_vec = self.coauth_dict[name]
                if coauth_vec is not None:
                    coauth_emb = np.vstack((coauth_emb,coauth_vec))
        if coauth_emb.size == 0:
            coauth_emb = np.zeros((1,64))
        return np.mean(coauth_emb, axis=0)
                
    
if __name__ == "__main__":
    x = [['tiwari', ' p'], ['rosen', ' muhhamed'], ['madabhushi', ' at']]
    coemb = CoAuthorEmbeddings()
    coemb.get_coauth_emb(x).shape



