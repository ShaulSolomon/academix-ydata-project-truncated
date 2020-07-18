import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from utils import PROJECT_ROOT, DATA_PATH
import re
from gensim.models import Word2Vec


'''Helper Functions'''
def csv_to_df(df):
    index_names = list(df.index)
    mesh_dict = {index_names[i].strip("'"):df.iloc[i].to_list() for i in range(df.shape[0])}
    return mesh_dict


class MeshEmbeddings_own():
        
    def __init__(self, path_file=PROJECT_ROOT+"data/mesh_data/ownmesh2vec.model"):
        print(path_file)
        model = Word2Vec.load(path_file)
        self.mesh_dict = {k : model[k] for k in model.wv.index2word}
        self.dict_freq = None
        self.mesh_missing = set()

    def set_mesh_freq(self, all_mesh: list):
        '''
        Given all the mesh terms, it creates a dictionary for "mesh_term":term_frequency

            :param list all_mesh - list with all of the possible mesh terms (a list of lists).
            Updates self.dict_freq instead of returning a value
        '''
        dict_freq = {}
        total = 0
        for mesh_list in all_mesh:
            if mesh_list is not None:
                for mesh in mesh_list:
                    mesh = re.sub(r"\/.*","",mesh)
                    dict_freq[mesh] = dict_freq.get(mesh, 1) + 1 
                    total = total + 1 
#         ## Was going to divide all by total, but no need ###
#         for key,value in dict_freq.items():
#             dict_freq[key] = value / total
        self.dict_freq = dict_freq 
        # Number of Documents - N
        self.N = len(all_mesh)
        
    def set_mesh_vectorizer(self, all_mesh: list):
        self.corpus, self.vocab = self.get_mesh_corpus_vocab(all_mesh)
        vocab = [str(x) for x in self.vocab]
        self.pipe = Pipeline([('count', CountVectorizer(vocabulary=vocab)),
                 ('tfid', TfidfTransformer())]).fit(mesh_emb.corpus) 
#         pipe['count'].transform(mesh_emb.corpus).toarray()
#         pipe['tfid'].idf_
        
        
    def get_mesh_vec(self, mesh_name: str) -> list:
        '''
        For a given mesh term, returns its 64 d vector embedding

            :param str mesh_name - name of the given mesh term
            :return list - vector embedding of mesh term
        '''
        mesh_name = re.sub(r"\/.*","",mesh_name.lower())
        if mesh_name in self.mesh_dict:
            return self.mesh_dict[mesh_name]
        else:
            self.mesh_missing.add("MESH NAME NOT FOUND: "+ mesh_name)
            return None

    def get_mesh_emb(self, mesh_names: list, method: str = "avg" ) -> np.array:
        '''
        Given mesh_names, join them using `method` for input into NN

            :param list mesh_names - list of all the mesh names
            :param str method - possible methods on joining vectors
                        :"avg" - getting the mean based off term frequency
            :return np.array - vector embedding of the mesh_terms
        '''
        mesh_emb = np.array([], dtype=np.float32).reshape(0,64)
        if mesh_names is None:
                return np.zeros((1,64),dtype=np.float)
        for mesh in mesh_names:
            mesh = re.sub(r"\/.*","",mesh)
            mesh_vec = self.get_mesh_vec(mesh)
            if mesh_vec is not None:
                mesh_emb = np.vstack((mesh_emb,mesh_vec))
        #check to see if mesh_emb is empty 
        if mesh_emb.shape[0]==0:
                return np.zeros((1,64),dtype=np.float)
        if method == "avg":
            freq_list = np.array([])
            if self.dict_freq is None:
                print("Need to set_mesh_freq.")
                return None
            for name in mesh_names:  
                name = re.sub(r"\/.*","",name)
                if name in self.mesh_dict.keys():
                    # If the word appears in the corpus before
                    if name in self.dict_freq.keys():
                        freq_list = np.append(freq_list, self.dict_freq[name])
                    # If it didnt exist before, give it a value of one
                    else:
                        freq_list = np.append(freq_list, [1])

            # Need to inverse the frequency of each of the terms. So most popular term gets least value.
            if len(freq_list) == 1:
                return mesh_emb
            # Inverse Document Frequency - log(N/df_i)
            IDF = np.log(self.N / freq_list.reshape(-1,1))

            #Not using the IDF
            return np.mean(mesh_emb,axis=0).reshape(1,-1)
        
#             return np.mean(mesh_emb * IDF,axis=0).reshape(1,-1)
        #Return the first of the embeddings
        elif method == "first":
            return mesh_emb[0,:]
            
        else:
            print("METHOD NOT FOUND")
            return None

    def get_feat_mesh(self, df_mesh: list) -> np.array:
        '''
                For Dataset - converts mesh row into (n,64) np array
                        :param list  - list of mesh terms per publication
                        :returns np.array - vector embeddings for mesh terms
        '''
        lst = []
        for mesh_terms in df_mesh:
            lst.append(self.get_mesh_emb(mesh_terms, method='avg').reshape(1,-1))
        return np.array(lst).squeeze()
    
    
    def get_mesh_count(self, df: pd.DataFrame) -> list:
        '''
        For each mesh row, return number of mesh terms.
        If there are no mesh terms, return 0

            :param pd.DataFrame df - our dataframe
            :return list - count of mesh terms
        '''
        all_mesh = df['mesh'].values.tolist()
        lst = []
        for mesh in all_mesh:
            if mesh is None:
                lst.append(0)
            else:
                lst.append(len(mesh))
        return np.array(lst)


    def mesh2int(self,row: list, dict_meshtoint: dict) -> str:
        '''
        CURRENTLY NOT IN USE - USING set_mesh_freq 

        Helper function for get_mesh_gram_freq. Turns mesh terms into label equivalent

            :param pd.Series row - list of mesh terms for given row
            :param dict dict_meshtoint - dict of mesh terms to their int labels
            :return None - if row is empty
                    list - list of labels
        '''
        if row is None:
            return "0"
        lst = ""
        for value in row:
            lst = lst + " " + str(dict_meshtoint[value])
        return lst.strip()

    def get_mesh_corpus_vocab(self, all_mesh: list) -> pd.DataFrame:
        '''
        CURRENTLY NOT IN USE - USING set_mesh_freq 


        For each of the mesh terms, get the IDF for 1-gram, until N-gram.

            :param pd.DataFrame df - Dataframe with each row the mesh terms
            :return pd.DataFrame - new column with frequency 
        '''
        set_mesh = set()
        for mesh in all_mesh:
            if mesh is not None:
                set_mesh.update(mesh)
        # dict_inttomesh = {i:mesh for i,mesh in enumerate(set_mesh)}
        dict_meshtoint = {mesh:i for i,mesh in enumerate(set_mesh,1)}
        df_mesh = [self.mesh2int(row, dict_meshtoint) for row in all_mesh]
        size_dict = len(dict_meshtoint.keys())
        return df_mesh, list(range(size_dict))


    
    
class MeshEmbeddings():

    def __init__(self,path_file=PROJECT_ROOT+"data/mesh_data/MeSHFeatureGeneratedByDeepWalk.csv"):
        df = pd.read_csv(path_file,header=None).set_index(0)
        self.mesh_dict = csv_to_df(df)
        self.dict_freq = None
        self.mesh_missing = set()

    def set_mesh_freq(self, all_mesh: list):
        '''
        Given all the mesh terms, it creates a dictionary for "mesh_term":term_frequency
            :param list all_mesh - list with all of the possible mesh terms (a list of lists).
            Updates self.dict_freq instead of returning a value
        '''
        dict_freq = {}
        total = 0
        for mesh_list in all_mesh:
            if mesh_list is not None:
                for mesh in mesh_list:
                    mesh = re.sub(r"\/.*","",mesh)
                    dict_freq[mesh] = dict_freq.get(mesh, 1) + 1 
                    total = total + 1 
#         ## Was going to divide all by total, but no need ###
#         for key,value in dict_freq.items():
#             dict_freq[key] = value / total
        self.dict_freq = dict_freq 
        # Number of Documents - N
        self.N = len(all_mesh)
        
    def set_mesh_vectorizer(self, all_mesh: list):
        self.corpus, self.vocab = self.get_mesh_corpus_vocab(all_mesh)
        vocab = [str(x) for x in self.vocab]
        self.pipe = Pipeline([('count', CountVectorizer(vocabulary=vocab)),
                 ('tfid', TfidfTransformer())]).fit(mesh_emb.corpus) 
#         pipe['count'].transform(mesh_emb.corpus).toarray()
#         pipe['tfid'].idf_
        
        
    def get_mesh_vec(self, mesh_name: str) -> list:
        '''
        For a given mesh term, returns its 64 d vector embedding
            :param str mesh_name - name of the given mesh term
            :return list - vector embedding of mesh term
        '''
        mesh_name = re.sub(r"\/.*","",mesh_name)
        if mesh_name in self.mesh_dict:
            return self.mesh_dict[mesh_name]
        else:
            self.mesh_missing.add("MESH NAME NOT FOUND: "+ mesh_name)
            return None

    def get_mesh_emb(self, mesh_names: list, method: str = "avg" ) -> np.array:
        '''
        Given mesh_names, join them using `method` for input into NN
            :param list mesh_names - list of all the mesh names
            :param str method - possible methods on joining vectors
                        :"avg" - getting the mean based off term frequency
            :return np.array - vector embedding of the mesh_terms
        '''
        mesh_emb = np.array([], dtype=np.float32).reshape(0,64)
        if mesh_names is None:
                return np.zeros((1,64),dtype=np.float)
        for mesh in mesh_names:
            mesh = re.sub(r"\/.*","",mesh)
            mesh_vec = self.get_mesh_vec(mesh)
            if mesh_vec is not None:
                mesh_emb = np.vstack((mesh_emb,mesh_vec))
        #check to see if mesh_emb is empty 
        if mesh_emb.shape[0]==0:
                return np.zeros((1,64),dtype=np.float)
        if method == "avg":
            freq_list = np.array([])
            if self.dict_freq is None:
                print("Need to set_mesh_freq.")
                return None
            for name in mesh_names:  
                name = re.sub(r"\/.*","",name)
                if name in self.mesh_dict.keys():
                    # If the word appears in the corpus before
                    if name in self.dict_freq.keys():
                        freq_list = np.append(freq_list, self.dict_freq[name])
                    # If it didnt exist before, give it a value of one
                    else:
                        freq_list = np.append(freq_list, [1])

            # Need to inverse the frequency of each of the terms. So most popular term gets least value.
            if len(freq_list) == 1:
                return mesh_emb
            # Inverse Document Frequency - log(N/df_i)
            IDF = np.log(self.N / freq_list.reshape(-1,1))

            #Not using the IDF
            return np.mean(mesh_emb,axis=0).reshape(1,-1)
        
#             return np.mean(mesh_emb * IDF,axis=0).reshape(1,-1)
        #Return the first of the embeddings
        elif method == "first":
            return mesh_emb[0,:]
            
        else:
            print("METHOD NOT FOUND")
            return None

    def get_feat_mesh(self, df_mesh: list) -> np.array:
        '''
                For Dataset - converts mesh row into (n,64) np array
                        :param list  - list of mesh terms per publication
                        :returns np.array - vector embeddings for mesh terms
        '''
        lst = []
        for mesh_terms in df_mesh:
            lst.append(self.get_mesh_emb(mesh_terms, method='avg').reshape(1,-1))
        return np.array(lst).squeeze()
    
    
    def get_mesh_count(self, df: pd.DataFrame) -> list:
        '''
        For each mesh row, return number of mesh terms.
        If there are no mesh terms, return 0
            :param pd.DataFrame df - our dataframe
            :return list - count of mesh terms
        '''
        all_mesh = df['mesh'].values.tolist()
        lst = []
        for mesh in all_mesh:
            if mesh is None:
                lst.append(0)
            else:
                lst.append(len(mesh))
        return np.array(lst)


    def mesh2int(self,row: list, dict_meshtoint: dict) -> str:
        '''
        CURRENTLY NOT IN USE - USING set_mesh_freq 
        Helper function for get_mesh_gram_freq. Turns mesh terms into label equivalent
            :param pd.Series row - list of mesh terms for given row
            :param dict dict_meshtoint - dict of mesh terms to their int labels
            :return None - if row is empty
                    list - list of labels
        '''
        if row is None:
            return "0"
        lst = ""
        for value in row:
            lst = lst + " " + str(dict_meshtoint[value])
        return lst.strip()

    def get_mesh_corpus_vocab(self, all_mesh: list) -> pd.DataFrame:
        '''
        CURRENTLY NOT IN USE - USING set_mesh_freq 
        For each of the mesh terms, get the IDF for 1-gram, until N-gram.
            :param pd.DataFrame df - Dataframe with each row the mesh terms
            :return pd.DataFrame - new column with frequency 
        '''
        set_mesh = set()
        for mesh in all_mesh:
            if mesh is not None:
                set_mesh.update(mesh)
        # dict_inttomesh = {i:mesh for i,mesh in enumerate(set_mesh)}
        dict_meshtoint = {mesh:i for i,mesh in enumerate(set_mesh,1)}
        df_mesh = [self.mesh2int(row, dict_meshtoint) for row in all_mesh]
        size_dict = len(dict_meshtoint.keys())
        return df_mesh, list(range(size_dict))    
    
if __name__ == "__main__":
  
    PATH_FILE = r"/home/ubuntu/Proj/AYP/data/mesh_data/MeSHFeatureGeneratedByDeepWalk.csv"
    msh = get_mesh_vec.MeshEmbeddings(PATH_FILE)
    msh.set_mesh_freq(list(df['mesh']))
    print(msh.get_mesh_emb(['Humans','Abdomen']).shape)