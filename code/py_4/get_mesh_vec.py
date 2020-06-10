import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from utils import PROJECT_ROOT, DATA_PATH
import re


'''Helper Functions'''
def csv_to_df(df):
    index_names = list(df.index)
    mesh_dict = {index_names[i].strip("'"):df.iloc[i].to_list() for i in range(df.shape[0])}
    return mesh_dict


class MeshEmbeddings():

    def __init__(self,path_file=PROJECT_ROOT+"data/mesh_data/MeSHFeatureGeneratedByDeepWalk.csv"):
        df = pd.read_csv(path_file,header=None).set_index(0)
        self.mesh_dict = csv_to_df(df)
    
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
            print("MESH NAME NOT FOUND: "+ mesh_name)
            return None

    def get_mesh_emb(self, mesh_names: list, method: str = "avg" ) -> np.array:
        '''
        Given mesh_names, join them using `method` for input into NN

            :param list mesh_names - list of all the mesh names
            :param str method - possible methods on joining vectors
                        :"avg" - getting the mean
            :return np.array - vector embedding of the mesh_terms
        '''
        mesh_emb = np.array([], dtype=np.float32).reshape(0,64)
        if mesh_names is None:
                return np.zeros((1,64),dtype=np.float)
        for mesh in mesh_names:
            mesh_vec = self.get_mesh_vec(mesh)
            if mesh_vec is not None:
                mesh_emb = np.vstack((mesh_emb,mesh_vec))
        #check to see if mesh_emb is empty 
        if mesh_emb.shape[0]==0:
                return np.zeros((1,64),dtype=np.float)
        if method == "avg":
            return np.mean(mesh_emb, axis=0).reshape(1,-1)
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
                lst.append(self.get_mesh_emb(mesh_terms))
        return np.array(lst).squeeze()

def mesh2int(row: list, dict_meshtoint: dict) -> list:
    '''
    Helper function for get_mesh_gram_freq. Turns mesh terms into label equivalent

        :param pd.Series row - list of mesh terms for given row
        :param dict dict_meshtoint - dict of mesh terms to their int labels
        :return None - if row is empty
                list - list of labels
    '''
    if row is None:
        return None
    lst = []
    for value in row:
        lst.append(dict_meshtoint[value])
    return lst

def get_mesh_gram_freq(df: pd.DataFrame, N: int) -> pd.DataFrame:
    '''
    For each of the mesh t6erms, get the IDF for 1-gram, until N-gram.

        :param pd.DataFrame df - Dataframe with each row the mesh terms
        :param int N - max size of N terms
        :return pd.DataFrame - new column with frequency 
    '''
    all_mesh = df['mesh'].values.tolist()
    set_mesh = set()
    for mesh in all_mesh:
        if mesh is not None:
            set_mesh.update(mesh)
    # dict_inttomesh = {i:mesh for i,mesh in enumerate(set_mesh)}
    dict_meshtoint = {mesh:i for i,mesh in enumerate(set_mesh)}
    df_mesh = pd.Series([mesh2int(row, dict_meshtoint) for row in df['mesh'].values])
    # TODO: Turn sparse matrix into frequency (avg?)
    return df_mesh

def get_mesh_count(df: pd.DataFrame) -> list:
    '''
    For each mesh row, return number of mesh terms.
    If there are no mesh terms, return 0

        :param pd.DataFrame df - our datafame
        :return list - count of mesh terms
    '''
    all_mesh = df['mesh'].values.tolist()
    lst = []
    for mesh in all_mesh:
        if mesh is None:
            lst.append(0)
        else:
            lst.append(len(mesh))
    return lst
    
if __name__ == "__main__":
  
    PATH_FILE = r"/home/ubuntu/Proj/AYP/data/mesh_data/MeSHFeatureGeneratedByDeepWalk.csv"

    msh = MeshEmbeddings(PATH_FILE)
    print(msh.get_mesh_emb(['Humans','Abdomen']).shape)