import pandas as pd

'''Helper Functions'''
def csv_to_df(df):
    index_names = list(df.index)
    mesh_dict = {index_names[i].strip("'"):df.iloc[i].to_list() for i in range(df.shape[0])}
    return mesh_dict


class MeshEmbeddings():

    def __init__(self,path_file):
        df = pd.read_csv(path_file,header=None).set_index(0)
        self.mesh_dict = csv_to_df(df)
    
    def get_mesh_vec(self, mesh_name):
        if mesh_name in self.mesh_dict:
            return self.mesh_dict[mesh_name]
        else:
            print("MESH NAME NOT FOUND")
            return None

if __name__ == "__main__":

    msh = MeshEmbeddings()
    print(msh.get_mesh_vec("Human"))