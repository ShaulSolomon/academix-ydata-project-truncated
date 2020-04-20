# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
"""
Notebook version of the new clustering algorithm
"""


# %%
from yuval_module.paper_clusterer import PaperClusterer
from yuval_module.paper_source import PaperSource
import sys
import pandas as pd
import numpy as np
import s3_functions as s3func


#get_ipython().run_line_magic('matplotlib', 'inline')


# %%
def load_dataset(set_name):
    ps=PaperSource()
    ps.load_dataset(set_name)
    return ps


# %%
def get_res_papers(ps,author_name):
    df=ps.get_dataset()
    return df[df['last_author_name']==author_name]


# %%
#ps=load_dataset('enriched_labeled')
ps=load_dataset('mini')


# %%
df=ps.get_dataset()
df.head()


# %%
df=get_res_papers(ps,'Szeszko, PR')


# %%
df.head()


# %%
def get_clusters(ps,author):
    author_df = get_res_papers(ps,author)
    paper_clusterer=PaperClusterer(eps=1.27)
    res_clusters, cluster_dfs=paper_clusterer.cluster_res(author_df)
    return cluster_dfs[0].sort_values(["cluster"])


# %%
def get_out_path(row):
    cur_name=row["last_author_name"]
    fields=cur_name.split(", ")
    return "/home/ubuntu/data/{}_{}.csv".format(fields[0], fields[1])


# %%
# def get_file_name(author_name):
#     author_name=author_name.replace(',',' ')
#     author_rebuilt='_'.join(author_name.split(' '))
#     return ''.join(["/home/ubuntu/data/", author_rebuilt, ".csv"])
# #%%
def main(author_list):
    print(author_list)
    
    for author in author_list:
        print(author)
        cluster_df=get_clusters(ps,author)
        # out_path=get_out_path(row)
        # cluster_df.to_csv(out_path, index=False, columns=["pmid", "cluster", "last_author_inst", "last_author_forename","pub_year","last_author_name"])
        
#         #pmid_df=pd.read_csv(list_path, dtype=str, usecols=["PMID"]).rename(columns={"PMID":"Highlight"})
#         #print(pmid_df.info())
#         #limit=min(limit, len(pmid_df))
#         #pmid_list=pmid_df.Highlight.tolist()[offset:offset+limit]
#         pmid_list=pmid_df.pmid.tolist()[:2]
#         paper_clusterer=PaperClusterer(eps=1.27)
#         res_clusters, cluster_dfs=paper_clusterer.infer_paper_author_data(pmid_list)
    
#     #res_clusters, cluster_dfs=paper_clusterer.infer_paper_author_data(pmid_list).sort_values(["rownum"])
#         return res_clusters, cluster_dfs
    #out_path="/home/ubuntu/data/rp_40000_authors.tsv"
    #out_path="/home/ubuntu/data/test_forenames_res.tsv"

    #res_clusters.to_csv(out_path, index=False, sep='\t')


# %%
author_list=['Madabhushi, A','Szeszko, PR']
main(author_list)


# %%
with open("enriched_labeled_dataset_mini.csv", encoding="utf-8") as f:
    s3func.upload_to_s3(f,'enriched_labeled_dataset_mini.csv')


# %%
print(cluster_dfs[0].cluster.value_counts())


# %%
out_path="/home/ubuntu/data/hecht_jr.csv"
#out_path="/home/ubuntu/data/gottlieb_pa.csv"
cluster_dfs[0].sort_values(["cluster"]).to_csv(out_path, index=False, columns=["pmid", "cluster", "last_author_inst", "last_author_forename","pub_year","last_author_name"])


# %%
# cur_clusters=cluster_dfs[0]
# print(cur_clusters.info())
# print(cur_clusters.cluster.value_counts())
# cur_clusters.to_csv("/home/ubuntu/data/friedman_n.csv", index=False)


# %%
# cur_clusters=cluster_dfs[0][["pmid", "last_author_id", "last_author_inst", 
#                              "last_author_country", "last_author_name",
#                              "last_author_forename", "cluster",
#                             "pub_year", "last_author_affiliation"]].sort_values(["cluster"])
# print(cur_clusters)


# %%
# cm = 'tab20b'
# cur_clusters=cluster_dfs[20][["cluster","pmid","last_author_inst",  "last_author_forename",
#                              "mesh_clean", "other_authors", "last_author_country", "last_author_name"]].reset_index()
# cur_clusters.sort_values('cluster').style.background_gradient(cmap=cm, subset=['cluster']).set_caption('Clustering output')


# %%
def forename_delta(n1, n2):
        if ' ' in n1:
            n1=n1.split()[0]
        if ' ' in n2:
            n2=n2.split()[0]
        if n1==n2:
            if len(n1)>1:
                return 1.0
            else:
                return 0.5
        elif n1 in n2 or n2 in n1:
            return 0.75
        else:
            return 0.0


# %%
forename_delta("Peter A","P A")


# %%


