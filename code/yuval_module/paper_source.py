"""
PaperSource supplies all the needed fields for clustering papers 
in order to identify researchers.
It gets the paper metadata from our ES index and from Pubmed.
"""
from elasticsearch import Elasticsearch
from elasticsearch_dsl import A, Search
from pymed import PubMed

import json
from time import time, sleep
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import s3_functions as s3func
#from elastic_index.balsamic.identify_inst_state import Grid

class PaperSource:
    def __init__(self):
        self.creds=s3func.get_creds()
        self.data=None
        self.cur_researcher_id=0
    def get_dataset(self):
            return self.data

    def load_dataset(self, set_name):
            """"
            fetch the data from S3
            """
            datasets={
                    "enriched_labeled" : "enriched_labeled_dataset.json",
                    "not_enriched_labeled": "not_enriched_labeled_dataset.csv",
                    "mini" : "academix_enriched_mini.json"
                    }
            self.data=s3func.get_dataframe_from_s3(self.creds['AWS_ACCESS_KEY'],
                self.creds['AWS_ACCESS_SECRET_KEY'],
                self.creds['BUCKET'],
                file=datasets[set_name],
                type='json'
            )
            pass
        
    def clean_email(self, orig_email):
        try:
            if isinstance(orig_email, str):
                return orig_email
            elif isinstance(orig_email, list):
                l = len(orig_email)
                if l==0:
                    return ""
                elif l==1:
                    return orig_email[0]
                else:
                    return orig_email[-1]
            elif orig_email is None or pd.isnull(orig_email):
                return ""
        except ValueError as e:
            print(e)
            print("could not process {}".format(orig_email))
            return ""

    def add_processed_fields(self,res_df):
            res_df.loc[:, "mesh_clean"]=res_df.apply(get_mesh_clean, axis=1)
            #res_df.loc[:, "num_mesh"]=res_df["mesh_clean"].apply(lambda l:len(l))
            res_df.loc[:, "other_authors"]=res_df.apply(get_other_authors, axis=1)
            #res_df.loc[:, "num_coauthors"]=res_df["other_authors"].apply(lambda l:len(l))
            res_df.loc[:, "inst_clean"]=res_df.apply(simplify_inst, axis=1)
            res_df.loc[:, "email_clean"]=res_df.last_author_email.apply(self.clean_email)
             #res_df.loc[:, "candidate_rank"]=res_df["num_coauthors"]+res_df["num_mesh"]
            return res_df.sort_values(["inst_clean"])

def simplify(val):
    if val is None or val=="" or not val:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return val[0]

def simplify_inst(row):
    initial=row["last_author_inst"]  
    return simplify(initial)

def get_other_authors(row):
    #print("in get_other_authors, pmid={}".format(row["pmid"]))
    #print(row["authors"])
    author_name_set=set([author["name"] for author in row["authors"]])
    #print(row["last_author_name"])
    author_name_set.discard(row["last_author_name"])
    return author_name_set

def get_mesh_clean(row):
    if isinstance(row['mesh'], list):
        cur_list=row['mesh'] 
    else:
        cur_list=[]

    if isinstance(row['mesh_major'], list):
        cur_list.extend(row['mesh_major'])
    res_lst=[term.split('/',1)[0] for term in cur_list]
    return res_lst
