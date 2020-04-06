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
        self.default_source = 'some text'

    def get_papers_df_s3(self, author_name):
            df=pd.DataFrame()
            return df

#     def get_max_researcher_id_from_index(self):
#         a = A('max', field='last_author_id')
#         self.search.aggs.bucket('max_author_id', a)
#         self.search = self.search.execute()
#         return int(self.search.aggs["max_author_id"]["value"])

#     def build_author_name(self, author, pmid):
#         # print("in build_author_name:")
#         # print("pmid:")
#         # print(pmid)
#         # print("author:")
#         # print(author)
#         if author is None:
#             #print("pmid is {}, author is {}, returning empty".format(pmid, author))
#             return ""
        
#         if "initials" not in author or author["initials"] is None:
#             author["initials"]=""

#         if "lastname" not in author or author["lastname"] is None:
#             author["lastname"]=""

#         if author["initials"]=="" and author["lastname"]=="":
#             #print("pmid is {}, author is {}, returning empty".format(pmid, author))
#             return ""

#         try:
#             full_name = ', '.join([author["lastname"], 
#                             author["initials"]])
#         except Exception as e:
#             print("build_author_name failed for pmid {}, author {}".format(pmid, author))
#             print(e)
#         return full_name


#     def get_papers_df(self, author, orig_paper_row=None):
#         """
#         Bring all papers in the index matching the author name
#         """
#         res = self.es.search(index="bulk_papers6",
#                         body={"query": {"match": {"last_author_name": author}},
#                             "size": 200},
#                         _source_include=["mesh", "mesh_major", "last_author_name", 
#                                         "pub_year",  "last_author_id","pmid", "last_author_inst",  
#                                         "last_author_inst_type", "last_author_country", "authors", "last_author_email",
#                                         "last_author.forename","last_author.affiliation"])
#         num_hits = res["hits"]["total"]
#         #print("found {} publications".format(num_hits))

#         #turn to df
#         hits_json=[hit["_source"] for hit in res["hits"]["hits"]]
#         #print(hits_json[:2])
#         hits_df=json_normalize(hits_json).rename(columns={"last_author.forename":"last_author_forename",
#                                                           "last_author.affiliation":"last_author_affiliation"})
#         # print("hits after normalize:")
#         # print(hits_df)
#         if hits_df.empty:
#             return hits_df
#         if orig_paper_row is not None and not orig_paper_row.pmid in hits_df.pmid: #paper not found in index
#             hits_df=hits_df.append(orig_paper_row)
#             #print("Added orig_paper_row to hits_df")

#         cur_columns=set(hits_df.columns)
#         diff_set=set(self.fields).difference(cur_columns)
#         for fl in diff_set:
#             hits_df.loc[:,fl]=np.NaN

#         output_cols=self.fields.copy()
#         output_cols.append("last_author_forename")
#         res_df=hits_df[output_cols].copy()
#         if self.got_paper_from_pubmed:
#             res_df=pd.concat([res_df, self.get_pubmed_paper_as_df()])

#         # print("res_df:")
#         # print(res_df[["pmid", "last_author.forename", "last_author_name"]])
#         filled_forename=res_df["last_author_forename"].fillna("")
#         res_df.loc[:,"last_author_forename"]=filled_forename
#         # print("after filling:")
#         # print(res_df[["pmid", "last_author.forename", "last_author_name", "last_author_forename"]])
#         return res_df

#     def get_pubmed_paper_as_df(self):
#         res=pd.Series(self.cached_paper).to_frame().transpose()
#         # print("pubmed_paper_df:")
#         # print(res)
#         return res

# #     def complement_last_author(self, res, aff):
# #         #aff=self.cached_paper["aff"]
# #         # print("in complement_last_author:")
# #         # print("aff={}".format(aff))
# #         country_code, syn, found_country = self.grid.find_country(aff)

# #         print("country results:")
# #         print("country={}, syn={}, found_country={}".format(country_code, syn, found_country))
# #         if not found_country:
# #             return res
# #         res["last_author_country"]=self.grid.code_country_map[country_code]
# #         doc=self.grid.nlp(aff)
# #         doc_w_ents=(doc.text, [ent for ent in doc.ents])
# #         email=self.grid.get_emails(aff)
# #         aff_ent_codes = [(aff, doc_w_ents[1], country_code)] 
    
# #         best_insts = [self.grid.locate_inst(aff, ents, country_code) for aff, ents, country_code  in aff_ent_codes]
# #         # prep_docs=[{'hit':hca[0],
# #         #         'country': grid.code_country_map[hca[1][0]],
# #         #         'inst':inst[0],
# #         #         'type':inst[3],
# #         #         'email':email} for hca, inst, email in zip(hit_country_aff, best_insts, emails)]
# #         print("best_insts:")
# #         print(best_insts)
# #         #res["best"]=best_insts
# #         best_inst,best_score,aff,best_inst_type=best_insts[0]
# #         if best_inst!="":
# #             res["last_author_inst"]=best_inst
# #             res["last_author_inst_type"]=best_inst_type
# #         return res
    
#     def get_paper_data(self, pubmed_hit):
#         """
#         Go over object returned from pymed,
#         and collect data fields from it.
#         """
#         res={"pmid":pubmed_hit.pubmed_id,
#         "authors":pubmed_hit.authors.copy()}

#         res["pub_year"]=pubmed_hit.publication_date.year

#         #if res["pmid"] in ["24597912", "24523217"]:
#         # print("in get_paper_data:")
#         # print("pmid={}".format(res["pmid"]))
#         #print("authors={}".format(res["authors"]))
        
#         #print(res["authors"])
#         if not res["authors"]:
#             return res
#         last_author=res["authors"][-1]
#         # print("last_author:")
#         # print(last_author)
#         res["last_author"]=last_author
#         if "firstname" in last_author and last_author["firstname"] is not None:
#             res["last_author_forename"]=last_author["firstname"]
#         else:
#             res["last_author_forename"]=""
#         print("last_author_forename from pubmed: {}".format(res["last_author_forename"]))
#         res["last_author_name"]=self.build_author_name(last_author, res["pmid"])
#         if res["last_author_name"]=="": #currently, cannot do much if no author name
#             return res

#         for author in res["authors"]:
#             author["name"]=self.build_author_name(author, res["pmid"])
#         mesh_headings=[]
#         if hasattr(pubmed_hit, "mesh_headings") and pubmed_hit.mesh_headings is not None:
#             mesh_headings=pubmed_hit.mesh_headings.copy()

#         aff=np.NaN
#         if "affiliation" in last_author:
#             aff=last_author["affiliation"]
#             #print("aff={}".format(aff))
#             if aff is not None:
#                 res["last_author_affiliation"]=aff
#                 res=self.complement_last_author(res, aff)
        
          
#         mesh_array=[]
#         try:
#             for mesh in mesh_headings:
#                 for key in mesh:
#                     mesh_array.append(key)
#         except:
#             pass

#         if mesh_array:
#             res["mesh"]=mesh_array

#         return res

#     def add_author_names(self, authors):
#         for author in authors:
#             author["name"]=self.build_author_name(author, 5)
#         return authors


#     def get_paper_data_from_index(self, bunch_df):
#         """
#         Bring papers with given pmids from the index, if they exist.
#         """
#         cur_query=self.get_papers_template
#         #print(bunch_df.head())
#         cur_query["terms"]["pmid"]=bunch_df["pmid"].tolist()
#         # print("cur_query:")
#         # print(cur_query)
#         resp=self.es.search(index="bulk_papers6",
#                             body={"query": cur_query,
#                                 "size": 200},
#                             _source_include=["mesh", "mesh_major", "last_author_name", 
#                                         "pub_year",  "last_author_id","pmid",
#                                         "last_author_inst",  "last_author_inst_type", 
#                                         "last_author_country", "authors", "last_author_email",
#                                         "last_author.forename", "last_author.affiliation"])


#         num_hits = resp["hits"]["total"]
#         print("found {} publications".format(num_hits))
#         hit_list=[hit["_source"] for hit in resp["hits"]["hits"]]
#         hit_df=json_normalize(hit_list).rename(columns={"last_author.forename":"last_author_forename",
#                                                         "last_author.affiliation":"last_author_affiliation"})
        
        
#         print("hit_df retrieved fields:")
#         print(hit_df.info())
#         print(hit_df.head())

        
#         merged_df = pd.DataFrame(columns=["mesh", "mesh_major", "last_author_name", 
#                                     "pub_year",  "last_author_id","pmid", "last_author_inst",  
#                                     "last_author_inst_type", "last_author_country", "authors", "last_author_email",
#                                     "last_author_forename","last_author_affiliation", "rownum"])
#         if not hit_df.empty:
#             hit_df.loc[:,"authors"]=hit_df.authors.apply(lambda a:self.add_author_names(a))
#             merged_df=pd.merge(hit_df, bunch_df, on="pmid")
#             merged_df.loc[:,"from_pubmed"]=False

#         print("merged_df from index:")
#         print(merged_df.info())
#         #print(merged_df.head())
#         # print("returning merged_df")
#         # print(merged_df[["pmid","last_author_name", "last_author_forename","last_author_inst"]])
#         return merged_df

#     def print_pubmed_names(self, hit_df):
#         authors=hit_df.authors.tolist()
#         pmids=hit_df.pmid.tolist()
#         res=[]
#         for a in authors:
#             auth_names=[]
#             for aa in a:
#                 if "name" in aa:
#                     auth_names.append(aa["name"])
#             res.append(auth_names)
#         pairs=[(p,a) for p,a in zip(pmids, res)]
#         print(pairs)

#     def get_paper_data_from_pubmed(self, bunch_df):
#         if len(bunch_df)==0:
#             merged_df = pd.DataFrame(columns=["mesh", "mesh_major", "last_author_name", 
#                                     "pub_year",  "last_author_id","pmid", "last_author_inst",  
#                                     "last_author_inst_type", "last_author_country", "authors", "last_author_email",
#                                     "last_author_forename", "last_author_affiliation", "rownum"])
#             return merged_df

#         qstr=' '.join(bunch_df.pmid)
#         results = self.pubmed.query(qstr, max_results=100)
#         sleep(0.4)
#         #print("pubmed results:")
        

#         hit_list=[self.get_paper_data(res) for res in results]
            
        
#         hit_df=pd.DataFrame.from_dict(hit_list)

#         #self.print_pubmed_names(hit_df)

#         merged_df = pd.DataFrame(columns=["mesh", "mesh_major", "last_author_name", 
#                                     "pub_year",  "last_author_id","pmid", "last_author_inst",  
#                                     "last_author_inst_type", "last_author_country", "authors", "last_author_email",
#                                     "last_author_forename","last_author_affiliation","rownum"])
#         if not hit_df.empty:
#             merged_df=pd.merge(hit_df, bunch_df, on="pmid")
#             merged_df.loc[:,"from_pubmed"]=True
        
#         print("merged_df from pubmed:")
#         print(merged_df.info())
#         # print(merged_df.head())
#         #self.print_pubmed_names(merged_df)
#         #print(merged_df[["pmid","authors"]])
#         return merged_df

#     def clean_email(self, orig_email):
#         try:
#             if orig_email is None or pd.isnull(orig_email):
#                 return ""
#             if isinstance(orig_email, str):
#                 return orig_email
#             if isinstance(orig_email, list):
#                 l = len(orig_email)
#                 if l==0:
#                     return ""
#                 elif l==1:
#                     return orig_email[0]
#                 else:
#                     return orig_email[-1]
#         except ValueError as e:
#             print(e)
#             print("could not process {}".format(orig_email))
#             return ""

#     def add_processed_fields(self,res_df):
#         res_df.loc[:, "mesh_clean"]=res_df.apply(get_mesh_clean, axis=1)
#         #res_df.loc[:, "num_mesh"]=res_df["mesh_clean"].apply(lambda l:len(l))
#         res_df.loc[:, "other_authors"]=res_df.apply(get_other_authors, axis=1)
#         #res_df.loc[:, "num_coauthors"]=res_df["other_authors"].apply(lambda l:len(l))
#         res_df.loc[:, "inst_clean"]=res_df.apply(simplify_inst, axis=1)
#         res_df.loc[:, "email_clean"]=res_df.last_author_email.apply(self.clean_email)
#         #res_df.loc[:, "candidate_rank"]=res_df["num_coauthors"]+res_df["num_mesh"]
#         return res_df.sort_values(["inst_clean"])

#     # def cluster_all(self, papers_df):
#     #     res=[self.cluster_author_name(row) for idx, row in papers_df.iterrows()]
#     #     return res

#     def report_missing_pmids(self, still_missing_ids_df):
#         res_lst=[]
#         for _idx, pmid_row in still_missing_ids_df.iterrows():
#             res={"pmid":pmid_row["pmid"],
#             "comment":"paper not found in pubmed"}
#             res_lst.append(pd.Series(res).to_frame().transpose())
#         return res_lst

   

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
