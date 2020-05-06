import os, re, sys

from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto import s3
import boto3

from yuval_module.paper_clusterer import PaperClusterer
from yuval_module.paper_source import PaperSource

import utils
from utils import PROJECT_ROOT, DATA_PATH

PATH = PROJECT_ROOT+ DATA_PATH
FILE = "enriched_labeled_dataset.json"

def load_dataset(set_name):
    ps=PaperSource()
    ps.load_dataset(set_name)
    return ps
def get_res_papers(ps,author_name):
    df=ps.get_dataset()
    return df[df['last_author_name']==author_name]

def top_authors(df, use_case):
  '''
  Finds for us the most published authors to use them in our dataset.

  input:
  df - dataframe with all the data stored
  use_case - possible use-cases:
    0) Base case (top 15 authors) // use_case = "base"
    1) 3 distinct authors (each having ~30 papers) // use_case = "3_dist_auth" 
    2) 2 distinct authors (one with ~30 papers, the other with ~10 papers) // use_case = "2_dist_dif_auth"
    3) 3 authors that share the same name (together has 50 papers) // use_case =  "1_auth"
  '''
  unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
  unique_authors = unique_authors[unique_authors["PI_IDS"] == 1].index
  indie_authors = df[df['last_author_name'].isin(unique_authors)].groupby('last_author_name')['pmid'].nunique().sort_values(ascending=False)
  if use_case == "base":
    indie_author = list(indie_authors.index)[:20]
  elif use_case == "3_dist_auth":
    indie_author = list(indie_authors.index)[15:18]
  elif use_case == "2_dist_dif_auth":
    indie_author = list(indie_authors.index)
    indie_author = [indie_author[15]] + [indie_author[1050]]
  elif use_case == "1_auth":
    unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
    unique_authors = unique_authors[unique_authors["PI_IDS"] == 3].index
    indie_authors = df[df['last_author_name'].isin(unique_authors)].groupby('last_author_name')['pmid'].nunique().sort_values(ascending=False)
    indie_author = [list(indie_authors.index)[0]]
  return df[df["last_author_name"].isin(indie_author)]

def get_use_case(df, use_case):
    '''
    In order to get accurate results, we need to run many scenarios for each use case. 
    This function just gets a list of all candidates for the use-cases
    '''
    if use_case == "3_ua_same":
        #Unique author where each is ~ 30 papers...
        unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
        unique_authors = unique_authors[unique_authors["PI_IDS"] == 1].index
        bool_authors_30 = df[df['last_author_name'].isin(unique_authors)].groupby('last_author_name')['pmid'].size().between(27,33)
        possible_authors_30 = list(unique_authors[bool_authors_30])
        return possible_authors_30
    elif use_case == "2_ua_dif":
         #Unique author where each is ~ 30 papers or ~10 papers...
        unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
        unique_authors = unique_authors[unique_authors["PI_IDS"] == 1].index
        bool_authors_30 = df[df['last_author_name'].isin(unique_authors)].groupby('last_author_name')['pmid'].size().between(27,33)
        bool_authors_10 = df[df['last_author_name'].isin(unique_authors)].groupby('last_author_name')['pmid'].size().between(8,12)
        possible_authors_30 = list(unique_authors[bool_authors_30])
        possible_authors_10 = list(unique_authors[bool_authors_10])   
        return zip(possible_authors_30,possible_authors_10)
    else use_case == '2_da_same':
    unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
    unique_authors = unique_authors[unique_authors["PI_IDS"] == 2].index
    indie_authors = df[df['last_author_name'].isin(unique_authors)].groupby(['last_author_name','PI_IDS'])[['pmid']].nunique().reset_index(1)
    indie_authors = indie_authors.join(indie_authors, lsuffix="_l", rsuffix='_r').reset_index()
    indie_authors = indie_authors[indie_authors["PI_IDS_l"] != indie_authors["PI_IDS_r"]].drop_duplicates("last_author_name",keep="first").set_index('last_author_name')
    possible_authors_same = indie_authors[(indie_authors["pmid_l"] > 5) & 
                                        (indie_authors["pmid_r"] > 5) &
                                        (np.abs(indie_authors["pmid_l"] - indie_authors["pmid_r"]) < 4)]
    else:
        print("USE CASE NOT FOUND -  PLEASE LOOK AT DOCUMENTATION")
        return None

def get_similarity_matrix(ps,authors_dfs,flag_remove_doubles = True):
  '''
  Using Yuval's code, we take the dataframe, and for:
  `Authors, Mesh, Forenames, Institutions, Emails, Countries`
  We compute similarities and return a similarity matrix.

  If flag_remove_doubles = True, we are trying to train the LR model, and therefore want to make sure we don't
  have any duplicates.

  If flag_remove_doubles = False, we are trying to get a distance matrix for the DBScan and need the 
  duplicates, because the algorithm takes a square matrix.

  Input: 
  ps - PaperSource instance
  authors_dfs - Dataframe of all features for given authors
  flag_remove_double - flag whether to delete duplicates

  Output:
  sim_matrix - Matrix based off the similarity of features for given pairs of documents.
  '''

  ### --- Getting general similarity matrix --- ###

  num_papers = authors_dfs.shape[0]
  print("Total number of papers: ", num_papers)

  print("Building Same Author Column")
  #get similarity column
  author_list = list(authors_dfs['PI_IDS'])
  pair_col = []

  for i in range(num_papers):
    for j in range(num_papers):
      if author_list[i] == author_list[j]:
        pair_col.append(0)
      else:
        pair_col.append(1)

  print("Number of paper combinations (pre-cleaning) is: ", len(pair_col))
  
  print("Getting Similarities")
  
  paper_clusterer=PaperClusterer(eps=1.27)
  #get dist matrix
  sim_matrix = paper_clusterer.get_dist_matrix(authors_dfs, True)
  sim_matrix['same_author'] = pair_col

  ### --- Removing Pairs --- ###
  if flag_remove_doubles:
    print("Removing Doubles")

    pairs = []
    for i in range(num_papers):
      for j in range(num_papers):
        if (i<j):
          pairs.append(True)
        else:
          pairs.append(False)

    sim_matrix = sim_matrix.iloc[pairs]
  else:
    print("Keeping Doubles")

  print("Returning Similarity Matrix.")
  print("Number of pairs after cleaning: ", len(sim_matrix.index))
  return sim_matrix
