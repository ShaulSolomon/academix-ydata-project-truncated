import os, re, sys

from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto import s3
import boto3

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

sys.path.append('code')
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

def base_authors(df, use_case):
    '''
    Finds for us the most published authors to use as training examples for LR model

    input:
    df - dataframe with all the data stored
    use_case - possible use-cases:
      1) UA case (top 20 UA authors) // use_case = "base_ua"
      2) DA case (3 DA's where each publisher has at least 4 papers)// use_case = "base_da

    TODO: Add possible base for disambiguated authors AND/OR one combined base
    '''
    # if use_case == "base_ua":
    #   unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
    #   unique_authors = unique_authors[unique_authors["PI_IDS"] == 1].index
    #   indie_authors = df[df['last_author_name'].isin(unique_authors)].groupby('last_author_name')['pmid'].nunique().sort_values(ascending=False)
    #   indie_author = list(indie_authors.index)[:20]
    #   return df[df["last_author_name"].isin(indie_author)]

    if use_case == "base_da":
      unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
      unique_authors = unique_authors[unique_authors["PI_IDS"] == 2].index
      #Combine rows based off last_author_name 
      indie_authors = df[df['last_author_name'].isin(unique_authors)].groupby(['last_author_name','PI_IDS'])[['pmid']].nunique().reset_index(1)

      indie_authors = indie_authors.join(indie_authors, lsuffix="_l", rsuffix='_r').reset_index()
      indie_authors = indie_authors[indie_authors["PI_IDS_l"] != indie_authors["PI_IDS_r"]] \
                                    .drop_duplicates("last_author_name",keep="first") \
                                    .set_index('last_author_name')
      #Each need to have more than 5 papers and need to have an equal number of papers
      possible_authors = list(indie_authors[((indie_authors["pmid_l"] > 33) |
                                            (indie_authors["pmid_l"] < 28)) &
                                            ((indie_authors["pmid_r"] > 33) |
                                            (indie_authors['pmid_r'] < 28)) &
                                            (indie_authors['pmid_l'] > 2) &
                                            (indie_authors['pmid_r'] > 2)].index)
      # unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
      # unique_authors = unique_authors[unique_authors["PI_IDS"] == 3].index
      # indie_authors = df[df['last_author_name'].isin(unique_authors)].groupby(['last_author_name','PI_IDS'])[['pmid']]                                                                  .nunique().reset_index(1)

      # indie_authors2 = indie_authors.join(indie_authors, lsuffix="_l", rsuffix='_r')
      # indie_authors = indie_authors2.join(indie_authors, lsuffix="_l", rsuffix='_r').reset_index()

      # indie_authors = indie_authors[(indie_authors["PI_IDS_l"] != indie_authors["PI_IDS_r"]) & \
      #                               (indie_authors["PI_IDS_l"] != indie_authors["PI_IDS"]) & \
      #                               (indie_authors["PI_IDS_r"] != indie_authors["PI_IDS"])].drop_duplicates("last_author_name",keep="first")                                                                                        .set_index('last_author_name')

      # possible_authors = list(indie_authors[(indie_authors["pmid_l"] > 3) & \
      #                                       (indie_authors["pmid_r"] > 3) & \
      #                                       (indie_authors["pmid"] > 3)].index)

      # unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
      # unique_authors = unique_authors[unique_authors["PI_IDS"] == 2].index
      # #Combine rows based off last_author_name 
      # indie_authors = df[df['last_author_name'].isin(unique_authors)].groupby(['last_author_name','PI_IDS'])[['pmid']].nunique().reset_index(1)
      # indie_authors = indie_authors.join(indie_authors, lsuffix="_l", rsuffix='_r').reset_index()
      # indie_authors = indie_authors[indie_authors["PI_IDS_l"] != indie_authors["PI_IDS_r"]].drop_duplicates("last_author_name",keep="first").set_index('last_author_name')
      # #Each need to have more than 5 papers and need to have an equal number of papers
      # possible_authors = np.concatenate((possible_authors,(list(indie_authors[(indie_authors["pmid_l"] > 4) & 
      #                                       (indie_authors["pmid_r"] > 4)].index))))
      return df[df["last_author_name"].isin(possible_authors)]
    else:
        print("USE CASE GIVEN NOT FAMILIAR - PLEASE CHECK DOCSTRING")

def get_use_case(df, use_case):
    '''
    In order to get accurate results, we need to run many scenarios for each use case. 
    This function just gets a list of all candidates for the use-cases.

    Parameters:
        df - Dataframe of publications
        use_case = possible use_cases
                    3_ua_same - 3 Unique Authors with similar num papers
                    2_ua_dif - 2 Unique Authors with dif. num papers
                    2_da_same - 2 Disambiguated Authors with same num papers
                    2_da_dif -  2 Disambiguated Authors with dif num papers

    Return:
        List of all possible author names that fit the use_case
    '''
    if use_case == "3_ua_same":
        #Three Unique author where each is ~ 30 papers...
        #Get Unique authors
        unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
        unique_authors = unique_authors[unique_authors["PI_IDS"] == 1].index
        #Take only whose papers are between 27 and 33 papers
        bool_authors_30 = df[df['last_author_name'].isin(unique_authors)].groupby('last_author_name')['pmid'].size().between(27,33)
        possible_authors_30 = list(unique_authors[bool_authors_30])
        return possible_authors_30
    elif use_case == "2_ua_dif":
         #Two Unique authors where one is ~ 30 papers and the other is ~10 papers
         #Get Unique authors
        unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
        unique_authors = unique_authors[unique_authors["PI_IDS"] == 1].index
        #Take only whose papers are between 27-33 and 8-12
        bool_authors_30 = df[df['last_author_name'].isin(unique_authors)].groupby('last_author_name')['pmid'].size().between(27,33)
        bool_authors_10 = df[df['last_author_name'].isin(unique_authors)].groupby('last_author_name')['pmid'].size().between(8,12)
        possible_authors_30 = list(unique_authors[bool_authors_30])
        possible_authors_10 = list(unique_authors[bool_authors_10])   
        return (possible_authors_30,possible_authors_10)
    elif use_case == '2_da_same':
        #Two disambiguated authors where both have more than 5 papers and have a close number of papers (3 or less)
        #Get disambiguated authors
        unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
        unique_authors = unique_authors[unique_authors["PI_IDS"] == 2].index
        #Combine rows based off last_author_name 
        indie_authors = df[df['last_author_name'].isin(unique_authors)].groupby(['last_author_name','PI_IDS'])[['pmid']].nunique().reset_index(1)
        indie_authors = indie_authors.join(indie_authors, lsuffix="_l", rsuffix='_r').reset_index()
        indie_authors = indie_authors[indie_authors["PI_IDS_l"] != indie_authors["PI_IDS_r"]].drop_duplicates("last_author_name",keep="first").set_index('last_author_name')
        #Each need to have more than 5 papers and need to have an equal number of papers
        possible_authors_same = indie_authors[(indie_authors["pmid_l"] > 5) & 
                                            (indie_authors["pmid_r"] > 5) &
                                        (np.abs(indie_authors["pmid_l"] - indie_authors["pmid_r"]) < 4)]
        return list(possible_authors_same.index)
    elif use_case == '2_da_dif':
        #Two disambiguated authors with at both at least 3 papers, but dif. number of clusters (dif > 5)
        #Get disambiguate authors
        unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
        unique_authors = unique_authors[unique_authors["PI_IDS"] == 2].index
        #Combine rows based off last_author_name
        indie_authors = df[df['last_author_name'].isin(unique_authors)].groupby(['last_author_name','PI_IDS'])[['pmid']].nunique().reset_index(1)
        indie_authors = indie_authors.join(indie_authors, lsuffix="_l", rsuffix='_r').reset_index()
        indie_authors = indie_authors[indie_authors["PI_IDS_l"] != indie_authors["PI_IDS_r"]].drop_duplicates("last_author_name",keep="first").set_index('last_author_name')
        #Each need at least 4 papers and dif. has to be greater than 6
        possible_authors_dif = indie_authors[(indie_authors["pmid_l"] > 3) & 
                                            (indie_authors["pmid_r"] > 3) &
                                            (np.abs(indie_authors["pmid_l"] - indie_authors["pmid_r"]) > 5)]
        return list(possible_authors_dif.index)
    else:
        print("USE CASE NOT FOUND -  PLEASE LOOK AT DOCUMENTATION")
        return None

def get_similarity_matrix(ps,dfs_authors,scaler=None,flag_base= True):
  '''
  Using Yuval's code, we take the dataframe, and for:
  `Authors, Mesh, Forenames, Institutions, Emails, Countries`
  We compute similarities and return a similarity matrix.

  If flag_base = True, we are trying to train the LR model, and therefore want to make sure we don't
  have any duplicates. We are also creating a Scaler instance to be used for future scaling.

  If flag_base = False, we are trying to get a distance matrix for the DBScan and need the 
  duplicates, because the algorithm takes a square matrix. We are also normalizing with Scaler instance.

  Input: 
    ps - PaperSource instance
    dfs_authors - Dataframe of all features for given authors
    flag_base - if base case - remove doubles and create scaler
    scaler - if not the base_case, the Scaler to normalize the values.

  Output:
    sim_matrix - Matrix based off the similarity of features for given pairs of documents.
    scaler - instance of Scaler
  '''
  print("Getting Similarities")
  paper_clusterer=PaperClusterer(eps=1.27)

  same_author_list = list(dfs_authors['last_author_name'].unique())
  total_df = pd.DataFrame()


  #get dist_matrix for every possible pair...
  for i, same_author in enumerate(same_author_list):
    print("Author {} within {}".format(i+1,len(same_author_list)))
    df_temp = dfs_authors[dfs_authors['last_author_name'] == same_author]

    num_papers = df_temp.shape[0]
    print("Total number of papers: ", num_papers)


    #add to pairs if they are the same author or not

    pid_list = list(df_temp['PI_IDS'])
    pair_col = []

    for j in range(num_papers):
      for k in range(num_papers):
        if pid_list[j] == pid_list[k]:
          pair_col.append(0)
        else:
          pair_col.append(1)

    sim_matrix = paper_clusterer.get_dist_matrix(df_temp, True)
    sim_matrix['same_author'] = pair_col    


    ### --- Removing Pairs --- ###

    #If we are learning our LR weights
    if flag_base:
      print("Removing Doubles")

      pairs = []
      for i in range(num_papers):
        for j in range(num_papers):
          if (i<j):
            pairs.append(True)
          else:
            pairs.append(False)
      sim_matrix = sim_matrix.iloc[pairs]
    total_df = pd.concat([total_df,sim_matrix])

  if flag_base:
    #Normalize the data
    scaler =  StandardScaler()
    total_df.iloc[:,:-1] = scaler.fit_transform(total_df.iloc[:,:-1])
  else:
    # Normalize the data
    total_df.iloc[:,:-1] = scaler.transform(total_df.iloc[:,:-1])

  print("Returning Similarity Matrix.")
  print("Number of pairs after cleaning: ", len(total_df.index))
  return total_df, scaler
