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

def get_use_case(df, use_case):
    '''
    In order to get accurate results, we need to run many scenarios for each use case. 
    This function just gets a list of all candidates for the use-cases.

    Parameters:
        df - Dataframe of publications
        use_case = possible use_cases
                    2_da_same - 2 Disambiguated Authors with same num papers
                    2_da_dif -  2 Disambiguated Authors with dif num papers

    Return:
        List of all possible author names that fit the use_case
    '''
    if use_case == '2_da_same':
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
    elif use_case == '3_da':
        #Three disambiguated authors, with atleast 3 papers each (all types, same and dif)
        unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
        unique_authors = unique_authors[unique_authors["PI_IDS"] == 3].index
        return unique_authors
    elif use_case == '1_da':
        unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
        unique_authors = unique_authors[unique_authors["PI_IDS"] == 1].index
        indie_authors = df[df['last_author_name'].isin(unique_authors)].groupby(['last_author_name','PI_IDS'])[['pmid']].nunique().reset_index(1)
        indie_authors = list(indie_authors[(indie_authors.pmid > 7) & (indie_authors.pmid < 40)].index)
        len_all = len(indie_authors)
        np.random.seed(42)
        rand_auth = np.random.choice(range(len_all),50,replace=False)
        return list(np.array(indie_authors)[rand_auth])
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


def split_authors(df: pd.DataFrame):
    '''
    Given a dataframe, we want to be able to create use-cases while ensuring there is no overlap between groups to ensure there is no bias.
    We take only cases of 2 and 3 DA (as they are the most common), and split them up 2/5,2/5,1/5 for Core [train_set], Use_Case [test_set],
    and Eps [val_set] respectively.
    
        :param pd.DataFrame df - our dataframe created from NIH dataset
        :list auth_core - list of all authors in train_set
        :list auth_eps - list of all authors in val_set
        :list auth_usecase - list of all authors in test_set
    '''
    #TWO DA CASE
    unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
    unique_authors = unique_authors[unique_authors["PI_IDS"] == 2].index
    #Combine rows based off last_author_name
    indie_authors = df[df['last_author_name'].isin(unique_authors)].groupby(['last_author_name','PI_IDS'])[['pmid']].nunique().reset_index(1)
    indie_authors = indie_authors.join(indie_authors, lsuffix="_l", rsuffix='_r').reset_index()
    indie_authors = indie_authors[indie_authors["PI_IDS_l"] != indie_authors["PI_IDS_r"]] \
                              .drop_duplicates("last_author_name",keep="first") \
                              .set_index('last_author_name')
    #Each need at least 4 papers and dif. has to be greater than 6
    possible_authors_2 = list(indie_authors[(indie_authors["pmid_l"] > 3) & (indie_authors["pmid_r"] > 3)].index)


    #THREE DA CASE
    unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
    unique_authors = unique_authors[unique_authors["PI_IDS"] == 3].index
    indie_authors = df[df['last_author_name'].isin(unique_authors)].groupby(['last_author_name','PI_IDS'])[['pmid']]                                                                  .nunique().reset_index(1)

    indie_authors2 = indie_authors.join(indie_authors, lsuffix="_l", rsuffix='_r')
    indie_authors = indie_authors2.join(indie_authors, lsuffix="_l", rsuffix='_r').reset_index()

    indie_authors = indie_authors[(indie_authors["PI_IDS_l"] != indie_authors["PI_IDS_r"]) & \
                                (indie_authors["PI_IDS_l"] != indie_authors["PI_IDS"]) & \
                                (indie_authors["PI_IDS_r"] != indie_authors["PI_IDS"])] \
                                .drop_duplicates("last_author_name",keep="first") \
                                .set_index('last_author_name')                     

    possible_authors_3 = list(indie_authors[((indie_authors["pmid_l"] > 3) & \
                                  (indie_authors["pmid_r"] > 3)) | \
                                  ((indie_authors["pmid_l"] > 3) & \
                                  (indie_authors["pmid"] > 3)) | \
                                  ((indie_authors["pmid_r"] > 3) & \
                                  (indie_authors["pmid"] > 3))].index)

    auth_core = []
    auth_usecase = []
    auth_eps = []
    np.random.seed(42)

    num_auth_2 = len(possible_authors_2)
    print("Total number of 2 DA authors: ", num_auth_2)
    rand2 = np.random.choice(range(num_auth_2),num_auth_2, replace=False)
    possible_authors_2 = list(np.array(possible_authors_2)[rand2])
    core_2, usecase_2, eps_2 = possible_authors_2[:int(num_auth_2*(2/5))], \
                            possible_authors_2[int(num_auth_2*(2/5)):2*int(num_auth_2*(2/5))],\
                            possible_authors_2[2*int(num_auth_2*(2/5)):]

    num_auth_3 = len(possible_authors_3)
    print("Total number of 3 DA authors: ",num_auth_3)
    rand3 = np.random.choice(range(num_auth_3),num_auth_3, replace=False)
    possible_authors_3 = list(np.array(possible_authors_3)[rand3])
    core_3, usecase_3, eps_3 = possible_authors_3[:int(num_auth_3*(2/5))], \
                            possible_authors_3[int(num_auth_3*(2/5)):2*int(num_auth_3*(2/5))],\
                            possible_authors_3[2*int(num_auth_3*(2/5)):]

    auth_core.append(core_2)
    auth_core.append(core_3)
    auth_core = (auth_core[0] + auth_core[1])

    auth_usecase.append(usecase_2)
    auth_usecase.append(usecase_3)
    auth_usecase = (auth_usecase[0] + auth_usecase[1])

    auth_eps.append(eps_2)
    auth_eps.append(eps_3)
    auth_eps = (auth_eps[0] + auth_eps[1])

    return auth_core, auth_eps, auth_usecase



