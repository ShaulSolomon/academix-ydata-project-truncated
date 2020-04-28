#Initializations

import os, re, sys

CWD = 'c:\\Users\\shaul\\Documents\\GitHub\\academix-ydata-project\\code'
if os.getcwd() != CWD:
    os.chdir("./code/")

from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto import s3
#import boto3
import importlib


import pandas as pd
import numpy as np
#import s3_functions as s3func

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from yuval_module.paper_clusterer import PaperClusterer
from yuval_module.paper_source import PaperSource

from sklearn.linear_model import LogisticRegression as LogR

#%matplotlib inline

PATH = "C:/Users/shaul/Documents/GitHub/academix-ydata-project/data/labeled_data/"
FILE = "enriched_labeled_dataset.json"

def load_dataset(set_name):
    ps=PaperSource()
    ps.load_dataset(set_name)
    return ps
def get_res_papers(ps,author_name):
    df=ps.get_dataset()
    return df[df['last_author_name']==author_name]


def top_N_indie_authors(df,N=5):
    '''
    input: df - dataframe of all pmid with all relevent details
            N - number of authors to take

    output: dataframe with the top N publishers who we know do not have name disambiguation
    '''
    unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
    unique_authors = unique_authors[unique_authors["PI_IDS"] == 1].index
    top_N_authors = list(df[df['last_author_name'].isin(unique_authors)].groupby('last_author_name')['pmid'].nunique().sort_values(ascending=False)[:N].index)
    df_authors = df[df['last_author_name'].isin(top_N_authors)]
    return df_authors


def get_similarity_matrix(ps,authors_dfs):

  ### --- Getting general similarity matrix --- ###

  num_papers = authors_dfs.shape[0]
  print("Total number of papers: ", num_papers)

  print("Building Same Author Column")
  #get similarity column
  author_list = list(authors_dfs['last_author_name'])
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

  # If we have N documents, the matrix will give us an NxN similarity matrix. 
  # We can simplify our task by removing N_i x N_i (a document with itself)
  # And by removing N_j x N_i (where j > i), because it will already be covered by N_i x N_j 
  
  print("Removing Doubles")
  #Get all possible pairs of combination expressed in "AA", "AB", etc.
  # pairs = []
  # for i in range(num_papers):
  #   outer_index = chr(65+i)
  #   for j in range(num_papers):
  #     inner_index = chr(65 + j)
  #     pairs.append(outer_index + inner_index)

  # #join it back to sim_matrix
  # sim_matrix['pairs'] = [pair for pair in pairs]

  # split_pairs = [list(pair) for pair in pairs]
  # #keep instance only where N_i x N_j (where i < j, which includes i == j and i < j)
  # split_pairs = ["".join(pair) for pair in split_pairs if ord(pair[0]) < ord(pair[1])]
  # #filter matrix
  # sim_matrix = sim_matrix.loc[sim_matrix['pairs'].isin(split_pairs)]
  # sim_matrix.set_index("pairs",inplace=True)
  pairs = []
  for i in range(num_papers):
    for j in range(num_papers):
      if (i<j):
        pairs.append(True)
      else:
        pairs.append(False)

  sim_matrix = sim_matrix.iloc[pairs]

  print("Returning Similarity Matrix.")
  return sim_matrix

def get_train_test(df,perc_change = 0.8):
  #Get equal number of same and different for Logistic Regression

  df_same = df[df['same_author'] == 0]
  df_dif = df[df['same_author'] == 1]
  num_dif = len(df_dif.index)
  num_same = len(df_same.index)
  idx_rand = list(np.random.choice(range(num_dif),num_same,replace=False))
  df_dif = df_dif.iloc[idx_rand]

  perc_train = int(len(df_dif.index)*perc_change)

  train_df = pd.concat((df_same[:perc_train],df_dif[:perc_train]))
  test_df = pd.concat((df_same[perc_train:],df_dif[perc_train:]))
  return train_df.iloc[:,:-1] , train_df.iloc[:,-1], test_df.iloc[:,:-1], test_df.iloc[:,-1]

def log_model(X_train,y_train,X_test,y_test):
  lr_model = LogR(penalty='l2',max_iter=1000,random_state=42)
  lr_model.fit(X_train,y_train)
  #train LR with 0- same author 1- dif author
  y_hat = lr_model.predict_proba(X_test)
  pred_prob = [max(vals) for vals in y_hat]
  return pred_prob

if __name__ == "__main__":
  CWD = 'c:\\Users\\shaul\\Documents\\GitHub\\academix-ydata-project\\code'
  if os.getcwd() != CWD:
      os.chdir("./code/")

  if os.path.exists(PATH + FILE):
      print("READING FROM LOCAL")
      df = pd.read_json(PATH+ FILE)
      ps = PaperSource()
  else:
      print("PULLING FROM S3")
      ps = load_dataset("enriched_labeled")
      df = ps.get_dataset()

  df.drop(columns="last_author_country",inplace=True)
  df.rename(columns={'ORG_STATE':'last_author_country'},inplace=True)

  #Get unique authors
  df = top_N_indie_authors(df,5)
  df = get_similarity_matrix(ps,df)
  X_train, y_train, X_test, y_test = get_train_test(df,0.8)
  y_hat = log_model(X_train,y_train,X_test,y_test)
  plt.hist(y_hat,bins=50)
  
  
