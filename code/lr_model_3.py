#Initializations

import os, re, sys

CWD = 'c:\\Users\\shaul\\Documents\\GitHub\\academix-ydata-project\\code'
if os.getcwd() != CWD:
    os.chdir("./code/")

from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto import s3
import boto3
import importlib
import math
from sklearn.model_selection import GridSearchCV


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
  
  print("Removing Doubles")

  pairs = []
  for i in range(num_papers):
    for j in range(num_papers):
      if (i<j):
        pairs.append(True)
      else:
        pairs.append(False)

  sim_matrix = sim_matrix.iloc[pairs]

  print("Returning Similarity Matrix.")
  print("Number of pairs after cleaning: ", len(sim_matrix.index))
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


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def log_model(X_train,y_train,X_test,y_test):
  '''
  input: X_train, y_train, X_test, y_test
  return: Score, predict_prob
  '''
  penalty = ['l1', 'l2']
  # Create regularization hyperparameter space
  C = np.logspace(0, 4, 10)
  # Create hyperparameter options
  hyperparameters = dict(C=C, penalty=penalty)
  log = LogR()
  clf = GridSearchCV(log, hyperparameters, cv=5, verbose=0)
  best_model = clf.fit(X_train, y_train)
  best_penalty = best_model.best_estimator_.get_params()['penalty']
  best_C = best_model.best_estimator_.get_params()['C']
  print('Best Penalty:', best_penalty)
  print('Best C:', best_C)
  weights = best_model.best_estimator_.coef_.flatten()
  bias = best_model.best_estimator_.intercept_.flatten()
  print("Calculating predict probability")
  predict_prob = [sigmoid(np.dot(x_test,weights) + bias[0]) for x_test in X_test.to_numpy()]
  score = clf.score(X_test,y_test)
  return score, predict_prob

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
  df = top_N_indie_authors(df,12)
  df = get_similarity_matrix(ps,df)
  X_train, y_train, X_test, y_test = get_train_test(df,0.8)
  score, y_hat = log_model(X_train,y_train,X_test,y_test)
  print(score)
  
