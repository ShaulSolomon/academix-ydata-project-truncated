#Initializations

import os, re, sys

CWD = 'c:\\Users\\shaul\\Documents\\GitHub\\academix-ydata-project\\code'
# if os.getcwd() != CWD:
#     os.chdir("./code/")

from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto import s3
import boto3
import importlib
import math
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN as DBS
from collections import Counter 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from collections import defaultdict

import pandas as pd
import numpy as np
#import s3_functions as s3func

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from yuval_module.paper_clusterer import PaperClusterer
from yuval_module.paper_source import PaperSource

from sklearn.linear_model import LogisticRegression as LogR

#%matplotlib inline
import utils as utils
from utils import PROJECT_ROOT

PATH = PROJECT_ROOT+ "data/labeled_data/"
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
    indie_author = list(indie_authors.index)[:15]
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

def get_train_test(df,perc_change = 0.8):
  '''
  Splits the dataframe into Train and Test data, splitting with `perc_change` %.
  To keep a balance of the data, we take the same # of same author pairs to the # of dif author pairs
  The dif author pairs is random, to protect against overfitting.

  Input:
  df - dataframe with all the data
  perc_change - percentage of data to be in Train set

  Output:
  X_train, y_train, X_test, y_test
  '''
  #Get equal number of same and different for Logistic Regression
  df_same = df[df['same_author'] == 0]
  df_dif = df[df['same_author'] == 1]
  num_dif = len(df_dif.index)
  num_same = len(df_same.index)
  #Randomize which pairs to take
  idx_rand = list(np.random.choice(range(num_dif),num_same,replace=False))
  df_dif = df_dif.iloc[idx_rand]

  perc_train = int(len(df_dif.index)*perc_change)

  print("There are {} pairs being used, half of them with the same author, {} of them as train data".format(num_same*2,perc_train*2))

  train_df = pd.concat((df_same[:perc_train],df_dif[:perc_train]))
  test_df = pd.concat((df_same[perc_train:],df_dif[perc_train:]))
  return train_df.iloc[:,:-1] , train_df.iloc[:,-1], test_df.iloc[:,:-1], test_df.iloc[:,-1]


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def log_model(X_train,y_train,X_test,y_test):
  '''
  Learns the best Logistic Regression model using GridSearhCV, and then implements it on the test data.

  input: 
  X_train, y_train, X_test, y_test

  return: 
  Score - Vanilla score provided by the clf.score() method. In LogR, its accuracy.
  Predict_prob - For the X_test data, get what their value would be with weights (and bias) of LogR model.
  best_model - the clf model we created.
  '''
  penalty = ['l1', 'l2','elasticnet']
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
  predict_prob = apply_weights(X_test,best_model)
  score = clf.score(X_test,y_test)
  return score, predict_prob, best_model

def apply_weights(X_test, best_model):
  '''
  Applies the weights and bias to X_test data.

  input:
  X_test
  best_model - model learned from the training data

  output:
  predict_prob - guesses for each X_test
  '''
  weights = best_model.best_estimator_.coef_.flatten()
  bias = best_model.best_estimator_.intercept_.flatten()
  predict_prob = [sigmoid(np.dot(x_test,weights) + bias[0]) for x_test in X_test.to_numpy()]
  return predict_prob

def get_dist_matrix(ps,df,model,flag_no_country = False):
  '''
  Get the estimated LogR value for each doc pair.

  input:
  ps - Papersource instance
  df - dataframe
  model - LogR model
  flag_no_country - flag if we want to include country similarity as feature

  output:
  sim_matrix - Similarity Matrix
  '''
  df_sim = get_similarity_matrix(ps,df,False)
  X_feat = df_sim.iloc[:,:-1]
  if flag_no_country:
    X_feat.drop(columns="country",inplace=True)
  X_feat_weights = apply_weights(X_feat,model)
  num_paper = int(np.sqrt(len(X_feat_weights)))
  #Need a square matrix for DBScan
  sim_matrix = np.array(X_feat_weights).reshape(num_paper,-1)
  return sim_matrix


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
  score, y_hat,_ = log_model(X_train,y_train,X_test,y_test)
  print(score)
  
