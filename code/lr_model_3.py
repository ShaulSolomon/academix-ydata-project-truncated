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

PATH = "C:/Users/shaul/Documents/GitHub/academix-ydata-project/data/labeled_data/"
FILE = "enriched_labeled_dataset.json"

def load_dataset(set_name):
    ps=PaperSource()
    ps.load_dataset(set_name)
    return ps
def get_res_papers(ps,author_name):
    df=ps.get_dataset()
    return df[df['last_author_name']==author_name]


def top_N_indie_authors(df,N=5, start_index=0, flag_DBS_authors = False):
    '''
    input: df - dataframe of all pmid with all relevent details
           start_index - where to start from, regarding authors
           N - number of authors to take  
           flag_DBS_authors - flag to see if want to take authors for the DBScan

    output: dataframe with the top N publishers who we know do not have name disambiguation
    '''
    #To stay safe, only use authors without disambiguation
    unique_authors = df.groupby('last_author_name')[["PI_IDS"]].nunique()
    unique_authors = unique_authors[unique_authors["PI_IDS"] == 1].index
    #Get top N
    top_N_authors = list(df[df['last_author_name'].isin(unique_authors)].groupby('last_author_name')['pmid'].nunique().sort_values(ascending=False)[start_index:N+start_index].index)
    df_authors = df[df['last_author_name'].isin(top_N_authors)]
    if flag_DBS_authors:
      DB_authors = list(df[df['last_author_name'].isin(unique_authors)].groupby('last_author_name')['pmid'].nunique().sort_values(ascending=False)[:start_index].index)
      df_dbscan = df[df['last_author_name'].isin(DB_authors)]
      return df_authors, df_dbscan
    else:
      return df_authors


def get_similarity_matrix(ps,authors_dfs,flag_remove_doubles = True):

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
  predict_prob = get_weights(X_test,best_model)
  score = clf.score(X_test,y_test)
  return score, predict_prob, best_model

def get_weights(X_test, best_model):
  weights = best_model.best_estimator_.coef_.flatten()
  bias = best_model.best_estimator_.intercept_.flatten()
  predict_prob = [sigmoid(np.dot(x_test,weights) + bias[0]) for x_test in X_test.to_numpy()]
  return predict_prob

def get_dist_matrix(ps,df,model):
  df_sim = get_similarity_matrix(ps,df,False)
  X_feat = df_sim.iloc[:,:-1]
  X_feat_weights = get_weights(X_feat,model)
  num_paper = int(np.sqrt(len(X_feat_weights)))
  sim_matrix = np.array(X_feat_weights).reshape(num_paper,-1)
  return sim_matrix

def assign_labels_to_clusters(df_core, num_clusters):
  K_dict = dict()
  set_of_clusters = set(num_clusters)
  #get pi_ids sorted by most productive
  pi_id_sorted = list(df_core.groupby('PI_IDS').size().sort_values(ascending=False).reset_index()['PI_IDS'])
  for pi_id in pi_id_sorted:
    #Get most popular clusters for each id
    cluster_for_id = [c_id for c_id, _ in Counter(df_core[df_core['PI_IDS'] == pi_id].cluster_pred).most_common()]
    for c in cluster_for_id:
      if c not in set_of_clusters:
        continue
      else:
        K_dict[pi_id] = c
        set_of_clusters.remove(c)
        break

  df_core['cluster_assigned'] = [K_dict[pid] for pid in df_core.PI_IDS]
  return df_core

def get_metrics(df):
  print("Precision score: {}, Recall score: {}".format(precision_score(df.cluster_assigned,df.cluster_pred,average='micro'),
                                                       recall_score(df.cluster_assigned,df.cluster_pred,average='micro')))
  mis_intergration_dict = defaultdict(int)
  mis_separation_dict = defaultdict(int)

  for clus_sep in df.groupby('PI_IDS')['cluster_pred'].nunique():
    mis_separation_dict[clus_sep] += 1
  mis_separation_dict = dict(mis_separation_dict)
  # total = sum(mis_separation_dict.values())
  # for key in mis_separation_dict.keys():
  #   mis_separation_dict[key] /= total

  print("Mis-Separation: ", mis_separation_dict)


  for clus_int in df.groupby('cluster_assigned')['cluster_pred'].nunique():
    mis_intergration_dict[clus_int] += 1
  mis_intergration_dict = dict(mis_intergration_dict)
  # total = sum(mis_intergration_dict.values())
  # for key in mis_intergration_dict.keys():
  #   mis_intergration_dict[key] /= total

  print("Mis-Integration: ",mis_intergration_dict)

  



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
  
