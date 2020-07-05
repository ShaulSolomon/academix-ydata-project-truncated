#Initializations

import math
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN as DBS
from collections import Counter 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from collections import defaultdict

import os, sys
import py_3.sim_matrix_3 as sim_matrix_3

import pandas as pd
import numpy as np
#import s3_functions as s3func

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


from sklearn.linear_model import LogisticRegression as LogR

def get_train_test(df,perc_change = 0.8):
  '''
  Splits the dataframe into Train and Test data, splitting with `perc_change` %.
  To keep a balance of the data, we take the same # of same author pairs to the # of dif author pairs
  The dif author pairs is random, to protect against overfitting.

  If we are dealing with DA case, we want to make sure that are enough cases with the same name, so that
  the model learns that name as a similarity feature is important.

  Input:
    df - dataframe with all the data
    perc_change - percentage of data to be in Train set
    da_samename_perc - perc of pairs not with the same author but have the same name

  Output:
  X_train, y_train, X_test, y_test
  '''

  #Get equal number of same and different for Logistic Regression
  df_same = df[df['same_author'] == 0]
  num_same = len(df_same.index)
  df_dif = df[df['same_author'] == 1]
  num_dif = len(df_dif.index)

  print("Same author #: {}, dif author #: {}".format(num_same,num_dif))

  #Randomize which pairs to take
  np.random.seed(42)
  if num_same < num_dif:
    idx_rand = list(np.random.choice(range(num_dif),num_same,replace=False))
    df_dif = df_dif.iloc[idx_rand]
  else:
    idx_rand = list(np.random.choice(range(num_same),num_dif,replace=False))
    df_same = df_same.iloc[idx_rand]

  perc_train = int(len(df_dif.index)*perc_change)

  print("There are {} pairs being used, half of them with the same author, {} of them as train data".format(min(num_same,num_dif)*2,perc_train*2))

  train_df = pd.concat((df_same[:perc_train],df_dif[:perc_train]))
  test_df = pd.concat((df_same[perc_train:],df_dif[perc_train:]))
  return train_df.iloc[:,:-1] , train_df.iloc[:,-1], test_df.iloc[:,:-1], test_df.iloc[:,-1]

def get_train_all(df):
    '''
    Ensures that there are an equal number of same_author and dif_author pairs so that the model wont be biased to dif. author cases.
    '''
    #Get equal number of same and different for Logistic Regression
    df_same = df[df['same_author'] == 0]
    num_same = len(df_same.index)
    df_dif = df[df['same_author'] == 1]
    num_dif = len(df_dif.index)

    print("Same author #: {}, dif author #: {}".format(num_same,num_dif))

    #Randomize which pairs to take
    np.random.seed(42)
    if num_same < num_dif:
        idx_rand = list(np.random.choice(range(num_dif),num_same,replace=False))
        df_dif = df_dif.iloc[idx_rand]
    else:
        idx_rand = list(np.random.choice(range(num_same),num_dif,replace=False))
        df_same = df_same.iloc[idx_rand]

    print("There are {} pairs being used, half of them with the same author.".format(min(num_same,num_dif)*2))

    train_df = pd.concat((df_same,df_dif))
    return train_df.iloc[:,:-1] , train_df.iloc[:,-1]

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
  solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
  hyperparameters = dict(C=C, penalty=penalty, solver=solver)
  log = LogR()
  clf = GridSearchCV(log, hyperparameters, cv=5, verbose=0)
  best_model = clf.fit(X_train, y_train)
  best_penalty = best_model.best_estimator_.get_params()['penalty']
  best_C = best_model.best_estimator_.get_params()['C']
  best_solver = best_model.best_estimator_.get_params()['solver']
  print('Best Penalty: ', best_penalty)
  print('Best C: ', best_C)
  print('Using solver: ',best_solver)
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

def get_dist_matrix(ps,df,scaler, model,flag_no_country = False):
  '''
  Get the estimated LogR value for each doc pair.

  input:
  ps - Papersource instance
  df - dataframe
  scaler - scaler to normalize data
  model - LogR model
  flag_no_country - flag if we want to include country similarity as feature

  output:
  dist_matrix - Similarity Matrix
  '''
  df_sim,_ = sim_matrix_3.get_similarity_matrix(ps,df,scaler,flag_base = False)
  X_feat = df_sim.iloc[:,:-1]
  if flag_no_country:
    X_feat.drop(columns="country",inplace=True)
  X_feat_weights = apply_weights(X_feat,model)
  num_paper = int(np.sqrt(len(X_feat_weights)))
  #Need a square matrix for DBScan
  dist_matrix = np.array(X_feat_weights).reshape(num_paper,-1)
  return dist_matrix


if __name__ == "__main__":
  path = os.getcwd()
  print(path)
  
