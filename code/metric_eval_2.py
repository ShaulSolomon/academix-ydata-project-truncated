import pandas as pd
import numpy as np

from collections import Counter 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from collections import defaultdict

def assign_labels_to_clusters(df_core, num_clusters):
  '''
  Based off algorithm in 2_metric_classification, use greedy algorithm to assign a PI_ID to a cluster.

  input:
  df_core - dataframe of doc pairs with dbscan approximation
  num_clusters - the number of clusters the dbscan assumed.

  output:
  df_core - return dataframe with the cluster_assigned column, which cluster the paper should be assigned to. 
  '''
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
  
  #give pi_id with no cluster, c = -1 (both where #clus > #pi_id and #pi _id > #clus)    
  for pi_id in pi_id_sorted:
    if pi_id not in K_dict:
      K_dict[pi_id] = -1

  df_core['cluster_assigned'] = [K_dict[pid] for pid in df_core.PI_IDS]
  return df_core

def get_metrics(df):
  '''
  Using cluster_pred (given by DBScan) and cluster_assigned (given by greedy algo),
  compute Precision, Recall, Mis-Integration, and Mis-Separation.

  Parameters:
    df - Dataframe with pmid, pi_id, cluster_pred, cluster_assigned

  Return:
    num_clusters_db - # of clusters created by DBScan
    num_authors - # of authors
    precision - precision score
    recall - recall score
    df_eval - DF with Mis-Integration and Mis_separation
  '''

  num_clusters_db = len(np.unique(df.cluster_pred))
  num_authors = len(np.unique(df.PI_IDS))
  precision = (precision_score(df.cluster_assigned,df.cluster_pred,average='weighted'))
  recall = recall_score(df.cluster_assigned,df.cluster_pred,average='weighted')
  #print("Number of clusters (DBS): {}\nNumber of unique authors: {}".format(num_clusters_db,num_authors))

  #print("Precision score: {}, Recall score: {}".format(precision, recall))
  mis_intergration_dict = dict()
  mis_separation_dict = dict()

  eval_dict = dict()
  eval_dict['mis_integration'] = mis_intergration_dict
  eval_dict['mis_separation'] = mis_separation_dict

  for clus_sep in df.groupby('PI_IDS')['cluster_pred'].nunique():
    if clus_sep in eval_dict['mis_separation'].keys():
      eval_dict['mis_separation'][clus_sep] += 1
    else:
      eval_dict['mis_separation'][clus_sep] = 1
  # total = sum(mis_separation_dict.values())
  # for key in mis_separation_dict.keys():
  #   mis_separation_dict[key] /= total
  # print("Mis-Separation: ", df_sep)

  for clus_int in df.groupby('cluster_assigned')['cluster_pred'].nunique():
    if clus_int in eval_dict['mis_integration'].keys():
      eval_dict['mis_integration'][clus_int] +=1
    else:
      eval_dict['mis_integration'][clus_int] = 1
  # total = sum(mis_intergration_dict.values())
  # for key in mis_intergration_dict.keys():
  #   mis_intergration_dict[key] /= total
  #print("Mis-Integration: ",mis_intergration_dict)
  df_eval = pd.DataFrame.from_dict(eval_dict, orient = "index")
  #sort the index values
  column = list(df_eval.columns)
  column.sort()
  new_col = ["{} cluster(s)".format(i) for i in column]
  df_eval = df_eval[column]
  df_eval.columns = new_col
  #print(df_eval)
  return num_clusters_db, num_authors, precision, recall, df_eval

def get_metrics_many(group_cases):
  '''
  Iterate through each of the cases and extract their individual metrics and get a total scoring

  Parameters:
    group_cases - list of all the dataframes of each DB case
  
  Return
    F1  - F1Score
  '''
  total_recall = np.array([])
  total_precision = np.array([])
  total_df_eval = pd.DataFrame()

  for i,group in enumerate(group_cases):
    df_core = assign_labels_to_clusters(group, group['cluster_pred'].unique())
    num_clusters_db, num_authors, precision, recall, df_eval = get_metrics(df_core)
    print("Situation {}".format(i))
    print("Num Clusters: ", num_clusters_db)
    print("Num Unique Authors: ", num_authors)
    print("Precision: ", precision)
    print("Recall: ",recall)
    print(df_eval.T)
    print("\n-------------------\n")
    total_precision = np.concatenate((total_precision,[precision]))
    total_recall = np.concatenate((total_recall,[recall]))
    total_df_eval = (total_df_eval.reindex_like(df_eval).fillna(0) + df_eval.fillna(0).fillna(0))

  
  # //TODO: Do we need a weighted mean or because the cases are similar enough, we can give them equal worth? 

  total_precision = np.mean(total_precision)
  total_recall = np.mean(total_recall)
  print("\n\nTotal Precision: {}\tTotal Recall: {}".format(total_precision,total_recall))
  total_df_eval = total_df_eval.T / total_df_eval.T.sum()
  print(total_df_eval.T)
  F1 = 2 * (total_precision * total_recall) / (total_precision + total_recall)
  return F1


if __name__ == "__main__":
    pass