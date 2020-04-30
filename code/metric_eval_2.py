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

  input:
  df - Dataframe with pmid, pi_id, cluster_pred, cluster_assigned

  '''

  num_clusters_db = len(np.unique(df.cluster_pred))
  num_authors = len(np.unique(df.PI_IDS))

  print("Number of clusters (DBS): {}\nNumber of unique authors: {}".format(num_clusters_db,
                                                                                                            num_authors))

  print("Precision score: {}, Recall score: {}".format(precision_score(df.cluster_assigned,df.cluster_pred,average='weighted'),
                                                       recall_score(df.cluster_assigned,df.cluster_pred,average='weighted')))
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
  print(df_eval)


if __name__ == "__main__":
    pass