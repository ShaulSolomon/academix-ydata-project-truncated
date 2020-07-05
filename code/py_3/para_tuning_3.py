import py_3.sim_matrix_3 as sim_matrix_3
import py_3.lr_model_3 as lr_model_3
import py_3.db_scan_3 as db_scan_3
import metric_eval_2

from itertools import product
from sklearn.cluster import DBSCAN as DBS
import numpy as np

import math
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN as DBS
from collections import Counter 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from collections import defaultdict

import itertools

from sklearn.linear_model import LogisticRegression as LogR

def pipeline(df, train_set,val_auth,ps):
    '''
    Pipelines from Author Names -> Pairs Weights -> Log R model -> DB Scan -> F1 Score
    '''
    sim_matrix_train, scaler = sim_matrix_3.get_similarity_matrix(ps,train_set,scaler=None,flag_base=True)    
    X_train, y_train = lr_model_3.get_train_all(sim_matrix_train)

    ###Possible hyper-parameters to tune
    lr__penalty = ['l1', 'l2','elasticnet']
    lr__C = np.logspace(0, 4, 10),
    lr__solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    db__eps = np.linspace(.43,.58,20)
    s = [lr__penalty, lr__C[0], lr__solver[0], db__eps]
    all_combinations = list(itertools.product((*s)))
    
    best_penalty = None
    best_C = None
    best_solver = None
    best_eps = None
    best_F1 = 0.0
    
    dict_auth = {}
    for auth in val_auth:
        dict_auth[auth] = {}
        dict_auth[auth]['df'] = df[df["last_author_name"] == auth]
        dict_auth[auth]['sim_mat'],_ = sim_matrix_3.get_similarity_matrix(ps,dict_auth[auth]['df'],scaler,flag_base = False)

    for (lr_penalty, lr_C, lr_solver, db_eps) in all_combinations:
        try:
            log = LogR(penalty=lr_penalty, C=lr_C, solver=lr_solver)
            model = log.fit(X_train,y_train)
            
            for auth in auth_eps:
                df_auth = dict_auth[auth]['df']
                df_sim =  dict_auth[auth]['sim_mat']
                X_feat = df_sim.iloc[:,:-1]
                X_feat_weights = apply_weights(X_feat,model)
                num_paper = int(np.sqrt(len(X_feat_weights)))
                dist_mat = np.array(X_feat_weights).reshape(num_paper,-1)
                df_all_cases.append([df_auth,dist_mat])

            y_hat_comb = []

            for case in df_all_cases:
                df_clus, df_case = case
                y_hat = DBS(eps=best_eps,min_samples=1,metric="precomputed").fit(df_case)
                df_clus = df_clus.loc[["pmid","PI_IDS"]]
                df_clus['cluster_pred'] = y_hat.labels_
                y_hat_comb.append(df_clus)

            f1, precision, recall, df_eval = metric_eval_2.get_metrics_many(y_hat_comb)
            if f1 > best_F1:
                best_F1 = f1
                best_penalty = lr_penalty
                best_C = lr_C
                best_solver = lr_solver
                best_eps = df_eps     
        except:
            continue
    return best_F1, best_penalty, best_C, best_solver, best_eps
    
    
    
