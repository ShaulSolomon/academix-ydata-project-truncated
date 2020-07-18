import py_3.sim_matrix_3 as sim_matrix_3
import py_3.lr_model_3 as lr_model_3
import py_3.db_scan_3 as db_scan_3
import metric_eval_2

from itertools import product
import numpy as np

import math
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN as DBS
from sklearn.linear_model import LogisticRegression as LogR
from collections import Counter 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from collections import defaultdict

from yuval_module.paper_clusterer import PaperClusterer


import itertools

from sklearn.linear_model import LogisticRegression as LogR

def pipeline(sim_matrix_train,dict_auth,ps) -> dict:
    '''
    Pipelines from Author Names -> Pairs Weights -> Log R model -> DB Scan -> F1 Score
    
        :param pd.DataFrame sim_matrix_train - the train data pre-converted to a similarity matrix
        :param dict dict_auth - per author combination of all the df's assosciated and their respective sim. matrices
        :param PaperSource ps - for yuvals code
        
        :return dict - the hyperparameters and the weights/bias after running every possible permutation permissible
    '''
    X_train, y_train = lr_model_3.get_train_all(sim_matrix_train)

    ###Possible hyper-parameters to tune
    lr__penalty = ['l1', 'l2','elasticnet']
    lr__C = np.logspace(0, 4, 10),
    lr__solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    db__eps = np.linspace(.3,2.58,20)
    s = [lr__penalty, lr__C[0], lr__solver[0], db__eps]
    all_combinations = list(itertools.product((*s)))
    
    best_penalty = None
    best_C = None
    best_weights = None
    best_bias = None
    best_solver = None
    best_eps = None
    best_F1 = 0.0

    for (lr_penalty, lr_C, lr_solver, db_eps) in all_combinations:
        try:
            log = LogR(penalty=lr_penalty, C=lr_C, solver=lr_solver)
            model = log.fit(X_train,y_train)
            weights = model.coef_[0]
            bias = model.intercept_[0]
            
            df_all_cases = []
            
            
            for auth in dict_auth.keys():
                df_auth = dict_auth[auth]['df']
                df_sim =  dict_auth[auth]['sim_mat']
                X_feat = df_sim.iloc[:,:-1]
                X_feat_weights = [lr_model_3.sigmoid(np.dot(x_test,weights) + bias) for x_test in X_feat.to_numpy()]
                num_paper = int(np.sqrt(len(X_feat_weights)))
                dist_mat = np.array(X_feat_weights).reshape(num_paper,-1)
                df_all_cases.append([df_auth,dist_mat])
                
                
    
            y_hat_comb = []

            for case in df_all_cases:
                df_clus, df_case = case
                y_hat = DBS(eps=db_eps,min_samples=1,metric="precomputed").fit(df_case)
                df_clus = df_clus[["pmid","PI_IDS"]]
                df_clus['cluster_pred'] = y_hat.labels_
                y_hat_comb.append(df_clus)
                   
            f1, precision, recall, df_eval = metric_eval_2.get_metrics_many(y_hat_comb)
                        
            if f1 > best_F1:
                best_F1 = f1
                best_penalty = lr_penalty
                best_C = lr_C
                best_solver = lr_solver
                best_eps = db_eps
                best_weights = weights
                best_bias = bias
        except:
            continue
            
    return {'best_F1':best_F1,
            'best_penalty':best_penalty,
            'best_C':best_C,
            'best_solver':best_solver,
            'best_eps':best_eps,
            'best_weights':best_weights,
            'best_bias':best_bias}


def pipeline_yuvals_code(sim_matrix_train,df,dict_auth,scaler,ps) -> dict:
    '''
    Pipelines from Author Names -> Pairs Weights -> Log R model -> DB Scan -> F1 Score
    
        :param pd.DataFrame sim_matrix_train - the train data pre-converted to a similarity matrix
        :param dict dict_auth - per author combination of all the df's assosciated and their respective sim. matrices
        :param PaperSource ps - for yuvals code
        
        :return dict - the hyperparameters and the weights/bias after running every possible permutation permissible
    '''
    
    X_train, y_train = lr_model_3.get_train_all(sim_matrix_train)

    ###Possible hyper-parameters to tune
    lr__penalty = ['l1', 'l2','elasticnet']
    lr__C = np.logspace(0, 4, 10),
    lr__solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    db__eps = np.linspace(0.1,2.0,30)
    s = [lr__penalty, lr__C[0], lr__solver[0], db__eps]
    all_combinations = list(itertools.product((*s)))
    
    best_penalty = None
    best_C = None
    best_solver = None
    best_weights = None
    best_bias = None
    best_eps = None
    best_F1 = 0.0

    for (lr_penalty, lr_C, lr_solver, db_eps) in all_combinations:
        try:
            log = LogR(penalty=lr_penalty, C=lr_C, solver=lr_solver)
            model = log.fit(X_train,y_train)
            weights = model.coef_[0]
            bias = model.intercept_[0]

            df_all_cases = []
            y_hat_comb = []

            for auth in list(dict_auth.keys()):
                df_auth = dict_auth[auth]['df']
                param_dict = {}
                param_dict['author'], param_dict['mesh'], param_dict['inst'], param_dict['email'], param_dict['country'], param_dict['forename'] = weights
                param_dict['email'] = 0


                paper_clusterer= PaperClusterer(db_eps, gammas=param_dict , scaler=scaler,bias=bias)
                # dist matrix
                combined_dist, combined_sim, total_df = paper_clusterer.get_dist_matrix(df_auth)
                # cluster
                res_clusters, cluster_dfs=paper_clusterer.cluster_res(total_df)
                cluster_dfs = cluster_dfs.rename(columns={'cluster':'cluster_pred'})

                y_hat_comb.append(cluster_dfs[["pmid","PI_IDS","cluster_pred"]])


            f1, precision, recall, df_eval = metric_eval_2.get_metrics_many(y_hat_comb)

            if f1 > best_F1:
                best_F1 = f1
                best_penalty = lr_penalty
                best_C = lr_C
                best_solver = lr_solver
                best_eps = db_eps
                best_weights = weights
                best_bias = bias
        except:
            continue
            
    return {'best_F1':best_F1,
            'best_penalty':best_penalty,
            'best_C':best_C,
            'best_solver':best_solver,
            'best_eps':best_eps,
            'best_weights':best_weights,
            'best_bias':best_bias}
