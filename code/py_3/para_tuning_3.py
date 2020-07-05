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

from sklearn.linear_model import LogisticRegression as LogR

from sklearn.pipeline import Pipeline

def pipeline(data_set,val_set,ps):
    '''
    Pipelines from Author Names -> Pairs Weights -> Log R model -> DB Scan -> F1 Score
    '''
    sim_matrix_train, scaler = sim_matrix_3.get_similarity_matrix(ps,train_set,scaler=None,flag_base=True)
    sim_matrix_val, scaler = sim_matrix_3.get_similarity_matrix(ps,val_set,scaler=scaler,flag_base=False)
    
    X_train, y_train = lr_model_3.get_train_all(sim_matrix)
    X_val, y_val = sim_matrix_val.iloc[:,:-1] , sim_matrix_val.iloc[:,-1]

    ###Possible hyper-parameters to tune
    lr__penalty = ['l1', 'l2','elasticnet']
    lr__C = np.logspace(0, 4, 10),
    lr__solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    db__eps = np.linspace(.43,.58,20)
    s = [lr__penalty, lr__C[0], lr__solver[0], db__eps]
    all_combinations = list(itertools.product((*s)))
    
    for (lr_penalty, lr_C, lr_solver, db_eps) in all_combinations:
        log = LogR(penalty=lr_penalty, C=lr_C, solver=lr_solver)
        model = log.fit(X_train,y_train)
        

    
    
    
