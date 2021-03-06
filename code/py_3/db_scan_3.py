import py_3.sim_matrix_3 as sim_matrix_3
import py_3.lr_model_3 as lr_model_3
import metric_eval_2

from itertools import product
from sklearn.cluster import DBSCAN as DBS
import numpy as np


def db_multiple(ps, df, scaler,authors, use_case, weights,bias,epsilon):
    '''
    Gathers several situations for each use case and calculates their y_hats

    Parameters:
        ps - PaperSource
        df - dataframe with all the details
        authors - list of authors who we draw from
        scaler - scaler to normalize our data
        use_case - the use_case we want to explore
        num_cases - the number of dif. cases we try (if num_auth < num_cases, we only do num_auth)
        model - model learned from the LR model

    Return:
        best_eps - best epsilon
        f1_scores - list of all f1 scores for epsilons
    '''
    #Get combinations of authors from the given use_case
    
    if use_case == "1_da" or use_case == "mix_bag":
        authors = sim_matrix_3.get_use_case(df,use_case)
    else:
        auth_df = df[df['last_author_name'].isin(authors)]   
        authors = sim_matrix_3.get_use_case(auth_df,use_case)
    
    num_cases = len(authors)

    df_all_cases = []
    all_papers = []

    for i,auth in enumerate(authors):
        print("Processing combination number {} from {}".format(i+1,num_cases))
        df_auth = df[df['last_author_name'] == auth]
        all_papers.append(df_auth.shape[0])
        #Calculate the distance matrix
        
        print(auth)
        df_sim,_ = sim_matrix_3.get_similarity_matrix(ps,df_auth,scaler,flag_base = False)
        X_feat = df_sim.iloc[:,:-1]
        X_feat_weights = [lr_model_3.sigmoid(np.dot(x_test,weights) + bias) for x_test in X_feat.to_numpy()]
        num_paper = int(np.sqrt(len(X_feat_weights)))
        #Need a square matrix for DBScan
        dist_mat = np.array(X_feat_weights).reshape(num_paper,-1)
        df_all_cases.append([df_auth,dist_mat])
    

    y_hat_comb = []
    
    for case in df_all_cases:
        df_clus, df_case = case
        y_hat = DBS(eps=epsilon,min_samples=1,metric="precomputed").fit(df_case)
        df_clus = df_clus[["pmid","PI_IDS"]]
        df_clus['cluster_pred'] = y_hat.labels_
        y_hat_comb.append(df_clus)
    
    return y_hat_comb, num_cases, np.mean(np.array(all_papers))


def find_epsilon(ps, df, scaler, authors, model,epsilons):

    np.random.seed(42)

    num_authors = len(authors)

    df_all_cases = []

    for i,auth in enumerate(authors):
        print("Processing combination number {} from {}".format(i+1,num_authors))
        df_auth = df[df['last_author_name'] == auth]
        #Calculate the distance matrix
        dist_mat = lr_model_3.get_dist_matrix(ps,df_auth,scaler, model,flag_no_country = False)
        df_all_cases.append([df_auth,dist_mat])



    best_eps = None
    best_F1 = 0.0
    memory_f1 = []
    for eps in epsilons:
        y_hat_comb = []
        for case in df_all_cases:
            df_clus, df_case = case
            y_hat = DBS(eps=eps,min_samples=1,metric="precomputed",).fit(df_case)
            df_clus = df_clus[["pmid","PI_IDS"]]
            df_clus['cluster_pred'] = y_hat.labels_
            y_hat_comb.append(df_clus)
        f1_score, _,_,_ = metric_eval_2.get_metrics_many(y_hat_comb)
        memory_f1.append(f1_score)
        if f1_score > best_F1:
            best_F1 = f1_score
            best_eps = eps

    return best_eps, memory_f1






