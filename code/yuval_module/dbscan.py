"""
Run DB scan using Yuval's original code 
"""
from yuval_module.paper_clusterer import PaperClusterer
from yuval_module.paper_source import PaperSource
import py_3.sim_matrix_3 as sim_matrix_3
import py_3.lr_model_3 as lr_model_3
import metric_eval_2

import sys
import pandas as pd
import numpy as np
import utils as utils


def run_db_scan(author_df: pd.DataFrame, 
                eps:float=1.27,
                gammas:dict={
                                "author":0.5,
                                "mesh":0.3,
                                "inst":0.1,
                                "email":0.1,
                                "country":0.0},
                scaler=None,
                bias = None
                ):
        """
        run DBscan using yuval's code
        params:
                author_df(dataframe): a data frame of publications 
                gammas(dict): dict of weight
                eps(float): epsilon
        """
        print("Running Yuval's DBscan\n")
        paper_clusterer=PaperClusterer(eps, gammas, scaler=scaler,bias=bias)
        # dist matrix
        combined_dist, combined_sim, total_df = paper_clusterer.get_dist_matrix(author_df)
        # cluster
        res_clusters, cluster_dfs=paper_clusterer.cluster_res(total_df)
        cluster_dfs = cluster_dfs.rename(columns={'cluster':'cluster_pred'})
        return cluster_dfs
    
    
def run_multiple_df_scan(ps, df, auth_df, use_case,scaler = None,eps = None,params=None,bias=None):
    
    #Get combinations of authors from the given use_case
    if use_case == "1_da" or use_case == "mix_bag":
        authors = sim_matrix_3.get_use_case(df,use_case)
        auth_df = df[df['last_author_name'].isin(authors)]
    else:
        authors = sim_matrix_3.get_use_case(auth_df,use_case)

    num_authors = len(authors)


    y_hat_comb = []
    all_papers = []

    for i,auth in enumerate(authors):
        print("Processing combination number {} from {}".format(i+1,num_authors))
        df_auth = auth_df[auth_df['last_author_name'] == auth]
        all_papers.append(df_auth.shape[0])
        #Calculate the distance matrix
        if params:
            cluster_dfs = run_db_scan(df_auth,eps =eps,gammas=params,bias=bias,scaler=scaler)
        else:
            cluster_dfs = run_db_scan(df_auth)
        y_hat_comb.append(cluster_dfs[["pmid","PI_IDS","cluster_pred"]])
    
    return y_hat_comb, num_authors, np.sum(all_papers)

