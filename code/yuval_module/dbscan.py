"""
Run DB scan using Yuval's original code 
"""
from yuval_module.paper_clusterer import PaperClusterer
from yuval_module.paper_source import PaperSource
import sys
import pandas as pd
import numpy as np
import utils as utils

def run_db_scan(author_df: pd.DataFrame, 
                gammas:dict={
                                "author":0.5,
                                "mesh":0.3,
                                "inst":0.1,
                                "email":0.1,
                                "country":0.0}, 
                eps:float=1.27 ):
        """
        run DBscan using yuval's code
        params:
                author_df(dataframe): a data frame of publications 
                gammas(dict): dict of weight
                eps(float): epsilon
        """
        print("Running Yuval's DBscan\n")
        paper_clusterer=PaperClusterer(eps, gammas)
        # dist matrix
        combined_dist, combined_sim, total_df = paper_clusterer.get_dist_matrix(author_df)
        # cluster
        res_clusters, cluster_dfs=paper_clusterer.cluster_res(total_df)
        cluster_dfs=cluster_dfs.rename(columns={'cluster':'cluster_pred'})
        return cluster_dfs
