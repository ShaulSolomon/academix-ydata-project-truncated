"""
Run DB scan using Yuval's original code 
"""
from yuval_module.paper_clusterer import PaperClusterer
from yuval_module.paper_source import PaperSource
import sys
import pandas as pd
import numpy as np
import utils as utils

def run_db_scan(author_df: pd.DataFrame):
        """
        run DBscan using yuval's code
        params:
                author_df(dataframe): a data frame of publications  
        """
        print("Running Yuval's DBscan\n")
        paper_clusterer=PaperClusterer(eps=1.27)
        # dist matrix
        combined_dist, combined_sim, total_df = paper_clusterer.get_dist_matrix(author_df)
        # cluster
        res_clusters, cluster_dfs=paper_clusterer.cluster_res(total_df)
        cluster_dfs=cluster_dfs.rename(columns={'cluster':'cluster_pred'})
        return cluster_dfs
