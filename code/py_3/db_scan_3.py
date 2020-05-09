import py_3.sim_matrix_3 as sim_matrix_3
import py_3.lr_model_3 as lr_model_3

from itertools import product
from sklearn.cluster import DBSCAN as DBS
import numpy as np


def db_multiple(ps, df, scaler, use_case, num_cases, model,epsilon):
    '''
    Gathers several situations for each use case and calculates their y_hats

    Parameters:
        ps - PaperSource
        df - dataframe with all the details
        scaler - scaler to normalize our data
        use_case - the use_case we want to explore
        num_cases - the number of dif. cases we try (of num cases - 66 was the smallest)
        model - model learned from the LR model
        epsilon - needs to be learned but the epsilon for the DB
    '''
    #Get combinations of authors from the given use_case
    authors = sim_matrix_3.get_use_case(df,use_case)
    if type(authors) == tuple:
        #If we are trying to get two dif. types of authors
        auth_a,auth_b = authors
        all_comb = list(product(auth_a,auth_b))
    elif type(authors) == list:
        all_comb = list(product(authors,authors))
        #remove duplicates
        all_comb = [[a, b] for i, [a, b] in enumerate(all_comb) if not any(((a == c and b==d) or (a==d and b==c)) for c, d in all_comb[:i])] 
    else:
        print("None of the above")
        return None
    
    #Take only `num_cases` number of cases
    np.random.seed(42)
    rand_idx = np.random.choice(range(len(all_comb)),num_cases,replace=False)
    all_comb = np.array(all_comb)[rand_idx]

    y_hat_comb = []

    for i,comb in enumerate(all_comb):
        print("Processing combination number {} from {}".format(i+1,num_cases))
        df_auth = df[df['last_author_name'].isin(comb)]
        #Calculate the distance matrix
        dist_mat = lr_model_3.get_dist_matrix(ps,df_auth,scaler, model,flag_no_country = False)
        #input it through DBS
        y_hat = DBS(eps=epsilon, min_samples=1, metric="precomputed").fit(dist_mat)
        df_clus = df_auth[["pmid","PI_IDS"]]
        df_clus['cluster_pred'] = y_hat.labels_
        y_hat_comb.append(df_clus)
        print("\n") 

    return y_hat_comb

