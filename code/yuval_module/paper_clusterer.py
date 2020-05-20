import pandas as pd
#import pandas_log
import numpy as np
from time import time
from gensim import corpora, models
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from yuval_module.paper_source import PaperSource

class PaperClusterer:
    def __init__(self, 
                eps=1.37,
                gammas= {
                        "author":0.5,
                        "mesh":0.3,
                        "inst":0.1,
                        "email":0.1,
                        "country":0.0},
                scaler=None
                ):
        
        self.paper_source = PaperSource()
        self.cur_researcher_id=self.paper_source.cur_researcher_id #first free researcher id
        self.cached_researchers={}
        self.eps=eps
        self.cur_dict=corpora.Dictionary.load("./yuval_module/mesh_dict.mm")
        self.tfidf_model=models.TfidfModel.load("./yuval_module/tfidf_corpus_papers4.bin")
        self.num_dict_terms=len(self.cur_dict.keys())
        self.gammas = gammas 
        self.scaler = scaler

    def empty_last_author_response(self, row):
        res_row={"pmid":row.pmid,
        "last_author_name":"",
        "rownum":row.rownum,
        "from_pubmed":row.from_pubmed}
        return pd.Series(res_row).to_frame().transpose()

    def fetch_forename(self,data):
            """fetch forename from authors list"""
            last_author=data[-1]
            if "forename" in last_author and last_author["forename"] is not None:
                    return last_author["forename"]
            else:
                    return ""

    def cluster_res(self, rows_and_cluster_dfs_lst):
            """Get distance matrix & cluster it"""
            start_time=time()
            # add forename
            res_df=[r[0] for r in rows_and_cluster_dfs_lst]
            #cluster_out_path="/home/ubuntu/data/cluster_res.tsv"
            # clustered_df.to_csv(cluster_out_path, index=False, sep='\t')
            print("handled {} papers".format(rows_and_cluster_dfs_lst.pmid.size))
            end_time=time()
            print("iteration time: {}".format(end_time-start_time))
            cluster_dfs=rows_and_cluster_dfs_lst
            return res_df, cluster_dfs

    def get_dist_matrix(self, author_papers_df, just_sim_matrix_flag = False):
            """
            Starting author dataframe (papers written by people with the same name)
            and running clutering to get researcher id, institute etc. 
            """
            if author_papers_df.empty:
                    print("empty author_papers_df")
                    return pd.DataFrame()
            author_papers_df['last_author_forename']=author_papers_df['authors'].apply(lambda x: self.fetch_forename(x))
            author_papers_df=self.paper_source.add_processed_fields(author_papers_df)
            #print("author_papers_df affiliation:")
            #print(author_papers_df[["pmid","last_author_name","last_author_inst"]])
            if just_sim_matrix_flag:
                return self.process_features_to_dist(author_papers_df, True)
            combined_dist, for_clustering_df, combined_sim =self.process_features_to_dist(author_papers_df, False)
            #print("***")
            #print(for_clustering_df['weight'])
            for_clustering_df=self.cluster_by_sim(combined_dist,for_clustering_df)
            total_df=for_clustering_df.rename(columns={"db_cluster":"cluster"})
            total_df["cluster"].fillna(-1.0, inplace=True)
            self.print_cluster_metrics(total_df)
            return combined_dist, combined_sim, total_df




    def extract_req_data(self, cluster_df, pmid, author_name):
        # print("in extract_req_data:")
        # print(cluster_df)
        # print(pmid, author_name)
        res={}
        res["last_author_id"]=cluster_df.last_author_id.dropna().min()
        #last_row=cluster_df.iloc[-1]
        inst_candidates_df=cluster_df.dropna(subset=["last_author_inst"])
        # print("inst_candidates_df:")
        # print(inst_candidates_df)
        
        chosen_inst=np.NaN
        chosen_type=np.NaN
        chosen_country=np.NaN
        if not inst_candidates_df.empty:
            chosen_row=inst_candidates_df.iloc[-1]
            chosen_inst=chosen_row["last_author_inst"]
            chosen_type=chosen_row["last_author_inst_type"]
            chosen_country=chosen_row["last_author_country"]
        res["inferred_last_author_inst"]=chosen_inst
        res["inferred_last_author_inst_type"]=chosen_type
        res["inferred_last_author_country"]=chosen_country
        
        res["pmid"]=pmid
        res["last_author_name"]=author_name
        return res

    def get_cluster_result(self, papers_df, author_name):
        # print("pmid={}".format(pmid))
        # print("papers_df:")
        # print(papers_df[["pmid", "last_author_id","cluster", "last_author_inst","other_authors","mesh_clean"]])
        #cluster_row=papers_df[papers_df.pmid==pmid].iloc[0]
        cur_cluster=cluster_row["cluster"]
        cluster_df=papers_df[papers_df.cluster==cur_cluster]
        #res=self.extract_req_data(cluster_df.sort_values("pmid"), pmid, author_name)
        res["last_author_forename"]=cluster_row["last_author_forename"]
        res["pub_year"]=cluster_row["pub_year"]
        res["last_author_country"]=cluster_row["last_author_country"]
        res["last_author_inst"]=cluster_row["last_author_inst"]
        return res

    def process_features_to_dist(self, author_papers_df, just_sim_matrix_flag = False):
        debug=False
        for_clustering_df=author_papers_df
        for_clustering_df.loc[:, "weight"]=1
        #print("before tfidf, mesh_clean is:")
        #print(for_clustering_df["mesh_clean"])
        for_clustering_df.loc[:, "mesh_as_tfidf"]=for_clustering_df["mesh_clean"].apply(self.mesh_tfidf_transform)
        #with pandas_log.enable():   
        if len(for_clustering_df)==1:
            for_clustering_df.loc[:, "db_cluster"]=0
        else:
            if just_sim_matrix_flag:
                return self.build_distance_matrix(for_clustering_df, True)
            combined_dist, combined_sim = self.build_distance_matrix(for_clustering_df, False)
        return combined_dist, for_clustering_df, combined_sim
        



    def infer_all(self, papers_df):
        res=[self.infer_author_data(row) for idx, row in papers_df.iterrows()]
        return res


    def report_missing_pmids(self, still_missing_ids_df):
        res_lst=[]
        for _idx, pmid_row in still_missing_ids_df.iterrows():
            res={"pmid":pmid_row["pmid"],
            "comment":"paper not found in pubmed"}
            res_lst.append(pd.Series(res).to_frame().transpose())
        return res_lst

    def cluster_by_sim(self, combined_dist, df):
            #show_dist_to_nearest_neighbor_dist(combined_dist)
            df["db_cluster"]=self.dbscan_cluster(combined_dist, df["weight"])
            return df

    def infer_paper_author_data(self, pmid_list):
            num_pmids=len(pmid_list)
            orig_df=pd.DataFrame({"pmid":pmid_list, 
            "rownum":list(range(0, num_pmids))})
            BUNCH_SIZE=100
            divisor=np.ceil(num_pmids/BUNCH_SIZE)

            bunches=np.array_split(orig_df, divisor)
            res=[]
            cluster_dfs=[]
            for i, b in enumerate(bunches):
                start_time=time()
                bunch_from_index_df=self.paper_source.get_paper_data_from_index(b)
                missing_ids_df=b[~b.pmid.isin(bunch_from_index_df.pmid)]
                #print("{} missing ids".format(len(missing_ids_df)))
                bunch_from_pubmed_df=self.paper_source.get_paper_data_from_pubmed(missing_ids_df)
                still_missing_ids_df=missing_ids_df[~missing_ids_df.pmid.isin(bunch_from_pubmed_df.pmid)]
                tot_df=pd.concat([bunch_from_index_df, bunch_from_pubmed_df], sort=False)
                
                #out_path="/home/ubuntu/data/test_retrieved_fields.tsv"
                #tot_df.to_csv(out_path, index=False, sep="\t")
                rows_and_cluster_dfs_lst=self.infer_all(tot_df)
                cluster_rows_lst=[r[0] for r in rows_and_cluster_dfs_lst]
                #cluster_out_path="/home/ubuntu/data/cluster_res.tsv"
                #clustered_df.to_csv(cluster_out_path, index=False, sep='\t')
                
                
                #print("handled {} papers".format(i*100))
                end_time=time()
                #print("iteration time: {}".format(end_time-start_time))
                res.extend(cluster_rows_lst)
                res.extend(self.report_missing_pmids(still_missing_ids_df))
                cluster_dfs.extend([r[1] for r in rows_and_cluster_dfs_lst])
            res_df=pd.concat(res, sort=False).sort_values(["rownum"])
            return res_df, cluster_dfs


    def forename_delta(self, n1, n2):
        if ' ' in n1:
            n1=n1.split()[0]
        if ' ' in n2:
            n2=n2.split()[0]
        if n1==n2:
            if len(n1)>1:
                return 1.0
            else:
                return 0.5
        elif n1 in n2 or n2 in n1:
            return 0.75
        else:
            return 0.0

    def get_forename_similarities(self, name_list):
        limit=len(name_list)
        res=np.zeros((limit, limit))
        for i in tqdm(range(0, limit)):
            for j in range(0, limit):
                res[i,j]=self.forename_delta(name_list[i], name_list[j])
        return res


    def get_forename_similarity(self, df):
        filled_fornames=df["last_author_forename"].fillna("").str.strip()
        jac_sim=self.get_forename_similarities(filled_fornames.tolist())
        return jac_sim


    def get_all_pair_similarity(self, val_list, func):
        limit=len(val_list)
        res=np.zeros((limit, limit))
        for i in range(0, limit):
            for j in range(0, limit):
                res[i,j]=func(val_list[i], val_list[j])
        return res

    def get_var_similarity(self, df, varname, func):
        filled_col=df[varname].fillna("").str.strip()
        sim_mat=self.get_all_pair_similarity(filled_col.tolist(), func)
        return sim_mat

    def delta_sim(self, first, second):
        if type(first)==type(second) and first==second:
            return 1
        else:
            return 0

    def extended_delta_sim(self, first, second):
        if first=="" or second=="":
            return 0.5
        elif type(first)==type(second) and first==second:
            return 1
        return 0

    def country_similarity(self, df):
        return self.get_var_similarity(df, "last_author_country", self.delta_sim)

    def email_similarity(self, df):
        return self.get_var_similarity(df, "email_clean", self.delta_sim)

    def inst_similarity(self, df):
        return self.get_var_similarity(df, "last_author_inst", self.inst_delta)


    def forename_similarity(self, df):
        return self.get_var_similarity(df, "last_author_forename", self.forename_delta)
        

    def dbscan_cluster(self, dist, paper_weights):
        #MAXIMAL_DIST=1.37
        if(self.scaler):
                print(dist.shape)
                dist= self.scaler.transform(dist)
        clustering = DBSCAN(eps=self.eps, min_samples=2, metric="precomputed").fit(dist, sample_weight=paper_weights)
        return clustering.labels_

    def build_distance_matrix(self, df, just_sim_matrix_flag = False):
        print("Comparing Authors\n")
        author_sim=self.get_author_similarity(df)
        print("Comparing Mesh\n")
        mesh_sim=self.get_mesh_similarity(df)
        # print("forenames:")
        # print(df["last_author.forename"])
        print("Comparing Forenames\n")
        forename_sim=self.forename_similarity(df)
        print("Comparing Institutions\n")
        inst_sim=self.inst_similarity(df)
        print("Comparing Emails\n")
        email_sim=self.email_similarity(df)
        #print(df[["pmid","last_author_email","email_clean"]])
        print("Comparing Countries\n")
        country_sim=self.country_similarity(df) 

        # import seaborn  as sns
        # import matplotlib.pyplot as plt
        # #f,(ax1,ax2,ax3,ax4,ax5,ax6, axcb) = plt.subplots(1,7, 
        #             gridspec_kw={'width_ratios':[1,1,1,1,1,1,0.08]})
        # ax1.get_shared_y_axes().join(ax2,ax3,ax4,ax5,ax6)
        # g1 = sns.heatmap(author_sim,
        #                 cmap="YlGnBu",
        #                 cbar=False,
        #                 ax=ax1,
        #                 xticklabels=df.pmid,
        #                 yticklabels=df.pmid)
        # g1.set_ylabel('')
        # g1.set_xlabel('author')
        # g2 = sns.heatmap(mesh_sim,
        # cmap="YlGnBu",
        # cbar=False,
        # ax=ax2,
        # xticklabels=df.pmid,
        # yticklabels=df.pmid
        # )
        # g2.set_ylabel('')
        # g2.set_xlabel('mesh')
        # g2.set_yticks([])
        # g3 = sns.heatmap(inst_sim,cmap="YlGnBu",ax=ax3, cbar_ax=axcb)
        # g3.set_ylabel('')
        # g3.set_xlabel('inst')
        # g3.set_yticks([])
        # g4 = sns.heatmap(email_sim,cmap="YlGnBu",ax=ax4, cbar_ax=axcb)
        # g4.set_ylabel('')
        # g4.set_xlabel('email')
        # g3.set_yticks([])
        # g5= sns.heatmap(country_sim,cmap="YlGnBu",ax=ax5, cbar_ax=axcb)
        # g5.set_ylabel('')
        # g5.set_xlabel('country')
        # g5.set_yticks([])
        # g6 = sns.heatmap(forename_sim,cmap="YlGnBu",ax=ax6, cbar_ax=axcb)
        # g6.set_ylabel('')
        # g6.set_xlabel('forename')
        # g6.set_yticks([])

        # # may be needed to rotate the ticklabels correctly:
        # for ax in [g1,g2,g3,g4,g5,g6]:
        #     tl = ax.get_xticklabels()
        #     ax.set_xticklabels(tl, rotation=90)
        #     tly = ax.get_yticklabels()
        #     ax.set_yticklabels(tly, rotation=0)


        #plt.show()


        similarities={"author":author_sim,
                      "mesh":mesh_sim,
                      "inst":inst_sim,
                      "email":email_sim,
                      "country":country_sim,
                      "forename":forename_sim}
        #print("sim matrices shapes for author and mesh:")
        num_items=len(df)*len(df)
        feat_df=pd.DataFrame({"author":author_sim.reshape(num_items,),
                              "mesh":mesh_sim.reshape(num_items,),
                              "inst":inst_sim.reshape(num_items,),
                              "email":email_sim.reshape(num_items,),
                              "country":country_sim.reshape(num_items,),
                              "forename":forename_sim.reshape(num_items,)
                              })
        
        if just_sim_matrix_flag:
            return feat_df
        
        #print("correlations:")
        #print(feat_df.corr())

        #print("similarity statistics:")
        #print(feat_df.describe())

    

        combined_sim=self.combine_similarities(similarities)
        combined_dist=self.sim_to_dist(combined_sim)
        combined_dist_vector=combined_dist.reshape(num_items,)
        #print(pd.Series(combined_dist_vector).describe())

        return combined_dist,combined_sim

 

    def inst_delta(self, n1, n2):
        if n1==n2:
            if len(n1)>1:
                return 1.0
            else:
                return 0.5
        elif n1 in n2 or n2 in n1:
            return 0.75
        else:
            return 0.0

    # def get_inst_similarities(self, inst_list):
    #     limit=len(inst_list)
    #     res=np.zeros((limit, limit))
    #     for i in range(0, limit):
    #         for j in range(0, limit):
    #             res[i,j]=self.inst_similarity(inst_list[i], inst_list[j])
    #     return res        
    
    # def get_inst_similarity(self, df):
    #     filled_insts=df["last_author_inst"].fillna("").str.strip()
    #     jac_sim=self.get_inst_similarities(filled_insts.tolist())
    #     return jac_sim

    def jaccard(self, l1, l2):
        if len(l1)==0 or len(l2)==0:
            return 0
        s1=set(l1)
        s2=set(l2)
        si=s1.intersection(s2)
        su=s1.union(s2)
        res=len(si)/len(su)
        return res

    def get_jaccard_similarities(self, author_lists):
        limit=len(author_lists)
        res=np.zeros((limit, limit))
        for i in range(0, limit):
            for j in range(0, limit):
                if i==j:
                    res[i,j]=1
                else:
                    res[i,j]=self.jaccard(author_lists[i], author_lists[j])
        return res

    def get_author_similarity(self, df):
        jac_sim=self.get_jaccard_similarities(df["other_authors"].tolist())
        return jac_sim

    def mesh_tfidf_transform(self, mesh_list):
        """
        Get mesh list, return sparse vector
        """
        rows=[]
        cols=[]
        vals=[]
        bow = self.cur_dict.doc2bow(mesh_list)
        tfidf_bow = self.tfidf_model[bow]
        for (id, tfidf) in tfidf_bow:
                rows.append(0)
                cols.append(id)
                vals.append(tfidf)

        res=csr_matrix((vals, (rows, cols)), shape=(1, self.num_dict_terms))
        return res

    def get_mesh_similarity(self, df):
        limit=len(df)
        meshes=df["mesh_as_tfidf"].tolist()
        res=np.zeros((limit, limit))
        for i in tqdm(range(0, limit)):
            for j in range(0, limit):
                if i==j:
                    res[i,j]=1
                else:
                    res[i,j]=cosine_similarity(meshes[i], meshes[j])
        
        return res

    def combine_similarities(self, similarity_map):
       
        
        key_set=set(similarity_map.keys())
        key_set.remove("forename")
        weighted_sims=[self.gammas[k]*similarity_map[k] for k in key_set]
        aa=[w.shape for w in weighted_sims]
        #print(aa)
        avg_sim=np.sum(w for w in weighted_sims)
        sim=np.minimum(avg_sim, similarity_map["forename"])
        return sim

    def sim_to_dist(self, sim):
        dist=np.sqrt(2.000002-2.0*sim)
        return dist

    def print_cluster_metrics(self, cluster_df):
        num_insts=len(cluster_df)
        num_noise=(cluster_df.cluster==-1).sum()
        percent_noise=100*((cluster_df.cluster==-1).mean())
        cluster_counts=cluster_df.cluster.value_counts()
        print("cluster_counts:")
        print(cluster_counts)
        top_cluster=cluster_counts.iloc[0]
        print("For this clustering, N={}".format(num_insts))
        print("{} instances, or {} percent, were clustered as noise".format(num_noise, percent_noise))
        print("top cluster is {} percent of the total".format(100*(top_cluster/num_insts)))




def main(list_path, offset, limit):
    # n1="Sanggyu"
    # n2="Sangkyu"
    # print(forename_similarity(n1,n2))

    #pmid_df=pd.read_excel(list_path, sheet_name="Sheet1", dtype=str)
    pmid_df=pd.read_csv(list_path, dtype=str, usecols=["PMID"]).rename(columns={"PMID":"Highlight"})
    #print(pmid_df.info())
    limit=min(limit, len(pmid_df))
    pmid_list=pmid_df.Highlight.tolist()[offset:offset+limit]
  
    paper_clusterer=PaperClusterer()
    res_clusters, cluster_dfs=paper_clusterer.infer_paper_author_data(pmid_list)
    
    #res_clusters, cluster_dfs=paper_clusterer.infer_paper_author_data(pmid_list).sort_values(["rownum"])
    return res_clusters, cluster_dfs
    #out_path="/home/ubuntu/data/rp_40000_authors.tsv"
    #out_path="/home/ubuntu/data/test_forenames_res.tsv"

    #res_clusters.to_csv(out_path, index=False, sep='\t')

if __name__=="__main__":
    #list_path="/home/ubuntu/data/testing_set.csv"
    list_path="/home/ubuntu/data/test_forenames.tsv"
    #list_path="/home/ubuntu/data/RP 2.xlsx"
    offset=0
    limit=1
#     if len(sys.argv)>1:
#         list_path=sys.argv[1]
    
#     if len(sys.argv)>2:
#         offset=int(sys.argv[2])

#     if len(sys.argv)>3:
#         limit=int(sys.argv[3])
   
    main(list_path, offset, limit)
