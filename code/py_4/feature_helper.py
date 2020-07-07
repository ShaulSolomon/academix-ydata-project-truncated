import pandas as pd
def get_names():
        """
        get names for training embedding
        """
        return pd.read_json('./py_4/last_names.json', orient='values')[0].to_list()
        #return ["Roy Granit", "Shaul Solomon", "clarke", "davies","davis","kelley","kelly","wood","macdonald","woods","mcdonald","rogers","thompson","rodgers","cooke","cook","stevens"]

def get_co_authors():
        """
        get co authors for training embedding
        """
        return pd.read_pickle('./py_4/co_authors.pkl')['names'].to_list()
        #return ["Roy Granit", "Shaul Solomon", "clarke", "davies","davis","kelley","kelly","wood","macdonald","woods","mcdonald","rogers","thompson","rodgers","cooke","cook","stevens"]
