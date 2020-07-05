from py_4.feature_helper import get_names
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
from os import makedirs, path
from utils import PROJECT_ROOT, DATA_PATH

class NameEmbeddings():
    def __init__(self):
        self.model_path=PROJECT_ROOT+"code/models/names_epochs_2_vectorSize_64_window_2.model"
        self.model = self.get_model()

    def get_model(self):
        if path.exists(self.model_path):
            print(f"'{self.model_path}' already exits. Using existing model to re-generate results.")
            model = Doc2Vec.load(self.model_path)
        else:
            print('error finding names model')
        return model

    def infer_vec(self, name: list):
        return self.model.infer_vector(name)


