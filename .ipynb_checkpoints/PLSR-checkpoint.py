import numpy as np
import pandas as pd
from Norm import Norm
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import cosine_similarity


class PLSR(Norm):
    """Class for handling PLSR regression model on human property knowledge

    Parameters
    ----------

    path : string 
        path to the csv file containing property knowledge data

    ----------
    """

    def __init__(self, path='data/cslb_feature_matrix.csv'):
        super().__init__(path=path)
        self._build_embeddings()
        self._trained = False
        
        
    def set_vocabulary(self, train_words):
        """Function to set training words, and use the test for testing"""
        if np.array_equal(train_words, []):
                train_words = self.concepts

        self._train_words = train_words
        self._test_words = [w for w in self.concepts if w not in train_words]

        self._train_data, self._train_response = self._build_data(
            self._train_words)
        self._test_data, self._test_response = self._build_data(self._test_words)
        
        
    def train(self, embedding_size=50, max_iter=2000):
        """method for training partial least squared regression model, also makes predictions 
           on both training and testing data which is saved. 

        Parameters
        ----------

        embedding_size : int
            size of embedding for partial least squared regression 

        max_iter : int 
            the maximum iterations of training befor stopping

        ----------
        """ 

        self.embedding_size = embedding_size
        self._regressor = PLSRegression(
            n_components=embedding_size)
        self._regressor.fit(self._train_data, self._train_response)
        self._trained = True

        self._train_preds = self._regressor.predict(self._train_data)
        self._test_preds = self._regressor.predict(self._test_data)
    
    
    def predictions(self, data_type='train'):
        """Function to get prediction data"""
        assert self.trained()

        if data_type == 'train':
            return self._train_preds
        elif data_type == 'test':
            return self._test_preds
        else:
            print('Not a valid data_type string')
    
    
    def rank_neighbours(self, vector, top=10):
        """Function to find the top neighbours for an embedding"""

        mat = cosine_similarity([vector], self.data_matrix)

        return np.flip([(self.id2concept[num], mat[0, num]) for num in np.argsort(mat[0, :])[-top:]])
    
    def top_features(self, vector, top = 10):
        """Function that gives the top cosine similar features for a word vector"""
        mat = self._regressor.predict([vector])

        return np.flip([(self.id2feature[num], mat[0,num]) for num in np.argsort(mat[0,:])[-top:]])
    
    
    def wvector(self, word):
        """Function to get word embedding from spacy """
    
        return self.embedding_matrix[:, self.concept2id[word]]
    
    @property
    def train_data(self):
        return self._train_data
    
    
    @property
    def test_data(self):
        return self._test_data
    
    
    @property
    def train_preds(self):
        return self._train_preds
    
    
    @property
    def test_preds(self):
        return self._test_preds
    
    
    @property
    def train_words(self):
        return self._train_words
    
    
    @property
    def test_words(self):
        return self._test_words
    
            
    @property
    def trained(self):
        return self._trained