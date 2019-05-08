import numpy as np
import pandas as pd 
from Norm import Norm 
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

class PLSR(Norm):
    def __init__(self, path = 'data/cslb_feature_matrix.csv'):
        '''
            Initial new embedding model or load in values from previous model
        '''
        
        super().__init__(path = path)
        self._buildEmbeddings()
        
    def train(self, embedding_size = 50, train_words = []):
        '''
            Function for training partial least squared regression model 
        '''
        
        if np.array_equal(train_words, []):
            train_words = self.concepts
        
        # get train and test words 
        self.train_words = train_words
        self.test_words = [w for w in self.concepts if w not in train_words]
        
        # build training data 
        self.train_data = np.zeros((len(self.train_words), self.embedding_matrix.shape[0]))
        self.train_response = np.zeros((len(self.train_words), self.data_matrix.shape[1]))
        for index, concept in enumerate(self.train_words):
            self.train_data[index,:] = self.embedding_matrix[:,self.concept2id[concept]]
            self.train_response[index,:] = self.data_matrix[self.concept2id[concept],:]
            
        self.test_data = np.zeros((len(self.test_words), self.embedding_matrix.shape[0]))
        self.test_response = np.zeros((len(self.test_words), self.data_matrix.shape[1]))
        for index, concept in enumerate(self.test_words):
            self.test_data[index,:] = self.embedding_matrix[:,self.concept2id[concept]]
            self.test_response[index,:] = self.data_matrix[self.concept2id[concept],:]
        
        self.embedding_size = embedding_size
        self.regressor = PLSRegression(n_components=embedding_size)
        self.regressor.fit(self.train_data, self.train_response)
        
        self.train_preds = self.regressor.predict(self.train_data)
        self.test_preds = self.regressor.predict(self.test_data)
        
    def _topFeatures(self, vector, top = 10):
        '''
            Function that gives the top cosine similar features for a word 
        '''

        return np.flip([(self.id2feature[num], vector[num]) for num in np.argsort(vector)[-top:]])
    
    def feature_score(self, type = 'train'):
        '''
            Scores model on its ability to retrieve correct features
        '''
        if type == 'train':
            preds = self.train_preds
            concepts = self.train_words
        if type == 'test':
            preds = self.test_preds
            concepts = self.test_words

        total_scores = []
        for value, concept in enumerate(concepts):
            features = [x[0] for x in self.concept_features[concept]]
            num_features = len(features)

            vector = preds[value,:]
            predicted_features = [x[1] for x in self._topFeatures(vector = vector, top = num_features)]

            # scores = [int(features[num] == predicted_features[num]) for num in range(num_features)]
            positives = [f for f in predicted_features if f in features]

            #total_scores.append(accuracy_score(np.ones(num_features), scores))
            total_scores.append(len(positives) / num_features)

        return total_scores
    
        
        