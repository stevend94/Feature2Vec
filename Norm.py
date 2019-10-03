import numpy as np 
import pandas as pd
import spacy

EMBEDDING_SIZE = 300

class Norm:
    """Class object for handling property norm data 

    Parameters
    ----------

    path : string 
        path to the csv file containing property knowledge data

    ----------
    """
    
    def __init__(self, path):
        self.feature_matrix, self.data_matrix = self._load(path = path)
        
        self.concepts = list(self.feature_matrix.index.values)
        self.concept2id, self.id2concept = self._build_index(self.concepts)
        
        self.features = list(self.feature_matrix.columns[1:])
        self.feature2id, self.id2feature = self._build_index(self.features)
        
        self.concept_features = {}
        for concept in self.concepts:
            self.concept_features[concept] = self._build_concept_features(concept)
            
        self.nlp = spacy.load('en_core_web_lg')
        self._correction_dict = self._build_corrections(self.concepts)
        self._build_embeddings()
        
        
    def _build_corrections(self, concepts):
        """A dictionary that corrects some spelling"""
        correction_dict = dict(zip(concepts, concepts))
        correction_dict['pennicillin'] = 'penicillin'
        correction_dict['castenets'] = 'castanets'
        
        return correction_dict
        
        
    def _build_data(self, words):
        """Function to build a matrix input and response data for training, given a
        set of words"""
        data = np.zeros((len(words), self.embedding_matrix.shape[0]))
        response = np.zeros((len(words), self.data_matrix.shape[1]))
        
        for index, concept in enumerate(words):
            data[index,:] = self.embedding_matrix[:,self.concept2id[concept]]
            response[index,:] = self.data_matrix[self.concept2id[concept],:]
            
        return data, response 
            
            
    def _build_index(self, items):
        """Function to give each item in a list an index value, encoded in a dictionary"""

        item_to_index = dict(
            zip(items, list(range(len(items)))))

        index_to_item = {k: v for v, k in item_to_index.items()}

        return item_to_index, index_to_item
    
    
    def _build_concept_features(self, concept):
        """Dictionary that maps concepts to a list of associated features according
        to the property knowledge data"""
        return [(f , self.data_matrix[self.concept2id[concept], self.feature2id[f]]) 
                for f in self.features if self.data_matrix[self.concept2id[concept], self.feature2id[f]] > 0]
        
    
    def _load(self, path):
        """Function to load .csv file of property knowledge given a path"""
        # open feature_matrix data and read in using csv
        feature_matrix = pd.read_csv(path, index_col=[0])
        data_matrix = feature_matrix.values[:, 1:]
        
        return feature_matrix, data_matrix
    
            
    def _build_embeddings(self):
        """Function to get embedding from spacy """

        self.embedding_matrix = np.zeros((EMBEDDING_SIZE, len(self.concepts)))
        for index, word in enumerate(self.concepts):
            token = self.nlp(u'' + self._correction_dict[word].replace('_', ' '))
            if token.has_vector:
                self.embedding_matrix[:,index] = token.vector
            else:
                print(token, 'has not vector')
                
    
    def _get_features(self, concept, max_features=0):
        """Function to extract feature for a given concept. If max_features is 0, take all features"""
        if max_features == 0:
            features = [x[0] for x in self.concept_features[concept]]
        else:
            features = [x[0] for x in sorted(
                self.concept_features[concept], key=lambda tup: tup[1], reverse=True)][:max_features]

        return features
