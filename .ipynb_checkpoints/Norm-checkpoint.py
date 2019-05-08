import numpy as np 
import pandas as pd
import spacy

class Norm:
    def __init__(self, path):
        self._load(path = path)
    
    def _load(self, path):
        '''
            Function to load cslb data 
        '''
        
         # Open CSLB data
        # open norm data and read in using csv
#         with open(path + '/' + 'norms.dat', 'rb') as f:
#             self.norms = pd.read_csv(f, delimiter = "\t")

        # open feature_matrix data and read in using csv
        self.feature_matrix = pd.read_csv(path, index_col=[0])

        self.data_matrix = self.feature_matrix.values[:, 1:]

        # get all associated concepts
        concepts = list(self.feature_matrix['Vectors'].values)           
        self.concept2id = {}
        self.id2concept = {}
        self.concepts = []
        for index, concept in enumerate(concepts):
            self.concepts.append(concept)
            self.concept2id[concept] = index
            self.id2concept[index] = concept

        # get all associated features 
        features = list(self.feature_matrix.columns[1:])
        self.feature2id = {}
        self.id2feature = {}
        self.features = []
        for index, feature in enumerate(features):
            self.features.append(feature)
            self.feature2id[feature] = index
            self.id2feature[index] = feature


        # get the features for each concept 
        self.concept_features = {}
        for concept in self.concepts:
            feature_concepts = [(f , self.data_matrix[self.concept2id[concept], self.feature2id[f]]) for f in self.features if self.data_matrix[self.concept2id[concept], self.feature2id[f]] > 0]
            self.concept_features[concept] = feature_concepts
            
    def _buildEmbeddings(self, embedding_size = 300):
        '''
            Function to get embedding from spacy 
        '''
        self.nlp = spacy.load('en_core_web_lg')
        
        correction_dict = dict(zip(self.concepts, self.concepts))
        correction_dict['pennicillin'] = 'penicillin'
        correction_dict['castenets'] = 'castanets'

        self.embedding_matrix = np.zeros((embedding_size, len(self.concepts)))
        for index, word in enumerate(self.concepts):
            token = self.nlp(u'' + correction_dict[word].replace('_', ' '))
            if token.has_vector:
                self.embedding_matrix[:,index] = token.vector
            else:
                print(token, 'has not vector')
