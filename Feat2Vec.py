import numpy as np 
import sys 
import pandas as pd
from Norm import Norm
import random 
from keras.models import Input, Model 
from keras.layers import Embedding, dot, Flatten, Activation
from keras.optimizers import Adam
import spacy
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import tensorflow as tf
from utils import *


class Feat2Vec(Norm):
    def __init__(self, path = 'data/cslb_feature_matrix.csv'):
        '''
            Initial new embedding model or load in values from previous model (train words must be constructed)
        '''
        
        super().__init__(path = path)
        self.model = self._build_model()
        self._trained = False
        
    
    def set_vocabulary(self, train_words):
        """Function to set training words, and use the test for testing"""
        if np.array_equal(train_words, []):
                train_words = self.concepts

        self._train_words = train_words
        self._test_words = [w for w in self.concepts if w not in train_words]
               
            
    def _positive_sample(self, words):
        """Function used for producing batches of positive/correct word-property co-occurrences 
            for training with an embedding model."""
        
        positive_couples = []
        positive_sample_weights = []

        for word in words:
            for (feat, value) in self.concept_features[word]:
                positive_couples.append([self.concept2id[word], self.feature2id[feat]])
                positive_sample_weights.append(self.data_matrix[self.concept2id[word], self.feature2id[feat]])
                
        return positive_couples, positive_sample_weights
      
        
    def _negative_sample(self, positive_couples, positive_sample_weights, negative_samples = 1, shuffle = True, seed = None): 
        """Function used for producing batches of word-property co-occurrences using negative sampling 
            for training with an embedding model. Needs positive couples and labels first """
        negative_couples = []
        negative_sample_weights = []
        
        if negative_samples > 0:
            num_negative_samples = int(len(positive_couples) * negative_samples)
            neg_words = [c[0] for c in positive_couples]
            random.shuffle(neg_words)

            negative_couples += [[neg_words[i % len(neg_words)],
                         random.randint(1, len(self.features) - 1)]
                        for i in range(num_negative_samples)]
            
            negative_sample_weights += [1.0 / negative_samples] * num_negative_samples
            
        return negative_couples, negative_sample_weights

    
    def _generate_samples(self, words, weight_positives = False, negative_samples = 20, shuffle = True, seed = None):
        """Perform negative sampling of training data to generate both positive and negative instances """
        positive_couples, positive_sample_weights = self._positive_sample(words=words)
        negative_couples, negative_sample_weights = self._negative_sample(positive_couples = positive_couples,
                                                    positive_sample_weights = positive_sample_weights,
                                                    negative_samples = negative_samples,
                                                    shuffle = shuffle,
                                                    seed = seed)
        
        if weight_positives == False:
            positive_sample_weights = [1] * len(positive_couples)
        
        couples = negative_couples + positive_couples
        labels = [0] * len(negative_couples) + [1] * len(positive_couples)
        sample_weights = negative_sample_weights + positive_sample_weights
        
        if shuffle:
            if seed is None:
                seed = random.randint(0, 10e6)
            random.seed(seed)
            random.shuffle(couples)
            random.seed(seed)
            random.shuffle(labels)
            random.seed(seed)
            random.shuffle(sample_weights)

        return couples, labels, sample_weights
        
    def _build_model(self):
        """Function to build neural-based embedding model (skip-gram)"""
        with tf.device('/cpu:0'):
            property_input = Input(shape =(1,), name = 'property_input')
            word_input = Input(shape = (1,), name = 'word_input')
            property_embs = Embedding(len(self.features), self.embedding_matrix.shape[0], 
                                      trainable = True, name = 'property_embeddings')(property_input)
            
            word_embs = Embedding(len(self.concepts), self.embedding_matrix.shape[0], weights = [self.embedding_matrix.T],
                                  trainable = True, name = 'words_embeddings')(word_input)

        product = dot([word_embs, property_embs], axes=-1, normalize=False, name="dot_product")
        product = Flatten()(product)
        output = Activation('sigmoid', name = 'output')(product)
        
        return Model(inputs = [word_input, property_input], outputs = output)
        
                    
    def train(self, epochs = 100, batch_size = 6144, lr = 5e-3, train_words = [],
              negative_samples = 1, seed = None, shuffle = True, verbose = 1):
        
        """method for training Feature2Vec model. 

        Parameters
        ----------

        epochs : int
            number of training epochs

        batch_size : int 
            size of training batches 
            
        lr : float
            learning rate for training model 
            
        negative_samples : int 
            number of negative samples per positive instance to generate 
            
        seed : int 
            random seed for shuffle 
            
        shuffle : boolean 
            whether to shuffle training data or not 
            
        verbose : int 
            whether to be verbose or not during training

        ----------
        """
        
        self.model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr=lr), metrics = ['accuracy'])
        
        if not self._trained:
            if np.array_equal(train_words, []):
                train_words = self.concepts

            # build skip-gram implementation for features 
            self._train_words = train_words
            self._test_words = [w for w in self.concepts if w not in train_words]
        
        self._trained = True 
        
        # get training data
        losses = []
        loss = 100
        for epoch in range(epochs):
            
            if verbose == 1:
                sys.stdout.write('\r' + 'Epoch: ' + str(epoch) + ' Loss: ' + str(loss))
            couples, labels, sample_weights = self._generate_samples(words=train_words,
                                                                     weight_positives=False,
                                                                     negative_samples=negative_samples,
                                                                     shuffle=shuffle,
                                                                     seed=seed)
            
            

            callback = self.model.fit([np.asarray(couples)[:,0], np.asarray(couples)[:,1]], labels,
                           batch_size = batch_size,
                           epochs = 1,
                           sample_weight = np.asarray(sample_weights),
                           verbose = 0)

            loss = callback.history['loss'][0]
            losses.append(loss)
        
        self.feature_vectors = np.asarray(self.model.get_layer('property_embeddings').get_weights())[0,:,:]
        return losses
    
    
    def fvector(self, feature):
        """Function to get property embedding for feature (IF TRAINED)"""
        
        if np.array_equal(self.feature_vectors, []):
            print('MUST RUN TRAINING FIRST')
            return None
        
        return self.feature_vectors[self.feature2id[feature],:]
    
    
    def wvector(self, word):
        """Function to get word embedding from spacy """
    
        return self.embedding_matrix[:, self.concept2id[word]]
    
    
    def top_features(self, vector, top = 10):
        """Function that gives the top cosine similar features for a word vector"""
        mat = cosine_similarity([vector], self.feature_vectors)

        return np.flip([(self.id2feature[num], mat[0,num]) for num in np.argsort(mat[0,:])[-top:]])
                                 

    def topFeatures_spacy(self, word, top = 10):
        '''
            Function that gives the top cosine similar features for a word for all spacy tokens
        '''
        
        token = self.nlp(u'' + word)
        mat = cosine_similarity([token.vector], self.feature_vectors)

        return np.flip([(self.id2feature[num], mat[0,num]) for num in np.argsort(mat[0,:])[-top:]])
    
    
    def rank_neighbours(self, vector, top = 10):
        '''
            Function to find the top neighbours for an embedding 
        '''

        mat = cosine_similarity([vector], self.embedding_matrix.T)

        return np.flip([(self.id2concept[num], mat[0,num]) for num in np.argsort(mat[0,:])[-top:]])
    
    
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
        return self._test_data
    
    
    @property
    def train_words(self):
        return self._train_words
    
    
    @property
    def test_words(self):
        return self._test_words
    
            
    @property
    def trained(self):
        return self._trained
    
    
if __name__ == '__main__':
    model = Feat2Vec()
    model.train(till_convergence = True, verbose = 0)