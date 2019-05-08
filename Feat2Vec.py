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


class Feat2Vec(Norm):
    def __init__(self, path = 'data/cslb_feature_matrix.csv', embedding_size = 300, load_path = ''):
        '''
            Initial new embedding model or load in values from previous model (train words must be constructed)
        '''
        
        super().__init__(path = path)
        self._buildEmbeddings(embedding_size = embedding_size)
                    
    def _positive_sample(self):
        '''
            Function to get positive instances from training data 
        '''
        
        positive_couples = []
        positive_labels = []

        for word in self.train_words:
            for (feat, value) in self.concept_features[word]:
                positive_couples.append([self.concept2id[word], self.feature2id[feat]])
                positive_labels.append(1)
                
        return positive_couples, positive_labels
        
    def _negative_sample(self, positive_couples, positive_labels, negative_samples = 1, shuffle = True, seed = None):
        '''
            Function used for producing batches of word-property co-occurrences using negative sampling 
            for training in an embedding model. Needs positive couples and labels first 
        ''' 
        
        negative_couples = []
        negative_labels = []
        
        if negative_samples > 0:
            num_negative_samples = int(len(positive_labels) * negative_samples)
            neg_words = [c[0] for c in positive_couples]
            random.shuffle(neg_words)

            negative_couples += [[neg_words[i % len(neg_words)],
                         random.randint(1, len(self.features) - 1)]
                        for i in range(num_negative_samples)]

            negative_labels += [0] * num_negative_samples
        
        couples = negative_couples + positive_couples
        labels = negative_labels + positive_labels
        
        if shuffle:
            if seed is None:
                seed = random.randint(0, 10e6)
            random.seed(seed)
            random.shuffle(couples)
            random.seed(seed)
            random.shuffle(labels)

        return couples, labels
                
    def _build_model(self, lr):
        '''
            Function to build neural-based embedding model (skip-gram)
        '''
        
        adam = Adam(lr = lr)
        
        with tf.device('/cpu:0'):
            property_input = Input(shape =(1,), name = 'property_input')
            word_input = Input(shape = (1,), name = 'word_input')

            property_embs = Embedding(len(self.features), self.embedding_matrix.shape[0], trainable = True, name = 'property_embeddings')(property_input)
            word_embs = Embedding(len(self.concepts), self.embedding_matrix.shape[0], weights = [self.embedding_matrix.T],
                                  trainable = True, name = 'words_embeddings')(word_input)

        product = dot([word_embs, property_embs], axes=-1, normalize=False, name="dot_product")
        product = Flatten()(product)

        output = Activation('sigmoid', name = 'output')(product)

        self.model = Model(inputs = [word_input, property_input], outputs = output)
        self.model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
        
                    
    def train(self, epochs = 100, batch_size = 6000, lr = 1e-4, till_convergence = True,  
              tolerence = 1e-5, max_epochs = 1000, negative_samples = 1, seed = None, shuffle = True, verbose = 1, train_words = []):
        '''
            function to train model 
        '''
        
        if np.array_equal(train_words, []):
            train_words = self.concepts
        
        # build skip-gram implementation for features 
        self.train_words = train_words
        self.test_words = [w for w in self.concepts if w not in train_words]
        positive_couples, positive_labels = self._positive_sample()
        self._build_model(lr = lr)
        class_weight = {1: 1.0, 0: 1.0 / negative_samples}
        
        if till_convergence:
            previous_loss = 100
            current_loss = 100
            diff = 100
            epoch = 0
            
            while diff >= tolerence:
                sys.stdout.write('\r' + 'Epoch: ' + str(epoch) + ' delta: ' + str(diff))
                
                # get training data
                couples, labels = self._negative_sample(positive_couples = positive_couples,
                                                        positive_labels = positive_labels,
                                                        negative_samples = negative_samples,
                                                        shuffle = shuffle,
                                                        seed = seed)
                
                self.model.fit([np.asarray(couples)[:,0], np.asarray(couples)[:,1]], labels,
                               batch_size = batch_size,
                               epochs = 1,
                               verbose = verbose,
                               class_weight = class_weight)
                
                current_loss = self.model.history.history['loss'][-1]
                diff = np.abs(previous_loss - current_loss)
                previous_loss = current_loss
                epoch += 1
            
        else:
            # get training data
            loss = 5
            for epoch in range(epochs):
                sys.stdout.write('\r' + 'Epoch: ' + str(epoch) + ' Loss: ' + str(loss))
                couples, labels = self._negative_sample(positive_couples = positive_couples,
                                                        positive_labels = positive_labels,
                                                        negative_samples = negative_samples,
                                                        shuffle = shuffle,
                                                        seed = seed)

                callback = self.model.fit([np.asarray(couples)[:,0], np.asarray(couples)[:,1]], labels,
                               batch_size = batch_size,
                               epochs = 1,
                               verbose = verbose,
                               class_weight = class_weight)
                
                loss = callback.history['loss'][0]
            
        self.feature_vectors = np.asarray(self.model.get_layer('property_embeddings').get_weights())[0,:,:]
        
    def save(self, path):
        '''
            Function to save property embeddings, along with words [EXPERIMENTAL]
        '''
        
        for feature in self.features:
            with open(path + '/feature_embeddings.txt', 'a') as f:
                f.write(feature + ' ')
                f.write(self.fvector(feature)[1:-1])
                f.write('\n')
        
        for word in self.concepts:
            with open(path + '/words.txt', 'a') as f:
                f.write(word + '\n')
                
        self.setup()
                
    def _load_vectors(self, path):
        '''
            Function to load feature vectors [EXPERIMENTAL]
        '''
        
        self.features =[]
        self.words = []
        self.feature_vectors = []
        self.features_parsed = True
        
        with open(path, 'r') as f:
            for line in f:
                if line != '':
                    split_line = line.split()
                    self.feature.append(split_line[0])
                    self.feature_vectors.append([float(v) for v in split_line[1:]])
                
        with open(path + '/words.txt', 'r') as f:
            words = f.read().split('\n')[:-1]
                
        
    def fvector(self, feature):
        '''
            Function to get property embedding for feature (IF TRAINED)
        '''
        
        if np.array_equal(self.feature_vectors, []):
            print('MUST RUN TRAINING FIRST')
            return None
        
        return self.feature_vectors[self.feature2id[feature],:]
    
    def wvector(self, word):
        '''
            Function to get word embedding from spacy 
        '''
        
        return self.embedding_matrix[:, self.concept2id[word]]
    
    def topFeatures(self, word, top = 10):
        '''
            Function that gives the top cosine similar features for a word 
        '''
        
        mat = cosine_similarity([self.wvector(word)], self.feature_vectors)

        return np.flip([(self.id2feature[num], mat[0,num]) for num in np.argsort(mat[0,:])[-top:]])
    
    def feature_score(self, type = 'test'):
        '''
            Scores model on its ability to retrieve correct features
        '''
        if type == 'train':
            concepts = self.train_words
        if type == 'test':
            concepts = self.test_words

        total_scores = []
        for concept in concepts:
            features = [x[0] for x in self.concept_features[concept]]
            num_features = len(features)

            predicted_features = [x[1] for x in self.topFeatures(concept, top = num_features)]

            # scores = [int(features[num] == predicted_features[num]) for num in range(num_features)]
            positives = [f for f in predicted_features if f in features]

            # total_scores.append(f1_score(np.ones(num_features), scores))
            total_scores.append(len(positives) / num_features)

        return total_scores

    def topFeatures_spacy(self, word, top = 10):
        '''
            Function that gives the top cosine similar features for a word 
        '''
        
        token = norm.nlp(u'' + word)
        mat = cosine_similarity([token.vector], self.feature_vectors)

        return np.flip([(self.id2feature[num], mat[0,num]) for num in np.argsort(mat[0,:])[-top:]])
    
    
if __name__ == '__main__':
    model = Feat2Vec()
    model.train(till_convergence = True, verbose = 0)