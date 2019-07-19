import sys
import numpy as np 
from Norm import Norm
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class NeuralNetwork(Norm):
    """Class for handling DNN model on human property knowledge

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
        
        self.model = self._build_model(output_shape = self.data_matrix.shape[1])
    
    
    def set_vocabulary(self, train_words):
        """Function to set training words, and use the test for testing"""
        if np.array_equal(train_words, []):
                train_words = self.concepts

        self._train_words = train_words
        self._test_words = [w for w in self.concepts if w not in train_words]

        self._train_data, self._train_response = self._build_data(
            self._train_words)
        self._test_data, self._test_response = self._build_data(self._test_words)
    
        
    def _build_model(self, output_shape):
        '''Function to build neural network model'''

        model = Sequential()
        model.add(Dense(300, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(800, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1500, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2000, activation = 'relu'))
        model.add(Dense(output_shape, activation = 'softmax'))
        
        return model

        
    def train(self, epochs = 120, batch_size = 20, verbose = 1):
        """method for training neural network model, also makes predictions 
           on both training and testing data which is saved. 

        Parameters
        ----------

        epochs : int
            number of training epochs

        batch_size : int 
            size of training batches 
            
        verbose : int 
            whether to be verbose or not during training

        ----------
        """
        
        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_crossentropy'])
        loss = 100
        losses = []
        
        for epoch in range(epochs):
            
            if verbose == 1:
                sys.stdout.write('\r' + 'Epoch: ' + str(epoch) + ' Loss: ' + str(loss))
                
            callback = self.model.fit(self._train_data, 
                           normalize(self._train_response, norm = 'l1'), 
                           epochs = 1, 
                           verbose = 0,
                           batch_size = batch_size)
        
            
            loss = callback.history['loss'][0]
            losses.append(loss)
        
        
        self._train_preds = self.model.predict(self.train_data)
        self._test_preds = self.model.predict(self.test_data)
        self._trained = True
        
        return losses
    
    
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
        mat = self.model.predict(vector.reshape(1, len(vector)))

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