import numpy as np
from Feature2Vec import Feature2Vec
from PLSR import PLSR
from NeuralNetwork import NeuralNetwork
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import os
from utils import *
import sys
import timeit
import pickle as pkl


def init_state():
    """Function to load and set numpy state for experiments
       used in the paper"""
    
    with open('state_zero.pkl', 'rb') as f:
        st0 = pkl.load(f)
        
    np.random.set_state(st0)
    

if __name__ == '__main__':
    start_time = timeit.default_timer()

    arg = sys.argv[1]

    SEED = 42
    np.random.seed(seed = SEED)

    PLSR_SMALL_DIMS = 50
    PLSR_LARGE_DIMS = 120

    if arg == 'mcrae':
        path = 'data/mcrae_feature_matrix.csv'
        prefix = 'MCRAE'
        concept_index = 400

    if arg == 'cslb':
        path = 'data/cslb_feature_matrix.csv'
        prefix = 'CSLB'
        concept_index = 500

    if arg not in ['mcrae', 'cslb']:
        print('Not a recognised command')
        sys.exit()

    # build and train feature2vec
    print('Using', prefix)
    print('Building feature2vec')
    model = Feature2Vec(path = path)
    
    init_state()
    shuffle = np.random.permutation(len(model.concepts))
    train_concepts = list(np.asarray(model.concepts)[shuffle][:concept_index])
    test_concepts = list(np.asarray(model.concepts)[shuffle][concept_index:])
    
    model.set_vocabulary(train_words = train_concepts)

    print('Training feature2vec')
    model.train(verbose = 1, epochs = 20, lr = 5e-3, negative_samples = 20)
    print('')

    # test for word dog
    print('Example features learned for word: dog')
    print(model.top_features(model.wvector('dog'), top = 10))
    print('')

    # build baseline model (50 and 120)
    print('Building partial least squared regression (baseline)')
    plsr_small = PLSR(path = path)
    plsr_small.set_vocabulary(train_words = train_concepts)
    plsr_small.train(embedding_size = PLSR_SMALL_DIMS)

    plsr_large = PLSR(path = path)
    plsr_large.set_vocabulary(train_words = train_concepts)
    plsr_large.train(embedding_size = PLSR_LARGE_DIMS)
    print('')

    concept_dict_plsr_small = {}
    for index, concept in enumerate(model.test_words):
        concept_dict_plsr_small[concept] = plsr_small.test_preds[index,:]

    concept_dict_plsr_large = {}
    for index, concept in enumerate(model.test_words):
        concept_dict_plsr_large[concept] = plsr_large.test_preds[index,:]
        
    print('Training Neural Network')
    nn = NeuralNetwork(path = path)
    nn.set_vocabulary(train_words = train_concepts)
    nn.train(verbose = 1, epochs = 150, batch_size = 20)
    
    concept_dict_nn = {}
    for index, concept in enumerate(nn.test_words):
        concept_dict_nn[concept] = nn.test_preds[index,:]

    print('')
    print('Evaluation: Retrieval of gold standard vectors')
    print('PLSR 50 neighbour scores')
    tops = [1, 5, 10, 20]
    for n in tops:
        print('Top', n, neighbour_score(concept_dict_plsr_small, plsr_small, top = n))
    print('')

    print('PLSR 120 neighbour scores')
    tops = [1, 5, 10, 20]
    for n in tops:
        print('Top', n, neighbour_score(concept_dict_plsr_large, plsr_large, top = n))
    print('')
    
    print('Neural Network neighbour scores')
    tops = [1, 5, 10, 20]
    for n in tops:
        print('Top', n, neighbour_score(concept_dict_nn, nn, top = n))
    print('')

    concept_dict_f2v = {}
    for index, concept in enumerate(model.test_words):
        concept_dict_f2v[concept] = construct_vector(concept, model)

    print('Feature2Vec neighbour scores')
    tops = [1, 5, 10, 20]
    for n in tops:
        print('Top', n, neighbour_score(concept_dict_f2v, model, top = n))
    print('')
    
    ############################################################################################################
    print('Evaluation: Feature retrieval scores: ACCURACY')
    print('PLSR 50 Scores')
    print('Train:', np.mean(feature_score(plsr_small, data_type = 'train', max_features = 0, score_func = accuracy_score))*100)
    print('Test:', np.mean(feature_score(plsr_small, data_type = 'test', max_features = 0, score_func = accuracy_score))*100)
    print('')

    print('PLSR 120 Scores')
    print('Train:', np.mean(feature_score(plsr_large, data_type = 'train', max_features = 0, score_func = accuracy_score))*100)
    print('Test:', np.mean(feature_score(plsr_large, data_type = 'test', max_features = 0, score_func = accuracy_score))*100)
    print('')
    
    print('Neural Network Scores')
    print('Train:', np.mean(feature_score(nn, data_type = 'train', max_features = 0, score_func = accuracy_score))*100)
    print('Test:', np.mean(feature_score(nn, data_type = 'test', max_features = 0, score_func = accuracy_score))*100)
    print('')

    print('Feature2Vec Scores')
    print('Train:', np.mean(feature_score(model, data_type = 'train', max_features = 0, score_func = accuracy_score))*100)
    print('Test:', np.mean(feature_score(model, data_type = 'test', max_features = 0, score_func = accuracy_score))*100)
    print('')
    ############################################################################################################
    
    print('')
    print('Time:', timeit.default_timer() - start_time)
