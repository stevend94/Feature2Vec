import numpy as np 
from Feat2Vec import Feat2Vec 
from PLSR import PLSR
from sklearn.metrics.pairwise import cosine_similarity
# save embeddings for intrinsic evaluations 
import os 
import vecto
import vecto.embeddings 
from utils import * 
import sys 
import timeit



if __name__ == '__main__':
    
    start_time = timeit.default_timer()
    
    arg = sys.argv[1]
    
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
    print('Building feat2vec')
    model = Feat2Vec(path = path)
    train_concepts = model.concepts[:concept_index]
    test_concepts = model.concepts[concept_index:]
    
    print('Training feat2vec')
    model.train(till_convergence = True, verbose = 0, tolerence = 1e-4, lr = 1e-4, negative_samples = 20, train_words = train_concepts)
    print('')
    
    # test for word dog
    print('Example features learned for word: dog')
    print(model.topFeatures('dog', top = 10))
    print('')
    
    # build baseline model (50 and 200)
    print('Building partial least squared regression (baseline)')
    plsr50 = PLSR(path = path)
    plsr50.train(train_words = train_concepts, embedding_size = 50)

    plsr200 = PLSR(path = path)
    plsr200.train(train_words = train_concepts, embedding_size = 200)
    print('')
    
    concept_dict_plsr50 = {}
    for index, concept in enumerate(model.test_words):
        concept_dict_plsr50[concept] = plsr50.test_preds[index,:]
        
    concept_dict_plsr200 = {}
    for index, concept in enumerate(model.test_words):
        concept_dict_plsr200[concept] = plsr200.test_preds[index,:]
    
    print('')
    print('Evaluation: Retrieval of gold standard vectors')
    print('PLSR 50 neighbour scores')
    tops = [1, 5, 10, 20]
    for n in tops:
        print('Top', n, neighbourScore(concept_dict_plsr50, plsr50, top = n, gsf = True))
    print('')
        
    print('PLSR 200 neighbour scores')
    tops = [1, 5, 10, 20]
    for n in tops:
        print('Top', n, neighbourScore(concept_dict_plsr200, plsr200, top = n, gsf = True))
    print('')
    
    concept_dict_f2v = {}
    for index, concept in enumerate(model.test_words):
        concept_dict_f2v[concept] = constructVector(concept, model)

    print('Feature2Vec neighbour scores')
    tops = [1, 5, 10, 20]
    for n in tops:
        print('Top', n, neighbourScore(concept_dict_f2v, model, top = n, gsf = False))
    print('')
            
    print('Evaluation: Feature retrieval scores')
    print('PLSR 50 Scores')
    print('Train:', np.mean(plsr50.feature_score(type = 'train'))*100)
    print('Test:', np.mean(plsr50.feature_score(type = 'test'))*100)
    print('')
    
    print('PLSR 200 Scores')
    print('Train:', np.mean(plsr200.feature_score(type = 'train'))*100)
    print('Test:', np.mean(plsr200.feature_score(type = 'test'))*100)
    print('')
    
    print('Feat2Vec Scores')
    print('Train:', np.mean(model.feature_score(type = 'train'))*100)
    print('Test:', np.mean(model.feature_score(type = 'test'))*100)
    print('')
    
    if not all([x in os.listdir('embeddings') for x in ['PLSR50_' + prefix, 'PLSR200_' + prefix, 'F2V_' + prefix]]):
        print('Building embedding vectors')
        plsr50_embs = dict(zip(plsr50.concepts, plsr50.regressor.predict(plsr50.embedding_matrix.T)))
        plsr200_embs = dict(zip(plsr200.concepts, plsr200.regressor.predict(plsr200.embedding_matrix.T)))

        f2v_embs = {}
        for index, concept in enumerate(model.concepts):
            f2v_embs[concept] = constructVector(concept, model)

        saveEmbs(plsr50_embs, name = 'PLSR50_' + prefix)
        saveEmbs(plsr200_embs, name = 'PLSR200_' + prefix)
        saveEmbs(f2v_embs, name = 'F2V_' + prefix)

        plsr50_vsm = vecto.embeddings.load_from_dir('embeddings/PLSR50_' + prefix)
        plsr200_vsm = vecto.embeddings.load_from_dir('embeddings/PLSR200_' + prefix)
        f2v_vsm = vecto.embeddings.load_from_dir('embeddings/F2V_' + prefix)
    
    print('')
    print('Scoring VSMs')
    plsr50_vsm = vecto.embeddings.load_from_dir('embeddings/PLSR50_' + prefix)
    plsr200_vsm = vecto.embeddings.load_from_dir('embeddings/PLSR200_' + prefix)
    f2v_vsm = vecto.embeddings.load_from_dir('embeddings/F2V_' + prefix)
    
    print('PLSR50 scores')
    vectoBenchmark(plsr50_vsm)
    print('')
    
    print('PLSR200 scores')
    vectoBenchmark(plsr200_vsm)
    print('')
    
    print('F2V scores')
    vectoBenchmark(f2v_vsm)
    print('')
    
    if not prefix in os.listdir('embeddings'):
        embs = {}
        for concept in model.concepts:
            embs[concept] = model.data_matrix[model.concept2id[concept],:]
            
        saveEmbs(embs, name = prefix)
        
    emb_vsm = vecto.embeddings.load_from_dir('embeddings/' + prefix)
    print(prefix, 'scores on similarity benchmarks')
    vectoBenchmark(emb_vsm)
    print('')
    print('Time:', timeit.default_timer() - start_time)
            
            
        