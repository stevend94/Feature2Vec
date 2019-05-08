import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
# save embeddings for intrinsic evaluations 
import os 
from vecto.utils.fetch_benchmarks import fetch_benchmarks
from vecto.benchmarks.similarity import Similarity
import spacy 

def vectoBenchmark(vsm):
    '''
        Function used to benchmark all embedding models in the vecto experimental folder on
        intrinsic word similarity evaluation benchmarks.
    '''
        
    benchmark_path = 'data/benchmarks/benchmarks/similarity/en'

    # if benchmarks are not there then get them
    if not os.path.exists('data/benchmarks'):
        fetch_benchmarks()

    evals = [
         'simlex999',
         'men',]

    sim = Similarity()
    
    for name in evals:
        score = sim.evaluate(vsm, sim.read_test_set(path = benchmark_path + '/' + name + '.txt'))[0:-1]
        print(name, '-', score)
        
def saveEmbs(embs, name):
    directory = 'embeddings/' + name
    save_path = directory+'/embeddings.txt'

    # check if directory exists first
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write embedding data
    with open(save_path, 'a', encoding = 'utf-8') as f:
        for word in list(embs.keys()):
            string = word
            for num in embs[word]:
                string += ' ' + str(num)
            f.write(string + '\n')
            
            
def topNeighbours(vector, model, top = 10, gsf = False):
    '''
        Function to find the top neighbours for an embedding 
    '''
    if gsf:
        matrix = model.data_matrix
    else:
        matrix = model.embedding_matrix.T
    
    mat = cosine_similarity([vector], matrix)

    return np.flip([(model.id2concept[num], mat[0,num]) for num in np.argsort(mat[0,:])[-top:]])

def neighbourScore(concept_dict, model, top = 10, gsf = False):
    '''
        Function to score the ability of a set of vectors to predict their 
        unseen golden feature representation when they are unseen.
    '''
    
    scores = []
    for concept in concept_dict.keys():
        top_concepts = [s[1] for s in topNeighbours(concept_dict[concept], model, top = top, gsf = gsf)]
        if concept in top_concepts:
            scores.append(1)
        else:
            scores.append(0)
            
    return np.mean(scores) * 100

def constructVector(concept, model):
    new_vector = np.zeros(model.embedding_matrix.shape[0])
    for feature in [s[0] for s in model.concept_features[concept]]:
        new_vector += model.fvector(feature)
        
    return  new_vector / len(model.concept_features[concept])