import os 
import spacy 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import pandas as pd 


def retrieve_info(dataframe):
    """Function to extract features and concepts from dataframe"""
    concepts = []
    features = []

    for num in range(dataframe.values.shape[0]):
        c = dataframe.iloc[num]['Concept']
        if c not in concepts:
            concepts.append(c)

    for num in range(dataframe.values.shape[0]):
        f = dataframe.iloc[num]['Feature']
        if f not in features:
            features.append(f)
            
    return concepts, features


def build_norms(path, save_path):
    """Function to build dataframe from feature norms and save as csv. Spefically 
       for the mcrae excel data from https://sites.google.com/site/kenmcraelab/norms-data"""
    
    dataframe = pd.read_excel(path)
    
    concepts, features = retrieve_info(dataframe)
    feature_to_id = dict(zip(features, list(range(len(features)))))
    concept_to_id = dict(zip(concepts, list(range(len(concepts)))))
    
    # get production frequencies in a matrix format
    matrix = np.zeros((len(concepts), len(features)))
    for num in range(dataframe.values.shape[0]):
        c = dataframe.iloc[num]['Concept']
        f = dataframe.iloc[num]['Feature']
        matrix[concept_to_id[c], feature_to_id[f]] = dataframe.iloc[num]['Prod_Freq']
    
    
    # build dataframe of norms using dictionary 
    data_dict = {}
    data_dict['Vectors'] = concepts

    for feature in features:
        data_dict[feature] = matrix[:, feature_to_id[feature]]
    
    
    data_frame = pd.DataFrame(data_dict)
    data_frame.to_csv(save)path


def neighbour_score(concept_dict, model, top = 10):
    """Function to score the ability of a set of vectors to predict their 
        unseen golden feature representation when they are unseen."""
    
    scores = []
    for concept in concept_dict.keys():
        top_concepts = [s[1] for s in model.rank_neighbours(concept_dict[concept], top = top)]
        if concept in top_concepts:
            scores.append(1)
        else:
            scores.append(0)
            
    return np.mean(scores) * 100


def score_features(top_feats, features, score_func):
    """Method to score the number of correct features predicted by a vector"""
    predicted_features = [x[1] for x in top_feats]
    correct_features = [1 if f in features else 0 for f in predicted_features]
    
    score = score_func(correct_features, np.ones(len(predicted_features)))

    return score


def feature_score(model, data_type='train', max_features=0, score_func = accuracy_score):
    """Scores model on its ability to retrieve correct features

    Parameters
    ----------

    model : object 
        Either PLSR or Feat2Vec model 

    data_type : string ("train" or "test")
        Indicate whether this evaluation is being performed on the training 
        or testing data 

    max_features : int 
        The maximum number of features to retrieve

    ----------

    returns 
    ----------
        score : float
            the percentage of correct features predicted
    ----------
    """

    if data_type == 'train':
        concepts = model.train_words
    if data_type == 'test':
        concepts = model.test_words

    total_scores = []
    for concept in concepts:
        features = model._get_features(concept=concept, max_features=max_features)
        
        vector = model.wvector(concept)
        top_feats = model.top_features(vector=vector, top = len(features))
        total_scores.append(score_features(top_feats, features=features, score_func = score_func))

    return total_scores
    

def construct_vector(concept, model):
    """Function to construct new word embeddings from property vectors"""
    new_vector = np.zeros(model.embedding_matrix.shape[0])
    for feature in [s[0] for s in model.concept_features[concept]]:
        new_vector += model.fvector(feature)
        
    return  new_vector / len(model.concept_features[concept])

