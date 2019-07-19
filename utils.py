import os 
import spacy 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score


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

