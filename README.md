# Feature2Vec
[![arxiv](http://img.shields.io/badge/math.CO-arXiv%3A1908.11439-B31B1B.svg)](https://arxiv.org/abs/1908.11439)

Code used in the paper Feature2Vec: Distributional semantic modelling of human property knowledge. 

![Feature2Vec](https://github.com/stevend94/Feature2Vec/blob/master/imgs/arch5.png)

## Requirements 
Code ran on ubuntu 16.04 with anaconda python distribution with python 3.6.9. Packages include,
* numpy
* pandas
* scikit-learn
* spacy 
* tensorflow-gpu
* keras-gpu
* jupyter lab 
* conda_nb 

Note that the model probably trains faster on CPU but GPU versions of tensorflow and keras were used during experiments. We have also included a neural network class based on recent work https://www.mdpi.com/2504-2289/3/2/30 . We compare it with our model using hyper parameters that we found to work well (although better paramters may be found which could improve results).

[![Spacy](http://img.shields.io/badge/spacy%3A-008000.svg)](https://spacy.io/models/en)
spacy requires the en_core_web_lg language model.

python -m spacy download en_core_web_lg 

## Pretrained Embeddings 
Pretrained embeddings for features, along with save/load functions, are now available. The embeddings are placed in the embeddings folder as .txt files. The embeddings were trained using all concepts and features. Feature2Vec still requires the spacy word embeddings and property norm datasets to run. See tsne below for example  of possible analysis.

For another good visualization, see the [tensorflow projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/wmvanvliet/Feature2Vec/tf_projector/embeddings/projector_config.json) kindly provided by [Marijn van Vliet](https://github.com/wmvanvliet).


<img align="center" width="800" height="600" src="https://github.com/stevend94/Feature2Vec/blob/master/imgs/tsne2.png">




