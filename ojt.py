# import necessary libraries
import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from collections import Counter

import bitermplus as btm
import tmplot as tmp

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from tqdm import tqdm
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

import texthero as hero
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

#LDA==========================================================================================================
#https://radimrehurek.com/gensim/models/ldamodel.html

# read the txt file
# with open(r'C:\Users\jimna\OJT\cleaned_tweets.txt') as file:
#     data = file.readlines()

# # create a dataframe
# df = pd.DataFrame(data, columns=['text'])

# # write dataframe to csv file
# df.to_csv('cleaned_tweets.csv', index=False)
# data = pd.read_csv('cleaned_tweets.csv')

# # preprocess text data
# stop_words = stopwords.words('english')

# def preprocess(text):
#     result = []
#     for token in simple_preprocess(text):
#         if token not in stop_words:
#             result.append(token)
#     return result

# data['processed_text'] = data['text'].apply(preprocess)

# # create dictionary and corpus
# dictionary = corpora.Dictionary(data['processed_text'])
# corpus = [dictionary.doc2bow(text) for text in data['processed_text']]

# # train LDA model
# num_topics = 10
# lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)

# # print top topics and their keywords
# for topic in lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=False):
#     print('Topic {}: {}'.format(topic[0], [word[0] for word in topic[1]]))

#Biterm========================================================================================================
#https://bitermplus.readthedocs.io/en/latest/index.html
# IMPORTING DATA
# df = pd.read_csv(
#     'cleaned_tweets.csv', header=None, names=['texts'])
# texts = df['texts'].str.strip().tolist()
# # PREPROCESSING
# # Obtaining terms frequency in a sparse matrix and corpus vocabulary
# X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
# tf = np.array(X.sum(axis=0)).ravel()
# # Vectorizing documents
# docs_vec = btm.get_vectorized_docs(texts, vocabulary)
# docs_lens = list(map(len, docs_vec))
# # Generating biterms
# biterms = btm.get_biterms(docs_vec)

# # INITIALIZING AND RUNNING MODEL
# model = btm.BTM(
#     X, vocabulary, seed=12321, T=10, M=20, alpha=50/8, beta=0.01)
# model.fit(biterms, iterations=20)
# p_zd = model.transform(docs_vec)

# METRICS
# perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, 8)
# coherence = btm.coherence(model.matrix_topics_words_, X, M=20)

# topics = btm.get_top_topic_words(
#     model,
#     words_num=100,)
# print(topics)

#distilbert===================================================================================================

#resets the visualization html
# if os.path.exists("TopicModels.html"):
#    os.remove("TopicModels.html")

# #Load data set
# prefix = 'Getting comments'
# pbar = tqdm(total=100, position=0, leave=True) #progress bar in terminal
# pbar.set_description(prefix)
# pbar.update(2)
# data = open('cleaned_tweets.txt', encoding="utf8").read().splitlines()
# #Set up UMAP and HDBSCAN models for dimension reductions and clustering
# prefix = 'Progress: Setting UMAP models and HDBSCAN models'
# pbar.set_description(prefix)
# pbar.update(18)
# umap_model = UMAP(n_neighbors=3, n_components=3, min_dist=0.05)
# hdbscan_model = HDBSCAN(min_cluster_size=5, min_samples=5, gen_min_span_tree=True, prediction_data=True)
# #loads embeddings from embedding model
# prefix = 'Loading embeddings'
# pbar.set_description(prefix)
# pbar.update(20)
# all_embeddings = np.load('embeddings.npy')

# #creates BERT model for topic modeling and fits the model to the data set
# prefix = 'Creating BERT Topic Model'
# pbar.set_description(prefix)
# pbar.update(30)

# topic_model = BERTopic(
#     umap_model=umap_model,
#     hdbscan_model=hdbscan_model,
#     top_n_words=5,
#     language='multilingual',
#     calculate_probabilities=True,
#     verbose=True
# )
# prefix = 'Fitting the model'
# pbar.set_description(prefix)
# pbar.update(10)
# topics, probs = topic_model.fit_transform(data, embeddings=all_embeddings)

# #Getting Topics for visualization and processing
# prefix = 'Getting Topics'
# pbar.set_description(prefix)
# pbar.update(10)

# freq = topic_model.get_topic_info()
# top_topics = freq.head(10) #topic_model.get_topic(10)
# len_of_topics = len(top_topics)

# prefix = 'Visualizing'
# pbar.set_description(prefix)
# pbar.update(5)

# fig1 = topic_model.visualize_topics(top_n_topics=len_of_topics)
# fig2 = topic_model.visualize_barchart(top_n_topics=len_of_topics)
# fig3 = topic_model.visualize_distribution(probs[200], min_probability=0.001)
# fig4 = topic_model.visualize_hierarchy(top_n_topics=len_of_topics)
# fig5 = topic_model.visualize_heatmap(top_n_topics=len_of_topics, width=1000, height=1000)
# fig6 = topic_model.visualize_term_rank()

# with open('TopicModels.html', 'a') as f:
#     f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
#     f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))
#     f.write(fig3.to_html(full_html=False, include_plotlyjs='cdn'))
#     f.write(fig4.to_html(full_html=False, include_plotlyjs='cdn'))
#     f.write(fig5.to_html(full_html=False, include_plotlyjs='cdn'))
#     f.write(fig6.to_html(full_html=False, include_plotlyjs='cdn'))

# print("\033[A                                                                                                                                                                     \033", end="\r")
# print(" ", end="\r")
# prefix = 'Topic Modeling Complete'
# pbar.set_description(prefix)
# pbar.update(5)
# pbar.close()
# #Coherence Metric
# print("\n\n==========================================COHERENCE=============================================\n\n")

# #Reg exp for tokenizing the data set
# tokenizer = lambda s: re.findall('\w+', s.lower())
# text = [tokenizer(t) for t in data]

# # Getting Topics
# all_topics = topic_model.get_topics()
# top = []
# keys = []
# for x in range(10):
#     keys.append(freq['Topic'].head(10)[x])

# #Tokenizing
# prefix = 'Getting Topics'
# pbar2 = tqdm(total=len(keys), position=0, leave=True)
# pbar2.set_description(prefix)
# for key in tqdm(keys, desc='Getting Topics', position=0, leave=True):
#     values = all_topics[key]
#     topic_1 = []
#     for value in tqdm(values, desc='Retrieving Values in topic ' + str(key), position=0, leave=True):
#         topic_1.append(value[0])
#     top.append(topic_1)

# # Creating a dictionary with the vocabulary
# word2id = Dictionary(text)
# vec = CountVectorizer()
# X = vec.fit_transform(data).toarray()
# vocab = np.array(vec.get_feature_names())
# # Coherence model
# cm = CoherenceModel(topics=top, texts=text, coherence='u_mass', dictionary=word2id)
# coherence_per_topic = cm.get_coherence_per_topic()
# #Results
# print("\n==========================================COHERENCE RESULTS=============================================\n")
# for index, x in enumerate(coherence_per_topic):
#     print("topic %2d : %5.2f" % (index + 1, x))

# coherence = cm.get_coherence()
# print(coherence)

#bertopic===================================================================================================

# Read Stopwords_EN_TL.txt and save it into a pandas DataFrame
stop_words_dataframe = pd.read_csv("Stopwords_EN_TL.txt")
stop_words = set(stop_words_dataframe.iloc[:,0])
# Read csv and save into a pandas DataFrame
docs_dataframe = pd.read_csv("cleaned_tweets.txt")
# Remove stopwords for every comment and clean the dataset
docs = []
index = 0
for w in docs_dataframe.iloc[:,0].items():
    series = hero.remove_stopwords(pd.Series(w[1]),stop_words)
    series = hero.preprocessing.clean(series)
    docs.append(series[0])
# Output the cleaned dataset to an excel file
cleaned_dataset = pd.DataFrame(docs)
cleaned_dataset.to_excel("cleaned_tweets.xlsx")
# Initialize the model and fit it to the data

# Hyperparameters:
# language - "english" or "multilingual"
# top_n_words - the top_n_words in each topic (no effect)
# n_gram_range - the n-gram to be used by the vectorizer in the model (no effect / incoherent)
# min_topic_size - how big a topic should be, adjusted to be similar to LDA
# nr_topics - topic reduction, made topics more incoherent

topic_model = BERTopic(min_topic_size=25, language = "multilingual")
topics, probs = topic_model.fit_transform(docs)
# Print the topics found by the model
topics = topic_model.get_topic_info()
topics.to_excel("output.xlsx")
topics
# Extract vectorizer and tokenizer from BERTopic
vectorizer = topic_model.vectorizer_model
tokenizer = vectorizer.build_tokenizer()

# Extract features for Topic Coherence evaluation
tokens = [tokenizer(doc) for doc in docs]
dictionary = corpora.Dictionary(tokens)

topic_words = [[words for words, _ in topic_model.get_topic(topic)] 
               for topic in range(len(set(topics))-1)]

# Evaluate
coherence_model = CoherenceModel(topics=topic_words,
                                 texts=tokens,
                                 dictionary=dictionary, 
                                 coherence='c_v')

# Print Coherence
coherence = coherence_model.get_coherence()
coherence
topic_model.visualize_barchart()