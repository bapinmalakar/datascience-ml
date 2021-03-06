#Word embedding is type of represntation of words, which allow machine learning model to understan simillar meaning of words
#Technically word embeding is mapping of words to vectors of real number using Neural network, probabilistic model or 
# dimesion reduction on word co-occurance matrix.
# Its is language modeling and feature learning technique
# Word embedding is way to perform mapping using neural network
# Various word embedding model word2vec(Google), fastest(facebook), Glove(Standford)
# Word Embedding also called Distributed Semantic Model or Distributed Represented Model or DistributeSemantic Vector Space Model or Vector Space model
# Semantic means categorius similar words together

# Here words are classified or group based on meaning and each group represent induvidual vector
#i.e apple, orage, pinepal are fruits, so they placed in fruit Vector

# Here we practise word2vec model

#It's shallow two-layer neural network
# Shallow neural network consist of one hidden layer between input and output layer, where Deep nural network have multiple
# hidden layer between input and output layer
# word2vec is a two-layer network where there is input one hidden layer and output.
# word2vec represent words in vector space representation
# words are represent in the form of vector, placement are done
# similar words together, dissimilar words far away
# this is known as semantic relationship
# Neural network doesn't understand number, only understand number, so it will also convert text to numaric vector
# word2vector also reconstruct the linguistic context of the words
# means it learn vector representation  of words throug context

#before word2vec we used Latin Semantic Approach

# fit vocability 
# transform to vector
# it will put 0 if word not present in vocabulary, or put count of the word

# from sklearn.feature_extraction.text import CountVectorizer
# sentence_1= "guru99 is the best sitefor online tutorials. I love to visit guru99."
# vectorizer = CountVectorizer()
# vocabulary = vectorizer.fit([sentence_1])
# X= vectorizer.transform([sentence_1])

# print('For sentence_1\n')
# print(X.toarray())
# print(vocabulary.get_feature_names())


#Cons
# ignore order of words
# ignore context of words, menas it can create two vector for different words but semantically they are same


#Word2vec learns word by predicting its surrounding context.
# suppose "He loves Football." and want to calculate word2vec for word love
# suppose
# loves =  Vin. P(Vout / Vin) is calculated
# Vin is input word
# P is probability of likelihood
# Vout is the output word
# 
# Word2vec Architecture
# Two architructure used by Word2vec
# 1. Continuous Bag of words (CBOW)
# 2. Skip gram 

#Learning word representation is unsupervised but we need target to train model
# The CBOW and Skip Gram convert unsupervised representation to supervised form for train the model

#In CBOW 
# prediction used window of sourrounding contxt windows, means predict word from given sequence of words or context
# ex: Wi-1, Wi-2, Wi+1, Wi+2 are given word then it will predict Wi
# if V is vocabulary size and N is the hidden layer size. Input is defined as {Xi-1,X i-2, Xi+1, Xi+2}
# so, weight metrix is V * N
# another metrix can obtain by multiply Input Vector with Weight Metrix
# h=Xit*W, Xit input vector and W weight vector
# calculate the match between contxt and next word used
# u= predicted_representation * h, predicted_representation is obtained model

#Skip gram is opposite of CBOW, it will predict sequence of words or context from given word
# ex: If Wi is given, this will predict the context or Wi-1,Wi-2,Wi+1,Wi+2.
# It's reverse of CBOW, here model provide the sequence from given word

#Word2vec give option to choice between CBOW and Skip gram. This parameter need to be provide during traing the model

# CBOW many times faster then Skip gram, better frequency for frequent words
# where Skip gram need small dataset for training

#NLTK vs Word2vec
#NLTK is Natural Language Tollkit. Used for preprocessing text and clear the text and prepare features from effective word
# Like POS tags, Lemmatizing, Stemming, Remove stop word, rmove rare word etc.

#Word2vec used for semantic and syntetic matching. Using this we can find similar words, dissimilar words, dimensional reduction
# Another feature is convert Higher dimensional representation of the text into lower dimensional vector
# usecase predicting word context, topic modeling, document similarity

#start with code
# pip3 install gensim

import nltk as nltk
from gensim.models import Word2Vec # model for topic modeling and document indexing
# from nltk.corpus import abc # import abc file for test

# print('setup done: ', abc.sents())

# model = Word2Vec(abc.sents()) # craete  Word2Vec by the abc file
# # print(model)
# X = list(model.wv.vocab) #Vcabulary is stored in the form of the variable
# # print(X) 

# data = model.most_similar('science') # find similar words of science by model prediction

# print(data)

#Activation function
# activation function define the output od neural network after a set of steps
# X1, X2, X3, ... Xn is the node of nural network and W1,W2,W3,...Wn is the weight
# Output= summation of all weigh and node value, act as activation function
# If no activativation then output will be liner. For complext or non-linear output we need activation function
# # Softmax is output layer function which activate each node.
# 
# Topic modeling discover hidden structure in the text body. 


# now build a chatbot using Word2Vec
# prepare a json file which have 3 component
# tag: define the topic or subject of the discussion
# patter: define the question may ask by user to bot
# responses: define the answer of cosrresponding question/pattern

#setp1 prepare data, input/data.json prepared

#setp2 load data
import string
import json
import pandas as pd
from nltk.corpus import stopwords
from textblob import Word

json_file = './input/data.json'
with open(json_file, 'r') as f: # open file in read mode, it will return content in encodded mode
    print(f)
    data = json.load(f) #load josn data
    print(data)

#step3 convert data in dataframe
dataFrame = pd.DataFrame(data)
print(dataFrame.head())
dataFrame['patterns'] = dataFrame['patterns'].apply(', '.join) # convert array to string
print(dataFrame.head())

#step4 now clear data, remove stop word, filtered punctuation, removed numbers and dots. for that we going to use nltk

stop = stopwords.words('english') #only english words
print('stopss: ', stop)

#convert to lower case
dataFrame['patterns'] = dataFrame['patterns'].apply(lambda x: ' '.join(x.lower() for x in x.split())) #x.split() divide the each row into words
print('\n lowercase\n', dataFrame.head())

# filterd punctuation of from each patterns row
dataFrame['patterns'] = dataFrame['patterns'].apply(lambda x: ' '.join(x for x in x.split() if x not in string.punctuation))
print('\n punctuation\n', dataFrame.head())

dataFrame['patterns'] = dataFrame['patterns'].str.replace('[^\w\s]', '') #remove dots
print('\n\n', dataFrame['patterns'])

#remove digits
dataFrame['patterns'] = dataFrame['patterns'].apply(lambda x: ' '.join(x for x in x.split() if not x.isdigit())) #remove digit those are act as word
print('after remove pattern:: \n', dataFrame['patterns'])

#remove stop words
dataFrame['patterns'] = dataFrame['patterns'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop))
print('\nremove stopword \n', dataFrame['patterns'])

#remove different form of same word using lemmatization
dataFrame['patterns'] = dataFrame['patterns'].apply(lambda x: ' '.join([Word(word).lemmatize() for word in x.split()]))
print('\n\nRemoved same word\n', dataFrame['patterns'])

#make input vectors
Big_List = []
for row in dataFrame['patterns']:
    li = list(row.split()) # break sentence into words
    Big_List.append(li) # append list of words into big list

print(Big_List) # big list contain list of each sentence words

model = Word2Vec(Big_List, min_count=1, size=300, workers=4)