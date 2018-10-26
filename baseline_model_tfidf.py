
#this is the version that worked, I had an issue when before that #dup[i] = none was nullifying its references in the other arrays as
#well

import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
from time import time
import random
import datetime
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.models import Model
from keras.layers.merge import Add

from nltk.tokenize import sent_tokenize, word_tokenize
from keras.models import Sequential
from keras.layers import Dense, LSTM, Merge, Input, Embedding, Reshape
import numpy
from keras.callbacks import ModelCheckpoint
import keras.backend as K

import tensorflow as tf
# global variables
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy.core.multiarray import dtype
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.text import Tokenizer, text_to_word_sequence


def readMyFile(filename):
  q_pair_id = []
  q1_id = []
  q2_id = []
  q1 = []
  q2 = []
  prediction = []

  with open(filename) as inputfile:
      csvReader = csv.reader(inputfile)
      count = 0
      for line in csvReader:
          q_pair_id.append(line[0])
          q1_id.append(line[1])
          q2_id.append(line[2])
          q1.append(line[3])
          q2.append(line[4])
          if line[5] == '0':
              prediction.append(0)
          else:
              prediction.append(1)
          count += 1
  inputfile.close()
  return q_pair_id, q1_id, q2_id, q1, q2, prediction


pid, q1id, q2id, q1, q2, dup = readMyFile("train.csv")
q1 = q1[1:]
q2 = q2[1:]
dup = dup[1:]
q1 = np.array(q1)
q2 = np.array(q2)
dup = np.array(dup)




print("I AM HERE:", len(q1), len(q2))
print("I am here again ", q1.shape, q2.shape)


vocabulary_list = np.append(q1, q2)
vectorizer = TfidfVectorizer()
tf = vectorizer.fit(vocabulary_list)


# print(type(tfidf))
# print(tfidf.toarray())
# print("FRESH")
# for row in tfidf.toarray():
#     print([val for val in row])
score = 0
for i in range(len(q1)):
    x1 = []
    x2 = []
    pred = 0
    x1.append(q1[i])
    x2.append(q2[i])
    q1_tf = tf.transform(x1).toarray()
    q2_tf = tf.transform(x2).toarray()
    sim = cosine_similarity(q1_tf, q2_tf)
    if(sim > 0.7):
        pred = 1
    else:
        pred = 0
    if(pred == dup[i]):
        score+=1
print("acc: ", (score/len(q1))*100)




