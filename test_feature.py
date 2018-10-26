
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import gensim, logging
from sklearn.model_selection import train_test_split
import csv
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
from plac_core import call
from keras.models import model_from_json


def readMyFile(filename):
   q_pair_id = []
   q1_id = []
   q2_id = []
   q1 = []
   q2 = []
   prediction = []

   with open(filename) as inputfile:
       csvReader = csv.reader(inputfile)
       for line in csvReader:
           q_pair_id.append(line[0])
           q1_id.append(line[1])
           q2_id.append(line[2])
           q1.append(line[3])
           q2.append(line[4])
           prediction.append(line[5])
   return q_pair_id, q1_id, q2_id, q1, q2, prediction

pid, q1id, q2id, q1, q2, dup = readMyFile("train.csv")

q1 = q1[1:]
q2 = q2[1:]
dup = dup[1:]
#printing out the first question in the pairs in a file
# with open ("test.txt","w")as fp:
#    for line in q1:
#        fp.write(line+"\n")

print("hi there mariem: ",len(q1))

#splitting of q1 to train, test and validation
q1_train, q1_test= train_test_split(q1, test_size=0.15, random_state=1)
q1_train, q1_val= train_test_split(q1, test_size=0.15, random_state=1)

#splitting of q2 to train, test and validation
q2_train, q2_test= train_test_split(q2, test_size=0.15, random_state=1)
q2_train, q2_val= train_test_split(q2, test_size=0.15, random_state=1)

#splitting of duplicate to train, test and validation
dup_train, dup_test= train_test_split(dup, test_size=0.15, random_state=1)
dup_train, dup_val= train_test_split(dup, test_size=0.15, random_state=1)

for i in range(len(q1_train)):
   q1[i] = word_tokenize(q1[i])
   q2[i] = word_tokenize(q2[i])


with open ("test2.txt","w")as fp:
 for list in q2:
     fp.write("\n")
     for item in list:
         fp.write(item+", ")

with open ("test1.txt","w")as fp:
 for list in q1:
     fp.write("\n")
     for item in list:
         fp.write(item+", ")

print("length q1: ", len(q1))
print("length q2: ", len(q2))



# print("q1: ", q1[105781])
# print("q2: ", q2[105781])
# print("q2 after: ", q2[105782])
# print("q2 b4: ", q2[105780])
x_train = []

for i in range((len(q1_train))):
   if((len(q1_train[i])==0) or (len(q2_train[i])==0)):
       length = 0
       sameFirstWord = 0
       sameLastWord = 0
   else:
       if(len(q1_train[i]) > len(q2_train[i])):
           length = round(len(q2_train[i])/(len(q1_train[i])*1.0), 1)
       else:
           length = round(len(q1_train[i])/(len(q2_train[i])*1.0), 1)
       sameFirstWord = (q1_train[i][0] == q2_train[i][0])
       sameLastWord = (q1_train[i][-1] == q2_train[i][-1])

   x_train += [[length, sameFirstWord, sameLastWord]]

x_train = np.array(x_train)
dup_train = np.array(dup_train)

x_val = []
#same for validation
for i in range((len(q1_val))):
   if((len(q1_val[i])==0) or (len(q2_val[i])==0)):
       length = 0
       sameFirstWord = 0
       sameLastWord = 0
   else:
       if(len(q1_val[i]) > len(q2_val[i])):
           length = round(len(q2_val[i])/(len(q1_val[i])*1.0), 1)
       else:
           length = round(len(q1_val[i])/(len(q2_val[i])*1.0), 1)
       sameFirstWord = (q1_val[i][0] == q2_val[i][0])
       sameLastWord = (q1_val[i][-1] == q2_val[i][-1])

   x_val += [[length, sameFirstWord, sameLastWord]]

x_val = np.array(x_val)
dup_val = np.array(dup_val)

x_test = []
#same for testing
for i in range((len(q1_test))):
   if((len(q1_test[i])==0) or (len(q2_test[i])==0)):
       length = 0
       sameFirstWord = 0
       sameLastWord = 0
   else:
       if(len(q1_test[i]) > len(q2_test[i])):
           length = round(len(q2_test[i])/(len(q1_test[i])*1.0), 1)
       else:
           length = round(len(q1_test[i])/(len(q2_test[i])*1.0), 1)
       sameFirstWord = (q1_test[i][0] == q2_test[i][0])
       sameLastWord = (q1_test[i][-1] == q2_test[i][-1])

   x_test += [[length, sameFirstWord, sameLastWord]]

x_test = np.array(x_test)
dup_test = np.array(dup_test)



json_file = open('feature_based_1_archi.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('feature_based_1_weights.h5')
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
scores = loaded_model.evaluate(x_test, dup_test)
print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1] * 100))
