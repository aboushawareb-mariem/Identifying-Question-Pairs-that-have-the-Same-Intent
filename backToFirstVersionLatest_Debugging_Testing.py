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


# printing out the first question in the pairs in a file
# with open ("test.txt","w")as fp:
#    for line in q1:
#        fp.write(line+"\n")


def reg_pre_proc(text):  # preprocessing the data using regular expressions
  ''' Pre process and convert texts to a list of words '''
  text = str(text)
  text = text.lower()

  # Clean the text
  text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
  text = re.sub(r"what's", "what is ", text)
  text = re.sub(r"\'s", " ", text)
  text = re.sub(r"\'ve", " have ", text)
  text = re.sub(r"can't", "cannot ", text)
  text = re.sub(r"n't", " not ", text)
  text = re.sub(r"i'm", "i am ", text)
  text = re.sub(r"\'re", " are ", text)
  text = re.sub(r"\'d", " would ", text)
  text = re.sub(r"\'ll", " will ", text)
  text = re.sub(r",", " ", text)
  text = re.sub(r"\.", " ", text)
  text = re.sub(r"!", " ! ", text)
  text = re.sub(r"\/", " ", text)
  text = re.sub(r"\^", " ^ ", text)
  text = re.sub(r"\+", " + ", text)
  text = re.sub(r"\-", " - ", text)
  text = re.sub(r"\=", " = ", text)
  text = re.sub(r"'", " ", text)
  text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
  text = re.sub(r":", " : ", text)
  text = re.sub(r" e g ", " eg ", text)
  text = re.sub(r" b g ", " bg ", text)
  text = re.sub(r" u s ", " american ", text)
  text = re.sub(r"\0s", "0", text)
  text = re.sub(r" 9 11 ", "911", text)
  text = re.sub(r"e - mail", "email", text)
  text = re.sub(r"j k", "jk", text)
  text = re.sub(r"\s{2,}", " ", text)

  return text

print("q1 shape BEFORE outside gen: ", q1.shape)
print("q2 shape BEFORE outside gen: ", q2.shape)
print("dup shape BEFORE outside gen: ", dup.shape)


val_end = 40429#80858


dup_test = []
q1_test = []
q2_test = []
count_test_1 = 0
count_test_0 = 0

dup_val = []
q1_val = []
q2_val = []
count_val = 0

dup_train = []
q1_train = []
q2_train = []
count_train = 0

for i in range(len(dup)):
  if((count_test_1 < int(40429/2)) and (dup[i] == 1)):
      dup_test.append(dup[i])
      q1_test.append(q1[i])
      q2_test.append(q2[i])
      count_test_1+=1
  else:
      if((count_test_0 < 40429/2) and (dup[i] == 0)):
          dup_test.append(dup[i])
          q1_test.append(q1[i])
          q2_test.append(q2[i])
          count_test_0+=1
      else:
          if(count_val<40429):
              dup_val.append(dup[i])
              q1_val.append(q1[i])
              q2_val.append(q2[i])
              count_val+=1
          else:
              dup_train.append(dup[i])
              q1_train.append(q1[i])
              q2_train.append(q2[i])


q1_train = np.array(q1_train)
q2_train = np.array(q2_train)
dup_train = np.array(dup_train)

q1_val = np.array(q1_val)
q2_val = np.array(q2_val)
dup_val = np.array(dup_val)

q1_test = np.array(q1_test)
q2_test = np.array(q2_test)
dup_test = np.array(dup_test)

vocabulary_list = np.append(q1, q2)

tokenizer1 = Tokenizer()

tokenizer1.fit_on_texts(vocabulary_list)

count = 0
max_seq_length = 35


gen_indx_mat1_test = tokenizer1.texts_to_sequences(q1_test)
gen_indx_mat2_test = tokenizer1.texts_to_sequences(q2_test)
test1 = pad_sequences(gen_indx_mat1_test, maxlen=max_seq_length)
test2 = pad_sequences(gen_indx_mat2_test, maxlen=max_seq_length)

json_file = open('backToFirstVersionLatest_Debugging_architecture_100epochs_300D.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('backToFirstVersionLatest_Debugging_100epochs_weights_300D.h5')
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
scores = loaded_model.evaluate([test1, test2], dup_test)
print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1] * 100))
