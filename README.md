# Identifying-Question-Pairs-that-have-the-Same-Intent
This is a project based on the Quora-Kaggle challenge for identifying question pairs that could be answered using the same answer
First thing, I would like to give credit to Elior Cohen from (https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07) for his tutorial on how to implement the architecture described in Stanford University's paper: http://web.stanford.edu/class/cs224n/reports/2759336.pdf. I would also like to thank Quora for releasing its first question-pair dataset which was used to train and test the models in this repo.

## **Three Implemented Models**
  
   &nbsp;&nbsp;&nbsp;**1. Baseline model:** Simple model depending on encoding words using TF-IDF encoding.
  
   &nbsp;&nbsp;&nbsp;**2. Feature based model:** A model based on set features extracted from the text.
  
   &nbsp;&nbsp;&nbsp;**3. Siamese architecture model:** A more complex model that uses Glove pretrained word embeddings, &nbsp;&nbsp;&nbsp;and follows the Siamese architecture described by Stanford's paper.

## **Baseline Model**

![githubbaselinemodel](https://user-images.githubusercontent.com/16010276/48630869-2f7eed80-e9b5-11e8-972a-772ec8a7a848.png)


* In this model, regular expressions were used for preprocessing of the text to remove any insignificant characters, normalize capitalization, etc.
* TF-IDF stands for Term Frequency Inverse Document frequency. TF-IDF assigns a score to each word based on its frequency in one document in a corpus (hence "document frequency"), as well as its frequency across all documents in the corpus (hence inverse document frequency). In general, TF-IDF is used to rank the imporatnce of a word in a document. Based on TF-IDF we can do ranked retrieval of documents with respect to a certain query. This is done by getting the sum of the TF-IDF score of each word in each document, then ordering the documents according to the summation of the TF-IDF scores of the query words in each document. The higher the TF-IDF total score, the more relevant the document is.

* In this model, we use TF-IDF encoding to encode words in a numerical form. The term frequency in this scope becomes the frequency of a word in a single instance of a question, and the document frequency becomes the total number of occurences of this word throughout the whole dataset. 

* Scikit-Learn's TF-IDFVectorizer was used to get the TF-IDF encodings of the words in our model.

* After encoding, cosine similarity was used to judge on the similarity of a question pair applying a threshold of 0.9.

## **Feature Based Model**

![featurebasedgithub](https://user-images.githubusercontent.com/16010276/48632881-f8f7a180-e9b9-11e8-8f17-91ec6f623efc.png)
* In this model, NLTK's tokenizer was used to tokenize each question into a list of words, and for the preprocessing as well.

* Three features were extracted from each questin pair:

  &nbsp;&nbsp;&nbsp;1. The ratio of the questions' lengths.

  &nbsp;&nbsp;&nbsp;2. Whether or not the first word in both questions match.

  &nbsp;&nbsp;&nbsp;3. Whether or not the last word in both questions match.

* These three features are then passed to the model. The model is a simple fully connected neural network with 3 input neurons, and 2 hidden layers.

## **Siamese Architecture Model**

![git](https://user-images.githubusercontent.com/16010276/48633845-1ded1400-e9bc-11e8-989a-fa7af7011b9a.png)

* Having a siamese architecture means that we have two or more identical branches of the network, one for each input passed to the model.

* In this model, Glove's pretrained word embeddings were used. Word embedding is a way of mapping words to the vector space. Thus, every word is represented using a vector, and semantically similar words have close vectors. This allows for a better and a more meaningful numerical repersentation of words.

* In this model also, Keras's Embedding layer was used in order to replace every word with its corresponding vector from the Glove model. In order to do this, the following steps were followed:
 
  &nbsp;&nbsp;&nbsp;1. A dictionary of the words of the dataset was created. That is the list of unique words in the dataset.
 
  &nbsp;&nbsp;&nbsp;2. A loop over the dictionary was done to extract and store relevant words and their corresponding         vectors from the Glove model, since it is very large and not all words are needed.
 
  &nbsp;&nbsp;&nbsp;3. A mapping between the indices of the words in the dictionary and the vectors of these words was done   using a list in which every word vector is stored in the index corresponding to the index of the word it represents in the    dictionary.
  &nbsp;&nbsp;&nbsp;4. This index-vector mapping is passed to the embedding layer as the "weights" parameter.
  &nbsp;&nbsp;&nbsp;5. Each question has to be first converted to a list of indices to be passed to the embedding layer. For this, Keras's predefined Tokenizer method text_to_sequences is used.
  &nbsp;&nbsp;&nbsp;6. List of indices (questions as lists of indices instead of lists of words), and index-vector mapping are passed to the embedding layer, and it takes care of replacing each index with its corresponding vector.
* The embedding layer passes the inputs to the model which consists of an Long Short-Term Memory neural network, which is a type of Recurrent neural networks that can remember farther in the past thanks to its gates. Long Short-Term Memory NNs have the benefit of taking context into consideration while computing the prediction.
* Then the outputs from each branch of the Siamese architecture are combined using Manhattan distance.
