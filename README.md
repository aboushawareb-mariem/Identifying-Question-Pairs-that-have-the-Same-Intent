# Identifying-Question-Pairs-that-have-the-Same-Intent
This is a project based on the Quora-Kaggle challenge for identifying question pairs that could be answered using the same answer
First thing, I would like to give credit to Elior Cohen from (https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07) for his tutorial on how to implement the architecture described in Stanford University's paper: http://web.stanford.edu/class/cs224n/reports/2759336.pdf. I would also like to thank Quora for releasing its first question-pair dataset which was used to train and test the models in this repo.

## **Three Implemented Models**
  
   &nbsp;&nbsp;&nbsp;**1. Baseline model:** Simple model depending on encoding words using TF-IDF encoding.
  
   &nbsp;&nbsp;&nbsp;**2. Feature based model:** A model based on set features extracted from the text.
  
   &nbsp;&nbsp;&nbsp;**3. Siamese architecture model:** A more complex model that uses Glove pretrained word embeddings, &nbsp;&nbsp;&nbsp;and follows the Siamese architecture described by Stanford's paper.

## **Baseline Model**
![Baseline Model](/home/mariem/Pictures/githubbaseline.png)
* In this model, regular expressions were used for preprocessing of the text to remove any insignificant characters, normalize capitalization, etc.
* TF-IDF stands for Term Frequency Inverse Document frequency. TF-IDF assigns a score to each word based on its frequency in one document in a corpus (hence "document frequency"), as well as its frequency across all documents in the corpus (hence inverse document frequency). In general, TF-IDF is used to rank the imporatnce of a word in a document. Based on TF-IDF we can do ranked retrieval of documents with respect to a certain query. This is done by getting the sum of the TF-IDF score of each word in each document, then ordering the documents according to the summation of the TF-IDF scores of the query words in each document. The higher the TF-IDF total score, the more relevant the document is.

* In this model, we use TF-IDF encoding to encode words in a numerical form. The term frequency in this scope becomes the frequency of a word in a single instance of a question, and the document frequency becomes the total number of occurences of this word throughout the whole dataset. 

* Scikit-Learn's TF-IDFVectorizer was used to get the TF-IDF encodings of the words in our model.

* After encoding, cosine similarity was used to judge on the similarity of a question pair applying a threshold of 0.9.

## **Feature Based Model**


