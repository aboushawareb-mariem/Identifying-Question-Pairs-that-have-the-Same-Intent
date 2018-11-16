# Identifying-Question-Pairs-that-have-the-Same-Intent
This is a project based on the Quora-Kaggle challenge for identifying question pairs that could be answered using the same answer
First thing, I would like to give credit to Elior Cohen from (https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07) for his tutorial on how to implement the architecture described in Stanford University's paper: http://web.stanford.edu/class/cs224n/reports/2759336.pdf. I would also like to thank Quora for releasing its first question-pair dataset which was used to train and test the models in this repo.

## **Three Implemented Models**:
  
   &nbsp;&nbsp;&nbsp;**1. Baseline model:** Simple model depending on encoding words using TF-IDF encoding.
  
   &nbsp;&nbsp;&nbsp;**2. Feature based model:** A model based on set features extracted from the text.
  
   &nbsp;&nbsp;&nbsp;**3. Siamese architecture model:** A more complex model that uses Glove pretrained word embeddings, &nbsp;&nbsp;&nbsp;and follows the Siamese architecture described by Stanford's paper.
