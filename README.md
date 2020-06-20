# Data Science Twitter Text Sentiment Classification: Project Overview
* Created a tool that identifies data science Twitter text sentiment (Accuracy: 80%) to help social media notice negative word in the future when users post status.
* Used data set of Twitter text that be labeled in Kaggle (https://www.kaggle.com/c/ml2020spring-hw4/data). 
* Engineered features from words to become vector by using bag of words (BOW) and word embedding (CBOW).
* Optimized Deep Neural Networks and Recurrent Neural Networks (Long Short-term Memory) tuning parameters, learning rate and the size of batch, to reach the best model.

## Code and Resources Used
**Python Version:** 3.7  
**Packages:** pandas, numpy, torch, matplotlib, wordcloud, nltk, gensim    
**Model Building Colab:** https://colab.research.google.com/drive/16d1Xox0OW-VNuxDn1pvy2UXFIPfieCb9#scrollTo=r67y9UpchZ38        

## Data Cleaning
The data set contains labeled training data and testing data. The labeled training data have labels: 1 is negative meaning and 0 is positive meaning. After I load the txt to list, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:
* Transformed words to vectors by using BOW and CBOW.
* Splitted the data into train and validation sets with validation size of 10%.
* Set up torch dataset for batching in Dataloader.

## EDA
I looked inside the sentence of the data and calculate the ratio of positive/negative labels. Below are a few highlights from the figures.
WordCloud of Twitter text:
![alt text](https://github.com/ILing82816/ds_twitter_proj/blob/master/word_cloud.png "wordcloud")    

## Model Building
I tried two different models and evaluated them using binary cross entropy loss. I chose binary cross entropy because it is relatively easy to interpret for this type of model.  
I tried three different models:  
* **BOW + DNN** - Baseline for the model
* **CBOW + RNN (LSTM)** - Because the order of word would affect the sentence sentiment, I thought a considering the order way like CBOW and a memorable model like long short-term memory would be effective. 

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets.
* **CBOW + RNN:** Accuracy = 80.066% 

* **BOW + DNN:** Accuracy = 75.98 

## Next Step
To improve the accuracy of the classification, I can try more different nueral network structures and try different ways to deal with the text like autoencoder.
