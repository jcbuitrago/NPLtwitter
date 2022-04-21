from datetime import datetime
import string
# from string import punctuation
import nltk
#nltk.download()
import pandas as pd
import nltk.corpus
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import blankline_tokenize
from nltk.probability import FreqDist
# from nltk.util import bigrams, trigrams, ngrams
from nltk.stem import wordnet, WordNetLemmatizer

##### TRAINING SECTION #####

def train_function():
    
    ## Import the csv
    train_df = pd.read_csv("C:/Users/jcbui/OneDrive/Documents/Documentos/Endava/DataS/train.csv")
    
    ## Create a str with all the emergencies, also create a variable "e" with the amount of emergencies to later use to find the error.
    emergency_tweets = ""
    e=0
    total_size = 7613
    for i in range(total_size):
        if train_df["target"][i] == 1:
            e+=1
            tweet = train_df["text"][i].translate(str.maketrans('','',string.punctuation))
            emergency_tweets += (tweet + "\n" + "\n")
    
    ## Create the necesary tokens to adress the problema, either by word or tweet.
    
    ## Word token without stopwords
    tokens = word_tokenize(emergency_tweets)
    tokens_without_stopwords = []
    for word in tokens:
        if not (word.lower() in stopwords.words("english")):
            tokens_without_stopwords.append(word.lower())
    
    ## tweet token
    # blank = blankline_tokenize(emergency_tweets)
    
    ## Frequency List of tokens without stopwords
    fdist = FreqDist()
    
    for word in tokens_without_stopwords:
        fdist[word.lower()]+=1
    
    word_frq_dict = dict(fdist.most_common(1000))
    
    word_list = list(word_frq_dict.keys())
    ## Stemming, return the root of the words, may not work because of ambiguities. 
    ## To fix this we may use Lemmatization. It uses a dictionary to understand the proper root of the word. 
    
    # word_lem = WordNetLemmatizer()
    
    ## Create the bigrams, trigrams o ngrams that result in a better solution. This must be done with tokens in the form of words.
    # emergency_bigrams = list(nltk.bigrams(tokens))
    # emergency_trigrams = list(nltk.trigrams(tokens))
    # emergency_ngrams = list(nltk.ngrams(tokens, 5))
    # print(emergency_bigrams[:10])
    
    return word_list
    

##### TEST SECTION #####
start = datetime.now()
emergency_words = train_function()
test_df = pd.read_csv("C:/Users/jcbui/OneDrive/Documents/Documentos/Endava/DataS/test.csv")
sample_submission = pd.read_csv("C:/Users/jcbui/OneDrive/Documents/Documentos/Endava/DataS/sample_submission.csv")
test_size = len(test_df)
prediction = []

for i in range(test_size):
    ref = 5
    score = 0
    tweet = test_df["text"][i].translate(str.maketrans('','',string.punctuation))
    tweet_token = word_tokenize(tweet)
    
    if 5 < len(tweet_token) <= 10:
        ref = 3
    elif len(tweet_token) <= 5:
        ref = 2
    
    for word in tweet_token:
        if word.lower() in emergency_words:
            score += 1
            
    if score >= ref:
        prediction.append(1)
    else:
        prediction.append(0)

sample_submission["target"] = prediction

duracion = datetime.now() - start    
print(duracion)
               