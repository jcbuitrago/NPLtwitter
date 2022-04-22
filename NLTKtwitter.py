import string
from string import punctuation
from datetime import datetime
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

start = datetime.now()
# def train_function_ngram():
## Import the csv
train_df = pd.read_csv("C:/Users/jcbui/OneDrive/Documents/Documentos/Endava/DataS/train.csv")

## Create a str with all the emergencies, also create a variable "e" with the amount of emergencies to later use to find the error.
emergency_tweets = ""
filter_list = []
tweetsize = []
for i in range(int(len(train_df)*0.7)):
    if train_df["target"][i] == 1:
        
        tweet_no_punct = train_df["text"][i].translate(str.maketrans('','',string.punctuation))
        
        tweet_token = word_tokenize(tweet_no_punct)
        tweetsize.append(len(tweet_token))
        tweet_token_rm_stopwords = []
        for word in tweet_token:
            if not (word.lower() in stopwords.words("english")):
                tweet_token_rm_stopwords.append(word.lower())
                
        tweet_bigrams = list(nltk.bigrams(tweet_token_rm_stopwords))
        
        for bigram in tweet_bigrams:            
            filter_list.append((" ").join(bigram))

## Create the necesary tokens to adress the problema, either by word or tweet.

# print(emergency_tweets[:200])
# print(tokens[:10])
# blank = blankline_tokenize(emergency_tweets)
# for text in blank:
#     prueba = text
#     print(text)
#     break
# emergency_bigrams = list(nltk.bigrams(prueba))
# print(emergency_bigrams)

# print(blank[:5])
# print(blank[:10])
fdist = FreqDist()

for word in filter_list:
    fdist[word.lower()]+=1

word_dict = dict(fdist.most_common(1000))
word_list = list(word_dict.keys())

for i in range(int(len(train_df)*0.7), len(train_df)):
        
        tweet_no_punct = train_df["text"][i].translate(str.maketrans('','',string.punctuation))
        
        tweet_token = word_tokenize(tweet_no_punct)
        tweet_token_rm_stopwords = []
        for word in tweet_token:
            if not (word.lower() in stopwords.words("english")):
                tweet_token_rm_stopwords.append(word.lower())
        
        tweet_final = (" ").join(tweet_token_rm_stopwords)
        
        
## Stemming, return the root of the words, may not work because of ambiguities. 
## To fix this we may use Lemmatization. It uses a dictionary to understand the proper root of the word. 

# word_lem = WordNetLemmatizer()

## Create the bigrams, trigrams o ngrams that result in a better solution. This must be done with tokens in the form of words.
# emergency_bigrams = list(nltk.bigrams(tokens))
# emergency_trigrams = list(nltk.trigrams(tokens))
# emergency_ngrams = list(nltk.ngrams(tokens, 5))
# print(emergency_bigrams[:10])

# train_function_ngram()
print(datetime.now() - start)