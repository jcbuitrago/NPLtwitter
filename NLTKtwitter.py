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

## Import the csv
train_df = pd.read_csv("C:/Users/jcbui/OneDrive/Documents/Documentos/Endava/DataS/train.csv")

## Create a str with all the emergencies, also create a variable "e" with the amount of emergencies to later use to find the error.
emergency_tweets = ""
e=0
num_rows = 7613
# punctuation=re.compile(r'[-.?!,;:()|0-9]')

for i in range(750):
    if train_df["target"][i] == 1:
        e+=1
        # print(e)
        tweet_no_punct = train_df["text"][i].translate(str.maketrans('','',string.punctuation))
        tweet_token = word_tokenize(tweet_no_punct)
        tweet_token_rm_stopwords = [word for word in tweet_token if not word in stopwords.words()]
        tweet = (" ").join(tweet_token_rm_stopwords)
        emergency_tweets += (tweet + "\n" + "\n")

## Create the necesary tokens to adress the problema, either by word or tweet.

# print(emergency_tweets[:200])
tokens = word_tokenize(emergency_tweets)
# print(tokens[:10])
blank = blankline_tokenize(emergency_tweets)
# print(blank[:10])
fdist = FreqDist()

for word in tokens:
    fdist[word.lower()]+=1

print(fdist.most_common(10))

## Stemming, return the root of the words, may not work because of ambiguities. 
## To fix this we may use Lemmatization. It uses a dictionary to understand the proper root of the word. 

word_lem = WordNetLemmatizer()

## Create the bigrams, trigrams o ngrams that result in a better solution. This must be done with tokens in the form of words.
emergency_bigrams = list(nltk.bigrams(tokens))
emergency_trigrams = list(nltk.trigrams(tokens))
emergency_ngrams = list(nltk.ngrams(tokens, 5))
# print(emergency_bigrams[:10])

print(datetime.now() - start)