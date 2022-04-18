from asyncore import read
import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

train_df = pd.read_csv("C:/Users/jcbui/OneDrive/Documents/Documentos/Endava/DataS/train.csv")
# print(train_df[train_df["target"] == 0]["text"].values[1])
# print(train_df.head())
# print(train_df["target"][0])
# print("True" if "#earthquake" in train_df["text"][0] else "False")

count_vectorizer = feature_extraction.text.CountVectorizer()
example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])
print(example_train_vectors[0].todense().shape)
print(example_train_vectors[0].todense())

train_vectors = count_vectorizer.fit_transform(train_df["text"])

clf = linear_model.RidgeClassifier()

scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

print(scores)
