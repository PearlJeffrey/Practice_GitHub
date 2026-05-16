import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("instagram_users.csv")
print(df)

print(df.columns)
print("******************************")

data = df[["Caption", "Hashtags"]]
print(data.head())

captions = data["Caption"].tolist()
"""TF-IDF  ---> Term Frequency and Inverse Document Frequency """
uni_tfidf = text.TfidfVectorizer(input=captions, stop_words="english")
uni_matrix = uni_tfidf.fit_transform(captions)
uni_sim = cosine_similarity(uni_matrix)

def recommend_post(x):
  return ", ".join(data["Caption"].loc[x.argsort()[-5:-1]])

data["Recommended Post"] = [recommend_post(x) for x in uni_sim]
print(data.head())



