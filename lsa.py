#i've used this file to practice a little bit with sklearn (the library the allows me to analyse data with lsa)

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from IPython.display import display
import pandas as pd 
import numpy as np 

body = [
    "the quick brown fox",
    "the slow brown dog",
    "the quick red dog",
    "the lazy yellow fox"
    "the lazy yellow fox"
]

"""
vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(body)

bag_of_words.todense() #see the matrix

svd = TruncatedSVD(n_components=2)
lsa = svd.fit_transform(bag_of_words)

topic_encoded_df = pd.DataFrame(lsa,columns=['topic 1','topic 2'])
topic_encoded_df['body'] = body

dictionary = vectorizer.get_feature_names_out() #unique values from body

encoding_matrix = pd.DataFrame(svd.components_,index=['topic 1',"topic 2"],columns=dictionary).T
encoding_matrix #shows the enconding matrix
"""

"""
vectorizer = CountVectorizer(min_df=1,stop_words='english')
bag_of_words = vectorizer.fit_transform(body)

svd = TruncatedSVD(n_components=1)
lsa = svd.fit_transform(bag_of_words)

topic_encoded_df = pd.DataFrame(lsa,columns=['topic 1','topic 2'])
topic_encoded_df['body'] = body

dictionary = vectorizer.get_feature_names_out() #unique values from body

encoding_matrix = pd.DataFrame(svd.components_,index=['topic 1',"topic 2"],columns=dictionary).T
encoding_matrix['abs_topic 1'] = np.abs(encoding_matrix['topic 1'])
encoding_matrix['abs_topic 2'] = np.abs(encoding_matrix['topic 2'])
display(encoding_matrix.sort_values('abs_topic 1',ascending=False)) #shows the enconding matrix
"""

"""
vectorizer = TfidfVectorizer(min_df=1,stop_words='english')
bag_of_words = vectorizer.fit_transform(body)

svd = TruncatedSVD(n_components=2)
lsa = svd.fit_transform(bag_of_words)

topic_encoded_df = pd.DataFrame(lsa,columns=['topic 1','topic 2'])
topic_encoded_df['body'] = body

dictionary = vectorizer.get_feature_names_out() #unique values from body

encoding_matrix = pd.DataFrame(svd.components_,index=['topic 1',"topic 2"],columns=dictionary).T
encoding_matrix['abs_topic 1'] = np.abs(encoding_matrix['topic 1'])
encoding_matrix['abs_topic 2'] = np.abs(encoding_matrix['topic 2'])
display(encoding_matrix.sort_values('abs_topic 1',ascending=False)) #shows the enconding matrix
"""

