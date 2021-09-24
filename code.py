import numpy as np
import pandas as pd
import itertools
import io
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Read the data
#Our data is in a csv file so we will use pandas.read_csv() method.
df=pd.read_csv('/content/news.csv',sep=',', engine='python', error_bad_lines=False)
#Get shape and head (this is just to check if our data is properly read)
# head() will print first 5 rows of entire dataset
df.shape
df.head()


#get labels, here we have only two values for lable real or fake, which we are gonna for to prediction
labels = df.label
#again head() will print first 5 labels
labels.head()

Name: label, dtype: object
In [0]:
#Split the dataset into train and test sets
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)
In [0]:
'''
Initialize a TfidfVectorizer
TF (Term Frequency): The number of times a word appears in a document is its Term Frequency. 
IDF (Inverse Document Frequency): Words that occur many times a document, but also occur many times in many others, may be irrelevant.
The TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features.
'''
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#Fit and transform train set, transform test set 
# we use x_train and x_test for this purpose

tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
In [0]:
'''
Passive Aggressive algorithms are online learning algorithms. 
Such an algorithm remains passive for a correct classification outcome, and turns aggressive in the event of a miscalculation,
updating and adjusting. Unlike most other algorithms, it does not converge. 
Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector.
'''
pac = PassiveAggressiveClassifier(max_iter=30)
pac.fit(tfidf_train,y_train)
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
Accuracy: 91.26%
In [0]:
#Build Confusion Matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
Out[0]:
array([[96,  7],
       [11, 92]])
In [0]:
'''
So with this model, we have 96 true positives, 92 true negatives, 7 false positives, and 11 false negatives.
'''
