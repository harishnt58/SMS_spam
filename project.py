import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn import metrics


df = pd.read_csv('SMSSpamCollection.tsv',sep='\t')
#df=pd.read_csv('spam.csv')
print(df.head())

print(df.describe())

print(df.info())

print(df.isnull().sum())

# print(df['label'].value_counts())
x=df['message']
y=df['label']

#x=df['v1']
#y=df['v2']
X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=0.4,random_state=100)

count_vect=CountVectorizer()

count_vect.fit(X_train)
X_train_counts=count_vect.transform(X_train)
X_train_counts
print(X_train.shape)


tfidf_transformer=TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)


vectorizer=TfidfVectorizer()
X_train_tfidf=vectorizer.fit_transform(X_train)


clf=LinearSVC()
clf.fit(X_train_tfidf,y_train)

tect_clf=Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])
tect_clf.fit(X_train,y_train)
import pickle
filename = 'finalized_model.sav'
pickle.dump(tect_clf, open(filename, 'wb'))

predictions=tect_clf.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))

y_pred=tect_clf.predict(["your account balance"])
print("Predicted:")
print(y_pred)
y_pred=tect_clf.predict(["Your mobile number has won cash award"])
print("Predicted:")
print(y_pred)      
