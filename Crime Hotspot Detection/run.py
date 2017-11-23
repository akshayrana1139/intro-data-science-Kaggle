
# coding: utf-8

# In[229]:


import warnings
warnings.filterwarnings("ignore")


# Data Manipulation
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn as sk
import itertools
from scipy import stats

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

# import xgboost as xgb
# from mlxtend.classifier import StackingClassifier

# Ensembling
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

# Metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# Loading data
train = pd.read_csv("./crimeDataCsv.csv")


# In[230]:


train.head()


# In[231]:


# As there are some columns that are irrelevant for detecting the hotspot. So these columns can be removed.

train=train.drop(["S. No.", "ID", "X Coordinate", "Y Coordinate", "Case Number", "Updated On", "Latitude",
                  "Longitude", "Location"], 1)


# In[232]:


# Getting the last part of a block and using that as a value. [Avenue, Roads, Streets - clubbing them into one entity]
j = []
for i in train["Block"]:
    s = i.split(" ")
    j.append(s[len(s)-1])
train["Block"] = j

# Getting months out of the date as it will effect the crime rate [year is same for all, dates will have no impact so ignoring]
j = []
for i in train["Date"]:
    i = i.split("/")[0]
    j.append(i)
train["Date"] = j


# In[233]:


train.head()


# In[234]:


# Use this to extract festures from latitude, longitude using the google map api.
# Since in this case, all the dataset lies in Chicago, so not useful for us. 
# Future Use : We might extract information like middle of street, corner, crowded place etc.

import json
from urllib.request import urlopen
def getplace(location):
    url = "http://maps.googleapis.com/maps/api/geocode/json?"
    url += "latlng=%s,%s&sensor=false" % (location[0], location[1])
    v = urlopen(url).read()
    j = json.loads(v)
    components = j['results'][0]['address_components']
    country = town = None
    for c in components:
        if "postal_town" in c['types']:
            town = c['long_name']
        elif "locality" in c['types']:
            town = c['long_name']
    return town

# j = []
# from ast import literal_eval
# for i in train["Location"]:
#     s = getplace(literal_eval(i))
#     print(s)


# In[236]:


# Encoding values for String by grouping them into buckets. 
# Although Location and Location Description have 100 unique values so its 
# better to study the impact and find a pattern and then group them into sub groups..

list_to_encode = ["Primary Type", "Year", "Severity", "Arrest", "Domestic", "FBI Code", "Description", "Location Description", "Block", "IUCR", "Beat"]
for values in list_to_encode:
    train[values] = preprocessing.LabelEncoder().fit(train[values]).transform(train[values])


# In[237]:


# Mean Normalization : Beat, IUCR : May need for Description, Location Description, Community Area, Block
# Helps to form a gaussian curve for dataset and speeds up training and improves accuracy..

list_to_normalize = ["Block", "IUCR", "Description", "Location Description", "Beat", "Ward", "Community Area"]
for i in list_to_normalize:
    train[i] = ((train[i]-train[i].mean())/(train[i].max()-train[i].min())).round(2)


# In[238]:


train.head()


# In[239]:


# Splitting the dataset in training and testing with the ratio 60:40

X_train, X_test, y_train, y_test = train_test_split(train[["Date", "Block", "IUCR", "Primary Type", "Description", "Severity", "Location Description", "Arrest", "Domestic", "Beat", "District", "Ward", "Community Area", "FBI Code", "Year"]],
                                                        train["Hotspot"], test_size=.4)


# In[240]:


# Trying out different classifiers and storing their accuracies on training and testing set repectively. 
# Based on high performing classifiers, we can create an ensemble model of those to further imporve the accuracy.

result = pd.DataFrame(columns = ("Classifiers","Training","Testing"))
classifier, train_scores, test_scores = [],[],[]

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
classifier.append("RandomForestClassifier")
train_scores.append(random_forest.score(X_train, y_train))
test_scores.append(random_forest.score(X_test, y_test))

cf = SVC()
cf.fit(X_train, y_train)
classifier.append("SVM")
train_scores.append(cf.score(X_train, y_train))
test_scores.append(cf.score(X_test, y_test))

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
classifier.append("KNeighborsClassifier")
train_scores.append(knn.score(X_train, y_train))
test_scores.append(knn.score(X_test, y_test))

clf_log = LogisticRegression()
clf_log = clf_log.fit(X_train,y_train)
classifier.append("LogisticRegression")
train_scores.append(clf_log.score(X_train, y_train))
test_scores.append(clf_log.score(X_test, y_test))

# pd.DataFrame(list(zip(X_train.columns, np.transpose(clf_log.coef_))))

clf_pctr = Perceptron(
    class_weight='balanced'
    )
clf_pctr = clf_pctr.fit(X_train,y_train)
classifier.append("Perceptron")
train_scores.append(clf_pctr.score(X_train, y_train))
test_scores.append(clf_pctr.score(X_test, y_test))


clf_tree = tree.DecisionTreeClassifier(
    #max_depth=3,\
    class_weight="balanced",\
    min_weight_fraction_leaf=0.01\
    )
clf_tree = clf_tree.fit(X_train,y_train)
classifier.append("DecisionTreeClassifier")
train_scores.append(clf_tree.score(X_train, y_train))
test_scores.append(clf_tree.score(X_test, y_test))


clf_ext = ExtraTreesClassifier(
    max_features='auto',
    bootstrap=True,
    oob_score=True,
    n_estimators=1000,
    max_depth=None,
    min_samples_split=10
    #class_weight="balanced",
    #min_weight_fraction_leaf=0.02
    )
clf_ext = clf_ext.fit(X_train,y_train)
classifier.append("ExtraTreesClassifier")
train_scores.append(clf_ext.score(X_train, y_train))
test_scores.append(clf_ext.score(X_test, y_test))


clf_gb = GradientBoostingClassifier(
            #loss='exponential',
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.5,
            random_state=0).fit(X_train,y_train)
classifier.append("GradientBoostingClassifier")
train_scores.append(clf_gb.score(X_train, y_train))
test_scores.append(clf_gb.score(X_test, y_test))


clf_ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)
clf_ada.fit(X_train,y_train)
classifier.append("AdaBoostClassifier")
train_scores.append(clf_ada.score(X_train, y_train))
test_scores.append(clf_ada.score(X_test, y_test))


for i,text in enumerate(classifier):
    result.loc[i+1] = [text,np.around(train_scores[i]*100, decimals = 1),np.around(test_scores[i]*100, decimals=1)]
result.sort_values("Testing", ascending = False)

