import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import random
import nltk
import string
from tempfile import mkdtemp
from shutil import rmtree
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from nltk import WordNetLemmatizer
from nltk.stem import SnowballStemmer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
import joblib
import seaborn as sn 
np.random.seed(42)
random.seed(42)


import sklearn.metrics as metrics


def clean(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    texter = re.sub(r"<br />", " ", text)
    texter = re.sub(r"&quot;", "\"",texter)
    texter = re.sub('&#39;', "\"", texter)
    texter = re.sub('\n', " ", texter)
    texter = re.sub(' u '," you ", texter)
    texter = re.sub('`',"", texter)
    texter = re.sub(' +', ' ', texter)
    texter = re.sub(r"(!)\1+", r"!", texter)
    texter = re.sub(r"(\?)\1+", r"?", texter)
    texter = re.sub('&amp;', 'and', texter)
    texter = re.sub('\r', ' ',texter)
    clean = re.compile('<.*?>')
    texter = texter.encode('ascii', 'ignore').decode('ascii')
    texter = re.sub(clean, '', texter)
    if texter == "":
        texter = ""
    return texter

def remove_digits_punctuation(text):
    text = text.strip(string.punctuation)
    text = re.sub(r'[\d]+', '', text)
    return text

def count_alphanum(s):
  return int(len(re.sub("[^a-zA-Z0-9]", "", s)))

def classification_metrics(test_labels, pred, multi=False):
    if multi:
        average = 'macro'
    else:
        average = 'binary'
    acc = metrics.accuracy_score(test_labels, pred)
    recall = metrics.recall_score(test_labels, pred, average=average, pos_label='climate')
    precision = metrics.precision_score(test_labels, pred, average=average, pos_label='climate')
    f1 = metrics.f1_score(test_labels, pred, average=average, pos_label='climate')
    return acc, recall, precision, f1

from sklearn.metrics import confusion_matrix
def plot_cm(true, pred, labels, title):
    cm = confusion_matrix(true, pred, labels=labels)
    # df_cm = pd.DataFrame(cm, index = [i for i in labels],
    #               columns = [i for i in labels])
    fig = plt.figure()
    sn.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()
    return cm

def reduce_NMF(train, test, n_components):
    nmf = NMF(n_components=n_components, random_state=42, max_iter=100)
    nmf.fit(train)
    W = nmf.transform(test)
    H = nmf.components_
    return W,H

def plot_roc(test_labels, DecScore, pos_label="climate", title="ROC curve"):
    print("changed")
    fpr, tpr, thres = metrics.roc_curve(test_labels, DecScore, pos_label=pos_label)
    auc = metrics.auc(tpr, fpr,)
    fig = plt.figure()
    
    plt.plot(tpr, fpr, label=f"AUC: {auc}")
    plt.legend()
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    plt.close()
    return auc

def lsi(train, test, topic_count):
    model = TruncatedSVD(n_components=topic_count, random_state=42)
    model.fit(train)
    reduced = model.transform(test)
    print(f"reduced: {type(reduced)} of shape: {reduced.shape}")
    print(f"test: {type(test)} of shape: {test.shape}")
    #explained_var = explained_variance_score(test, reduced)
    #ASK HOW TO GET EXPLAINED VARIANCE FOR TEST
    explained_var = model.explained_variance_ratio_.sum()
    print(explained_var)
    return reduced, explained_var