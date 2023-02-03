import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import random
import nltk
from nltk import WordNetLemmatizer
from nltk import SnowballStemmer
from utils import clean
from tempfile import mkdtemp
from shutil import rmtree
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn import svm
import joblib
import json
np.random.seed(42)
random.seed(42)



def remove_digits_punctuation(text):
    text = text.strip(string.punctuation)
    text = re.sub(r'[\d]+', '', text)
    return text


def grid_search():
    print("in grid search")
    df = pd.read_csv("Project1-Classification.csv")
    vect = TfidfVectorizer(stop_words='english')
    scaler = StandardScaler(with_mean=False)
    dim_red = (TruncatedSVD(random_state=42),
               NMF(max_iter=500, random_state=42))
    clf = (svm.LinearSVC(C=100000, random_state=42),
           LogisticRegression(solver='saga', penalty='l1',
                              C=10, random_state=42, max_iter=20000),
           LogisticRegression(solver='saga', penalty='l2',
                              C=500, random_state=42, max_iter=20000),
           GaussianNB())

    train, test = train_test_split(
        df[["full_text", "root_label", "leaf_label"]], test_size=0.2, random_state=42)

    params = [{'vect__min_df': (3, 5), 'dim_red': dim_red,
               'dim_red__n_components': (5, 30, 80), 'clf': clf}]
    steps = [("vect", vect), ("scaler", scaler),
             ("dim_red", None), ("clf", None)]

    all_results = {}

    total_start = time()
    # clean vs not clean
    for cl in [False, True]:
        train1, test1 = train, test
        start = time()
        if cl:
            train1['full_text'] = train['full_text'].apply(clean)
            test1['full_text'] = test['full_text'].apply(clean)
        end = time()
        print(f"cleaned/not cleaned in {end-start} secs")

        # different ways of prepping the text

        lemmatizer = WordNetLemmatizer()
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        for prep in ["none", "stem", "lemma"]:
            start = time()
            train2, test2 = train1, test1
            if prep == 'stem':
                train1['full_text'] = train1['full_text'].apply(
                    nltk.word_tokenize)
                stemmerTemp = []
                for article in train1["full_text"]:
                    stemmerTemp.append(' '.join(stemmer.stem(word)
                                       for word in article))
                train2['full_text'] = stemmerTemp
                test1['full_text'] = test1['full_text'].apply(
                    nltk.word_tokenize)
                stemmerTemp2 = []
                for article in test1["full_text"]:
                    stemmerTemp2.append(' '.join(stemmer.stem(word)
                                        for word in article))
                test2['full_text'] = stemmerTemp2
                # train2['full_text'] = train1['full_text'].apply(stemmerTot)
                # test2['full_text'] = test1['full_text'].apply(stemmerTot)
            elif prep == 'lemma':
                train2['full_text'] = train2['full_text'].apply(
                    nltk.word_tokenize)
                lemmaTemp = []
                for article in train2["full_text"]:
                    lemmaTemp.append(
                        ' '.join(lemmatizer.lemmatize(word) for word in article))
                # print(f"LEMMA: {lemmaTemp[0]}")
                #print(f"TRAIN2: {train2['full_text'][0]}")
                train2['full_text'] = lemmaTemp

                test1['full_text'] = test1['full_text'].apply(
                    nltk.word_tokenize)
                lemmaTemp2 = []
                for article in test1["full_text"]:
                    lemmaTemp2.append(
                        ' '.join(lemmatizer.lemmatize(word) for word in article))
                test2['full_text'] = lemmaTemp2

                # train2['full_text'] = train1['full_text'].apply(lemmatizeTot)
                # test2['full_text'] = test1['full_text'].apply(lemmatizeTot)

            train2['full_text'] = train2['full_text'].apply(
                remove_digits_punctuation)
            test2['full_text'] = test2['full_text'].apply(
                remove_digits_punctuation)
            end = time()
            print(f"prepped in {end-start} secs")

            cachedir = mkdtemp(
                dir='/Users/ineshchakrabarti/ECE-219/')
            mem = joblib.Memory(cachedir=cachedir, verbose=0)
            pipe = Pipeline(steps=steps, memory=mem)
            print(pipe)

            # no need to swap out the classifiers as params takes care of that
            search = GridSearchCV(pipe, params, n_jobs=-1,
                                  cv=5, verbose=5, scoring='accuracy')
            start = time()
            print('started gridsearch')
            search.fit(train2['full_text'], train2['root_label'])
            end = time()
            #f.write(f"time: {end-start}s\n")
            print((f"grid search time: {end-start}s"))
            # store results
            if cl:
                key = 'cleaned_'+prep
            else:
                key = 'dirty_'+prep
            all_results[key] = search.cv_results_
            result_df = pd.DataFrame(search.cv_results_)
            result_df.to_csv(f"{key}.csv")
            rmtree(cachedir)

    total_end = time()
    print("FINALLY DONE")
    print(f"total_time: {total_end-total_start}")
    top5 = []
    for key, result in all_results.items():
        for i in range(len(result['mean_test_score'])):
            if len(top5) < 5 or result['mean_test_score'][i] > top5[-1][1]:
                top5.pop()
                top5.append(
                    ((key,)+tuple(result['params'][i]), result['mean_test_score'][i]))
                top5 = sorted(top5, key=lambda x: x[1], reverse=True)
    print(top5)
    with open('top5.txt', 'w') as f:
        f.write(top5)
        f.close()
    with open("all_results.json", "w") as outfile:
        json.dump(all_results, outfile)
    return all_results


def get_top5(all_results, output_path):
    top5 = []
    for key, result in all_results.items():
        for i in len(result['mean_test_score']):
            if len(top5) < 5 or result['mean_test_score'][i] > top5[-1][1]:
                top5.pop()
                top5.append(
                    ((key,)+tuple(result['params'][i]), result['mean_test_score'][i]))
                top5 = sorted(top5, key=lambda x: x[1], reverse=True)
    print(top5)
    with open(output_path, 'w') as f:
        f.write(top5)
        f.close()


if __name__ == '__main__':
    print("started search")
    all_results = grid_search()
    print("finished search")
    get_top5(all_results, "all_results.txt")
    print("got top5")

