
"""
:Author: Arid Hasan
:script name: sentiment classification for Bangla language
"""

import os
import re
import csv
import random
import numpy as np
from time import time
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import warnings
import optparse
import os, errno
from datetime import datetime

def tokenize(text):
    """
    tokenizing text for Bangla language
    :param text:
    :return: tokenized text
    """
    text_tok = re.sub('([,!?।‘’“”`()])', r' \1 ', text)
    text_tok = re.sub('\s{2,}', ' ', text_tok)
    return text_tok


def read_bn_stopwords(loc='etc/stopwords-bn.txt'):
    stop_words = []
    with open(loc, 'r', encoding='utf-8') as f:
        for line in f:
            stop_words.append(line.strip())
    return stop_words


def remove_stopwords(text, lang='bn'):
    """
    removing stopwords from Bangla text.
    :param text:
    :param lang:
    :return:
    """
    if lang == 'bn':
        stop_words = read_bn_stopwords()
    else:
        stop_words = set(stopwords.words('english'))
    words = text.split()
    final_words = []
    for word in words:
        if word not in stop_words:
            final_words.append(word)
    final_text = " ".join(final_words)
    return final_text


def read_data(file_loc, delimiter=',', lang='bn'):
    """
    Reading files and preparing data to training and test

    :param file_loc:
    :param delimiter:
    :param lang:
    :return: train_x, train_y, label_encoder
    """
    data = []
    labels = []
    with open(file_loc, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        next(reader)
        print('Header not included')
        for row in reader:
            # line = line.strip()
            if len(row) < 3:
                continue
            # cols = line.split(delimiter)
            """
            if lang == 'bn':
                tokenized_text = tokenize(row[1].strip().lower())
            else:
                tokenized_text = row[1].strip().lower()
            processed_text = remove_stopwords(tokenized_text, lang)
            """
            if len(row[1]) > 0:
                labels.append(row[2].strip())
                data.append(row[1])
    print(labels)
    label_encoder = preprocessing.LabelEncoder()
    train_y = label_encoder.fit_transform(labels)
    # train_y = label_encoder.fit(labels)
    # train_y = labels
    train_x = np.array(data)
    # print("encoder" + str(label_encoder.classes_))
    return train_x, train_y, label_encoder




def calculate_performance(y_true, y_pred, labels):
    """
    Calculating performances of our model
    :param y_true:
    :param y_pred:
    :param labels:
    :return: accuracy, precision, recall, f1 score and classification report
    """
    (acc, P, R, F1) = (0.0, 0.0, 0.0, 0.0)
    acc = metrics.accuracy_score(y_true, y_pred)
    P = metrics.precision_score(y_true, y_pred, average='weighted')
    R = metrics.recall_score(y_true, y_pred, average='weighted')
    F1 = metrics.f1_score(y_true, y_pred, average='weighted')
    report = metrics.classification_report(y_true, y_pred, target_names=labels)

    return acc * 100, P * 100, R * 100, F1 * 100, report


if __name__ == '__main__':
    print('running---svc')
    start_time = time()

    warnings.filterwarnings("ignore")
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="train_data", default=None, type="string")
    parser.add_option('-v', action="store", dest="val_data", default=None, type="string")
    parser.add_option('-t', action="store", dest="test_data", default=None, type="string")
    parser.add_option('-o', action="store", dest="output_file", default=None, type="string")
    parser.add_option('-m', action="store", dest="model_file", default=None, type="string")
    parser.add_option('-a', action="store", dest="model", default=None, type="string")

    options, args = parser.parse_args()
    a = datetime.now().replace(microsecond=0)

    train_file = options.train_data
    dev_file = options.val_data
    test_file = options.test_data
    results_file = options.output_file
    best_model_path = options.model_file
    model = options.model
    outFile = open(results_file, "w")

    model_dir = os.path.dirname(best_model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    output_file = options.output_file


    lang = 'bn'
    n_features = 1500

    delimiter = '\t'
    #model = 'svc'

    random_seed = 2814
    random.seed(random_seed)

    X_train, y_train, label_encoder = read_data(train_file, delimiter, lang)
    X_test, y_test, test_le = read_data(test_file, delimiter, lang)

    # all_train = label_encoder.fit_transform(y_train + y_test)
    # y_train = test_le.fit_transform(y_train)
    # y_test = test_le.fit_transform(y_test)

    tfidf_vectorize = TfidfVectorizer(ngram_range=(1, 5), max_features=n_features)
    # tfidf_vectorizer = TfidfVectorizer(encoding='utf-8',lowercase=True, ngram_range=(1, 2), norm='l2', use_idf=True, max_df=0.95, min_df=3, max_features=n_features)

    t_start = time()
    X_train_feat = tfidf_vectorize.fit_transform(X_train)
    print(X_train_feat.shape)
    print("done in %0.4fs." % (time() - t_start))

    """
    :Initializing classifier/model
    """
    if model.lower() == 'rf':
        classifier = RandomForestClassifier(n_estimators=200, n_jobs=20, random_state=random_seed)\
            .fit(X_train_feat, y_train)
    elif model.lower() == 'svm:
        classifier = LinearSVC(random_state=random_seed).fit(X_train_feat, y_train)
        # classifier = LinearSVC()
    else:
        print("Only svm and rf supported")
    """
    :Creating tfidf-vectorizer features for test data
    :Predicting output from classifier/model
    """
    X_test_feat = tfidf_vectorize.transform(X_test)

    y_test_pred = classifier.predict(X_test_feat)
    y_test_pred = label_encoder.inverse_transform(y_test_pred)
    labels = list(label_encoder.classes_)  # + list(test_le.classes_)
    # labels = list(test_le.classes_)

    #y_test_pred = test_le.inverse_transform(y_test_pred)
    #labels = list(test_le.classes_)

    y_test_true = test_le.inverse_transform(y_test)
    """
    :Calculating performance for our model
    """
    acc, precision, recall, F1, report = calculate_performance(y_test_true, y_test_pred, labels)
    # acc, precision, recall, F1, report = calculate_performance(y_test_true, y_test_pred, labels)
    result = str("{0:.2f}".format(acc)) + "\t" + str("{0:.2f}".format(precision)) + "\t" + str(
        "{0:.2f}".format(recall)) + "\t" + str("{0:.2f}".format(F1)) + "\n"

    out = open(output_file, "w")
    out.write("Acc\tPrecision\tRecall\tF1\n");
    out.write(result)
    print("Test set:\t" + result)
    print(report)
    print ("time taken: ")
    print(time() - start_time)

