# -*- coding: utf-8 -*-
"""
Created on July 12, 2019

@author: kphowell
"""

import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import numpy
from scipy import sparse
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import copy


NOTE_TEXT_INDEX = 1
GOC_INDEX = 2
INPATIENT_INDEX = 3
DATASET_INDEX = 4
NOTE_TYPE_INDEX = 7
FEATURE_INDEX = 8


class BackdoorAdjustment:
    """This class is adapted from github.com/tapilab/aaai-2016-robust"""
    def __init__(self):
        self.clf = LogisticRegression(class_weight='balanced')

    def predict_proba(self, X):
        # build features with every possible confounder
        l = X.shape[0]
        rows = range(l * self.count_c)
        cols = list(range(self.count_c)) * l
        data = [self.c_ft_value] * (l * self.count_c)
        c = sparse.csr_matrix((data, (rows, cols)))
        # build the probabilities to be multiplied by
        p = numpy.array(self.c_prob).reshape(-1, 1)
        p =numpy.tile(p, (X.shape[0], 1))

        # combine the original features and the possible confounder values
        repeat_indices = numpy.arange(X.shape[0]).repeat(self.count_c)
        X = X[repeat_indices]
        Xc = sparse.hstack((X, c), format='csr')
        proba = self.clf.predict_proba(Xc)
        # multiply by P(z) and sum over the confounder for every instance in X
        proba *= p
        proba = proba.reshape(-1, self.count_c, self.count_y)
        proba = numpy.sum(proba, axis=1)
        # normalize
        norm = numpy.sum(proba, axis=1).reshape(-1, 1)
        proba /= norm
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return numpy.array(proba.argmax(axis=1))

    def fit(self, X, y, c, c_ft_value=1.):
        self.c_prob = numpy.bincount(c) / len(c)
        self.c_ft_value = c_ft_value
        self.count_c = len(set(c))
        self.count_y = len(set(y))

        rows = range(len(c))#kph changed from rang(len(c))
        cols = c
        data = [c_ft_value] * len(c)
        c_fts = sparse.csr_matrix((data, (rows, cols)))
        Xc = sparse.hstack((X, c_fts), format='csr')

        self.clf.fit(Xc, y)


def backdoor_adjustment_var_C(X, y, z, c, rand, feature_names):
    clf = BackdoorAdjustment()
    clf.fit(X, y, z, c_ft_value=c)
    return clf


def show_most_informative_features(vectorizer, clf, n=50):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf._final_estimator.coef_[0], feature_names)) # coef can actually be found in a few places in the object, but it looks the same in each
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

def get_top_features(vectorizer, clf, class_labels):
    feature_names = vectorizer.get_feature_names()
    top10 = numpy.argsort(clf._final_estimator.coef_[0])[-20:]
    print("%s: %s" % ('pos', "\n".join(feature_names[j] for j in top10)))
    print('\n')
    bottom10 = numpy.argsort(clf._final_estimator.coef_[0])[:20]
    print("%s: %s" % ('neg', "\n".join(feature_names[j] for j in bottom10)))
    print('\n')
    return [feature_names[j] for j in top10], [feature_names[j] for j in bottom10]

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=pyplot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = pyplot.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=numpy.arange(cm.shape[1]),
           yticks=numpy.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def print_results(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred)
    print(report)

def add_binary_confound_train(featureset, confound_value, confound_names, confound_weight=1):
    if len(confound_names) != 2:
        print('You must enter two features for the binary case')
    if str(confound_names[0]) in featureset:
        print('confound already in featureset')
    if confound_names[0] in str(confound_value):
        featureset[str(confound_names[0])+'_note'] = confound_weight
        featureset['not_'+str(confound_names[0])+'_note'] = 0
    else:
        featureset[str(confound_names[0]) + '_note'] = 0
        featureset['not_' + str(confound_names[0]) + '_note'] = confound_weight
    return featureset

def create_binary_confound_test_sets(featureset, trackingset, confound_names, confound_weight=1):
    confound_featureset = []
    confound_trackingset = []
    featureset_confound_true = copy.deepcopy(featureset)
    trackingset_confound_true = copy.deepcopy(trackingset)
    for i in range(0, len(featureset_confound_true)):
        featureset_confound_true[i][0][str(confound_names[0])+'_note'] = confound_weight
        trackingset_confound_true[i][1][str(confound_names[0])+'_note'] = confound_weight
        featureset_confound_true[i][0]['not_'+str(confound_names[0])+'_note'] = 0
        trackingset_confound_true[i][1]['not_'+str(confound_names[0])+'_note'] = 0
    confound_featureset.append(featureset_confound_true)
    confound_trackingset.append(trackingset_confound_true)

    featureset_confound_false = copy.deepcopy(featureset)
    trackingset_confound_false = copy.deepcopy(trackingset)
    for i in range(0, len(featureset_confound_false)):
        featureset_confound_false[i][0][str(confound_names[0]) + '_note'] = 0
        trackingset_confound_false[i][1][str(confound_names[0]) + '_note'] = 0
        featureset_confound_false[i][0]['not_' + str(confound_names[0]) + '_note'] = confound_weight
        trackingset_confound_false[i][1]['not_' + str(confound_names[0]) + '_note'] = confound_weight
    confound_featureset.append(featureset_confound_false)
    confound_trackingset.append(trackingset_confound_false)

    return confound_featureset, confound_trackingset

def print_confound_goc_distribution(dataset, confounders):
    for index, confounder in confounders:
        goc_pos_count = 0
        goc_neg_count = 0
        for note in dataset:
            if confounder[0] in str(note[index]):
                if note[GOC_INDEX] == 'pos':
                    goc_pos_count += 1
                else:
                    goc_neg_count += 1
        print(confounder[0] + ' notes:\nGOC\+: ' + str(goc_pos_count) + '\nGOC-: ' + str(goc_neg_count))

def main(train_notes, test_notes, confound_index, confound_values, baseline=False):
    train_featuresets = []
    train_tracking = []
    confound_counts = [0 for x in range(0,len(confound_values))]
    for i in range(0, len(train_notes)):
        if not baseline:
            if len(confound_values) == 2: #these counts will work for binary confounds
                if confound_values[0] in str(train_notes[i][confound_index]):
                    confound_counts[0] += 1
                else:
                    confound_counts[1] += 1
            else: #this probably won't work out of the box for admit-ip admit-op, etc
                if str(train_notes[i][confound_index]) in confound_values:
                    confound_counts[confound_values.index(str(train_notes[i][confound_index]))] += 1
        doc_features = train_notes[i][FEATURE_INDEX].split(', ')
        doc_featureset = {}
        # Create a featureset (a dictionary with each features and the number of times it occurs in the document),
        # excluding any numbers or empty strings
        deduped_features = {}
        for feat in doc_features:
            if any(char.isdigit() for char in feat):
                continue
            if feat == '':
                continue
            if feat not in doc_featureset:
                doc_featureset[feat] = 1
            else:
                doc_featureset[feat] += 1
            if feat not in deduped_features:
                deduped_features[feat] = 1

        train_featuresets.append([deduped_features, train_notes[i][2], str(train_notes[i][confound_index])])
        train_tracking.append([train_notes[i][0], deduped_features])
    #kbest_featuresets = SelectKBest(chi2, k=200).fit_transform(train_featuresets)

    test_featuresets = []
    test_tracking = []
    for i in range(0, len(test_notes)):
        doc_features = test_notes[i][FEATURE_INDEX].split(', ')
        doc_featureset = {}
        deduped_features = {}
        # Create a featureset (a dictionary with each features and the number of times it occurs in the document),
        # excluding any numbers or empty strings
        for feat in doc_features:
            if any(char.isdigit() for char in feat):
                continue
            if feat == '':
                continue
            if feat not in doc_featureset:
                doc_featureset[feat] = 1
            else:
                doc_featureset[feat] += 1
            if feat not in deduped_features:
                deduped_features[feat] = 1

        test_featuresets.append([deduped_features, test_notes[i][2]])
        test_tracking.append([test_notes[i][0], deduped_features])

    predictions = []
    gold = []
    train_vect = DictVectorizer().fit([i[0] for i in train_featuresets])
    if not baseline:
        backdoor_model = backdoor_adjustment_var_C(train_vect.transform([i[0] for i in train_featuresets]), [1 if i[1] == 'pos' else 0  for i in train_featuresets], [1 if confound_values[0] in i[2] else 0 for i in train_featuresets], 1., '', '')
        for i in range(0, len(test_featuresets)):
            prediction = backdoor_model.predict(train_vect.transform(test_featuresets[i][0]))[0]
            predictions.append(prediction)
            true_val = 1 if test_featuresets[i][1] == 'pos' else 0
            gold.append(true_val)
    else:
        baseline_model = LogisticRegression(class_weight='balanced')
        baseline_model.fit(train_vect.transform([i[0] for i in train_featuresets]), [1 if i[1] == 'pos' else 0 for i in train_featuresets])
        for i in range(0, len(test_featuresets)):
            prediction = baseline_model.predict(train_vect.transform(test_featuresets[i][0]))
            predictions.append(prediction)
            true_val = 1 if test_featuresets[i][1] == 'pos' else 0
            gold.append(true_val)


    plot_confusion_matrix(gold, predictions, classes=[0,1],
                          title='Logistic Regression Confusion matrix, without normalization')
    pyplot.show()
    print_results(gold, predictions, classes=[0,1])
    return



#database = '../cambia_nlp_corpus.db'
database = '../merged_splits.db'
conn = sqlite3.connect(database)
curs = conn.cursor()
train_notes = curs.execute("SELECT * FROM TRAINING_SET").fetchall()
test_notes = curs.execute("SELECT * FROM DEV_SET").fetchall()
held_out_notes = curs.execute("SELECT * FROM TEST_SET").fetchall()
conn.close
model_top_features = []
model_bottom_features = []
confounders = [[INPATIENT_INDEX, ['1','2']],[INPATIENT_INDEX, ['2','1']],[DATASET_INDEX, ['PCC','not_PCC']],
               [DATASET_INDEX, ['PICSI','not_PICSI']],[DATASET_INDEX, ['FCS','not_FCS']],
               [NOTE_TYPE_INDEX, ['ADMITNOTE','not_ADMITNOTE']], [NOTE_TYPE_INDEX, ['EXCLUDE','not_EXCLUDE']],
               [NOTE_TYPE_INDEX, ['CODESTATUS','not_CODESTATUS']], [NOTE_TYPE_INDEX, ['DCSUMMARY','not_DCSUMMARY']],
               [NOTE_TYPE_INDEX, ['ED','not_ED']], [NOTE_TYPE_INDEX, ['PN','not_PN']], [NOTE_TYPE_INDEX, ['SW','not_SW']],
               [NOTE_TYPE_INDEX, ['OTHERSUMMARY','not_OTHERSUMMARY']]]

print('Baseline ')
main(train_notes, test_notes, 0,[], baseline=True)
for index, values in confounders:
    print(values[0])
    main(train_notes, test_notes, index,values, baseline=False)
#
# print('Train Notes:')
# print_confound_goc_distribution(train_notes, confounders)
# print('Test Notes:')
# print_confound_goc_distribution(test_notes, confounders)