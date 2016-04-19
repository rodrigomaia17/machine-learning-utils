from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import cross_validation


algs = [GaussianNB(), LogisticRegression(), LinearSVC(),
        RandomForestClassifier(n_estimators=100)]


def classify(data, target):
    for alg in algs:
        result = cross_validation.cross_val_score(alg, data, target, cv=10)
        print("{0} : {1}".format(type(alg).__name__, result.mean()))
