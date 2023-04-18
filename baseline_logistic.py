import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

os.system('cls' if os.name == 'nt' else 'clear')

def euclidean_distance(vec1, vec2):
    return [(e1-e2)**2 for e1,e2 in zip(vec1, vec2)]

def read_data(PATH):
    with open(PATH, "r", encoding="utf-8") as f:
        tourneys = f.readlines()
    
    encodings = []
    labels = []
    tourneys = [tourney.strip('\n') for tourney in tourneys]
    for tourney in tourneys:
        encoding1, encoding2, label = tourney.split(";")
        encoding1 = [float(e) for e in encoding1.split(",")]
        encoding2 = [float(e) for e in encoding2.split(",")]

        encoding = encoding1 + encoding2 # NOTE contatenation
        #encoding = euclidean_distance(encoding1, encoding2)
        encodings.append(encoding)
        labels.append(int(label))

    return encodings, labels


def assemble_matrices(encodings, labels):
    return np.array(encodings), np.array(labels)


def logistic_regression(X_train, y_train, X_test, y_test):
    X, Y = assemble_matrices(X_train, y_train)
    testX, testY = assemble_matrices(X_test, y_test)

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X, Y)

    prediction = model.predict(testX)
    return prediction

def naive_bayes(X_train, y_train, X_test, y_test):
    X, Y = assemble_matrices(X_train, y_train)
    testX, testY = assemble_matrices(X_test, y_test)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    testX = scaler.fit_transform(testX)

    model = MultinomialNB()
    model.fit(X, Y)

    prediction = model.predict(testX)
    return prediction


if __name__ == "__main__":

    TRAIN = "pooled_train"
    TEST = "pooled_test"
    X_train, y_train = read_data(TRAIN)
    X_test, y_test = read_data(TEST)

    lr_predictions = logistic_regression(X_train, y_train, X_test, y_test)
    print(lr_predictions)
    lr_report = classification_report(y_test, lr_predictions)
    print(accuracy_score(y_test, lr_predictions))
    print(lr_report)

    #Naive-Bayes
    nb_predictions = naive_bayes(X_train, y_train, X_test, y_test)
    print(nb_predictions)
    nb_report = classification_report(y_test, nb_predictions)
    print(accuracy_score(y_test, nb_predictions))
    print(nb_report)
