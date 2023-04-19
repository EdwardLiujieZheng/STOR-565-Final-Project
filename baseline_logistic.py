import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


os.system('cls' if os.name == 'nt' else 'clear')

def extract_player_features(encodings: list[float]):
    player1 = []
    player2 = []
    for i in range(1, 31):
        if i%2 == 1:
            player1.append(encodings[i])
        else:
            player2.append(encodings[i])
    return player1, player2


def difference_squared(encodings):
    player1, player2 = extract_player_features(encodings)
    return [(e1-e2)^2 for e1,e2 in zip(player1, player2)]


def difference(encodings):
    player1, player2 = extract_player_features(encodings)
    return [(e1-e2) for e1,e2 in zip(player1, player2)]


def read_data(PATH, encoding_func=None):
    """Reads column for each player, substract player encodings to get match encoding."""
    with open(PATH, "r", encoding="utf-8") as f:
        matches = f.readlines()
    
    encodings = []
    labels = []
    matches = [match.strip('\n') for match in matches]
    for match in matches:
        encoding, label = match.split(";")
        encoding = [float(e) for e in encoding.split(",")]
        
        if encoding_func:
            encoding = encoding_func(encoding)
        encodings.append(encoding)
        labels.append(int(label))
    return np.array(encodings), np.array(labels)


def logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return prediction

def lda(X_train, y_train, X_test, y_test):
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return prediction

def qda(X_train, y_train, X_test, y_test):
    model = QuadraticDiscriminantAnalysis()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return prediction

def naive_bayes(X_train, y_train, X_test, y_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    return prediction


if __name__ == "__main__":

    TRAIN = "modified_train"
    TEST = "modified_test"
    X_train, y_train = read_data(TRAIN)
    X_test, y_test = read_data(TEST)

    # Logistic Regression 
    lr_predictions = logistic_regression(X_train, y_train, X_test, y_test)
    lr_report = classification_report(y_test, lr_predictions)
    print("Logistic Regression:")
    print(accuracy_score(y_test, lr_predictions))
    print(lr_report)

    # Naive-Bayes
    nb_predictions = naive_bayes(X_train, y_train, X_test, y_test)
    nb_report = classification_report(y_test, nb_predictions)
    print("Naive Bayes:")
    print(accuracy_score(y_test, nb_predictions))
    print(nb_report)

    # LDA 
    lda_predictions = lda(X_train, y_train, X_test, y_test)
    lda_report = classification_report(y_test, lda_predictions)
    print("LDA:")
    print(accuracy_score(y_test, lda_predictions))
    print(lda_report)

    # QDA
    qda_predictions = qda(X_train, y_train, X_test, y_test)
    qda_report = classification_report(y_test, qda_predictions)
    print("QDA:")
    print(accuracy_score(y_test, qda_predictions))
    print(qda_report)

