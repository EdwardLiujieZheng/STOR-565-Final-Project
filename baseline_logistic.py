import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score


DATA = "training"
TEST = "test"

def read_data(PATH):
    with open(PATH, "r", encoding="utf-8") as f:
        tourneys = f.readlines()
    
    encodings = []
    labels = []
    tourneys = [tourney.strip('\n') for tourney in tourneys]
    for tourney in tourneys:
        encoding,label = tourney.split(";")
        encoding = [float(e) for e in encoding.split(",")]
        encodings.append(encoding)
        labels.append(int(label))
    return encodings, labels


def assemble_matrices(encodings, labels):
    return np.array(encodings), np.array(labels)


def logistic_regression(training_datapath, test_datapath):
    train_encodings, train_labels = read_data(training_datapath)
    X, Y = assemble_matrices(train_encodings, train_labels)

    test_encodings, test_labels = read_data(test_datapath)
    testX, testY = assemble_matrices(test_encodings, test_labels)

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X, Y)

    prediction = model.predict(testX)
    print(prediction)
    score = model.score(testX, testY)
    print(score)

if __name__ == "__main__":
    logistic_regression(DATA, TEST)

    # Split the dataset into training and testing sets
    trainX, trainY = read_data(DATA)
    X_train, y_train = assemble_matrices(trainX, trainY)
    testX, testY = read_data(TEST)
    X_test, y_test = assemble_matrices(testX, testY)

    # Create and train a QDA classifier
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)

    # Create and train an LDA classifier
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Use the trained classifiers to make predictions on the test data
    qda_pred = qda.predict(X_test)
    lda_pred = lda.predict(X_test)

    # Calculate the accuracy of the classifiers
    qda_acc = accuracy_score(y_test, qda_pred)
    lda_acc = accuracy_score(y_test, lda_pred)

    print("QDA Accuracy:", qda_acc)
    print("LDA Accuracy:", lda_acc)
