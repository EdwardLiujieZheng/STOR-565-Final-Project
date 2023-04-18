import os
import numpy as np
from keras import Input 
from random import shuffle
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import classification_report

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

    return np.array(encodings), np.array(labels)


def run_model(X_train, y_train, X_test, y_test):
    input_length = len(X_train[0])
    print(input_length)
    # Model Architecture 
    model = Sequential(name="SimpleFFN") # Model
    model.add(Input(shape=(input_length,), name='Input-Layer'))
    model.add(Dense(30, activation='relu', name='Hidden-Layer1')) 
    model.add(Dense(30, activation='softplus', name='Hidden-Layer2')) 
    model.add(Dense(1, activation='sigmoid', name='Output-Layer')) 

    # Compile model
    model.compile(optimizer='adam',
                loss='binary_crossentropy', 
                metrics=['Accuracy', 'Precision', 'Recall'], 
                run_eagerly=None,
                steps_per_execution=None 
                )


    # Fit model
    model.fit(X_train, 
            y_train, 
            batch_size=5, 
            epochs=10, 
            verbose='auto', 
            callbacks=None, 
            validation_split=0.2,  
            )


    # Prediction
    predictions = model.predict(X_test)
    pred_labels_te = (predictions > 0.5).astype(int)

    print(predictions)
    print(pred_labels_te)

    print("")
    print('-------------------- Model Summary --------------------')
    model.summary() # print model summary
    print("")

    print('---------- Evaluation on Test Data ----------')
    print(classification_report(y_test, pred_labels_te))
    print("")


if __name__ == "__main__":
    TRAIN = "pooled_train"
    TEST = "pooled_test"
    X_train, y_train = read_data(TRAIN)
    X_test, y_test = read_data(TEST)

    run_model(X_train, y_train, X_test, y_test)
