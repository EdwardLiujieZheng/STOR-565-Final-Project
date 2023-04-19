import os
import numpy as np
from keras import Input 
from random import shuffle
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import classification_report

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


def run_model(X_train, y_train, X_test, y_test):
    input_length = len(X_train[0])
    print(input_length)
    # Model Architecture 
    model = Sequential(name="SimpleFFN") # Model
    model.add(Input(shape=(input_length,), name='Input-Layer'))
    model.add(Dense(60, activation='relu', name='Hidden-Layer1')) 
    model.add(Dense(60, activation='softplus', name='Hidden-Layer2')) 
    model.add(Dense(60, activation='softplus', name='Hidden-Layer3')) 
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
            batch_size=10, 
            epochs=12, 
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


def run_ffn(TRAIN, TEST):
    X_train, y_train = read_data(TRAIN)
    X_test, y_test = read_data(TEST)
    run_model(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    TRAIN = "modified_train"
    TEST = "modified_test"

    run_ffn(TRAIN, TEST)
