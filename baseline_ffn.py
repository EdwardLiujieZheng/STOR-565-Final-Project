import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras import Input 
from keras.layers import Dense
from sklearn.metrics import classification_report

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
    return np.array(encodings), np.array(labels)


def run_model(training_datapath, test_datapath):
    X_train, y_train = read_data(training_datapath)
    X_test, y_test = read_data(test_datapath)

    # Model Architecture 
    model = Sequential(name="Model-with-One-Input") # Model
    model.add(Input(shape=(27,), name='Input-Layer')) # Input Layer - need to speicfy the shape of inputs
    model.add(Dense(30, activation='softplus', name='Hidden-Layer')) # Hidden Layer, softplus(x) = log(exp(x) + 1)
    model.add(Dense(1, activation='sigmoid', name='Output-Layer')) # Output Layer, sigmoid(x) = 1 / (1 + exp(-x))


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
            epochs=3, 
            verbose='auto', 
            callbacks=None, 
            validation_split=0.2,  
            )


    # Prediction
    pred_labels_tr = (model.predict(X_train) > 0.5).astype(int)
    pred_labels_te = (model.predict(X_test) > 0.5).astype(int)


    #Performance Summary
    # print("")
    # print('-------------------- Model Summary --------------------')
    # model.summary() # print model summary
    # print("")
    # print('-------------------- Weights and Biases --------------------')
    # for layer in model.layers:
    #     print("Layer: ", layer.name) # print layer name
    #     print("  --Kernels (Weights): ", layer.get_weights()[0]) # weights
    #     print("  --Biases: ", layer.get_weights()[1]) # biases
        
    print("")
    print('---------- Evaluation on Training Data ----------')
    print(classification_report(y_train, pred_labels_tr))
    print("")

    print('---------- Evaluation on Test Data ----------')
    print(classification_report(y_test, pred_labels_te))
    print("")

if __name__ == "__main__":
    run_model(DATA, TEST)
