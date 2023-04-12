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
    model.compile(optimizer='adam', # default='rmsprop', an algorithm to be used in backpropagation
                loss='binary_crossentropy', # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                metrics=['Accuracy', 'Precision', 'Recall'], # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. 
                loss_weights=None, # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
                weighted_metrics=None, # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
                run_eagerly=None, # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
                steps_per_execution=None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
                )


    # Fit model
    model.fit(X_train, 
            y_train, 
            batch_size=10, 
            epochs=3, 
            verbose='auto', # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
            callbacks=None, # default=None, list of callbacks to apply during training. See tf.keras.callbacks
            validation_split=0.2, # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. 
            steps_per_epoch=None, # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined. 
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
