import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.layers import Input # type: ignore
from tensorflow.keras.metrics import MeanSquaredError # type: ignore
import pandas as pd
import matplotlib.pyplot as plt

# Read dataset 
dataset = pd.read_csv("alzheimers_disease_data.csv")

# Αφαίρεση μη χρήσιμων στηλών
dataset = dataset.drop(columns=["PatientID", "DoctorInCharge"])

# Split into input and output
X = dataset.drop(columns=["Diagnosis"])
Y = dataset["Diagnosis"]

# Features normalization
X = StandardScaler().fit_transform(X=X)

I = X.shape[1]  # Number of columns
H_list = [I // 2, 2 * I//3, I, 2 * I]  # Number of neurons in the hidden layer

H = I // 2 # Number of neurons in the hidden layer

# Split the data to training and testing data 5-Fold
kfold = KFold(n_splits=5, shuffle=True)


h_list = [0.001, 0.001, 0.05, 0.1]
m_list = [0.2, 0.6, 0.6, 0.6]

h = 0.001
m = 0.2

for j in range(4):
    # print("Learning rate: ", h_list[j], " Momentum: ", m_list[j])
    print("Number of neurons in the hidden layer: ", H_list[j])
    for i, (train, test) in enumerate(kfold.split(X)):
        lossList = []
        mseList = []
        accuracyList = []
        
        # Create model
        model = Sequential()

        model.add(Input(shape=(I,)))
        model.add(Dense(H_list[j], activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        # Compile model
        optimizer = keras.optimizers.SGD(learning_rate=h, momentum=m)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[MeanSquaredError(), 'accuracy'])

        # Fit model
        model.fit(X[train], Y[train], epochs=32, batch_size=32, verbose=0)

        # Evaluate model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        lossList.append(scores[0])  # Append loss, which is the first element in scores
        mseList.append(scores[1])   # Append MSE, which is the second element in scores
        accuracyList.append(scores[2])  # Append accuracy, which is the third element in scores
        print("Fold :", i, " Loss:", scores[0], " MSE:", scores[1], " Accuracy:", scores[2])

    print("Loss: ", np.mean(lossList))
    print("MSE: ", np.mean(mseList))
    print("Accuracy: ", np.mean(accuracyList))
    print("\n")

