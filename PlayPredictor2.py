import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def PlayPredictor(data_path, kValues):

    #Step 1 : Loading data
    data = pd.read_csv(data_path, index_col=0)

    print("Actual Data size is ", len(data))

    #Step 2: Clean, prepare and manipulate the data
    feature_names = ['Whether', 'Temperature']

    print("Names of features", feature_names)

    Whether = data.Whether
    Temperature = data.Temperature
    Play = data.Play

    #create label Encoder
    le = preprocessing.LabelEncoder()

    #converting string into numbers
    Whether_encoded = le.fit_transform(Whether)
    print(Whether_encoded)

    Temp_encoded = le.fit_transform(Temperature)
    label = le.fit_transform(Play)
    print(Temp_encoded)

    #combine weather and temp into a single list of tuples
    features = list(zip(Whether_encoded, Temp_encoded))

    #Step 3: Train data
    model = KNeighborsClassifier(n_neighbors=3)

    #Training the model
    model.fit(features, label)

    #Step 4 : Testing the data
    predicted = model.predict([[0, 2]])
    print(predicted)

    #Step 5 : Check Accuracy   

    #Splitting the data
    data_train, data_test, target_train, target_test = train_test_split(features, label, test_size=0.5)

    accuracy_scores = []
    for k in kValues: 
        classifiers = KNeighborsClassifier(n_neighbors=k)
        classifiers.fit(data_train, target_train)
        #Making predictions to check accuracy
        predictions = classifiers.predict(data_test)
        accuracy = accuracy_score(target_test, predictions)
        accuracy_scores.append(accuracy)

    return accuracy_scores

def main():
    print("------HA14------")
    print("Play predictor ML case study")

    data_path = "C:/Users/DELL/OneDrive/Desktop/Programming/Python/HA/PlayPredictor.csv"

    kValues = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    Accuracy = PlayPredictor(data_path, kValues)

    for iCnt, k in enumerate(kValues):
        print(f"Accuracy for k={k}: {Accuracy[iCnt] * 100:.2f}%")

if __name__ == "__main__":
    main()
