import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def MarvellousPlayPredictor(data_path):

    #Step 1 : Load the data
    data = pd.read_csv(data_path, index_col=0)

    print("Actual Size of the dataset", len(data))

    #Step 2 : Clean, Prepare and manipulate the data
    feature_names = ['Whether', 'Temperature']

    print("Names of features", feature_names)

    Whether = data.Whether
    Temperature = data.Temperature
    Play = data.Play

    #create label Encoder
    le = preprocessing.LabelEncoder()

    #Converting string label into numbers
    Whether_encoded=le.fit_transform(Whether)
    print(Whether_encoded)

    #Converting string label into numbers
    temp_encoded=le.fit_transform(Temperature)
    label=le.fit_transform(Play)

    print(temp_encoded)

    #combining weather and temp into single list of tuples
    features=list(zip(Whether_encoded, temp_encoded))

    #Step 3 : Train Data
    model = KNeighborsClassifier(n_neighbors=3)

    #Train the model using the training sets
    model.fit(features,label)

    #Step 4 : Test data
    predicted = model.predict([[0,2]]) #0 : Overcast, 2: Mild, 
    print(predicted)

def main():
    
    print("-----Maitreya Gangurde-----")

    print("Machine Learning Application")

    print("Play predictor application using K nearest algorithm")

    data_path = "C:/Users/DELL/OneDrive/Desktop/Programming/Python/Class/ML/PlayPredictor.csv"

    MarvellousPlayPredictor(data_path)

if __name__ =="__main__":
    main()    