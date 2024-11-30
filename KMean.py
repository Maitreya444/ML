import numpy as np
import pandas as pd
from copy import deepcopy
from matplotlib import pyplot as plt

def MyKMean():
    print("---------------------------------------------")

    #Setting 3 centres, the model should predict similar results

    centre_1 = np.array([1,1])
    print(centre_1)

    print("---------------------------------------------")
    
    centre_2 = np.array([5,5])
    print(centre_2)

    print("---------------------------------------------")

    centre_3 = np.array([3,3])
    print(centre_3)

    print("---------------------------------------------")

    #Generate random data and centre it to three centres

    data_1 = np.random.randn(7,2) + centre_1
    print("Elements of first cluster with size"+str(len(data_1)))
    print(data_1)

    print("---------------------------------------------")

    data_2 = np.random.randn(7,2) + centre_2
    print("Elements of second cluster with size"+str(len(data_2)))
    print(data_2)

    print("---------------------------------------------")

    data_3 = np.random.randn(7,2) + centre_3
    print("Elements of third cluster with size"+str(len(data_3)))
    print(data_3)

    print("---------------------------------------------")
    
    data = np.concatenate((data_1, data_2, data_3), axis=0)
    print("Size of complete dataset "+str(len(data)))
    print(data)

    print("---------------------------------------------")

    plt.scatter(data[:,0], data[:,1], s=7)
    plt.title("Maitreya Gangurde : Input Dataset")
    plt.show()

    print("---------------------------------------------")
    #Number of Clusters

    k =3
    #Number of training data

    n = data.shape[0]
    print("Total number of elements are",n)
    print("---------------------------------------------")

    #Number of features in the data
    c = data.shape[1]
    print("Total number of features are",c)
    print("---------------------------------------------")

    #Generate random centres, here we use sigma and mean to ensure it represent the whole data

    mean = np.mean(data, axis=0)
    print("Value of mean",mean)
    print("---------------------------------------------")

    #Calculate stabdard deviation
    std = np.std(data,axis=0)
    print("Value of std",std)
    print("---------------------------------------------")

    centres = np.random.randn(k,c)*std+mean
    print("Random points are ",centres)
    print("---------------------------------------------")

    #Plot the data and centres generated as random

    plt.scatter(data[:,1], data[:,1], c='r', s=7)
    plt.scatter(centres[:,0], centres[:,1], marker='*', c='g', s=150)
    plt.title("Maitreya Gangurde : Input database with random centroid *")
    plt.show()
    print("---------------------------------------------")

    centres_old = np.zeros(centres.shape)   #To store old centres
    centres_new = deepcopy(centres)         #To store new centres

    print("Value of centroids")
    print(centres_new)
    print("---------------------------------------------")

    data.shape
    clusters = np.zeros(n)
    distances = np.zeros((n,k))

    print("Initial distances are")
    print(distances)
    print("---------------------------------------------")

    error = np.linalg.norm(centres_new -centres_old)
    print("value of error is ",error)
    #When after an update, the estimate of that centre stays the same, exit loop

    while error !=0:
        print("Value of error is ",error)
        #Measure the distance to every centre
        print("Measure the distance to every centre")
        for i in range(k):
            print("Iteration number",i)
            distances[:,i] = np.linalg.norm(data-centres[i], axis=1)

        #Assign all traing data to closest centre
        clusters = np.argmin(distances, axis=1)

        centres_old = deepcopy(centres_new)

        #Calculate mean for every cluster and update its centre

        for i in range(k):
            centres_new[i] = np.mean(data[clusters ==1],axis=0)
        error = np.linalg.norm(centres_new - centres_old)
    #End of while
    centres_new

    #Plot the data and centres generated as random

    plt.scatter(data[:,0], data[:,1], s=7)
    plt.scatter(centres_new[:,0], centres_new[:,1], marker='*', c='g', s=150)
    plt.title("Maitreya Ganurde : Final data with centroids")
    plt.show()

def main():
    print("------Maitreya Gangurde------")

    print("Unsupervised Machine Learning")

    print("Clustering using K Mean Algorithm")

    MyKMean()

if __name__ =="__main__":
    main()