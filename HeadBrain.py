import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def HeadBrainPredictor():
    #Loading Data
    data = pd.read_csv("MarvellousHeadBrain.csv")

    print("Size of the dataset", data.shape)

    X = data['Head Size(cm^3)'].values
    Y = data['Brain Weight(grams)'].values

    #Least Square method
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    n = len(X)

    numerator = 0
    denominator = 0

    #Equation of line is Y = mx + c

    for i in range(n):
        numerator += (X[i] -mean_x)*(Y[i]-mean_y)
        denominator += (X[i] - mean_x)**2

    m = numerator / denominator

    c = mean_y - (m*mean_x)

    print("Slope of regression line is",m)
    print("Y intercept of regression line is",c)

    max_x = np.max(X)+100
    min_x = np.min(X)-100

    #Display plotting of above points
    x = np.linspace(min_x, max_x, n)

    y = c + m*x 

    plt.plot(x,y,color='#58b790', label='Regression Line')

    plt.scatter(X,Y, color='#ef5423', label='scatter plot')

    plt.xlabel('Head size in cm3')

    plt.ylabel('Brain weight in gram')

    plt.legend()
    plt.show()

    #Findout goodness of fit ie R Square

    ss_t = 0
    ss_r = 0

    for i in range(n):
        y_pred = c + m * X[i]
        ss_t +=(Y[i] - mean_y)**2
        ss_r +=(Y[i] - y_pred)**2

    r2 = 1-(ss_r/ss_t)

    print(r2)

def main():
    print("-----------------------------------------")

    print("Supervised Machine Learning")

    print("Linear Regression on head and brain size dataset")

    HeadBrainPredictor()

if __name__=="__main__":
    main()