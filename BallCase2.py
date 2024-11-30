from sklearn import *

#Rough      1
#Smooth     0

def main():
    print("Ball classification case study")

    #Load the data
    BallFeatures = [[35, 1], [47, 1], [90, 0], [48, 1], [90, 0], [35, 1],[92, 0],[35, 1],[96,0], [43, 1], [110, 0], [35, 1], [95, 0], [36,1], [82,0]]
    
    Labels = ["Tennis","Tennis","Cricket","Tennis","Cricket","Tennis", "Cricket","Tennis","Tennis","Tennis","Cricket","Tennis","Cricket", "Tennis","Cricket"]

    obj = tree.DecisionTreeClassifier()         #Decide the algorithm

    obj = obj.fit(BallFeatures, Labels)         #Train the model

    print(obj.predict([[36,"1"], [91, "0"]]))   #Test the model

if __name__ =="__main__":
    main()
