from sklearn import *

def main():
    print("Ball classification case study")

    #Load the data
    BallFeatures = [[35, "Rough"], [47, "Rough"], [90, "Smooth"], [48, "Rough"], [90, "Smooth"], [35, "Rough"],[92, "Smooth"],[35, "Rough"],[96,"Smooth"], [43, "Rough"], [110, "Smooth"], [35, "Rough"], [95, "Smooth"]]
    
    Labels = ["Tennis","Tennis","Cricket","Tennis","Cricket","Tennis", "Cricket","Tennis","Tennis","Tennis","Cricket","Tennis","Cricket", "Tennis","Cricket"]

    obj = tree.DecisionTreeClassifier()         #Decide the algorithm

    obj = obj.fit(BallFeatures, Labels)         #Train the model

    print(obj.predict([[36,"Rough"], [91, "Smooth"]])) #Test the model

if __name__ =="__main__":
    main()