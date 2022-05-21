# PRE PROCESSING DATA FOR MLP
import numpy as np
import pandas as pd
import activation_functions as act
import MLP

# DATA PRE PROCESSING
dictSexSelector = {
    'male': 0,
    'female': 1
}

titanicData = pd.read_csv("Data/train.csv", usecols=["PassengerId", "Survived", "Pclass", "Sex", "Age"])
titanicData['Sex'] = titanicData['Sex'].apply(lambda x: dictSexSelector[x])
features = titanicData[['Pclass', 'Sex', 'Age']].to_numpy()
response = titanicData[['Survived']].to_numpy()

# percep = MLP.MLP(features, response, act.sigmoid)
# CREATING AN INSTANCE OF MLP
xis = np.array([[1, 2],
                [3, 4]])
firsthidden = np.array([[0.1, 0.1],[0.1, 0.1]],dtype=object)
secondhidden = np.array([[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]],dtype=object)
hidden = np.array( [firsthidden, secondhidden, 0], dtype=object)
percep = MLP.MLP(xis, [[0, 1],[0, 1]], act.sigmoid, 0.2, 2, 3)
# percep.perceptron_colum(xis[1:],hidden[0])
# print(f"valor da func é {percep.perceptron_node(xis, 0.1) }")

# print(features)
# print(response)
