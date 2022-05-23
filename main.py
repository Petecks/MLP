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
ypsu= np.array([[0, 1],
                [0, 1]])
firsthidden = np.array([[0.1, 0.1],[0.1, 0.1]],dtype=object)
secondhidden = np.array([[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]],dtype=object)
hidden = np.array( [firsthidden, secondhidden, 0], dtype=object)

percep = MLP.MLP(xis, ypsu, act.sigmoid, 0.2, 2, 3)
percep.test_model_epoch()
percep.training_mode()
# print(percep.test_model_epoch())



# for name in dir():
#     if not name.startswith('_'):
#         del globals()[name]