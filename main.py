# PRE PROCESSING DATA FOR MLP
import numpy as np
import pandas as pd
import activation_functions as act
import MLP

# DATA PRE PROCESSING

numData = pd.read_csv("Data/data.csv", header=None)
parameters = numData.to_numpy()

# percep = MLP.MLP(features, response, act.sigmoid)
# CREATING AN INSTANCE OF MLP
# xis = np.array([[1, 2],
#                 [3, 4]])
# ypsu= np.array([[0, 1],
#                 [0, 1]])
# firsthidden = np.array([[0.1, 0.1],[0.1, 0.1]],dtype=object)
# secondhidden = np.array([[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]],dtype=object)
# hidden = np.array( [firsthidden, secondhidden, 0], dtype=object)
#
# percep = MLP.MLP(xis, ypsu, act.sigmoid, 0.2, 2, 3)
# percep.test_model_epoch()
# percep.training_mode()
# d = np.reshape(parameters[:,0].transpose() ,(10,-1))
percep2 = MLP.MLP(parameters[:,1:],parameters[:,0], act.sigmoid, 0.1, 200, 10, 10, 10)
percep2.training_mode()

# print(percep.test_model_epoch())
print(percep2.test_model_epoch(1,1))


# for name in dir():
#     if not name.startswith('_'):
#         del globals()[name]