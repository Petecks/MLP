# PRE PROCESSING DATA FOR MLP
import numpy as np
import pandas as pd
import activation_functions as act
import MLP
import matplotlib.pyplot as plt
import random
# DATA PRE PROCESSING

numData = pd.read_csv("Data/data.csv", header=None)
parameters = numData.to_numpy()
# randomize parameters
# random.shuffle(parameters)

#instances of different parameters
percep1 = MLP.MLP(parameters[:,1:],parameters[:,0], act.sigmoid, 0.5, 500 , 10, 10, 10)
#training with holdout setup
percep1.training_mode(1)
percep2 = MLP.MLP(parameters[:,1:],parameters[:,0], act.sigmoid, 0.5, 500 , 10, 10)
percep2.training_mode(1)
percep3 = MLP.MLP(parameters[:,1:],parameters[:,0], act.sigmoid, 0.5, 500 , 10, 5, 10)
percep3.training_mode(1)
percep4 = MLP.MLP(parameters[:,1:],parameters[:,0], act.sigmoid, 0.5, 500 , 5, 10)
percep4.training_mode(1)
percep5 = MLP.MLP(parameters[:,1:],parameters[:,0], act.sigmoid, 0.2, 500 , 10, 10, 10)
percep5.training_mode(1)
percep6 = MLP.MLP(parameters[:,1:],parameters[:,0], act.sigmoid, 0.2, 500 , 10, 10)
percep6.training_mode(1)
percep7 = MLP.MLP(parameters[:,1:],parameters[:,0], act.sigmoid, 0.2, 500 , 10, 5, 10)
percep7.training_mode(1)
percep8 = MLP.MLP(parameters[:,1:],parameters[:,0], act.sigmoid, 0.2, 500 , 5, 10)
percep8.training_mode(1)

# show the percentage of error for each input until 10
# for i in range(10):
#     print(percep1.test_model_epoch(i,1))

# print results of errors
plt.plot(percep1.error_count,label='setup 1')
plt.plot(percep2.error_count,label='setup 2')
plt.plot(percep3.error_count,label='setup 3')
plt.plot(percep4.error_count,label='setup 4')
plt.plot(percep5.error_count,label='setup 5')
plt.plot(percep6.error_count,label='setup 6')
plt.plot(percep7.error_count,label='setup 7')
plt.plot(percep8.error_count,label='setup 8')



################## HOLDOUT SETUP #########################
erro1=0
erro2=0
erro3=0
erro4=0
erro5=0
erro6=0
erro7=0
erro8=0
for i in range(105,210):
    shaped_d = np.zeros(10)
    shaped_d[parameters[i,0]] = 1
    max_index_d = np.argmax(shaped_d)
    out1 = percep1.test_model_epoch(i)
    max_index_y = np.argmax(out1)
    if max_index_d != max_index_y:
        erro1 += 1
    out2 = percep2.test_model_epoch(i)
    max_index_y = np.argmax(out2)
    if max_index_d != max_index_y:
        erro2 += 1
    out3 = percep3.test_model_epoch(i)
    max_index_y = np.argmax(out3)
    if max_index_d != max_index_y:
        erro3 += 1
    out4 = percep4.test_model_epoch(i)
    max_index_y = np.argmax(out4)
    if max_index_d != max_index_y:
        erro4 += 1
    out5 = percep5.test_model_epoch(i)
    max_index_y = np.argmax(out5)
    if max_index_d != max_index_y:
        erro5 += 1
    out6 = percep6.test_model_epoch(i)
    max_index_y = np.argmax(out6)
    if max_index_d != max_index_y:
        erro6 += 1
    out7 = percep7.test_model_epoch(i)
    max_index_y = np.argmax(out7)
    if max_index_d != max_index_y:
        erro7 += 1
    out8 = percep8.test_model_epoch(i)
    max_index_y = np.argmax(out8)
    if max_index_d != max_index_y:
        erro8 += 1
print("erro1",erro1,"erro2",erro2,"erro3",erro3,"erro4",erro4,"erro5",erro5,"erro6",erro6,"erro7",erro7,"erro8",erro8)

# print labels
plt.xlabel('Épocas')
plt.ylabel('Erros por época')
plt.legend(loc='upper right')
plt.show()