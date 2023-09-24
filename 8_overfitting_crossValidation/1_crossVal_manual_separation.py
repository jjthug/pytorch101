import torch
import torch.nn as nn
import numpy as np

import seaborn as sns
iris = sns.load_dataset('iris')

data = torch.tensor(iris[iris.columns[0:4]].values).float()
labels = torch.zeros(len(data), dtype = torch.long)
labels[iris.species == 'versicolor'] = 1
labels[iris.species == 'virginica'] = 2

propTraining = .8
nTraining =int(len(labels)*propTraining)
trainTestBool = np.zeros(len(labels),dtype=bool)

# trainTestBool[range(nTraining)] = True
# print(trainTestBool)

items2use4train = np.random.choice(range(len(labels)), nTraining, replace=False)
trainTestBool[items2use4train] = True

print('Average of full data:')
print(torch.mean(labels.float()))

print('Average of training data:')
print(torch.mean(labels[trainTestBool].float()))

print('Average of test data:')
print(torch.mean(labels[~trainTestBool].float()))


#NETWORK

ANNiris = nn.Sequential(
    nn.Linear(4,64),
    nn.ReLU(),
    nn.Linear(64,64),
    nn.ReLU(),
    nn.Linear(64,3),
    # nn.Softmax(dim=1) # no need
)

lossfn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(ANNiris.parameters(),lr=.01)

print(data.shape)
print(data[trainTestBool,:].shape)
print(data[~trainTestBool,:].shape)

# TRAINING
numepochs = 1000

losses = torch.zeros(numepochs)
ongoingAcc = []

for epochi in range(numepochs):
    yHat = ANNiris(data[trainTestBool,:])
    ongoingAcc.append(100 * torch.mean((torch.argmax(yHat,axis=1) == labels[trainTestBool]).float()))

    # compute loss
    loss = lossfn(yHat,labels[trainTestBool])
    losses[epochi] = loss

    #backward prop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# train
# ON TRAIN DATA
predictions = ANNiris(data[trainTestBool,:])
trainacc = 100*torch.mean((torch.argmax(predictions,axis=1) == labels[trainTestBool]).float())

#ON TEST DATA
predictions = ANNiris(data[~trainTestBool,:])
testacc = 100 * torch.mean((torch.argmax(predictions,axis=1) == labels[~trainTestBool]).float())

print("Final train acc =>",trainacc)
print("Final test acc =>",testacc)
