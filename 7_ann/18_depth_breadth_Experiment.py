import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

import torch.nn.functional as F

from IPython import display
display.set_matplotlib_formats('svg')

# import dataset (comes with seaborn)
import seaborn as sns
iris = sns.load_dataset('iris')
print( iris.head() )

# some plots to show the data
# sns.pairplot(iris, hue='species')
# plt.show()


data = torch.tensor(iris[iris.columns[0:4]].values).float()
labels = torch.zeros(len(data), dtype=torch.long)
# setosa by default = 0
labels[iris.species == 'versicolor'] = 1
labels[iris.species == 'virginica'] = 2

class ANNiris(nn.Module):
    def __init__(self,nUnits,nLayers):
        super().__init__()

        # create a dictionary to store layers
        self.layers = nn.ModuleDict()
        self.nLayers= nLayers

        ## input layer
        self.layers['input'] = nn.Linear(4,nUnits)

        ## hidden layer
        for i in range(nLayers):
            self.layers[f'hidden{i}'] = nn.Linear(nUnits,nUnits)

        # output layer
        self.layers['output'] = nn.Linear(nUnits,3)


    def forward(self,x):
        x = self.layers['input'](x)

        for i in range(self.nLayers):
            x = F.relu( self.layers[f'hidden{i}'](x) )

        x = self.layers['output'](x)

        # x = F.softmax(x) no need as we are using the crossEntropyLoss function which does softmax to the output internally

        return x


nUnitsPerLayer = 12
nLayers = 4
net = ANNiris(nUnitsPerLayer,nLayers)
print("net=",net)


# quick test

tmpx = torch.randn(10,4)

y = net(tmpx)

print("y.shape=",y.shape)

print("y=",y)


# TRAINING

epochsNum = 100

def trainTheModel(theModel):
    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(theModel.parameters(),lr = .01)

    # loop over epochs
    for i in range(epochsNum):
        yHat = theModel(data)

        loss = lossfun(yHat,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predictions = net(data)
    predlabels = torch.argmax(predictions,axis=1)
    acc = 100 * torch.mean((predlabels == labels).float())

    # number of traininable params
    nParams = sum([p.numel() for p in net.parameters() if p.requires_grad])

    return acc, nParams


# todo im getting 1971 trainable parameters, supposedly its 723
acc = trainTheModel(net)
print("accuracy=>",acc)


## define the model params
numlayers = range(1,6)
numunits = np.arange(4,101,3)

accuracies = np.zeros((len(numunits),len(numlayers)))
totalparams = np.zeros((len(numunits),len(numlayers)))

numEpochs = 500

for unitidx in range(len(numunits)):
    for layeridx in range(len(numlayers)):

        net = ANNiris(numunits[unitidx],numlayers[layeridx])

        acc, numParams = trainTheModel(net)
        accuracies[unitidx,layeridx] = acc

        totalparams[unitidx,layeridx] = numParams


fig,ax = plt.subplots(1,figsize=(12,6))

ax.plot(numunits,accuracies,'o-',markerfacecolor='w',markersize=9)
ax.plot(numunits[[0,-1]],[33,33],'--',color=[.8,.8,.8])
ax.plot(numunits[[0,-1]],[67,67],'--',color=[.8,.8,.8])
ax.legend(numlayers)
ax.set_ylabel('accuracy')
ax.set_xlabel('Number of hidden units')
ax.set_title('Accuracy')
plt.show()