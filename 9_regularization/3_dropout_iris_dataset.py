# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset

import seaborn as sns
iris = sns.load_dataset('iris')

data = torch.tensor( iris[iris.columns[0:4]].values ).float()
# transform species to number
labels = torch.zeros(len(data), dtype=torch.long)
# labels[iris.species=='setosa'] = 0 # don't need!
labels[iris.species=='versicolor'] = 1
labels[iris.species=='virginica'] = 2

print(data)
print(labels)

# fig = plt.figure(figsize=(5,5))
# plt.plot(data[np.where(labels==0)[0],0],data[np.where(labels==0)[0],1],'bs')
# plt.plot(data[np.where(labels == 1)[0],0], data[np.where(labels==1)[0],1],'ko')
# plt.title("The qwerties' doughnuts!")
# plt.xlabel('qwerty dimension 1')
# plt.ylabel('qwerty dimension 2')
# plt.show()


# split the data

train_data,test_data,train_labels,test_labels= train_test_split(data,labels,test_size=.2)

train_data = TensorDataset(train_data,train_labels)
test_data = TensorDataset(test_data,test_labels)

#loaders

batchsize =16
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True)
test_loader = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])


# CREATE THE MODEL CLASS

class theModelClass(nn.Module):
    def __init__(self,dropoutRate):
        super().__init__()

        # layers
        self.input = nn.Linear(4,12)
        self.hidden = nn.Linear(12,12)
        self.output = nn.Linear(12,3)

        self.dr = dropoutRate

    def forward(self,x):
        x = F.relu(self.input(x))
        x = F.dropout(x,p=self.dr,training=self.training)
        x = F.relu(self.hidden(x))
        x = F.dropout(x,p=self.dr,training=self.training)
        x = self.output(x)
        ## no dropout here
        return x


## test the model
tmpnet = theModelClass(.25)

tmpdata = torch.randn((10,4))

yHat = tmpnet.forward(tmpdata)
print("yHat=>",yHat)



def createANewModel(dropoutrate):
    ANNQC = theModelClass(dropoutrate)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(ANNQC.parameters(),lr=0.002)

    return ANNQC,loss_fn,optimizer

dropoutrate = .5
ANNQC,lossfn,optimizer = createANewModel(dropoutrate)


numepochs = 1000


def trainTheModel(ANNQC,lossfn,optimizer):
    # initialize accuracies as empties (not storing losses here)
    trainAcc = []
    testAcc = []

    # loop over epochs
    for epochi in range(numepochs):

        # switch learning on
        ANNQC.train()

        # loop over training data batches
        batchAcc = []
        for X, y in train_loader:
            # forward pass and loss
            yHat = ANNQC(X)
            loss = lossfn(yHat, y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute training accuracy just for this batch
            batchAcc.append(100 * torch.mean((torch.argmax(yHat, axis=1) == y).float()).item())
        # end of batch loop...

        # now that we've trained through the batches, get their average training accuracy
        trainAcc.append(np.mean(batchAcc))

        # test accuracy
        ANNQC.eval()
        X, y = next(iter(test_loader))  # extract X,y from test dataloader
        predlabels = torch.argmax(ANNQC(X), axis=1)
        testAcc.append(100 * torch.mean((predlabels == y).float()).item())

    # function output
    return trainAcc, testAcc

# def trainTheModel(ANNQC,lossfn,optimizer):
#     trainAcc = []
#     testAcc = []
#
#     for epochi in range(numepochs):
#         ANNQC.train()
#
#         batchAcc = []
#
#         for x,y in train_loader:
#             yHat = ANNQC.forward(x)
#             loss = lossfn(yHat,y)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             batchAcc.append(100*torch.mean((torch.argmax(yHat,axis=1) ==y).float()).item())
#
#         trainAcc.append(np.mean(batchAcc))
#
#         ANNQC.eval()
#         x,y = next(iter(test_loader))
#         yHat = ANNQC(x)
#         testAcc.append(100*torch.mean((torch.argmax(yHat,axis=1) == y).float()).item())
#
#     # function output
#     return trainAcc, testAcc

# create a model
dropoutrate = .0

# train the model (note the inputs!)
trainAcc,testAcc = trainTheModel(ANNQC,lossfn,optimizer)

# create a 1D smoothing filter
def smooth(x,k=5):
  return np.convolve(x,np.ones(k)/k,mode='same')

# plot the results
fig = plt.figure(figsize=(10,5))
plt.plot(smooth(trainAcc),'bs-')
plt.plot(smooth(testAcc),'ro-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend(['Train','Test'])
plt.title('Dropout rate = %g'%dropoutrate)

plt.show()


# run an experiment (note: takes ~3.5 mins)

dropoutRates = np.arange(10)/10
results = np.zeros((len(dropoutRates),2))

for di in range(len(dropoutRates)):

  # create and train the model
  ANNQC,lossfun,optimizer = createANewModel(dropoutRates[di])
  trainAcc,testAcc = trainTheModel(ANNQC,lossfun,optimizer)

  # store accuracies from last 100 epochs
  results[di,0] = np.mean(trainAcc[-100:])
  results[di,1] = np.mean(testAcc[-100:])


# plot the experiment results
fig,ax = plt.subplots(1,2,figsize=(15,5))

ax[0].plot(dropoutRates,results,'o-')
ax[0].set_xlabel('Dropout proportion')
ax[0].set_ylabel('Average accuracy')
ax[0].legend(['Train','Test'])

ax[1].plot(dropoutRates,-np.diff(results,axis=1),'o-')
ax[1].plot([0,.9],[0,0],'k--')
ax[1].set_xlabel('Dropout proportion')
ax[1].set_ylabel('Train-test difference (acc%)')

plt.show()