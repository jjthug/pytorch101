import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from IPython import display

display.set_matplotlib_formats('svg')

nPerClust = 100
blur = 1

A = [1, 1]
B = [5, 1]

# generate the data

a = [A[0] + np.random.randn(nPerClust) * blur, A[1] + np.random.randn(nPerClust) * blur]
b = [B[0] + np.random.randn(nPerClust) * blur, B[1] + np.random.randn(nPerClust) * blur]

labels_np = np.vstack((np.zeros((nPerClust, 1)), np.ones((nPerClust, 1))))

data_np = np.hstack((a, b)).T

data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

fig = plt.figure(figsize=(5, 5))
plt.plot(data[np.where(labels == 0)[0], 0], data[np.where(labels == 0)[0], 1], 'bs')
plt.plot(data[np.where(labels == 1)[0], 0], data[np.where(labels == 1)[0], 1], 'ko')
plt.title('binary data, qwerties')
plt.xlabel('x dim')
plt.ylabel('y dim')
plt.show()


def createANNmodel(lr):
    ANNclassify = nn.Sequential(
        nn.Linear(2, 1),
        nn.ReLU(),
        nn.Linear(1, 1)
    )

    lossfn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(ANNclassify.parameters(), lr=lr)

    return ANNclassify, lossfn, optimizer


numEpochs = 1000


def trainTheModel(ANNmodel, lossfn, optimizer):
    losses = np.zeros(numEpochs)

    for epochi in range(numEpochs):
        yHat = ANNmodel(data)

        # loss
        loss = lossfn(yHat, labels)
        losses[epochi] = loss

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predictions = ANNmodel(data)

    totalAcc = 100 * torch.mean(((predictions>0) == labels).float())

    return losses, predictions, totalAcc




# create the experiment

lrs = np.linspace(0.001,.1,40)
accByLR = np.zeros((len(lrs),1))

allLosses = np.zeros((len(lrs),numEpochs))

for i,lr in enumerate(lrs):
    annClassify,lossfn,optimizer = createANNmodel(lr)
    losses, preds, totalAcc = trainTheModel(annClassify,lossfn,optimizer)

    accByLR[i] = totalAcc
    allLosses[i,:] = losses

print(allLosses)

fig,ax = plt.subplots(1,2,figsize=(5,5))

ax[0].plot(lrs,accByLR)
ax[0].set_xlabel('lr')
ax[0].set_ylabel('accuracy')
ax[0].set_title('accuracy by learning rate')

ax[1].plot(allLosses.T)
ax[1].set_title('losses by learning rate')
ax[1].set_xlabel('Epoch number')
ax[1].set_ylabel('loss')
plt.show()

print(sum(torch.tensor(accByLR)>70)/len(accByLR))


# run n number of experiments

numExp = 50

accMeta = np.zeros((numExp,len(lrs)))

for expi in range(numExp):
    for i,lr in enumerate(lrs):
        annModel,loss,optimizer = createANNmodel(lr)
        losses, predictions, totalAcc = trainTheModel(annModel,loss,optimizer)

        accMeta[expi,i] = totalAcc

plt.plot(lrs,np.mean(accMeta,axis=0),'s-')
plt.xlabel('lrs')
plt.ylabel('accuracy mean')
plt.title('Accuracy by lr averaged')
plt.show()

