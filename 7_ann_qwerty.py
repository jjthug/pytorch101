import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

from IPython import display
display.set_matplotlib_formats('svg')

# create the data
nPerCluster = 100
blur = 1

A = [ 1, 1 ]
B = [ 5, 1 ]

# generate data

a = [A[0] + np.random.randn(nPerCluster)*blur, A[1] + np.random.randn(nPerCluster)*blur]
b = [B[0] + np.random.randn(nPerCluster)*blur, B[1] + np.random.randn(nPerCluster)*blur]

labels_np = np.vstack((np.zeros((nPerCluster,1)), np.ones((nPerCluster,1))))

data_np = np.hstack((a,b)).T

data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

fig = plt.figure(figsize=(5,5))
plt.plot(data[np.where(labels==0)[0],0], data[np.where(labels==0)[0],1],'bs')
plt.plot(data[np.where(labels==1)[0],0], data[np.where(labels==1)[0],1],'ko')
plt.title('the qwerties')
plt.xlabel('x dimension')
plt.ylabel('y dimension')
plt.show()


print(type(data_np))
print(np.shape(data_np))
print(' ')

print(type(data))
print(np.shape(data))





##

ANNclassify = nn.Sequential(
    nn.Linear(2,1),
    nn.ReLU(),
    nn.Linear(1,1),
    nn.Sigmoid()
)

lr = 0.1
lossfn = nn.BCELoss()

optimizer = torch.optim.SGD(ANNclassify.parameters(),lr=lr)

# train the model

numEpochs = 1000

losses = np.zeros(numEpochs)

for epochi in range(numEpochs):
    yHat = ANNclassify(data)
    loss = lossfn(yHat,labels)
    losses[epochi] = loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# show the losses

plt.plot(losses,'o',markerfacecolor = 'w', linewidth = .1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


#compute the predictions
predictions = ANNclassify(data)

predlabels = predictions>.5

misclassified = np.where(predlabels != labels)[0]

 # total accuracy
totalacc = 100-(100-len(misclassified))/nPerCluster*2

# plot
fig = plt.figure(figsize=(5,5))

plt.plot(data[misclassified,0], data[misclassified,1], 'rx', markersize=12)
plt.plot(data[np.where(~predlabels)[0],0], data[np.where(~predlabels)[0],1], 'bs')
plt.plot(data[np.where(predlabels)[0],0], data[np.where(predlabels)[0],1], 'ko')

plt.legend(['Misclassified,blue,black'])
plt.title(f'Total accuracy {totalacc}')
plt.show()