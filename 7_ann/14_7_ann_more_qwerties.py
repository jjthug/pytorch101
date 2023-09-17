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
C = [ 9, 1]

# generate data

a = [A[0] + np.random.randn(nPerCluster)*blur, A[1] + np.random.randn(nPerCluster)*blur]
b = [B[0] + np.random.randn(nPerCluster)*blur, B[1] + np.random.randn(nPerCluster)*blur]
c = [C[0] + np.random.randn(nPerCluster)*blur, C[1] + np.random.randn(nPerCluster)*blur]

labels_np = np.vstack((np.zeros((nPerCluster,1)), np.ones((nPerCluster,1)), 2*np.ones((nPerCluster,1))))
data_np = np.hstack((a,b,c)).T

data = torch.tensor(data_np).float()
labels_mid = torch.tensor(labels_np).long()
print("size of labels mid=",labels_mid.size())
labels = torch.squeeze(labels_mid)
print("size of labels =",labels.size())

print(labels)

fig = plt.figure(figsize=(5,5))
plt.plot(data[np.where(labels==0)[0],0], data[np.where(labels==0)[0],1],'bs')
plt.plot(data[np.where(labels==1)[0],0], data[np.where(labels==1)[0],1],'ko')
plt.plot(data[np.where(labels==2)[0],0], data[np.where(labels==2)[0],1],'ro')
plt.title('the qwerties')
plt.xlabel('x dimension')
plt.ylabel('y dimension')
plt.show()

ANNclassify = nn.Sequential(
    nn.Linear(2,18),
    nn.ReLU(),
    nn.Linear(18,3)
)

# loss fn
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(ANNclassify.parameters(),lr=0.1)


#training

numEpochs = 1000
losses = torch.zeros(numEpochs)
ongoingAcc = []


for epochi in range(numEpochs):
    yHat = ANNclassify(data)

    loss = loss_fn(yHat,labels)
    losses[epochi] = loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    matches = torch.argmax(yHat,axis=1) == labels
    matchesNumeric = matches.float()
    accPct = 100 * torch.mean(matchesNumeric)
    ongoingAcc.append(accPct)

# predictions after training
predictions = ANNclassify(data)

# final accuracy
matches = torch.argmax(predictions,axis=1) == labels
matchesNumeric = matches.float()
totalacc = 100 * torch.mean(matchesNumeric)
print("final accuracy =",totalacc)


print('Final accuracy: %g%%' %totalacc)

fig,ax = plt.subplots(1,2,figsize=(13,4))

ax[0].plot(losses.detach())
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('epoch')
ax[0].set_title('Losses')

ax[1].plot(ongoingAcc)
ax[1].set_ylabel('accuracy')
ax[1].set_xlabel('epoch')
ax[1].set_title('Accuracy')
plt.show()

# confirm that all model predictions sum to 1, but only when converted to softmax
sm = nn.Softmax(1)
print(torch.sum(sm(yHat),axis=1))

fig = plt.figure(figsize=(10,4))

plt.plot(yHat.detach(),'s-',markerfacecolor='w')
plt.xlabel('Stimulus number')
plt.ylabel('Probability')
plt.legend(['1','2','3'])
plt.show()
