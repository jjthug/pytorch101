# import libraries
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

# import dataset (comes with seaborn)
import seaborn as sns
iris = sns.load_dataset('iris')
print( iris.head() )

# some plots to show the data
sns.pairplot(iris, hue='species')
plt.show()

# organize the data

# convert from pandas dataframe to tensor
data = torch.tensor( iris[iris.columns[0:4]].values ).float()

# transform species to number
labels = torch.zeros(len(data), dtype=torch.long)
# labels[iris.species=='setosa'] = 0 # don't need!
labels[iris.species=='versicolor'] = 1
labels[iris.species=='virginica'] = 2

print(labels)

# Note the input into the function!
def createIrisModel(nHidden):

  # model architecture (with number of units soft-coded!)
  ANNiris = nn.Sequential(
      nn.Linear(4,nHidden),      # input layer
      nn.ReLU(),                 # activation unit
      nn.Linear(nHidden,nHidden),# hidden layer
      nn.ReLU(),                 # activation unit
      nn.Linear(nHidden,3),      # output unit
      #nn.Softmax(dim=1),        # final activation unit (here for conceptual purposes, note the CEL function)
        )

  # loss function
  lossfun = nn.CrossEntropyLoss()

  # optimizer
  optimizer = torch.optim.SGD(ANNiris.parameters(),lr=.01)

  return ANNiris,lossfun,optimizer


# a function to train the model

def trainTheModel(ANNiris):
    # initialize losses
    losses = torch.zeros(numepochs)
    ongoingAcc = []

    # loop over epochs
    for epochi in range(numepochs):
        # forward pass
        yHat = ANNiris(data)

        # compute loss
        loss = lossfun(yHat, labels)
        losses[epochi] = loss

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # final forward pass
    predictions = ANNiris(data)

    predlabels = torch.argmax(predictions, axis=1)
    return 100 * torch.mean((predlabels == labels).float())


numepochs  = 150
numhiddens = np.arange(1,129)
accuracies = []

for nunits in numhiddens:

  # create a fresh model instance
  ANNiris,lossfun,optimizer = createIrisModel(nunits)

  # run the model
  acc = trainTheModel(ANNiris)
  accuracies.append( acc )


# report accuracy
fig,ax = plt.subplots(1,figsize=(12,6))

ax.plot(accuracies,'ko-',markerfacecolor='w',markersize=9)
ax.plot(numhiddens[[0,-1]],[33,33],'--',color=[.8,.8,.8])
ax.plot(numhiddens[[0,-1]],[67,67],'--',color=[.8,.8,.8])
ax.set_ylabel('accuracy')
ax.set_xlabel('Number of hidden units')
ax.set_title('Accuracy')
plt.show()
