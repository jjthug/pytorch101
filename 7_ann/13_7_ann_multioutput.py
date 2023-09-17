import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')

#import iris dataset
import seaborn as sns
iris = sns.load_dataset('iris')

# check out the first few lines
print(iris.head())

sns.pairplot(iris, hue='species')
plt.show()

data = torch.tensor(iris[iris.columns[0:4]].values).float()

labels = torch.zeros(len(data), dtype=torch.long)
print("size_of_labels=",labels.size())
# no need for setosa labels = 0
labels[iris.species == 'versicolor'] = 1
labels[iris.species == 'virginica'] = 2

print(f"labels=",labels)


# create the ann

ANNiris = nn.Sequential(
    nn.Linear(4,64),
    nn.ReLU(),
    nn.Linear(64,64),
    nn.ReLU(),
    nn.Linear(64,4)
)

# loss fn
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(ANNiris.parameters(),lr=0.1)



#training

numEpochs = 1000
losses = torch.zeros(numEpochs)
ongoingAcc = []


for epochi in range(numEpochs):
    yHat = ANNiris(data)

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
predictions = ANNiris(data)

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
plt.legend(['setosa','versicolor','virginica'])
plt.show()
