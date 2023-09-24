import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import seaborn as snb

iris = snb.load_dataset('iris')
data = torch.tensor(iris[iris.columns[0:4]].values).float()

labels = torch.zeros(len(data),dtype=torch.long)
labels[iris.species == 'versicolor'] = 1
labels[iris.species == 'virginica'] = 2

fakedata = np.tile(np.array([1,2,3,4]),(10,1)) + np.tile(10*np.arange(1,11),(4,1)).T
fakelabels = np.arange(10) > 4
print(fakedata)
print(' ')
print(fakelabels)


train_data,test_data,train_labels,test_labels = train_test_split(fakedata,fakelabels,test_size=.2) # train_size, shuffle

print("Training data =>",train_data)
print("Train data shape =>",train_data.shape)
print("Training labels =>",train_labels)
print('===')
print("Test data =>",test_data)
print("Test data shape =>",test_data.shape)
print("Test labels =>",test_labels)


def createANNModel():
    ANNiris = nn.Sequential(
        nn.Linear(4,64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,3)
    )

    lossfn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ANNiris.parameters(),lr=.01)

    return ANNiris,lossfn,optimizer


ANNiris,lossfn,optimizer = createANNModel()

epochs = 1000
def trainTheModel(test_prop):

    train_acc = []
    test_acc = []
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_prop)

    for epochi in range(epochs):

        yHat = ANNiris(x_train)

        loss = lossfn(yHat,y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc.append(100*torch.mean((torch.argmax(yHat,axis=1) == y_train).float()).item())

        test_acc.append(100*torch.mean((torch.argmax(ANNiris(x_test),axis=1) == y_test).float()).item())

    return train_acc,test_acc

train_acc,test_acc = trainTheModel(.2)
print("train_acc =>",train_acc)
print(" ")
print("test_acc =>",test_acc)

# plot the results
fig = plt.figure(figsize=(10,5))

plt.plot(train_acc,'ro-')
plt.plot(test_acc,'bs-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend(['Train','Test'])
plt.show()


# train_prop experiment

train_props = np.linspace(.2,.95,10)
train_accs = np.zeros((len(train_props),epochs))
test_accs = np.zeros((len(train_accs),epochs))

for i in range(len(train_accs)):
    ANNiris,lossfn,optimizer = createANNModel()

    train_acc,test_acc = trainTheModel(1-train_props[i])

    train_accs[i,:] = train_acc
    test_accs[i,:] = test_acc


fig,ax = plt.subplots(1,2,figsize=(13,5))

ax[0].imshow(train_accs,aspect='auto',
             vmin=50,vmax=90, extent=[0,epochs,train_props[-1],train_props[0]])
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Training size proportion')
ax[0].set_title('Training accuracy')

p = ax[1].imshow(test_accs,aspect='auto',
             vmin=50,vmax=90, extent=[0,epochs,train_props[-1],train_props[0]])
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Training size proportion')
ax[1].set_title('Test accuracy')
fig.colorbar(p,ax=ax[1])

plt.show()
