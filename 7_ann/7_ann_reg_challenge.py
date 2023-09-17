import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')


def buildAndTrainModel(x,y):
  ANNreg = nn.Sequential(
    nn.Linear(1,1),
    nn.ReLU(),
    nn.Linear(1,1)
  )
  lr = 0.05

  lossfn = nn.MSELoss()
  optimizer = torch.optim.SGD(ANNreg.parameters(),lr=lr)

  epochs = 500
  losses = torch.zeros(epochs)

  for i in range(epochs):
    pred = ANNreg(x)

    loss = lossfn(pred,y)
    losses[i] = loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  predictions = ANNreg(x)

  return predictions, losses


def createTheData(m):
  N = 500
  x = torch.randn(N,1)
  y = m*x + torch.randn(N,1)/2

  return x,y


m =45
x,y = createTheData(m)
pred,losses = buildAndTrainModel(x,y)


fix,ax = plt.subplots(1,2,figsize=(12,4))

ax[0].plot(losses.detach(),'bs',markerfacecolor = 'w',linewidth=.1,label='Predictions')
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')


ax[1].plot(x,y,'bo',label='real data')
ax[1].plot(x,pred.detach(),'rs',label='preds')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title(f'pred corr coeff => {np.corrcoef(y.T, pred.detach().T)[0,1]:.2f}')
ax[1].legend()

plt.show()



slopes = np.linspace(-2,2,21)
numExp = 50

results = torch.zeros(len(slopes),numExp,2)

for slopei in range(slopes):
  for N in range(numExp):
    x,y = createTheData(slopes[slopei])
    pred,loss = buildAndTrainModel(x,y)

    results[slopei,N,0] = loss[-1]
    results[slopei,N,1] = np.corrcoef(y.T,pred.detach().T)[0,1]


results[np.isnan(results)] = 0

# plot


fig,ax = plt.subplots(1,2,figsize=(12,4))

ax[0].plot(slopes,np.mean(results[:,:,0],axis=1),'ko-',markerfacecolor='w',markersize=10)
ax[0].set_xlabel('Slope')
ax[0].set_ylabel('Loss')

ax[1].plot(slopes,np.mean(results[:,:,1],axis=1),'ms-',markerfacecolor='w',markersize=10)
ax[1].set_xlabel('slope')
ax[1].set_ylabel('corr coef')
plt.plot()



